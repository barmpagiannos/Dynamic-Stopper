# Periodically pause a running Autowalk every M meters for N seconds,
# perform a lateral sidestep (left/right/alternate) using body-frame v_y,
# and run a background CPU-load thread (optional) to simulate light workload.

from google.protobuf.message import Message
import argparse
import logging
import math
import os
import sys
import time
import threading
from typing import List, Tuple

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.autowalk import AutowalkClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient, power_on_motors
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.mission.client import MissionClient
from bosdyn.client.time_sync import TimedOutError

from bosdyn.api import image_pb2
from bosdyn.api.autowalk import walks_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2
from bosdyn.api.mission import mission_pb2
from bosdyn.api.graph_nav import nav_pb2 as gnav_nav_pb2

# frame helpers (name differs across SDKs)
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME
try:
    from bosdyn.client.frame_helpers import get_se3_a_tform_b as get_tf
except ImportError:
    from bosdyn.client.frame_helpers import get_a_tform_b as get_tf

LOGGER = logging.getLogger("dynamic_stopper")


# ---------- background CPU-load thread ----------
def bg_load_worker(stop_event: threading.Event, load_pct: float, cycle_ms: int) -> None:
    """
    Very light synthetic CPU load.
    Each cycle of length 'cycle_ms':
      - stays busy for (load_pct * cycle_ms)
      - sleeps for the remainder.
    Also does a tiny NumPy op to touch the FPU/CPU a bit.
    """
    load_pct = max(0.0, min(1.0, float(load_pct)))
    cycle_s = max(0.005, cycle_ms / 1000.0)  # at least 5 ms cycle
    busy_s = cycle_s * load_pct
    idle_s = max(0.0, cycle_s - busy_s)

    # tiny array (fast)
    arr = np.random.rand(64).astype(np.float32)

    LOGGER.info("Background load thread started (%.0f%%, cycle=%d ms).", load_pct * 100, cycle_ms)
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            # Busy slice
            while (time.perf_counter() - t0) < busy_s and not stop_event.is_set():
                # small math
                arr = arr * 1.0001 + 0.0001  # trivial vector op
                arr.sum()  # prevent optimization
            # Idle slice
            if idle_s > 0:
                stop_event.wait(idle_s)
    finally:
        LOGGER.info("Background load thread exiting.")


# ---------- I/O helpers ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_walk_from_disk(walk_dir: str, walk_filename: str) -> Message:
    path = os.path.join(walk_dir, "missions", walk_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing mission file: {path}")
    walk = walks_pb2.Walk()
    with open(path, "rb") as f:
        walk.ParseFromString(f.read())
    return walk


def upload_graph_and_snapshots(logger: logging.Logger, graph_nav: GraphNavClient, walk_dir: str) -> None:
    graph_file = os.path.join(walk_dir, "graph")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Missing graph file: {graph_file}")

    graph = map_pb2.Graph()
    with open(graph_file, "rb") as f:
        graph.ParseFromString(f.read())
    logger.info("Graph has %d waypoints / %d edges", len(graph.waypoints), len(graph.edges))

    wp_snaps = {}
    for wp in graph.waypoints:
        if not wp.snapshot_id:
            continue
        p = os.path.join(walk_dir, "waypoint_snapshots", wp.snapshot_id)
        with open(p, "rb") as f:
            s = map_pb2.WaypointSnapshot()
            s.ParseFromString(f.read())
            wp_snaps[s.id] = s

    edge_snaps = {}
    for e in graph.edges:
        if not e.snapshot_id:
            continue
        p = os.path.join(walk_dir, "edge_snapshots", e.snapshot_id)
        with open(p, "rb") as f:
            s = map_pb2.EdgeSnapshot()
            s.ParseFromString(f.read())
            edge_snaps[s.id] = s

    no_anchors = not len(graph.anchoring.anchors)
    resp = graph_nav.upload_graph(graph=graph, generate_new_anchoring=no_anchors)
    logger.info("Uploaded graph.")
    for sid in resp.unknown_waypoint_snapshot_ids:
        graph_nav.upload_waypoint_snapshot(waypoint_snapshot=wp_snaps[sid])
    for sid in resp.unknown_edge_snapshot_ids:
        graph_nav.upload_edge_snapshot(edge_snapshot=edge_snaps[sid])
    logger.info("Uploaded snapshots.")


# ---------- ODOM helpers ----------
def get_odom_xy(state_client: RobotStateClient) -> Tuple[float, float]:
    state = state_client.get_robot_state()
    snap = state.kinematic_state.transforms_snapshot
    odom_T_body = get_tf(snap, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    return float(odom_T_body.x), float(odom_T_body.y)


def distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def stop_now(cmd_client: RobotCommandClient) -> None:
    cmd_client.robot_command(RobotCommandBuilder.stop_command())


# ---------- image capture (optional) ----------
def save_spot_image(resp, out_dir: str, prefix: str = "") -> str:
    img = resp.shot.image
    t_ms = int(time.time() * 1000)
    name = resp.source.name.replace("/", "_")
    base = f"{prefix + '_' if prefix else ''}{name}_{t_ms}"

    if img.format == image_pb2.Image.FORMAT_JPEG:
        p = os.path.join(out_dir, f"{base}.jpg")
        with open(p, "wb") as f:
            f.write(img.data)
        return p

    if img.format == image_pb2.Image.FORMAT_RAW and img.pixel_format in (
        image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16,
    ):
        arr = np.frombuffer(img.data, dtype=np.uint16).reshape((img.rows, img.cols))
        p = os.path.join(out_dir, f"{base}.png")
        cv2.imwrite(p, arr)
        return p

    buf = np.frombuffer(img.data, dtype=np.uint8)
    decoded = cv2.imdecode(buf, -1)
    if decoded is not None:
        if decoded.ndim == 2:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        p = os.path.join(out_dir, f"{base}.jpg")
        cv2.imwrite(p, decoded)
        return p

    raise RuntimeError(f"Unsupported image format: {resp.source.name}")


def capture_images(image_client: ImageClient, sources: List[str], out_root: str,
                   pause_idx: int, jpeg_quality: int = 85) -> List[str]:
    out_dir = os.path.join(out_root, f"pause_{pause_idx:04d}")
    ensure_dir(out_dir)
    reqs = [build_image_request(s, quality_percent=jpeg_quality, resize_ratio=1.0) for s in sources]
    try:
        resps = image_client.get_image(reqs)
    except TimedOutError:
        resps = image_client.get_image(reqs)
    saved = []
    for r in resps:
        try:
            saved.append(save_spot_image(r, out_dir, prefix=f"pause{pause_idx:04d}"))
        except Exception as e:
            LOGGER.warning("Save failed for %s: %s", r.source.name, e)
    return saved


# ---------- lateral step via body-frame v_y ----------
def perform_lateral_step(cmd_client: RobotCommandClient,
                         meters: float,
                         speed_mps: float,
                         settle_sec: float = 0.3) -> None:
    """
    Move sideways by sending a body-frame velocity command in v_y for a fixed duration.
    Sign convention (Spot body frame): +y = left, -y = right.
    """
    meters = float(meters)
    speed_mps = max(0.05, float(speed_mps))  # guard
    if abs(meters) < 1e-3:
        return

    v_y = speed_mps if meters > 0 else -speed_mps
    duration = abs(meters) / speed_mps

    end_time = time.time() + duration
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.0, v_y=v_y, v_rot=0.0)
    cmd_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(duration)
    stop_now(cmd_client)
    time.sleep(settle_sec)


def main() -> bool:
    p = argparse.ArgumentParser(
        description="Autowalk with periodic pauses and a lateral sidestep at each pause."
    )
    p.add_argument("--hostname", required=True)
    p.add_argument("--walk_directory", required=True)
    p.add_argument("--walk_filename", default="autogenerated.walk")

    p.add_argument("--stop-every-m", type=float, default=5.0)
    p.add_argument("--pause-sec", type=float, default=2.0)
    p.add_argument("--mission-timeout", type=float, default=2.0)
    p.add_argument("--noloc", action="store_true")

    # photos (optional)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--image-sources", action="append")
    p.add_argument("--image-service", default=ImageClient.default_service_name)
    p.add_argument("--jpeg-quality-percent", type=int, default=85)
    p.add_argument("--save-dir", default=None)

    # sidestep controls
    p.add_argument("--sidestep-m", type=float, default=1.5,
                   help="Magnitude of lateral shift in meters (positive=left, negative=right).")
    p.add_argument("--sidestep-speed", type=float, default=0.3,
                   help="Lateral speed (m/s) used for the sidestep.")
    p.add_argument("--sidestep-mode", choices=["alternate", "left", "right"], default="alternate",
                   help="Direction policy per pause: always left, always right, or alternate.")

    # NEW: background load controls (this was missing)
    p.add_argument("--bg-load-pct", type=float, default=0,
                   help="Background CPU load (0.0–1.0). 0 disables the thread.")
    p.add_argument("--bg-cycle-ms", type=int, default=100,
                   help="Cycle period for the background load (ms).")

    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    LOGGER.info(
        "Params: stop-every=%.2f m | pause=%.2f s | mission-timeout=%.1f s | "
        "sidestep=%.2f m @ %.2f m/s (%s) | bg-load=%.0f%% cycle=%d ms",
        args.stop_every_m, args.pause_sec, args.mission_timeout,
        args.sidestep_m, args.sidestep_speed, args.sidestep_mode,
        args.bg_load_pct * 100.0, args.bg_cycle_ms
    )

    # SDK / robot
    sdk = bosdyn.client.create_standard_sdk("DynamicStopper", [MissionClient])
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    power_client = robot.ensure_client(PowerClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
    mission_client = robot.ensure_client(MissionClient.default_service_name)
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    autowalk_client = robot.ensure_client(AutowalkClient.default_service_name)

    # optional image client
    image_client = None
    img_sources: List[str] = []
    save_root = ""
    if args.save_images:
        image_client = robot.ensure_client(args.image_service)
        img_sources = args.image_sources or ["frontleft_fisheye_image", "frontright_fisheye_image"]
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_root = args.save_dir or os.path.join(args.walk_directory, f"captures_{ts}")
        ensure_dir(save_root)

    # optional background thread
    bg_stop = threading.Event()
    bg_thread = None
    if args.bg_load_pct > 0.0:
        bg_thread = threading.Thread(
            target=bg_load_worker,
            args=(bg_stop, args.bg_load_pct, args.bg_cycle_ms),
            daemon=True,
        )
        bg_thread.start()

    LOGGER.info("Acquiring lease…")
    lease = lease_client.take()
    with LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True):
        LOGGER.info("Clearing GraphNav state and uploading map…")
        graph_nav_client.clear_graph()
        upload_graph_and_snapshots(LOGGER, graph_nav_client, args.walk_directory)

        LOGGER.info("Loading Walk and compiling Autowalk…")
        walk_proto = load_walk_from_disk(args.walk_directory, args.walk_filename)
        resp = autowalk_client.load_autowalk(walk_proto)
        if resp.status != resp.STATUS_OK:
            raise RuntimeError(f"Autowalk load failed: {resp.status} ({resp.STATUS.Name(resp.status)})")

        LOGGER.info("Powering on…")
        power_on_motors(power_client)
        LOGGER.info("Standing…")
        blocking_stand(cmd_client, timeout_sec=20)

        if not args.noloc:
            try:
                loc_guess = gnav_nav_pb2.Localization()
                graph_nav_client.set_localization(
                    initial_guess_localization=loc_guess,
                    ko_tform_body=None,
                    max_distance=None,
                    max_yaw=None,
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
                )
                LOGGER.info("Localization request sent.")
            except Exception as e:
                LOGGER.warning("Localization attempt failed/unsupported: %s", e)

        prev_xy = get_odom_xy(state_client)
        accum_m = 0.0
        pause_until = 0.0
        last_keepalive = 0.0
        pause_idx = 0
        alternate_sign = +1  # start left (+y)

        def keepalive():
            nonlocal last_keepalive
            pause_t = time.time() + args.mission_timeout
            mission_client.play_mission(pause_t, settings=mission_pb2.PlaySettings())
            last_keepalive = time.time()

        keepalive()
        LOGGER.info("Starting mission with dynamic pauses + lateral sidestep. Press Ctrl+C to stop.")

        try:
            while True:
                st = mission_client.get_state()
                if st.status in (
                    mission_pb2.State.STATUS_SUCCESS,
                    mission_pb2.State.STATUS_FAILURE,
                    mission_pb2.State.STATUS_STOPPED,
                ):
                    LOGGER.info("Mission finished with status: %s",
                                mission_pb2.State.Status.Name(st.status))
                    break

                cur_xy = get_odom_xy(state_client)
                accum_m += distance_xy(prev_xy, cur_xy)
                prev_xy = cur_xy

                now = time.time()
                if now < pause_until:
                    stop_now(cmd_client)
                    time.sleep(0.05)
                    continue

                if accum_m >= args.stop_every_m:
                    pause_idx += 1
                    LOGGER.info("Reached %.2f m since last pause → lateral shift + pause (pause #%d)",
                                accum_m, pause_idx)
                    stop_now(cmd_client)

                    # decide sidestep direction
                    if args.sidestep_mode == "left":
                        meters = abs(args.sidestep_m)
                    elif args.sidestep_mode == "right":
                        meters = -abs(args.sidestep_m)
                    else:  # alternate
                        meters = abs(args.sidestep_m) * (1 if alternate_sign > 0 else -1)
                        alternate_sign *= -1

                    # perform sidestep (body-frame v_y)
                    try:
                        perform_lateral_step(cmd_client, meters=meters, speed_mps=args.sidestep_speed)
                    except Exception as e:
                        LOGGER.warning("Sidestep failed: %s", e)

                    # optional photo burst
                    if args.save_images and image_client is not None:
                        try:
                            saved = capture_images(image_client, img_sources, save_root,
                                                   pause_idx, jpeg_quality=args.jpeg_quality_percent)
                            LOGGER.info("Saved %d image(s).", len(saved))
                        except Exception as e:
                            LOGGER.warning("Image capture error: %s", e)

                    # pause window (do NOT send keepalive during pause)
                    pause_until = time.time() + args.pause_sec
                    accum_m = 0.0
                    time.sleep(0.05)
                    continue

                if (now - last_keepalive) > (args.mission_timeout * 0.5):
                    keepalive()

                time.sleep(0.05)

        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt. Stopping.")
        except (RpcError, ResponseError) as e:
            LOGGER.error("Run error: %s", e)
        finally:
            stop_now(cmd_client)
            # stop background thread if any
            if bg_thread is not None:
                bg_stop.set()
                bg_thread.join(timeout=1.0)
            LOGGER.info("Done.")

    return True


if __name__ == "__main__":
    ok = main()
    if not ok:
        sys.exit(1)
