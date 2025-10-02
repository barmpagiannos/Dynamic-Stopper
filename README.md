# Dynamic Stopper for Spot

This repository contains a Python script extending Boston Dynamics Spot’s Autowalk with **dynamic runtime behaviors**.

The code loads a recorded Autowalk mission and injects dynamic pauses, sidesteps, and image capture into the robot’s navigation. It also supports running background CPU tasks (e.g., lightweight inference threads such as YOLO) to simulate real-time decision making.

Developed for research and experimental purposes in the context of autonomous robotics and dynamic navigation.

---

## Features

### 1. Periodic Dynamic Pauses  
**Goal:** Stop the robot every *M* meters along the path for *N* seconds.  
**Techniques:**  
- ODOM distance tracking  
- Mission keepalive handling  
- Controlled stop commands  

**File:** `dynamic_stopper.py`

---

### 2. Lateral Sidestep Maneuvers  
**Goal:** Shift the robot sideways (left, right, or alternating) by a given distance.  
**Techniques:**  
- Body-frame velocity command (v_y)  
- Configurable sidestep magnitude and speed  
- Alternate or fixed direction policy  

**File:** `dynamic_stopper.py`

---

### 3. Image Capture at Pauses  
**Goal:** Capture and save images from Spot’s cameras at each pause.  
**Techniques:**  
- ImageClient API with multiple sources  
- JPEG/PNG saving with timestamps  
- Directory auto-creation per pause  

**File:** `dynamic_stopper.py`

---

### 4. Background CPU Load Thread  
**Goal:** Simulate lightweight background tasks (e.g., YOLO object detection).  
**Techniques:**  
- Configurable load percentage and cycle  
- NumPy operations for CPU/FPU activity  
- Runs in parallel thread during mission  

**File:** `dynamic_stopper.py`

---

## Requirements

- **Python 3.8+**  
- **Boston Dynamics Spot SDK**  
- External libraries:  
  - `numpy`  
  - `opencv-python`  
  - `protobuf`  

---

## Example Configuration

An example JSON launch configuration is provided:

```json
{
  "name": "DynamicStopper — sidestep + light BG load",
  "program": "examples/dynamic_stopper.py",
  "args": [
    "--hostname", "10.0.0.3",
    "--walk_directory", "recordings/Vasilis_Outside_22_08_2025_Test2.walk",
    "--walk_filename", "Vasilis_Outside_22_08_2025_Test2.walk",
    "--stop-every-m", "5.0",
    "--pause-sec", "2.0",
    "--sidestep-m", "1",
    "--sidestep-speed", "0.3",
    "--sidestep-mode", "alternate",
    "--save-images",
    "--image-sources", "frontleft_fisheye_image",
    "--image-sources", "frontright_fisheye_image",
    "--save-dir", "recordings/.../captures",
    "--bg-load-pct", "0.10",
    "--bg-cycle-ms", "100"
  ]
}
