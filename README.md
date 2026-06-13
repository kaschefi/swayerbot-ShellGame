# Sawyer Robot Shell Game Visual Servoing Simulation

## 1. Project Overview

- **Name:** Sawyer Robot Shell Game Visual Servoing Simulation
- **Environment:** ROS Noetic, Ubuntu 20.04 (via WSL 2 on Windows), Gazebo, MoveIt 1.
- **Objective:** An orange ball is hidden beneath one of three green cups on a table. A custom Gazebo controller shuffles the cups using wide arcs to prevent physical/visual collisions. A camera on Sawyer's wrist streams the bird's-eye view of the game arena. Computer vision tracks the cups and the hidden ball, resolves frame-to-frame data association, and commands the Sawyer arm via MoveIt to dive down along the Z-axis and point directly over the winning cup.

## 2. Software Architecture & Components

The repository is structured around three main scripts that orchestrate the simulation, vision, and kinematics pipelines:

- `sim_main.py`: The master orchestration node. Handles the state machine (CALIBRATING, LOCKED, TRACKING, REVEALING, IDLE). Manages the OpenCV GUI window. Captures a 4-corner workspace boundary calibration grid from the camera viewport when 'S' is pressed, uses a custom HSV color thresholding filter to isolate the ball/cups, and passes pixel coordinate targets to the motion interface.
- `src/sawyer_interface.py`: The MoveIt and trajectory backend wrapper. Converts raw warped pixel positions into Sawyer base frame coordinates (X, Y) using calibrated linear scaling metrics. Sets joint velocity scaling factors to 0.8 and acceleration to 0.7 for rapid execution. Bypasses standard C++ TF bottlenecks by enforcing a hardcoded valid downward orientation quaternion (x=-0.7071, y=0.7071, z=0, w=0) to guarantee Inverse Kinematics (IK) convergence, and drops the planning budget to 1.5 seconds via RRTConnect.
- `src/gazebo_game_controller.py`: The simulation driver. Runs a service interface (`/shell_game/hide_ball` and `/shell_game/toggle_shuffle`). Animates the cups. Crucially, uses a wide X-axis clearance arc radius (0.16m) to prevent green cups from overlapping/merging on camera, and applies an S-Curve Cosine Blend (`(1.0 - np.cos(progress * np.pi)) / 2.0`) to smoothly interpolate velocity transitions between swapping pairs, eliminating jerky stutters.

## 3. Key Engineering & Troubleshooting Milestones

During the development of this pipeline, several technical hurdles were encountered and resolved. The following milestones highlight the debugging and engineering process:

- **WSL 2 Network Overrides:** Resolved an issue where the standard `intera.sh` startup script repeatedly forced `ROS_MASTER_URI` to `localhost`, breaking communication with the simulation master. Fixed by enforcing downstream runtime environment variable overrides targeting the true virtual machine bridge subnet IP (`172.19.20.224`).
- **Python Version Translation Linkage:** Fixed a legacy `exit code 127` failure where the core simulation node `cameras_sim_io_node.py` died instantly because Ubuntu 20.04/Noetic dropped generic `python` command mapping. Patched by linking system path interpreters (`python-is-python3`) to recover camera data stream delivery.
- **Transform (TF) Tree Snapping:** Diagnosed startup coordinate failures where the `endpoint_state` topic emitted dummy zeros (`valid: False`). Discovered that `robot_state_publisher` requires ~15 seconds on initialization to stitch detached simulation geometry links into a unified kinematic transform tree.
- **Degenerate Inverse Kinematics Avoidance:** Resolved a `MoveIt Planning Failed: ABORTED` error caused by the hardware fallback layer outputting broken zero-magnitude quaternions (w=0). Fixed by intercepting the stream and injecting mathematically robust orientation bounds.

## 4. Interactive Keyboard Map Controls

The master orchestration node supports the following real-time keyboard inputs for simulation control:

| Key | Action |
|:---:|:---|
| **S** | Save 4-Corner Arena Calibration Grid Coordinates |
| **X** | Lock Target Ball (Transitions to tracking phase) |
| **H** | Call Simulation Service to Hide Ball under Cup 1 |
| **1** | Trigger Wide-Arc S-Curve Shuffle Routine [START] |
| **2** | Stop Shuffle Routine [STOP] (Prints logical cup winner index to console) |
| **E** | Initiate MoveIt Visual Servoing Approach Trajectory (Robot drops to Z = table + 4cm) |
| **Q** | Safe-Terminate Node and clear buffers |

## 5. Deployment Instructions

To launch the entire visual servoing pipeline, open five separate terminal sessions and execute the following sequences:

> **Note:** Ensure your ROS workspace is built and sourced before proceeding.

**Terminal 1 (Gazebo):**
```bash
cd ~/ros_ws && . ./intera.sh && roslaunch sawyer_gazebo sawyer_world.launch
```

**Terminal 2 (Action Server):**
```bash
cd ~/ros_ws && . ./intera.sh && rosparam set /use_sim_time true && rosrun intera_interface joint_trajectory_action_server.py
```

**Terminal 3 (MoveIt Planner):**
```bash
cd ~/ros_ws && . ./intera.sh && rosparam set /use_sim_time true && roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=false
```

**Terminal 4 (Game Controller Node):**
Launch the shuffle node:
```bash
cd ~/ros_ws && . ./intera.sh && python3 src/shuffle_cups.py
```

**Terminal 5 (Master Script):**
Source workspace, export IP configurations, and run the master tracker:
```bash
cd ~/ros_ws && . ./intera.sh
export ROS_IP=172.19.20.224
export ROS_HOSTNAME=172.19.20.224
export ROS_MASTER_URI=http://172.19.20.224:11311
python3 sim_main.py
```
