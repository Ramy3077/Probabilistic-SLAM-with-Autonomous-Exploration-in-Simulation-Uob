# Probabilistic SLAM with Autonomous Exploration in Simulation

## Overview
This project implements a complete **Simultaneous Localisation and Mapping (SLAM)** system from scratch in Python, deployed on a simulated differential-drive robot in **Webots**. 

The goal is to enable a robot to map an unknown environment using a **Particle Filter (FastSLAM)** and compare two exploration strategies:
1.  **Frontier-Based Exploration**: Autonomous decision-making to target boundaries between known and unknown space.
2.  **Random Exploration**: A baseline reactive random-walk strategy.

## Authorship & Roles
* **Ramy Mekhzer**: SLAM Core (Particle Filter, Motion/Measurement Models), Step-Window Inverse Sensor Model.
* **Bassel Kenaan**: Exploration Strategies (Frontier Detection, Clustering, Path Planning Integration), Evaluation Metrics.
* **Sahib Singh**: Robot Control Interface (Velocity Controllers, Webots Sensor Handling), Motion Primitives.

## Dependencies & Requirements
*   **Simulator**: [Webots R2023b](https://cyberbotics.com/) (or newer)
*   **Language**: Python 3.8+
*   **External Packages**:
    *   `numpy` (Matrix operations, particle handling)
    *   `scipy` (Optional, mostly for specific helper functions if used)
    *   `matplotlib` (For live visualization and plot generation)

## Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/RoboticsProject.git
    cd RoboticsProject
    ```
2.  Install python dependencies:
    ```bash
    pip install numpy matplotlib
    ```

## Usage
1.  Open the world file in Webots:
    *   `webots_worlds/Week3_Hexagon.wbt` (or similar)
2.  The robot controller is set to `robot_controller`.
3.  To run the **Frontier Exploration** (Main Agent):
    *   Ensure the controller script points to `live_slam_frontier.py`.
    *   Start the simulation in Webots.
    *   The robot will undock, spin to initialize LiDAR, and begin mapping.
4.  To run the **Random Baseline**:
    *   Ensure the controller script points to `live_slam_random.py` 

## Code Structure
*   `controllers/robot_controller/`
    *   `scripts/`: Executable scripts (e.g., `live_slam_frontier.py`).
    *   `slam/`: Core SLAM logic.
        *   `fastslam.py`: Particle Filter implementation.
        *   `occupancy.py`: Grid mapping and Inverse Sensor Model.
        *   `motion.py`: Probabilistic motion models.
    *   `explore/`: Exploration behaviors.
        *   `frontiers.py`: Frontier detection algorithms.
        *   `planner.py`: High-level target selection.
    *   `control/`: Low-level robot movement (PID, waypoint following).
    *   `models/`: Sensor and Actuator models.

## Implementation Attribution
To comply with coursework requirements, we explicitly state the source of each component:

### Self-Implemented
*   **Core Systems**:
    *   `robot_controller.py`: Main state machine and controller logic.
    *   `slam/occupancy.py`: Custom **Step-Window Inverse Sensor Model** and grid data structure.
    *   `slam/fastslam.py`: Particle filter lifecycle (resample, predict, update) implemented from probabilistic theory.
    *   `explore/frontiers.py`: Frontier detection logic (ROI extraction, clustering).
*   **Scripts**:
    *   `benchmark_exploration.py`: Custom benchmarking suite for quantitative analysis.
    *   `live_slam_frontier.py` / `live_slam_random.py`: Real-time execution loops.

### External Packages & References
*   **PythonRobotics (Atsushi Sakai)**: used as a reference for the **Motion Model** equations and standard particle filter structure.
    *   *Reference*: [PythonRobotics GitHub](https://github.com/AtsushiSakai/PythonRobotics)
*   **Webots API**: Used for all low-level hardware interfacing (Lidar, Motors, Encoders).
*   **NumPy**: Used for efficient matrix operations.
*   **Matplotlib**: Used for real-time visualization.

---
*University of Birmingham - Intelligent Robotics 06-30227*
