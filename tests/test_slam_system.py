import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from controllers.robot_controller.slam.fastslam import FastSLAM, FastSLAMConfig
from controllers.robot_controller.slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from controllers.robot_controller.slam.particles import Pose

def mock_motion_model(pose: Pose, control: tuple, dt: float, rng=None) -> Pose:
    
    #A temporary Kinematic Model stub.
    
    v, omega = control
    x, y, theta = pose
    
    # Adding a tiny bit of noise to prove particles spread out
    noise_scale = 0.01 if rng else 0.0
    if rng:
        n_x = rng.normal(0, noise_scale)
        n_y = rng.normal(0, noise_scale)
        n_th = rng.normal(0, noise_scale)
    else:
        n_x = n_y = n_th = 0.0

    theta_new = theta + omega * dt + n_th
    x_new = x + v * dt * np.cos(theta) + n_x
    y_new = y + v * dt * np.sin(theta) + n_y
    
    # Normalize angle to [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([x_new, y_new, theta_new])

def test_full_slam_loop():
    print(" Initializing SLAM System Test ")

    # Setup the Map
    spec = GridSpec(
        resolution=0.1,      # 10cm per pixel
        width=50,            # 5x5 meter map
        height=50,
        origin_x=-2.5,
        origin_y=-2.5
    )
    grid = OccupancyGrid(spec)
    
    # Setup SLAM Configuration
    start_pose = np.array([0.0, 0.0, 0.0])
    slam = FastSLAM(
        grid=grid,
        init_pose=start_pose,
        motion_model=mock_motion_model,
        config=FastSLAMConfig(num_particles=50)
    )

    # Create Mock Sensor Data (LaserScan)
    ranges = np.full(360, 5.0) # Default to far away (5m)
    # Place a "wall" in the front center (indices 0-10 and 350-360)
    ranges[0:10] = 2.0
    ranges[350:360] = 2.0
    
    scan = LaserScan(
        ranges=ranges,
        angle_min=0.0,  
        angle_inc=np.radians(1.0),
        range_min=0.1,
        range_max=10.0
    )

    # Run Simulation Loop
    print("\nRunning 10-step simulation...")
    print(f"Start Pose: {slam.best_pose()}")

    control_input = (0.5, 0.1) # Drive 0.5 m/s, turn 0.1 rad/s
    dt = 0.5                   

    for step in range(1, 11):

        estimated_pose = slam.step(control=control_input, dt=dt, scan=scan)
        
        x, y, theta = estimated_pose
        print(f"Step {step}: Pose X={x:.3f}, Y={y:.3f}, Theta={theta:.3f}")

    # Final Assertions 
    final_pose = slam.best_pose()
    

    dist_traveled = np.linalg.norm(final_pose[:2])
    print(f"\nTotal Distance Traveled: {dist_traveled:.3f}m (Expected ~2.5m)")
    
    if 2.0 < dist_traveled < 3.0:
        print(" Motion Model is driving the particles.")
    else:
        print(" Robot barely moved or teleported!")

    map_probs = grid.probabilities()
    occupied_count = np.sum(map_probs > 0.6) 
    
    print(f"Occupied Grid Cells: {occupied_count}")
    
    if occupied_count > 0:
        print(" Map is being updated with obstacles.")
    else:
        print("Map is still  empty!")

    print("Test Complete")

if __name__ == "__main__":
    test_full_slam_loop()