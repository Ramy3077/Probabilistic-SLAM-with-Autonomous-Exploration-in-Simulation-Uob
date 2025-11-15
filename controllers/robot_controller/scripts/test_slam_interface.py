# scripts/test_slam_interface.py

import numpy as np
from slam.interface import SLAMInterface
from slam.occupancy import LaserScan
from collections import namedtuple

# Fake odometry struct like Sahib's packet
Odometry = namedtuple("Odometry", ["dt", "v_l", "v_r"])

def main():
    print(">>> Initialising SLAMInterface")
    slam = SLAMInterface(grid_size=(50,50))


    # --- Fake odometry ---
    odom = Odometry(dt=0.1, v_l=0.1, v_r=0.1)

    # --- Fake LiDAR scan ---
    ranges = np.ones(360) * 1.0
    scan = LaserScan(
        ranges=ranges,
        angle_min=0.0,
        angle_inc=np.deg2rad(1.0),
        range_min=0.05,
        range_max=3.5,
    )

    print(">>> Running 5 SLAM steps")
    for i in range(5):
        slam.predict(odom)
        slam.update(scan)
        grid, pose = slam.get_map_and_pose()

        print(f"[step {i}] pose={pose} grid_shape={grid.shape}")

    print(">>> Test complete: SLAMInterface runs without errors.")

if __name__ == "__main__":
    main()
