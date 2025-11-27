#!/usr/bin/env python3
# Test script for SLAM measurement update and visualization

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from slam.fastslam import FastSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from slam.particles import Pose
from slam.visualize import plot_map
from models.motion import sample_motion_simple


def create_test_environment():
    # Create simple test environment with wall obstacles
    
    # Create a 10m x 10m grid with 5cm resolution
    spec = GridSpec(
        resolution=0.05,
        width=200,  # 10m / 0.05m
        height=200,
        origin_x=-5.0,
        origin_y=-5.0,
    )
    
    grid = OccupancyGrid(spec)
    
    # Add some obstacles (walls in log-odds space)
    # Vertical wall at x=2
    for i in range(60, 140):
        grid.mark_occupied(i, 140)  # j=140 -> x=2m
    
    # Horizontal wall at y=-1
    for j in range(20, 100):
        grid.mark_occupied(80, j)  # i=80 -> y=-1m
    
    # Mark some free space around origin
    for i in range(90, 110):
        for j in range(90, 110):
            grid.mark_free(i, j)
    
    return grid


def generate_mock_scan(pose: Pose, true_obstacles) -> LaserScan:
    # Generate synthetic laser scan by ray-casting to obstacles
    x_r, y_r, th_r = pose
    
    num_beams = 180
    angle_min = -np.pi / 2
    angle_max = np.pi / 2
    angle_inc = (angle_max - angle_min) / num_beams
    range_max = 5.0
    range_min = 0.1
    
    ranges = np.full(num_beams, range_max, dtype=float)
    
    # Simple ray casting to obstacles
    for k in range(num_beams):
        beam_angle = th_r + (angle_min + k * angle_inc)
        
        # Check intersection with known obstacles
        min_dist = range_max
        for (x_obs, y_obs) in true_obstacles:
            dx = x_obs - x_r
            dy = y_obs - y_r
            
            # Project onto beam direction
            dist = dx * np.cos(beam_angle) + dy * np.sin(beam_angle)
            
            if dist > 0:  # In front of robot
                # Check if obstacle is close to beam
                cross_track = abs(-dx * np.sin(beam_angle) + dy * np.cos(beam_angle))
                if cross_track < 0.1:  # Within 10cm of beam
                    min_dist = min(min_dist, dist)
        
        ranges[k] = min_dist
    
    # Add noise
    ranges += np.random.normal(0, 0.02, num_beams)
    ranges = np.clip(ranges, range_min, range_max)
    
    return LaserScan(
        ranges=ranges,
        angle_min=angle_min,
        angle_inc=angle_inc,
        range_min=range_min,
        range_max=range_max,
    )


def main():
    print("Testing SLAM measurement update and visualization...")
    
    # Create test environment
    grid = create_test_environment()
    
    # Define obstacles for mock scan generation
    obstacles = [
        (2.0, y) for y in np.linspace(-2, 2, 20)  # Vertical wall
    ] + [
        (x, -1.0) for x in np.linspace(-3, 0, 15)  # Horizontal wall
    ]
    
    # Initialize FastSLAM
    init_pose = np.array([0.0, 0.0, 0.0])
    config = FastSLAMConfig(
        num_particles=50,
        beam_subsample=3,
    )
    
    slam = FastSLAM(
        grid=grid,
        init_pose=init_pose,
        motion_model=sample_motion_simple,
        config=config,
    )
    
    # Simulate a few SLAM steps
    print("\nRunning SLAM simulation...")
    
    steps = [
        # (control, dt, true_pose_for_scan)
        ((0.5, 0.5), 0.1, np.array([0.05, 0.0, 0.0])),
        ((0.5, 0.5), 0.1, np.array([0.10, 0.0, 0.0])),
        ((0.5, 0.5), 0.1, np.array([0.15, 0.0, 0.0])),
        ((0.5, 0.6), 0.1, np.array([0.20, 0.01, 0.05])),
        ((0.5, 0.6), 0.1, np.array([0.25, 0.02, 0.10])),
    ]
    
    for idx, (control, dt, true_pose) in enumerate(steps):
        # Generate mock scan from approximate true position
        scan = generate_mock_scan(true_pose, obstacles)
        
        # Run SLAM step
        est_pose = slam.step(control=control, dt=dt, scan=scan)
        
        print(f"Step {idx + 1}: est_pose = {est_pose}, true_pose = {true_pose}")
        
        # Visualize every step
        if idx % 2 == 0:
            output_path = f"eval_logs/slam_viz_step_{idx:02d}.png"
            plot_map(
                grid=slam.grid,
                pose=est_pose,
                particles=slam.particles,
                scan=scan,
                title=f"SLAM Step {idx + 1}",
                output_path=output_path,
            )
    
    # Final visualization
    final_pose = slam.best_pose()
    print(f"\nFinal estimated pose: {final_pose}")
    plot_map(
        grid=slam.grid,
        pose=final_pose,
        particles=slam.particles,
        scan=None,
        title="Final SLAM State",
        output_path="eval_logs/slam_viz_final.png",
    )
    
    print("\n✅ Test complete! Check eval_logs/ for visualizations.")


if __name__ == "__main__":
    main()
