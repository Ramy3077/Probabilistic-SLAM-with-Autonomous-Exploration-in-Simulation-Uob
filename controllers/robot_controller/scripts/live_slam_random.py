#!/usr/bin/env python3
# Live SLAM with random exploration in Webots

import sys
import os
from pathlib import Path
import numpy as np

# Webots controller imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from robots.robot import MyRobot

# SLAM imports
from slam.fastslam import FastSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from slam.particles import Pose
from slam.visualize import plot_map
from models.motion import sample_motion_simple


class RandomExplorationSLAM:
    def __init__(self):
        # Initialize Webots robot
        self.robot = MyRobot()
        
        # Create occupancy grid matching actual arena size (4m x 4m arena)
        # Using 5m x 5m grid to have some margin around walls
        grid_spec = GridSpec(
            resolution=0.05,     # 5cm resolution (good detail for 4m arena)
            width=100,           # 100 cells × 0.05m = 5m
            height=100,          # 100 cells × 0.05m = 5m
            origin_x=-2.5,       # Center the 5m grid
            origin_y=-2.5,
            l_occ=0.85,
            l_free=-0.4,
        )
        self.grid = OccupancyGrid(grid_spec)
        
        # Initialize FastSLAM with fewer particles for large environment (faster)
        init_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        config = FastSLAMConfig(
            num_particles=30,      # Reduced for speed in large environment
            beam_subsample=10,     # More aggressive subsampling
            resample_threshold_ratio=0.5,
        )
        
        self.slam = FastSLAM(
            grid=self.grid,
            init_pose=init_pose,
            motion_model=sample_motion_simple,
            config=config,
        )
        
        # Exploration state - FASTER movement
        self.step_count = 0
        self.last_control = (6.0, 6.0)  # Default forward at full speed
        self.direction_change_interval = 100  # Change direction RARELY (was 50)
        self.obstacle_distance_threshold = 0.5  # 50cm safety
        
        # Diagonal backup state (for escaping tight spaces)
        self.diagonal_backup_active = False
        self.diagonal_backup_steps = 0
        self.diagonal_backup_direction = None
        
        # 90° turn state (after backing up)
        self.turning_90 = False
        self.turn_steps = 0
        self.turn_direction = None
        
        # Output directory
        self.output_dir = Path("eval_logs/live_slam")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old visualizations
        for old_file in self.output_dir.glob("*.png"):
            old_file.unlink()
        
        print("✅ Initialized random exploration SLAM")
        print(f"Grid: {grid_spec.width}x{grid_spec.height} cells, {grid_spec.resolution}m resolution")
        print(f"Particles: {config.num_particles}")
        print(f"Obstacle threshold: {self.obstacle_distance_threshold}m")
        print(f"Output: {self.output_dir}")
    
    def check_obstacle_ahead(self, lidar_ranges):
        # Check if there's an obstacle in the front sector
        # LiDAR typically has beams from -90° to +90°
        # Check front 60° (±30°)
        
        if len(lidar_ranges) == 0:
            return False
        
        num_beams = len(lidar_ranges)
        center = num_beams // 2
        front_sector = num_beams // 6  # ±30° sector
        
        # Check front beams
        start = max(0, center - front_sector)
        end = min(num_beams, center + front_sector)
        front_beams = lidar_ranges[start:end]
        
        if len(front_beams) > 0:
            # Filter out invalid readings (inf, nan)
            valid_beams = [r for r in front_beams if np.isfinite(r) and r > 0.01]
            
            if len(valid_beams) > 0:
                min_distance = min(valid_beams)
                
                # Debug output
                if min_distance < self.obstacle_distance_threshold:
                    print(f"⚠️  OBSTACLE DETECTED! Min distance: {min_distance:.2f}m (threshold: {self.obstacle_distance_threshold}m)")
                
                return min_distance < self.obstacle_distance_threshold
        
        return False
    
    def random_control(self, lidar_ranges, current_pose):
        # Generate random exploration control with obstacle avoidance
        
        # Execute 90° turn if active
        if self.turning_90:
            self.turn_steps += 1
            
            if self.turn_steps >= 7:  # 7 steps for ~90° (8 was 110-120°)
                # Done turning
                self.turning_90 = False
                self.turn_steps = 0
                self.turn_direction = None
                # Resume forward
                return (6.0, 6.0)
            else:
                # Continue turning - reverse one wheel for sharp turn
                if self.turn_direction == 'left':
                    return (-6.0, 6.0)  # Left wheel back, right wheel forward (sharp left)
                else:
                    return (6.0, -6.0)  # Left wheel forward, right wheel back (sharp right)
        
        # Continue diagonal backup if active
        if self.diagonal_backup_active:
            self.diagonal_backup_steps += 1
            
            # Check if we've cleared the obstacle zone
            valid_ranges = [r for r in lidar_ranges if np.isfinite(r) and r > 0.01]
            min_dist = min(valid_ranges) if valid_ranges else 1.0
            
            if min_dist > self.obstacle_distance_threshold + 0.1:  # Cleared threshold + margin
                # Done backing up - initiate 90° turn
                print(f"✅ Cleared obstacle zone ({min_dist:.2f}m), executing 90° turn")
                self.diagonal_backup_active = False
                self.diagonal_backup_steps = 0
                
                # Start 90° turn
                self.turning_90 = True
                self.turn_steps = 0
                self.turn_direction = self.diagonal_backup_direction
                
                if self.turn_direction == 'left':
                    return (-6.0, 6.0)  # Sharp left
                else:
                    return (6.0, -6.0)  # Sharp right
            elif self.diagonal_backup_steps >= 10:  # Shorter timeout (was 20) - limits backup distance
                # Give up backing - do the turn anyway
                print(f"⚠️  Backup timeout, doing 90° turn anyway")
                self.diagonal_backup_active = False
                self.diagonal_backup_steps = 0
                
                # Start 90° turn anyway
                self.turning_90 = True
                self.turn_steps = 0
                self.turn_direction = self.diagonal_backup_direction
                
                if self.turn_direction == 'left':
                    return (-6.0, 6.0)  # Sharp left
                else:
                    return (6.0, -6.0)  # Sharp right
            else:
                # Continue backing straight up (not diagonal, more effective)
                return (-5.0, -5.0)
        
        # Obstacle avoidance takes priority
        obstacle_ahead = self.check_obstacle_ahead(lidar_ranges)
        
        if obstacle_ahead:
            # Check distance to obstacle
            valid_ranges = [r for r in lidar_ranges if np.isfinite(r) and r > 0.01]
            min_dist = min(valid_ranges) if valid_ranges else 1.0
            
            if min_dist < self.obstacle_distance_threshold:
                # Too close! Initiate backup sequence
                if not self.diagonal_backup_active:
                    self.diagonal_backup_active = True
                    self.diagonal_backup_steps = 0
                    self.diagonal_backup_direction = np.random.choice(['left', 'right'])
                    print(f"🔙 OBSTACLE at {min_dist:.2f}m! Backing up to clear threshold...")
                    return (-5.0, -5.0)  # Start backing up
        
        # No obstacle - do random exploration HEAVILY biased toward FORWARD
        if self.step_count % self.direction_change_interval == 0:
            # 95% forward, 5% turns (much less chaotic)
            rand = np.random.rand()
            
            if rand < 0.95:  # 95% forward
                control = (6.0, 6.0)  # Fast forward
                action = "forward"
            elif rand < 0.975:  # 2.5% turn left
                control = (3.0, 6.0)
                action = "turn_left"
            else:  # 2.5% turn right
                control = (6.0, 3.0)
                action = "turn_right"
            
            self.last_control = control
            print(f"🎲 Random action: {action}")
        
        return self.last_control
    
    def run(self, max_steps=500, viz_every=20):
        print(f"\n🚀 Starting random exploration for {max_steps} steps...")
        print(f"📊 Visualizing every {viz_every} steps\n")
        
        dt = self.robot.timestep / 1000.0  # Convert to seconds
        
        while self.robot.robot.step(self.robot.timestep) != -1 and self.step_count < max_steps:
            # 1. Read sensors first
            sensor_packet = self.robot.create_sensor_packet(dt)
            lidar_data = sensor_packet['lidar']
            lidar_ranges = np.array(lidar_data['ranges'], dtype=float)
            
            # 2. SLAM update (do first to get current pose)
            odom = sensor_packet['odometry']
            odom_control = (odom['v_l'], odom['v_r'])
            
            scan = LaserScan(
                ranges=np.array(lidar_data['ranges'], dtype=float),
                angle_min=lidar_data['angle_min'],
                angle_inc=lidar_data['angle_increment'],
                range_min=0.1,
                range_max=5.0,
            )
            
            estimated_pose = self.slam.step(
                control=odom_control,
                dt=dt,
                scan=scan
            )
            
            # 3. Generate control with obstacle avoidance (needs pose for stuck detection)
            control = self.random_control(lidar_ranges, estimated_pose)
            
            # 4. Execute control
            self.robot.set_wheel_speeds(control[0], control[1])
            
            # 5. Logging
            if self.step_count % 10 == 0:  # More frequent for debugging
                coverage = self._compute_coverage()
                print(f"[{self.step_count:4d}] Pose: ({estimated_pose[0]:6.2f}, {estimated_pose[1]:6.2f}, {estimated_pose[2]:6.2f}), "
                      f"Coverage: {coverage:.1f}%")
            
            # LiDAR debugging every 50 steps
            if self.step_count % 50 == 0:
                valid_ranges = [r for r in lidar_ranges if np.isfinite(r) and r > 0.01]
                if len(valid_ranges) > 0:
                    print(f"   📡 LiDAR: {len(valid_ranges)}/{len(lidar_ranges)} beams valid, "
                          f"min={min(valid_ranges):.2f}m, max={max(valid_ranges):.2f}m, "
                          f"avg={np.mean(valid_ranges):.2f}m")
            
            # 6. Visualization
            if self.step_count % viz_every == 0:
                output_path = self.output_dir / f"slam_step_{self.step_count:04d}.png"
                plot_map(
                    grid=self.slam.grid,
                    pose=estimated_pose,
                    particles=self.slam.particles,
                    scan=scan,
                    title=f"Random Exploration - Step {self.step_count}",
                    output_path=str(output_path),
                    show_particles=True,
                    show_scan=True,
                    scale=5,  # Smaller scale for faster generation
                )
            
            self.step_count += 1
        
        # Final visualization
        final_pose = self.slam.best_pose()
        final_path = self.output_dir / "slam_final.png"
        plot_map(
            grid=self.slam.grid,
            pose=final_pose,
            particles=self.slam.particles,
            scan=None,
            title=f"Final Map - {self.step_count} steps",
            output_path=str(final_path),
            show_particles=True,
            show_scan=False,
            scale=10,
        )
        
        coverage = self._compute_coverage()
        print(f"\n✅ Exploration complete!")
        print(f"📊 Final statistics:")
        print(f"   Steps: {self.step_count}")
        print(f"   Coverage: {coverage:.1f}%")
        print(f"   Final pose: ({final_pose[0]:.2f}, {final_pose[1]:.2f}, {final_pose[2]:.2f})")
        print(f"   Visualizations: {self.output_dir}")
    
    def _compute_coverage(self):
        # Count any cell that has been observed (log_odds != 0)
        # This is "Explored Area" rather than "High Confidence Area"
        # It prevents coverage from dropping when the robot gets uncertain
        known_cells = np.count_nonzero(self.slam.grid.log_odds)
        
        # Normalize by arena size (4x4m = 16m^2)
        # Grid is 100x100 (10000 cells) covering 5x5m
        # Arena is 16m^2, so approx 6400 cells
        total_grid_cells = self.slam.grid.log_odds.size
        arena_cells = int(total_grid_cells * (16.0 / 25.0))
        
        # Debug print once in a while (e.g. if coverage is low but map looks full)
        if self.step_count % 500 == 0:
            print(f"   [Debug] Explored: {known_cells} / {arena_cells} (Arena Cells)")

        return min(100.0, 100.0 * known_cells / arena_cells)


def main():
    print("="*60)
    print("  LIVE SLAM WITH RANDOM EXPLORATION")
    print("  Optimized for LARGE environments")
    print("="*60)
    
    explorer = RandomExplorationSLAM()
    
    # Run for 5000 steps (~5-6 minutes in simulation)
    # Visualize every 50 steps (less I/O for large maps)
    explorer.run(max_steps=5000, viz_every=50)
    
    print("\n✨ Done! Check eval_logs/live_slam/ for visualizations.")


if __name__ == "__main__":
    main()
