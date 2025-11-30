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
            resolution=0.05,     # 5cm resolution
            width=300,           # 300 cells × 0.05m = 15m (covers 10x10m arena with margin)
            height=300,          # 300 cells × 0.05m = 15m
            origin_x=-7.5,       # Center the 15m grid
            origin_y=-7.5,
            l_occ=0.85,
            l_free=-0.4,
        )
        self.grid = OccupancyGrid(grid_spec)
        
        # Initialize FastSLAM with fewer particles for large environment (faster)
        init_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        config = FastSLAMConfig(
            num_particles=50,      # Increased from 30 for robustness in degenerate scenarios
            beam_subsample=3,      # Reduced from 10 - improves coverage speed
            resample_threshold_ratio=0.5,
        )
        
        self.slam = FastSLAM(
            grid=self.grid,
            init_pose=init_pose,
            motion_model=sample_motion_simple,
            config=config,
        )
        
        # Exploration state - FASTER movement
        # Exploration Parameters
        self.step_count = 0  # Re-added missing initialization
        self.obstacle_distance_threshold = 0.8  # Increased to 0.8m to avoid getting too close
        self.stagnation_check_interval = 20  # Check for stuck every 20 steps (was 50)
        
        # Diagonal backup state (for escaping tight spaces)
        self.diagonal_backup_active = False
        self.diagonal_backup_steps = 0
        self.diagonal_backup_direction = None
        
        # 90° turn state (after backing up)
        self.turning_90 = False
        self.turn_steps = 0
        self.turn_direction = None
        
        # Force-escape state (when wedged into geometry)
        self.force_escape_active = False
        self.force_escape_steps = 0
        self.force_escape_phase = 0  # 0=back, 1=forward, 2=turn
        
        # Stuck detection
        self.blind_steps = 0
        self.last_pose_check = None
        self.last_pose_step = 0
        self.stuck_check_interval = 20

        # === NEW STATE MACHINE VARIABLES ===
        self.state = 'FORWARD'  # States: FORWARD, TURN_RIGHT, BACKUP, TURN_LEFT
        self.state_steps = 0
        self.last_stagnation_pose = None

        
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
                if min_distance < self.obstacle_distance_threshold + 0.2:
                    print(f"  [SENSORS] Front Min: {min_distance:.2f}m (Thresh: {self.obstacle_distance_threshold}m)")
                
                if min_distance < self.obstacle_distance_threshold:
                    print(f"⚠️  OBSTACLE DETECTED! Min distance: {min_distance:.2f}m < {self.obstacle_distance_threshold}m")
                
                return min_distance < self.obstacle_distance_threshold
        
        return False
    
    def check_obstacle_behind(self, lidar_ranges):
        """Check if there's an obstacle behind the robot (for safe backing up)"""
        if len(lidar_ranges) == 0:
            return False
        
        num_beams = len(lidar_ranges)
        # Rear beams are at the edges (first and last ~15% of beams)
        rear_sector = num_beams // 6
        
        # Check left-rear and right-rear beams
        left_rear = lidar_ranges[:rear_sector]
        right_rear = lidar_ranges[-rear_sector:]
        rear_beams = list(left_rear) + list(right_rear)
        
        if len(rear_beams) > 0:
            valid_beams = [r for r in rear_beams if np.isfinite(r) and r > 0.01]
            
            if len(valid_beams) > 0:
                min_distance = min(valid_beams)
                # Use half the threshold for rear detection (more conservative)
                return min_distance < (self.obstacle_distance_threshold * 0.6)
        
        return False
    
    def simple_control(self, lidar_ranges, current_pose):
        """
        Rule-based exploration state machine:
        1. FORWARD: Go straight until obstacle or stuck.
        2. TURN_RIGHT: If obstacle ahead, turn 90 deg right.
        3. BACKUP: If stuck (no move for 3s), back up.
        4. TURN_LEFT: After backup, turn 90 deg left.
        """
        self.state_steps += 1
        
        # Filter valid ranges
        valid_ranges = [r for r in lidar_ranges if np.isfinite(r) and r > 0.01]
        min_dist = min(valid_ranges) if valid_ranges else 0.0
        
        # --- STATE: FORWARD ---
        if self.state == 'FORWARD':
            # 1. Check for Obstacle
            if self.check_obstacle_ahead(lidar_ranges):
                print(f"⚠️ Obstacle ahead ({min_dist:.2f}m) -> Switching to TURN_RIGHT")
                self.state = 'TURN_RIGHT'
                self.state_steps = 0
                return (0.0, 0.0) # Stop briefly
            
            # 2. Check for Stagnation (Stuck)
            if self.step_count % self.stagnation_check_interval == 0:
                if self.last_stagnation_pose is not None:
                    dist = np.linalg.norm(current_pose[:2] - self.last_stagnation_pose[:2])
                    if dist < 0.1: # Moved < 10cm in 3s
                        print(f"🚨 Stuck (moved {dist:.2f}m) -> Switching to TURN_LEFT (Spot Turn)")
                        self.state = 'TURN_LEFT'
                        self.state_steps = 0
                        self.last_stagnation_pose = np.copy(current_pose)
                        return (0.0, 0.0)
                self.last_stagnation_pose = np.copy(current_pose)
            
            # Action: Drive Forward
            # Reduced speed from 6.0 to 3.0 to accommodate short LiDAR range (2.0m)
            return (3.0, 3.0)

        # --- STATE: TURN_RIGHT ---
        elif self.state == 'TURN_RIGHT':
            # Action: Turn Right (Left wheel fwd, Right wheel back)
            if self.state_steps >= 7: # Approx 90 degrees (reduced from 12)
                print("✅ Turn complete -> Switching to FORWARD")
                self.state = 'FORWARD'
                self.state_steps = 0
                self.last_stagnation_pose = np.copy(current_pose) # Reset stuck check
                return (3.0, 3.0)
            return (3.0, -3.0)  # Slower turn

        # --- STATE: BACKUP ---
        elif self.state == 'BACKUP':
            # Check for obstacle behind
            if self.check_obstacle_behind(lidar_ranges):
                print("🚨 Obstacle behind! -> Switching to TURN_LEFT immediately")
                self.state = 'TURN_LEFT'
                self.state_steps = 0
                return (0.0, 0.0)
            
            # Action: Reverse
            if self.state_steps >= 20: # Back up for ~1.2s
                print("✅ Backup complete -> Switching to TURN_LEFT")
                self.state = 'TURN_LEFT'
                self.state_steps = 0
                return (0.0, 0.0)
            return (-5.0, -5.0)

        # --- STATE: TURN_LEFT ---
        elif self.state == 'TURN_LEFT':
            # Action: Turn Left (Left wheel back, Right wheel fwd)
            if self.state_steps >= 7: # Approx 90 degrees (reduced from 12)
                print("✅ Turn complete -> Switching to FORWARD")
                self.state = 'FORWARD'
                self.state_steps = 0
                self.last_stagnation_pose = np.copy(current_pose) # Reset stuck check
                return (3.0, 3.0)
            return (-3.0, 3.0)  # Slower turn
            
        return (0.0, 0.0) # Should not reach here
    
    def run(self, max_steps=500, viz_every=20):
        print(f"\n🚀 Starting random exploration for {max_steps} steps...")
        print(f"📊 Visualizing every {viz_every} steps\n")
        
        dt = self.robot.timestep / 1000.0  # Convert to seconds
        
        # === WARMUP PHASE ===
        print("⏳ Warming up sensors...")
        for _ in range(20): # Wait 20 steps (~1.2s) for sensors to stabilize
            self.robot.robot.step(self.robot.timestep)
            
        # Wait for valid LiDAR data
        print("⏳ Waiting for valid LiDAR data...")
        while self.robot.robot.step(self.robot.timestep) != -1:
            sensor_packet = self.robot.create_sensor_packet(dt)
            ranges = np.array(sensor_packet['lidar']['ranges'], dtype=float)
            
            # Check if we have ANY data (even if it's all inf/max range)
            if len(ranges) > 0:
                # Debug print to see what the sensor is seeing
                valid_finite = [r for r in ranges if np.isfinite(r)]
                min_val = min(ranges) if len(ranges) > 0 else 0
                max_val = max(ranges) if len(ranges) > 0 else 0
                finite_count = len(valid_finite)
                
                print(f"✅ Sensors ready! Received {len(ranges)} beams.")
                print(f"   Finite (hits): {finite_count}")
                print(f"   Range: [{min_val:.2f}, {max_val:.2f}]")
                break
        # ====================
        
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
                range_max=2.0,       # Updated to match user's Webots config (2.0m)
            )
            
            # --- DEBUG START ---
            if self.step_count % 10 == 0:
                valid_scan_points = np.sum((scan.ranges >= scan.range_min) & (scan.ranges <= scan.range_max))
                # Assuming subsample is 3 as per init
                subsample = 3 
                used_points = valid_scan_points // subsample
                print(f"[SCAN DEBUG] Raw valid points: {valid_scan_points}, Approx used points: {used_points}")
            # --- DEBUG END ---
            
            estimated_pose = self.slam.step(
                control=odom_control,
                dt=dt,
                scan=scan
            )
            
            # 3. Generate control with obstacle avoidance (needs pose for stuck detection)
            # 3. Generate control with rule-based state machine
            control = self.simple_control(lidar_ranges, estimated_pose)
            
            # 4. Execute control
            self.robot.set_wheel_speeds(control[0], control[1])
            
            # 5. Logging - VERBOSE
            if self.step_count % 1 == 0:  # Log EVERY step
                coverage = self._compute_coverage()
                known_cells = np.count_nonzero(self.slam.grid.log_odds)
                valid_beams = len([r for r in lidar_ranges if np.isfinite(r) and r > 0.01])
                print(f"STEP {self.step_count:04d} | "
                      f"Pose: ({estimated_pose[0]:5.2f}, {estimated_pose[1]:5.2f}, {estimated_pose[2]:5.2f}) | "
                      f"Cov: {coverage:5.2f}% ({known_cells} cells) | "
                      f"Beams: {valid_beams:3d} | "
                      f"Ctrl: ({control[0]:.1f}, {control[1]:.1f})")
            
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
        
        # Normalize by arena size (10x10m = 100m^2)
        # Grid is 300x300 (90000 cells) covering 15x15m = 225m^2
        # Arena is 100m^2, so approx 40000 cells
        total_grid_cells = self.slam.grid.log_odds.size
        arena_cells = int(total_grid_cells * (100.0 / 225.0))
        
        # --- DEBUG START ---
        if self.step_count % 10 == 0:
            print(f"\n[COVERAGE DEBUG] Step {self.step_count}")
            print(f"  Known Cells (log_odds != 0): {known_cells}")
            print(f"  Total Grid Cells: {total_grid_cells}")
            print(f"  Target Arena Cells: {arena_cells}")
            
            # Analyze grid values
            lo = self.slam.grid.log_odds
            occupied = np.count_nonzero(lo > 0)
            free = np.count_nonzero(lo < 0)
            print(f"  Occupied Cells (>0): {occupied}")
            print(f"  Free Cells (<0): {free}")
            
            if known_cells > 0:
                rows, cols = np.where(lo != 0)
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()
                print(f"  Map Bounding Box: [{r_min}:{r_max}, {c_min}:{c_max}]")
                print(f"  Box Size: {r_max-r_min} x {c_max-c_min}")
        # --- DEBUG END ---

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
