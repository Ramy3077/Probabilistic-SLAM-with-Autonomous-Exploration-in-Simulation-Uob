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
from slam.fastslam import ParticleFilterSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from slam.particles import Pose
from slam.visualize import plot_map
from models.motion import sample_motion_simple

# Trap recovery imports
from control.trap_recovery import TrapDetector, EscapeController, check_and_escape



class RandomExplorationSLAM:
    def __init__(self):
        # Initialize Webots robot
        self.robot = MyRobot()
        
        # Create occupancy grid matching actual arena size (10m x 10m arena)
        # Using 20m x 20m grid to cover full Lidar range (4m) + margin
        grid_spec = GridSpec(
            resolution=0.05,     # 5cm resolution
            width=400,           # 400 cells × 0.05m = 20m
            height=400,          # 400 cells × 0.05m = 20m
            origin_x=-10.0,      # Center the 20m grid
            origin_y=-10.0,
            l_occ=0.85,
            l_free=-0.4,
        )
        self.grid = OccupancyGrid(grid_spec)
        
        # Initialize ParticleFilterSLAM with fewer particles for large environment (faster)
        init_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        config = FastSLAMConfig(
            num_particles=200,     # Increased from 50 to 200 to prevent "getting lost"
            beam_subsample=5,      # Increased subsample slightly to maintain performance
            resample_threshold_ratio=0.5,
        )
        
        self.slam = ParticleFilterSLAM(
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
        
        # Initialize Trap Recovery System
        self.trap_detector = TrapDetector(
            robot_radius=0.225,  # meters (physical robot radius)
            wedge_threshold=0.9,  # Detect wedge if distance < 0.9 * radius
            valid_beam_threshold=0.15,  # Require 15% valid beams minimum (relaxed from 30%)
            stagnation_threshold=0.15,  # Stuck if moved < 0.15m (relaxed from 0.1m)
            check_interval_steps=20  # Check every 20 steps
        )
        self.escape_controller = EscapeController(
            max_escape_steps=40,
            shake_amplitude=5.0,
            backup_speed=-4.0,
            forward_speed=4.0,
            turn_speed=4.0
        )
        
        print("✅ Initialized random exploration SLAM with trap recovery")
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
    
    def simple_control(self, lidar_ranges, current_pose, angle_min, angle_inc):
        """
        Simple Random Exploration Logic.
        Rule: Go Straight, Turn Randomly (Left/Right) at Obstacle.
        """
        self.state_steps += 1
        
        # Check front sector for obstacles
        obstacle_ahead = self.check_obstacle_ahead(lidar_ranges)
        
        if self.state == 'FORWARD':
            if obstacle_ahead:
                # Pick random direction (50/50)
                if np.random.random() < 0.5:
                    print(f"⚠️ Obstacle detected -> Switching to TURN_RIGHT")
                    self.state = 'TURN_RIGHT'
                else:
                    print(f"⚠️ Obstacle detected -> Switching to TURN_LEFT")
                    self.state = 'TURN_LEFT'
                self.state_steps = 0
                return (0.0, 0.0) # Stop briefly
            else:
                # Go Straight
                return (4.0, 4.0)
        
        elif self.state == 'TURN_RIGHT':
            # Turn Right until obstacle clears AND minimum duration passes
            # (min 10 steps to ensure we actually turn away)
            if self.state_steps > 10 and not obstacle_ahead:
                 print("✅ Path clear -> Switching to FORWARD")
                 self.state = 'FORWARD'
                 self.state_steps = 0
                 return (4.0, 4.0)
            return (3.0, -3.0)

        elif self.state == 'TURN_LEFT':
            # Turn Left until obstacle clears AND minimum duration passes
            if self.state_steps > 10 and not obstacle_ahead:
                 print("✅ Path clear -> Switching to FORWARD")
                 self.state = 'FORWARD'
                 self.state_steps = 0
                 return (4.0, 4.0)
            return (-3.0, 3.0)
            
        return (0.0, 0.0)
    
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
                range_max=4.0,       # Updated to match 360 Lidar config (4.0m)
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
            
            # 3. Generate control with trap-aware state machine
            control = self.simple_control(
                lidar_ranges,
                estimated_pose,
                scan.angle_min,
                scan.angle_inc
            )
            
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
        # Grid is 400x400 (160000 cells) covering 20x20m = 400m^2
        # Arena is ~110m^2 (including walls/margin), so approx 44000 cells
        total_grid_cells = self.slam.grid.log_odds.size
        arena_cells = int(total_grid_cells * (110.0 / 400.0))
        
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
