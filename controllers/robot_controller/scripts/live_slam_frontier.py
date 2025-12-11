#!/usr/bin/env python3
# Live SLAM with SMART FRONTIER exploration (Simplified: No complex trap recovery)

import sys
from pathlib import Path
import numpy as np

# --- Make local packages importable ---
sys.path.insert(0, str(Path(__file__).parent.parent))

# Robot + SLAM imports
from robots.robot import MyRobot, Waypoint
from slam.fastslam import ParticleFilterSLAM as FastSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from slam.visualize import plot_map
from models.motion import sample_motion_simple

# Exploration + control imports
from control.waypoint_follow import GoToGoal
from control.path_planner import AStarPlanner, simplify_path
from explore.frontiers import detect_frontiers, cluster_frontiers
from explore.planner import choose_frontier
from eval.metrics import coverage_percent, entropy_proxy
from eval.logger import CsvLogger


class FrontierExplorationSLAM:
    def __init__(self):
        # --- Webots robot wrapper ---
        self.robot = MyRobot()
        self.waypoint = Waypoint(self.robot)

        # --- Occupancy grid (loaded from config) ---
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

        # --- FastSLAM config ---
        init_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        config = FastSLAMConfig(
            num_particles=50,
            beam_subsample=3,
            resample_threshold_ratio=0.5,
        )
        self.slam = FastSLAM(
            grid=self.grid,
            init_pose=init_pose,
            motion_model=sample_motion_simple,
            config=config,
        )

        # --- Navigation & Planning ---
        self.origin_xy = (grid_spec.origin_x, grid_spec.origin_y)
        self.resolution = grid_spec.resolution
        
        self.path_planner = AStarPlanner(
            safety_distance=0.28,
            allow_diagonal=True
        )
        
        # Controller
        self.goto_goal = GoToGoal(v_max=0.25, w_max=2.0)
        
        # State
        self.current_path = [] # List of (x,y) world waypoints
        self.current_target_frontier = None 
        self.blacklist = [] 
        self.planning_fail_cooldown = 0
        self.stuck_counter = 0  # Counter for safety stop duration
        self.backup_counter = 0 # Simple backup maneuver state
        self.turn_counter = 0   # Turn after backup state
        self.consecutive_planning_failures = 0 # Track failures to clear blacklist
        
        # --- Logging for evaluation ---
        self.logger = CsvLogger("eval_logs/live_frontier.csv")
        self.step_count = 0
        self.output_dir = Path("eval_logs/live_slam_frontier")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for old in self.output_dir.glob("*.png"):
            old.unlink()
        
        print("✅ Initialized SMART FRONTIER exploration (Simplified Mode)")
        print(f"Grid: {grid_spec.width}x{grid_spec.height} @ {grid_spec.resolution} m")

    # ---------- helpers ----------

    def _grid_codes(self) -> np.ndarray:
        lo = self.slam.grid.log_odds
        codes = np.zeros(lo.shape, dtype=int)
        codes[lo < 0.0] = -1
        codes[lo > 0.0] = 1
        return codes

    def _front_obstacle_min(self, ranges: np.ndarray) -> float | None:
        if ranges.size == 0:
            return None
        num_beams = ranges.size
        center = num_beams // 2
        front_sector = num_beams // 6  # ±30°
        
        start = max(0, center - front_sector)
        end = min(num_beams, center + front_sector)
        front = ranges[start:end]
        
        valid = front[np.isfinite(front) & (front > 0.01)]
        if valid.size == 0:
            return None
        return float(valid.min())

    # ---------- main loop ----------

    def run(self, max_steps: int = 5000, viz_every: int = 50):
        print("\n🚀 Starting SMART FRONTIER exploration...")
        dt = self.robot.timestep / 1000.0

        print("⏳ Warming up sensors...")
        for _ in range(20):
            self.robot.robot.step(self.robot.timestep)

        print("⏳ Waiting for valid LiDAR data...")
        while self.robot.robot.step(self.robot.timestep) != -1:
            packet = self.robot.create_sensor_packet(dt)
            ranges = np.array(packet["lidar"]["ranges"], dtype=float)
            if ranges.size > 0 and np.isfinite(ranges).sum() > 0:
                print("✅ LiDAR ready")
                break

        while (
            self.robot.robot.step(self.robot.timestep) != -1
            and self.step_count < max_steps
        ):
            # 1) Read sensors
            pkt = self.robot.create_sensor_packet(dt)
            odom = pkt["odometry"]
            lidar = pkt["lidar"]
            lidar_ranges = np.array(lidar["ranges"], dtype=float)

            # 2) SLAM update
            control = (odom["v_l"], odom["v_r"])
            scan = LaserScan(
                ranges=lidar_ranges,
                angle_min=lidar["angle_min"],
                angle_inc=lidar["angle_increment"],
                range_min=0.1,
                range_max=4.0, 
            )
            pose = self.slam.step(control=control, dt=dt, scan=scan)
            x, y, theta = pose
            pose_ij = self.slam.grid.world_to_grid(x, y)

            # --- Recovery Maneuvers ---
            # Priority 1: Backup
            if self.backup_counter > 0:
                self.backup_counter -= 1
                self.waypoint.set_velocity_vw(-0.15, 0.0)
                self.current_path = [] # Clear path
                self.stuck_counter = 0
                self.step_count += 1
                
                # Transition to turn when done
                if self.backup_counter == 0:
                     self.turn_counter = 30 # Turn for ~1s
                
                continue
            
            # Priority 2: Turn (Re-orient)
            if self.turn_counter > 0:
                self.turn_counter -= 1
                # Spin left or right? Let's just spin left.
                self.waypoint.set_velocity_vw(0.0, 1.0)
                self.current_path = []
                self.stuck_counter = 0
                self.step_count += 1
                continue

            # 3) Replanning Logic
            grid_codes = self._grid_codes()
            
            if self.planning_fail_cooldown > 0:
                self.planning_fail_cooldown -= 1
            
            elif not self.current_path or (self.step_count % 100 == 0):
                # Detect
                frontiers = detect_frontiers(grid_codes, unknown_val=0, free_val=-1)
                clusters = cluster_frontiers(frontiers)
                
                curr_target_centroid = None
                if self.current_target_frontier:
                    c = self.current_target_frontier.centroid
                    cx = self.origin_xy[0] + c[1] * self.resolution
                    cy = self.origin_xy[1] + c[0] * self.resolution
                    curr_target_centroid = (cx, cy)

                # Select & Plan
                selected_frontier, new_path = choose_frontier(
                    clusters,
                    pose_ij,
                    grid_codes,
                    self.resolution,
                    self.origin_xy,
                    self.path_planner,
                    current_target_xy=curr_target_centroid,
                    blacklist=self.blacklist
                )
                
                if new_path is not None:
                    self.current_path = simplify_path(new_path, tolerance=0.1)
                    self.current_target_frontier = selected_frontier
                    print(f"✅ New plan to frontier: {len(new_path)} nodes")
                    
                    if self.current_path:
                        start_node = self.current_path[0]
                        dist_to_start = np.hypot(start_node[0]-x, start_node[1]-y)
                        if dist_to_start < 0.30: 
                            self.current_path.pop(0)
                    
                    self.stuck_counter = 0 # Reset stuck counter on new plan
                    self.consecutive_planning_failures = 0 # Reset failure counter
                else:
                    self.planning_fail_cooldown = 20
                    self.consecutive_planning_failures += 1
                    
                    if self.step_count % 50 == 0:
                        print("⚠️ No reachable frontier found. Planning cooled down.")
                    
                    # Desperation Mode: If we fail too many times, clear blacklist AND move
                    if self.consecutive_planning_failures > 3:
                        print("⚠️ Stagnation detected (3+ failures). Clearing blacklist & Forcing Turn.")
                        self.blacklist = []
                        self.consecutive_planning_failures = 0
                        self.turn_counter = 45 # Force a move to break the loop

            # 4) Path Following Control
            obstacle_min = self._front_obstacle_min(lidar_ranges)
            
            if self.current_path:
                target_wp = self.current_path[0]
                
                v, w, reached_wp = self.goto_goal.step(
                    (x, y, theta),
                    target_wp,
                    obstacle_min=obstacle_min
                )
                
                # Check STUCK condition (Safety Stop Triggered)
                # If v is 0 because of obstacle, but we haven't reached goal
                if v == 0.0 and obstacle_min is not None and obstacle_min < self.goto_goal.stop_dist:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Decay if moving

                # If stuck too long, blacklist, backup and replan
                if self.stuck_counter > 10: # ~0.3s - 0.5s
                    print(f"🛑 STUCK at obstacle ({obstacle_min:.2f}m). Starting Recovery (Backup+Turn).")
                    if self.current_target_frontier:
                        c = self.current_target_frontier.centroid
                        cx = self.origin_xy[0] + c[1] * self.resolution
                        cy = self.origin_xy[1] + c[0] * self.resolution
                        self.blacklist.append((cx, cy))
                    
                    self.current_path = [] # Force replan check next step
                    self.current_target_frontier = None
                    self.stuck_counter = 0
                    self.planning_fail_cooldown = 0
                    self.backup_counter = 60 # Reverse for 60 steps (~2s)
                
                # Check distance to waypoint to pop
                dist_to_wp = np.hypot(target_wp[0]-x, target_wp[1]-y)
                if dist_to_wp < 0.15: 
                    self.current_path.pop(0) 
                
                self.waypoint.set_velocity_vw(v, w)
            else:
                self.waypoint.set_velocity_vw(0.0, 0.0)

            # 5) Metrics + logging
            cov = coverage_percent(
                grid_codes, 
                resolution=self.resolution, 
                navigable_area_m2=self.slam.grid.spec.navigable_area,
                unknown_val=0 
            )
            exp_area = (grid_codes != 0).sum() * (self.resolution ** 2) 
            ent = entropy_proxy(grid_codes, unknown_val=0)

            self.logger.log(
                pose_x=x, pose_y=y, pose_theta=theta,
                chosen_frontier_i=None, 
                chosen_frontier_j=None, 
                goal_x=self.current_path[-1][0] if self.current_path else None,
                goal_y=self.current_path[-1][1] if self.current_path else None,
                num_frontiers=len(frontiers) if 'frontiers' in locals() else 0,
                coverage_pct=cov,
                explored_m2=exp_area,
                entropy_proxy=ent,
                strategy="smart_frontier_simple",
            )

            # 6) Console debug
            if self.step_count % 20 == 0:
                n_clusters = len(clusters) if 'clusters' in locals() else 0
                path_len = len(self.current_path)
                print(
                    f"[step {self.step_count:04d}] "
                    f"pose=({x:.2f},{y:.2f}) "
                    f"clusters={n_clusters} "
                    f"path_nodes={path_len} "
                    f"cov={cov:.1f}%"
                )

            # 7) Map visualisation
            if self.step_count % viz_every == 0:
                out = self.output_dir / f"frontier_step_{self.step_count:04d}.png"
                plot_map(
                    grid=self.slam.grid,
                    pose=pose,
                    particles=self.slam.particles,
                    scan=scan,
                    title=f"Smart Frontier (Simple) - Step {self.step_count}",
                    output_path=str(out),
                    show_particles=True,
                    show_scan=True,      
                    scale=5,            
                )

            self.step_count += 1

        print("\n✅ Exploration complete")
        self.logger.close()


def main():
    print("=" * 60)
    print("  LIVE SLAM WITH SIMPLE FRONTIER EXPLORATION")
    print("=" * 60)
    explorer = FrontierExplorationSLAM()
    explorer.run(max_steps=5000, viz_every=50)

if __name__ == "__main__":
    main()
