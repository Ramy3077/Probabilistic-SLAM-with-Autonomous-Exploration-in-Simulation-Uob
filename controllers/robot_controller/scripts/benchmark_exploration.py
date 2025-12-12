
import sys
import os
import time
from pathlib import Path
import numpy as np

# --- Webots/Project Imports ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from robots.robot import MyRobot, Waypoint
from slam.fastslam import ParticleFilterSLAM as FastSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, GridSpec, LaserScan
from models.motion import sample_motion_simple
from eval.metrics import coverage_percent, entropy_proxy
from eval.logger import CsvLogger

# --- Strategies Imports ---
from control.waypoint_follow import GoToGoal
from control.path_planner import AStarPlanner, simplify_path
from explore.frontiers import detect_frontiers, cluster_frontiers
from explore.planner import choose_frontier

class BenchmarkRunner:
    def __init__(self, mode: str, max_steps: int = 1000):
        self.mode = mode.upper() # "FRONTIER" or "RANDOM"
        self.max_steps = max_steps
        
        # --- Robot & Grid Setup ---
        self.robot = MyRobot()
        self.waypoint = Waypoint(self.robot)
        
        grid_spec = GridSpec(
            resolution=0.05,
            width=400,
            height=400,
            origin_x=-10.0,
            origin_y=-10.0,
            l_occ=0.85,
            l_free=-0.4,
        )
        self.grid = OccupancyGrid(grid_spec)
        self.origin_xy = (grid_spec.origin_x, grid_spec.origin_y)
        self.resolution = grid_spec.resolution
        
        # --- SLAM Setup ---
        init_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        config = FastSLAMConfig(num_particles=50, beam_subsample=3, resample_threshold_ratio=0.5)
        self.slam = FastSLAM(self.grid, init_pose, sample_motion_simple, config)
        
        # --- Logging ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = f"eval_logs/benchmark_{self.mode.lower()}_{timestamp}.csv"
        self.logger = CsvLogger(self.log_file)
        
        # --- Strategy Specific Init ---
        if self.mode == "FRONTIER":
            self.init_frontier()
        elif self.mode == "RANDOM":
            self.init_random()
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        print(f"✅ Initialized Benchmark: {self.mode} for {self.max_steps} steps")
        print(f"📝 Logging to: {self.log_file}")

    def init_frontier(self):
        self.path_planner = AStarPlanner(safety_distance=0.28, allow_diagonal=True)
        self.goto_goal = GoToGoal(v_max=0.25, w_max=2.0)
        
        self.current_path = []
        self.current_target_frontier = None
        self.blacklist = []
        self.planning_fail_cooldown = 0
        self.stuck_counter = 0
        self.backup_counter = 0
        self.turn_counter = 0
        self.consecutive_planning_failures = 0

    def init_random(self):
        # State machine for Random: FORWARD, TURN_RIGHT, TURN_LEFT
        self.state = 'FORWARD'
        self.state_steps = 0
        self.obstacle_thresh = 0.8
        
    # --- Loop ---
    def run(self):
        dt = self.robot.timestep / 1000.0
        step_count = 0
        
        # Warmup
        print("⏳ Warming up sensors...")
        for _ in range(20): self.robot.robot.step(self.robot.timestep)
        
        # Wait for LiDAR
        print("⏳ Waiting for valid LiDAR...")
        while self.robot.robot.step(self.robot.timestep) != -1:
            pkt = self.robot.create_sensor_packet(dt)
            if len(pkt["lidar"]["ranges"]) > 0: break
            
        print("🚀 Starting Benchmark Run...")
        
        while self.robot.robot.step(self.robot.timestep) != -1 and step_count < self.max_steps:
            # 1. Sensors
            pkt = self.robot.create_sensor_packet(dt)
            odom = pkt["odometry"]
            lidar = pkt["lidar"]
            ranges = np.array(lidar["ranges"], dtype=float)
            
            # 2. SLAM
            scan = LaserScan(ranges, lidar["angle_min"], lidar["angle_increment"], 0.1, 4.0)
            pose = self.slam.step((odom["v_l"], odom["v_r"]), dt, scan)
            x, y, theta = pose
            
            # 3. Strategy Execution
            if self.mode == "FRONTIER":
                self.step_frontier(pose, ranges)
            else:
                self.step_random(pose, ranges)
                
            # 4. Logging
            grid_codes = self._grid_codes()
            cov = coverage_percent(grid_codes, self.resolution, self.grid.spec.navigable_area, 0)
            ent = entropy_proxy(grid_codes, unknown_val=0)
            exp_area = (grid_codes != 0).sum() * (self.resolution ** 2)
            
            self.logger.log(
                pose_x=x, pose_y=y, pose_theta=theta,
                coverage_pct=cov,
                explored_m2=exp_area,
                entropy_proxy=ent,
                strategy=self.mode,
                step=step_count # helper column
            )
            
            if step_count % 100 == 0:
                print(f"Step {step_count}/{self.max_steps}: Cov={cov:.1f}% Ent={ent:.1f}")
                
            step_count += 1
            
        print(f"🏁 Benchmark Complete. Saved to {self.log_file}")
        self.logger.close()

    # --- Helpers ---
    def _grid_codes(self):
        lo = self.slam.grid.log_odds
        codes = np.zeros(lo.shape, dtype=int)
        codes[lo < 0.0] = -1
        codes[lo > 0.0] = 1
        return codes

    def _front_obstacle(self, ranges, sector_factor=6):
        if ranges.size == 0: return None
        n = ranges.size
        # Center sector
        center = n // 2
        sector = n // sector_factor
        front = ranges[max(0, center-sector):min(n, center+sector)]
        valid = front[np.isfinite(front) & (front > 0.01)]
        return float(valid.min()) if valid.size > 0 else None


    def step_frontier(self, pose, ranges):
        x, y, theta = pose
        pose_ij = self.slam.grid.world_to_grid(x, y)
        
        
        if self.backup_counter > 0:
            self.backup_counter -= 1
            self.waypoint.set_velocity_vw(-0.15, 0.0)
            if self.backup_counter == 0: self.turn_counter = 30
            return
        if self.turn_counter > 0:
            self.turn_counter -= 1
            self.waypoint.set_velocity_vw(0.0, 1.0)
            return

        # Planning
        if not self.current_path or (self.planning_fail_cooldown == 0 and len(self.current_path) < 2):
             if self.planning_fail_cooldown > 0: self.planning_fail_cooldown -= 1
             else:
                grid_codes = self._grid_codes()
                frontiers = detect_frontiers(grid_codes, 0, -1)
                clusters = cluster_frontiers(frontiers)
                
                # Target existing centroid if valid
                curr_xy = None
                if self.current_target_frontier:
                     c = self.current_target_frontier.centroid
                     curr_xy = (self.origin_xy[0] + c[1]*self.resolution, self.origin_xy[1] + c[0]*self.resolution)

                sel, new_path = choose_frontier(
                    clusters, 
                    pose_ij, 
                    grid_codes, 
                    self.resolution, 
                    self.origin_xy, 
                    self.path_planner, 
                    current_target_xy=curr_xy, 
                    blacklist=self.blacklist
                )
                
                if new_path:
                    self.current_path = simplify_path(new_path, 0.1)
                    self.current_target_frontier = sel
                    self.consecutive_planning_failures = 0
                else:
                    self.planning_fail_cooldown = 10
                    self.consecutive_planning_failures += 1
                    if self.consecutive_planning_failures > 3:
                        self.blacklist = []
                        self.turn_counter = 45 
                        self.consecutive_planning_failures = 0

        
        if self.current_path:
            obs = self._front_obstacle(ranges)
            target = self.current_path[0]
            v, w, reached = self.goto_goal.step((x,y,theta), target, obstacle_min=obs)
            
            
            if v == 0 and obs and obs < 0.28: self.stuck_counter += 1
            else: self.stuck_counter = 0
            
            if self.stuck_counter > 10:
                self.stuck_counter = 0
                self.backup_counter = 40
                self.current_path = []
                self.blacklist = [] 
            
            if np.hypot(target[0]-x, target[1]-y) < 0.15:
                self.current_path.pop(0)
            
            self.waypoint.set_velocity_vw(v, w)
        else:
            self.waypoint.set_velocity_vw(0.0, 0.0)


    def step_random(self, pose, ranges):

        obs = self._front_obstacle(ranges, sector_factor=6) # 60 deg
        has_obs = obs is not None and obs < self.obstacle_thresh
        
        self.state_steps += 1
        
        if self.state == 'FORWARD':
            if has_obs:
                self.state = 'TURN_RIGHT' if np.random.random() < 0.5 else 'TURN_LEFT'
                self.state_steps = 0
                self.robot.set_wheel_speeds(0, 0)
            else:
                self.robot.set_wheel_speeds(4.0, 4.0) # Full speed forward
                
        elif self.state in ['TURN_RIGHT', 'TURN_LEFT']:
            if self.state_steps > 10 and not has_obs:
                self.state = 'FORWARD'
                self.state_steps = 0
                self.robot.set_wheel_speeds(4.0, 4.0)
            else:
                if self.state == 'TURN_RIGHT':
                    self.robot.set_wheel_speeds(3.0, -3.0)
                else:
                    self.robot.set_wheel_speeds(-3.0, 3.0)

# --- Entry Point ---
def main():
    # CHANGE THIS TO "RANDOM" or "FRONTIER"
    MODE = "FRONTIER" 
    
    if len(sys.argv) > 1:
        MODE = sys.argv[1].upper()
        
    print(f"Running Benchmark with mode: {MODE}")
    runner = BenchmarkRunner(MODE, max_steps=1000)
    runner.run()

if __name__ == "__main__":
    main()
