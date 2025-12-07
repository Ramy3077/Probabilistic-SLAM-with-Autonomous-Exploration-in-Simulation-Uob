#!/usr/bin/env python3
# Live SLAM with FRONTIER-based exploration in Webots

import sys
from pathlib import Path
import numpy as np

# --- Make local packages importable ---
sys.path.insert(0, str(Path(__file__).parent.parent))

# Robot + SLAM imports
# Robot + SLAM imports
from robots.robot import MyRobot, Waypoint
from slam.fastslam import FastSLAM, FastSLAMConfig
from slam.occupancy import OccupancyGrid, LaserScan
from slam.visualize import plot_map
from models.motion import sample_motion_simple
from configs.config_loader import load_grid_spec

# Exploration + control imports (YOUR modules)
from control.waypoint_follow import GoToGoal
from explore.frontiers import detect_frontiers
from explore.planner import choose_frontier
from explore.utils import ij_to_xy
from eval.metrics import coverage_percent, entropy_proxy, explored_area_m2
from eval.logger import CsvLogger


class FrontierExplorationSLAM:
    def __init__(self):
        # --- Webots robot wrapper ---
        self.robot = MyRobot()
        self.waypoint = Waypoint(self.robot)  # we will only use set_velocity_vw()

        # --- Occupancy grid (loaded from config) ---
        grid_spec = load_grid_spec()
        self.grid = OccupancyGrid(grid_spec)

        # --- FastSLAM config (same as random script) ---
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

        # --- Frontier planner + controller state ---
        self.goto_goal = GoToGoal()
        self.origin_xy = (grid_spec.origin_x, grid_spec.origin_y)
        self.resolution = grid_spec.resolution

        # --- Logging for evaluation ---
        self.logger = CsvLogger("eval_logs/live_frontier.csv")

        self.step_count = 0
        self.output_dir = Path("eval_logs/live_slam_frontier")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clear old PNGs
        for old in self.output_dir.glob("*.png"):
            old.unlink()

        print("✅ Initialized FRONTIER exploration SLAM")
        print(f"Grid: {grid_spec.width}x{grid_spec.height} @ {grid_spec.resolution} m")
        print(f"Particles: {config.num_particles}")
        print(f"Log file: eval_logs/live_frontier.csv")

    # ---------- helpers ----------

    def _grid_codes(self) -> np.ndarray:
        """
        Convert log-odds grid → integer codes {-1: unknown, 0: free, 1: occupied}
        so it can be consumed by frontier + metrics modules.
        """
        lo = self.slam.grid.log_odds
        codes = np.full(lo.shape, -1, dtype=int)   # unknown by default
        codes[lo > 0.0] = 1                        # occupied
        codes[lo < 0.0] = 0                        # free
        return codes

    def _front_obstacle_min(self, ranges: np.ndarray) -> float | None:
        """
        Minimal distance in a front sector of the LiDAR.
        Used as obstacle_min for GoToGoal.
        """
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
        print("\n🚀 Starting FRONTIER exploration...")
        dt = self.robot.timestep / 1000.0  # 64 ms → 0.064 s

        # --- Warmup sensors ---
        print("⏳ Warming up sensors...")
        for _ in range(20):
            self.robot.robot.step(self.robot.timestep)

        # --- Wait for first valid LiDAR scan ---
        print("⏳ Waiting for valid LiDAR data...")
        while self.robot.robot.step(self.robot.timestep) != -1:
            packet = self.robot.create_sensor_packet(dt)
            ranges = np.array(packet["lidar"]["ranges"], dtype=float)
            if ranges.size > 0:
                finite = np.isfinite(ranges)
                print(f"✅ LiDAR ready: {finite.sum()}/{ranges.size} finite beams")
                break

        # --- Main control loop ---
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
                range_max=2.0,
            )
            pose = self.slam.step(control=control, dt=dt, scan=scan)
            x, y, theta = pose

            # 3) Frontier detection on current map
            grid_codes = self._grid_codes()
            pose_i, pose_j = self.slam.grid.world_to_grid(x, y)
            frontiers = detect_frontiers(grid_codes)
            goal_ij = choose_frontier(frontiers, (pose_i, pose_j))

            if goal_ij is not None:
                goal_xy = ij_to_xy(
                    goal_ij[0],
                    goal_ij[1],
                    origin_xy=self.origin_xy,
                    resolution=self.resolution,
                )
            else:
                goal_xy = None

            # 4) Go-to-goal controller
            obstacle_min = self._front_obstacle_min(lidar_ranges)

            if goal_xy is not None:
                v, w, reached = self.goto_goal.step(
                    (x, y, theta),
                    goal_xy,
                    obstacle_min=obstacle_min,
                )
                # convert (v, w) → wheel speeds using your existing helper
                self.waypoint.set_velocity_vw(v, w)
            else:
                # No frontier: stop (or you could fall back to a tiny random motion)
                self.waypoint.set_velocity_vw(0.0, 0.0)
                reached = True

            # 5) Metrics + logging
            # Calculate metrics using new signatures
            cov = coverage_percent(
                grid_codes, 
                resolution=self.resolution, 
                navigable_area_m2=self.slam.grid.spec.navigable_area
            )
            exp_m2 = (grid_codes != -1).sum() * (self.resolution ** 2)
            ent = entropy_proxy(grid_codes)

            self.logger.log(
                pose_x=x,
                pose_y=y,
                pose_theta=theta,
                chosen_frontier_i=goal_ij[0] if goal_ij else None,
                chosen_frontier_j=goal_ij[1] if goal_ij else None,
                goal_x=goal_xy[0] if goal_xy else None,
                goal_y=goal_xy[1] if goal_xy else None,
                num_frontiers=len(frontiers),
                coverage_pct=cov,
                explored_m2=exp_m2, # Log absolute area
                entropy_proxy=ent,
                strategy="frontier",
            )

            # 6) Console debug every few steps
            if self.step_count % 10 == 0:
                print(
                    f"[step {self.step_count:04d}] "
                    f"pose=({x:5.2f},{y:5.2f},{theta:5.2f})  "
                    f"frontiers={len(frontiers):3d}  "
                    f"cov={cov:5.2f}%  ent={ent:4.2f}  "
                    f"goal={goal_xy}"
                )

            # 7) Map visualisation
            if self.step_count % viz_every == 0:
                out = self.output_dir / f"frontier_step_{self.step_count:04d}.png"
                plot_map(
                    grid=self.slam.grid,
                    pose=pose,
                    particles=self.slam.particles,
                    scan=scan,
                    title=f"Frontier Exploration - Step {self.step_count}",
                    output_path=str(out),
                    show_particles=True,
                    show_scan=True,
                    scale=5,
                )

            self.step_count += 1

        # --- Final visualisation + summary ---
        final_pose = self.slam.best_pose()
        final_path = self.output_dir / "frontier_final.png"
        plot_map(
            grid=self.slam.grid,
            pose=final_pose,
            particles=self.slam.particles,
            scan=None,
            title=f"Frontier Final Map - {self.step_count} steps",
            output_path=str(final_path),
            show_particles=True,
            show_scan=False,
            scale=10,
        )
        self.logger.close()

        print("\n✅ Frontier exploration complete")
        print(f"   Steps: {self.step_count}")
        print(f"   Final pose: ({final_pose[0]:.2f}, {final_pose[1]:.2f}, {final_pose[2]:.2f})")
        print("   Logs: eval_logs/live_frontier.csv")
        print(f"   Figures: {self.output_dir}")


def main():
    print("=" * 60)
    print("  LIVE SLAM WITH FRONTIER-BASED EXPLORATION")
    print("=" * 60)
    explorer = FrontierExplorationSLAM()
    explorer.run(max_steps=5000, viz_every=50)
    print("\n✨ Done! Check eval_logs/live_frontier.csv and eval_logs/live_slam_frontier/.")


if __name__ == "__main__":
    main()
