"""Browser demo driver.

The control flow follows FrontierExplorationSLAM.run() in
controllers/robot_controller/scripts/live_slam_frontier.py, one tick per
call to step(), with the Webots robot replaced by sim.SimRobot. Every
algorithmic piece is imported from the project unchanged:

    slam.fastslam      ParticleFilterSLAM (particle filter + occupancy map)
    slam.occupancy     OccupancyGrid, LaserScan, inverse sensor model
    models.motion      sample_motion_simple (odometry motion model)
    explore.frontiers  detect_frontiers, cluster_frontiers
    explore.planner    choose_frontier
    control.*          AStarPlanner, simplify_path, GoToGoal
"""
from __future__ import annotations

import math

import numpy as np

from control.path_planner import AStarPlanner, simplify_path
from control.waypoint_follow import GoToGoal
from explore.frontiers import cluster_frontiers, detect_frontiers
from explore.planner import choose_frontier
from models.motion import sample_motion_simple
from slam.fastslam import FastSLAMConfig, ParticleFilterSLAM
from slam.occupancy import GridSpec, LaserScan, OccupancyGrid

from sim import LIDAR_MAX_RANGE, SimRobot, World

DT = 0.064  # Webots basic timestep used by the controller (64 ms)


class InflatedAStarPlanner(AStarPlanner):
    """AStarPlanner with the safety-margin test answered from a precomputed
    obstacle dilation instead of scanning a (2m+1)^2 window per node.
    Same answer as AStarPlanner._is_valid, fast enough for WebAssembly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_grid = None
        self._cache = {}

    def _blocked(self, grid: np.ndarray, m: int) -> np.ndarray:
        if grid is not self._cache_grid:
            self._cache_grid = grid
            self._cache = {}
        if m not in self._cache:
            occ = (grid == 1).astype(np.int32)
            pad = np.pad(occ, m + 1)
            s = pad.cumsum(0).cumsum(1)
            H, W = grid.shape
            k = 2 * m + 1
            win = s[k:k + H, k:k + W] - s[0:H, k:k + W] - s[k:k + H, 0:W] + s[0:H, 0:W]
            self._cache[m] = (win > 0) | (grid != -1)
        return self._cache[m]

    def _is_valid(self, pos, grid, margin_cells=0):
        i, j = pos
        rows, cols = grid.shape
        if not (0 <= i < rows and 0 <= j < cols):
            return False
        return not self._blocked(grid, int(margin_cells))[i, j]


def _quiet(*_a, **_k):
    pass


class Demo:
    def __init__(self, map_spec: dict, num_particles: int = 40, noise: float = 1.0, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.world = World(map_spec)
        self.robot = SimRobot(self.world, self.rng, noise)

        spec = GridSpec(
            resolution=self.world.res,
            width=self.world.width,
            height=self.world.height,
            origin_x=self.world.origin_x,
            origin_y=self.world.origin_y,
            l_occ=0.85,
            l_free=-0.4,
        )
        self.grid = OccupancyGrid(spec)
        self.slam = ParticleFilterSLAM(
            grid=self.grid,
            init_pose=self.world.start.copy(),
            motion_model=sample_motion_simple,
            config=FastSLAMConfig(num_particles=num_particles, beam_subsample=3, resample_threshold_ratio=0.5),
            rng=self.rng,
        )
        self.origin_xy = (spec.origin_x, spec.origin_y)
        self.resolution = spec.resolution
        self.path_planner = InflatedAStarPlanner(safety_distance=0.45, allow_diagonal=True)
        self.goto_goal = GoToGoal(kp_lin=0.8, v_max=0.5, w_max=2.5, stop_dist=0.40)

        self.current_path: list = []
        self.current_target_frontier = None
        self.blacklist: list = []
        self.planning_fail_cooldown = 0
        self.stuck_counter = 0
        self.backup_counter = 0
        self.turn_counter = 0
        self.consecutive_planning_failures = 0
        self.step_count = 0

        self.pose = self.world.start.copy()
        self.frontier_cells: list = []
        self.goal_xy = None
        self.last_ranges = np.zeros(0)
        self._odom = (0.0, 0.0)
        self.done = False

    # -- helpers copied from FrontierExplorationSLAM --
    def _grid_codes(self) -> np.ndarray:
        lo = self.slam.grid.log_odds
        codes = np.zeros(lo.shape, dtype=int)
        codes[lo < 0.0] = -1
        codes[lo > 0.0] = 1
        return codes

    @staticmethod
    def _front_obstacle_min(ranges: np.ndarray):
        if ranges.size == 0:
            return None
        n = ranges.size
        c = n // 2
        sector = n // 6
        front = ranges[max(0, c - sector):min(n, c + sector)]
        valid = front[np.isfinite(front) & (front > 0.01)]
        return float(valid.min()) if valid.size else None

    def _centroid_xy(self, f):
        c = f.centroid
        return (self.origin_xy[0] + c[1] * self.resolution, self.origin_xy[1] + c[0] * self.resolution)

    def set_noise(self, noise: float) -> None:
        self.robot.noise = float(noise)

    # -- one controller tick --
    def step(self) -> None:
        # 1) sensors (encoders were read when the robot moved last tick)
        v_l, v_r = self._odom
        ranges, angle_min, angle_inc = self.robot.scan()
        self.last_ranges = ranges

        # 2) SLAM update
        scan = LaserScan(ranges=ranges, angle_min=angle_min, angle_inc=angle_inc,
                         range_min=0.1, range_max=LIDAR_MAX_RANGE)
        pose = self.slam.step(control=(v_l, v_r), dt=DT, scan=scan)
        self.pose = np.array(pose, dtype=float)
        x, y, theta = self.pose
        pose_ij = self.slam.grid.world_to_grid(x, y)

        v, w = self._control(x, y, theta, pose_ij, ranges)
        self.robot.set_velocity_vw(v, w)
        self._odom = self.robot.advance(DT)

        self.step_count += 1

    def _control(self, x, y, theta, pose_ij, ranges):
        if self.backup_counter > 0:
            self.backup_counter -= 1
            self.current_path = []
            self.stuck_counter = 0
            if self.backup_counter == 0:
                self.turn_counter = 30
            return -0.30, 0.0
        if self.turn_counter > 0:
            self.turn_counter -= 1
            self.current_path = []
            self.stuck_counter = 0
            return 0.0, 1.0

        if self.planning_fail_cooldown > 0:
            self.planning_fail_cooldown -= 1
        elif len(self.current_path) < 15 or (self.step_count % 100 == 0):
            self._replan(x, y, pose_ij)

        obstacle_min = self._front_obstacle_min(ranges)
        if not self.current_path:
            return 0.0, 0.0

        lookahead = 0.60
        target_wp, target_idx = self.current_path[0], 0
        for i, wp in enumerate(self.current_path):
            target_wp, target_idx = wp, i
            if math.hypot(wp[0] - x, wp[1] - y) > lookahead:
                break
        while self.current_path:
            wp0 = self.current_path[0]
            d0 = math.hypot(wp0[0] - x, wp0[1] - y)
            if d0 < 0.3:
                if len(self.current_path) > 1:
                    self.current_path.pop(0)
                else:
                    if d0 < 0.1:
                        self.current_path.pop(0)
                    break
            else:
                break

        v, w, _ = self.goto_goal.step((x, y, theta), target_wp, obstacle_min=obstacle_min)
        is_final = target_idx == len(self.current_path) - 1
        if not is_final and v < self.goto_goal.v_max and abs(w) < 1.0:
            v = self.goto_goal.v_max

        if v == 0.0 and obstacle_min is not None and obstacle_min < self.goto_goal.stop_dist:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        if self.stuck_counter > 10:
            if self.current_target_frontier:
                self.blacklist.append(self._centroid_xy(self.current_target_frontier))
            self.current_path = []
            self.current_target_frontier = None
            self.stuck_counter = 0
            self.planning_fail_cooldown = 0
            self.backup_counter = 40
        return v, w

    def _replan(self, x, y, pose_ij):
        grid_codes = self._grid_codes()
        frontiers = detect_frontiers(grid_codes, unknown_val=0, free_val=-1)
        self.frontier_cells = frontiers
        clusters = cluster_frontiers(frontiers)
        curr = self._centroid_xy(self.current_target_frontier) if self.current_target_frontier else None
        selected, new_path = choose_frontier(
            clusters, pose_ij, grid_codes, self.resolution, self.origin_xy,
            self.path_planner, current_target_xy=curr, blacklist=self.blacklist,
        )
        if new_path is not None:
            self.current_path = simplify_path(new_path, tolerance=0.1)
            self.current_target_frontier = selected
            self.goal_xy = self.current_path[-1] if self.current_path else None
            if self.current_path:
                s = self.current_path[0]
                if math.hypot(s[0] - x, s[1] - y) < 0.30:
                    self.current_path.pop(0)
            self.stuck_counter = 0
            self.consecutive_planning_failures = 0
        else:
            self.goal_xy = None
            self.planning_fail_cooldown = 20
            self.consecutive_planning_failures += 1
            if self.consecutive_planning_failures > 3:
                self.blacklist = []
                self.consecutive_planning_failures = 0
                self.turn_counter = 45
            if not clusters:
                self.done = True

    # -- data for the renderer --
    def frame(self) -> dict:
        """Scalars plus numpy arrays; the worker reads the arrays as raw buffers."""
        poses, _ = self.slam.particles.as_arrays()
        true = self.robot.pose
        est = self.pose
        codes = self._grid_codes()
        free_world = ~self.world.occ
        ranges = np.where(np.isfinite(self.last_ranges), self.last_ranges, 0.0)
        return {
            "step": self.step_count,
            "t": self.step_count * DT,
            "true": np.asarray(true, dtype=np.float64),
            "est": np.asarray(est, dtype=np.float64),
            "particles": poses.astype(np.float32).ravel(),
            "map": (self.grid.probabilities() * 255).astype(np.uint8),
            "frontiers": np.asarray(self.frontier_cells, dtype=np.int32).reshape(-1),
            "path": np.asarray(self.current_path, dtype=np.float64).reshape(-1),
            "ranges": ranges.astype(np.float32),
            "err": float(math.hypot(true[0] - est[0], true[1] - est[1])),
            "herr": float(abs((true[2] - est[2] + math.pi) % (2 * math.pi) - math.pi)),
            "spread": float(np.sqrt(poses[:, 0].var() + poses[:, 1].var())),
            "n": len(poses),
            "explored": float(((codes != 0) & free_world).sum() / max(1, free_world.sum())),
            "done": self.done,
        }


def silence_prints():
    """GoToGoal prints on every call; shadow print in those modules only
    (their source is untouched) to keep the browser console readable."""
    import control.path_planner as pp
    import control.waypoint_follow as wf
    wf.print = _quiet
    pp.print = _quiet
