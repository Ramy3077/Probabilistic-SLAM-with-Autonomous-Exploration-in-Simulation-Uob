"""Browser demo driver.

The control flow follows FrontierExplorationSLAM.run() in
controllers/robot_controller/scripts/live_slam_frontier.py, one tick per
call to step(), with the Webots robot replaced by sim.SimRobot. Every
algorithmic piece is imported from the project unchanged:

    slam.fastslam      ParticleFilterSLAM (particle filter + occupancy map)
    slam.occupancy     OccupancyGrid, LaserScan, inverse sensor model
    models.motion      sample_motion_diffdrive (odometry motion model)
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
from models.motion import sample_motion_diffdrive
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
            motion_model=sample_motion_diffdrive,
            config=FastSLAMConfig(num_particles=num_particles, beam_subsample=1, resample_threshold_ratio=0.5),
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
        self.done_reason = ""
        # demo-side recovery state (see _approach_goal and _watchdog)
        self.goal_best_dist = math.inf
        self.goal_best_step = 0
        self.target_cell = None
        self.approach_until = 0
        self.looked: list = []
        self.anchor = (self.world.start[0], self.world.start[1], 0)
        self.ignored = np.zeros((self.world.height, self.world.width), dtype=bool)
        self._fmask = None
        self.known_cells = 0
        self.progress_step = 0

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
        if self.done:
            return 0.0, 0.0
        self._watchdog()
        self._goal_timeout(x, y)
        self._no_motion(x, y)
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
        elif self.current_path and self.approach_until > self.step_count:
            pass  # following a fallback path; it is re-planned every 30 ticks
        elif not self.current_path or self.step_count % 10 == 0:
            # The Webots loop re-plans every tick while the path is short; here
            # it is every 10 ticks (0.64 s) so failed A* searches cost less.
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
            self._drop_target()
            self.current_path = []
            self.stuck_counter = 0
            self.planning_fail_cooldown = 0
            self.backup_counter = 40
        return v, w

    def _replan(self, x, y, pose_ij):
        grid_codes = self._grid_codes()
        frontiers = detect_frontiers(grid_codes, unknown_val=0, free_val=-1)
        fmask = np.zeros(grid_codes.shape, dtype=bool)
        if frontiers:
            fi, fj = np.asarray(frontiers).T
            fmask[fi, fj] = True
        self._fmask = fmask
        # Frontier cells the demo has given up on (see _give_up) are left out.
        live = [c for c in frontiers if not self.ignored[c]]
        self.frontier_cells = live
        clusters = cluster_frontiers(live)
        if not clusters:
            self._finish("no reachable frontiers left")
            return
        curr = self._cell_xy(self.target_cell) if self.target_cell is not None else None
        selected, new_path = choose_frontier(
            clusters, pose_ij, grid_codes, self.resolution, self.origin_xy,
            self.path_planner, current_target_xy=curr, blacklist=self.blacklist,
        )
        target = None
        if new_path is not None:
            target = self.slam.grid.world_to_grid(*new_path[-1])
            if self._path_len(new_path, x, y) < 0.3:
                # Standing on the chosen goal and the frontier is still there:
                # turn on the spot once, then give up on it.
                if self._looked_already(target):
                    self._give_up(target)
                else:
                    self.turn_counter = 50
                    return
                new_path = None
        if new_path is None:
            # explore.planner only scores the three nearest clusters and drops
            # any whose centroid is within 35 cm of an occupied cell. The last
            # frontiers of a room are usually thin LiDAR fans seen through a
            # doorway, which fail both tests, so the robot used to park. Fall
            # back to driving to the reachable cell closest to any frontier.
            new_path, target = self._approach_goal(clusters, pose_ij, grid_codes, x, y)
            if target == "looking":
                return
            if new_path is not None:
                self.approach_until = self.step_count + 30
        if new_path is not None:
            if self.target_cell is None or max(abs(target[0] - self.target_cell[0]), abs(target[1] - self.target_cell[1])) > 10:
                self.goal_best_dist = math.inf
                self.goal_best_step = self.step_count
            self.target_cell = target
            self.current_target_frontier = selected
            self.current_path = simplify_path(new_path, tolerance=0.1)
            self.goal_xy = self.current_path[-1] if self.current_path else None
            if self.current_path:
                s = self.current_path[0]
                if math.hypot(s[0] - x, s[1] - y) < 0.30:
                    self.current_path.pop(0)
            self.stuck_counter = 0
            self.consecutive_planning_failures = 0
        else:
            self.goal_xy = None
            self.target_cell = None
            self.planning_fail_cooldown = 20
            self.consecutive_planning_failures += 1
            if self.consecutive_planning_failures == 3:
                self.turn_counter = 45  # look around once from here
            elif self.consecutive_planning_failures > 6:
                # Still nothing reachable after turning on the spot.
                self._finish("no reachable frontiers left")

    # -- demo-side recovery: stuck goals, unreachable frontiers, termination --
    @staticmethod
    def _path_len(path, x, y):
        pts = [(x, y)] + list(path)
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    def _cell_xy(self, ij):
        return self.path_planner._grid_to_world(ij, self.origin_xy, self.resolution)

    def _give_up(self, cell, radius=0.75):
        """Stop targeting the current frontier cells within radius of cell."""
        if cell is None or self._fmask is None:
            return
        r = int(round(radius / self.resolution))
        i, j = cell
        H, W = self.ignored.shape
        i0, i1, j0, j1 = max(0, i - r), min(H, i + r + 1), max(0, j - r), min(W, j + r + 1)
        di, dj = np.mgrid[i0 - i:i1 - i, j0 - j:j1 - j]
        disk = di * di + dj * dj <= r * r
        self.ignored[i0:i1, j0:j1] |= disk & self._fmask[i0:i1, j0:j1]

    def _looked_already(self, cell):
        if any(max(abs(cell[0] - a), abs(cell[1] - b)) <= 10 for a, b in self.looked):
            return True
        self.looked.append(cell)
        return False

    def _drop_target(self):
        self._give_up(self.target_cell)
        self.target_cell = None
        self.current_target_frontier = None
        self.goal_best_dist = math.inf

    def _goal_timeout(self, x, y):
        """Give up on a goal the robot has not got 25 cm closer to in 12 s."""
        if self.goal_xy is None:
            return
        d = math.hypot(self.goal_xy[0] - x, self.goal_xy[1] - y)
        if d < self.goal_best_dist - 0.25:
            self.goal_best_dist = d
            self.goal_best_step = self.step_count
        elif (self.step_count - self.goal_best_step) * DT > 12.0:
            self._drop_target()
            self.current_path = []
            self.goal_xy = None
            self.planning_fail_cooldown = 0
            self.goal_best_step = self.step_count

    def _no_motion(self, x, y):
        """Wedged against a wall that the front sector does not see: the
        estimate has not moved 20 cm in 8 s while a path is being followed."""
        if math.hypot(x - self.anchor[0], y - self.anchor[1]) > 0.2 or not self.current_path:
            self.anchor = (x, y, self.step_count)
        elif (self.step_count - self.anchor[2]) * DT > 8.0:
            self._drop_target()
            self.current_path = []
            self.goal_xy = None
            self.backup_counter = 40
            self.anchor = (x, y, self.step_count)

    def _watchdog(self):
        """Stop the run if the estimated map has not grown by 0.5% of the
        grid in 60 s of simulated time. Uses the robot's own map only."""
        if self.step_count % 25:
            return
        known = int((self.slam.grid.log_odds != 0).sum())
        if known > self.known_cells + 0.005 * self.slam.grid.log_odds.size:
            self.known_cells = known
            self.progress_step = self.step_count
        elif (self.step_count - self.progress_step) * DT > 60.0:
            self._finish("map stopped growing")

    def _finish(self, reason):
        self.done = True
        self.done_reason = reason
        self.current_path = []
        self.goal_xy = None

    @staticmethod
    def _wavefront(passable, seeds, max_iter=2000):
        """8-connected BFS distance (in cells) from the seed mask through passable cells."""
        dist = np.full(passable.shape, -1, dtype=np.int32)
        cur = seeds & passable
        dist[cur] = 0
        k = 0
        while cur.any() and k < max_iter:
            k += 1
            pad = np.pad(cur, 1)
            nxt = (pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] | pad[1:-1, :-2] | pad[1:-1, 2:]
                   | pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:])
            cur = nxt & passable & (dist < 0)
            dist[cur] = k
        return dist

    def _approach_goal(self, clusters, pose_ij, grid_codes, x, y):
        fmask = np.zeros(grid_codes.shape, dtype=bool)
        for c in clusters:
            ci, cj = zip(*c.cells)
            fmask[list(ci), list(cj)] = True
        fmask &= ~self.ignored
        if not fmask.any():
            return None, None
        # distance to the nearest frontier cell, not passing through walls
        d_front = self._wavefront(grid_codes != 1, fmask)
        H, W = grid_codes.shape
        i0, j0 = pose_ij
        for safety in (self.path_planner.safety_distance, 0.35, 0.30):
            free = ~self.path_planner._blocked(grid_codes, int(math.ceil(safety / self.resolution)))
            # start from the nearest drivable cell (the estimate can sit in a wall smear)
            r = 10
            win = free[max(0, i0 - r):i0 + r + 1, max(0, j0 - r):j0 + r + 1]
            if not win.any():
                continue
            wi, wj = np.nonzero(win)
            wi = wi + max(0, i0 - r)
            wj = wj + max(0, j0 - r)
            k = int(np.argmin((wi - i0) ** 2 + (wj - j0) ** 2))
            seed = np.zeros_like(free)
            seed[int(wi[k]), int(wj[k])] = True
            d_robot = self._wavefront(free, seed)
            ok = (d_robot >= 0) & (d_front >= 0)
            if not ok.any():
                continue
            # closest to a frontier first, then shortest drive
            score = np.where(ok, d_front * 1000 + d_robot, np.iinfo(np.int32).max)
            gi, gj = (int(v) for v in np.unravel_index(int(np.argmin(score)), score.shape))
            fi, fj = np.nonzero(fmask)
            n = int(np.argmin((fi - gi) ** 2 + (fj - gj) ** 2))
            target = (int(fi[n]), int(fj[n]))  # the frontier cell being approached
            # walk the distance field back from the goal to the start
            path = [(gi, gj)]
            ci, cj = gi, gj
            while d_robot[ci, cj] > 0:
                nxt = next((ci + di, cj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                           if 0 <= ci + di < H and 0 <= cj + dj < W and d_robot[ci + di, cj + dj] == d_robot[ci, cj] - 1)
                ci, cj = nxt
                path.append(nxt)
            path.reverse()
            path_xy = [self._cell_xy(p) for p in path]
            if self._path_len(path_xy, x, y) >= 0.3:
                return path_xy, target
            if d_front[gi, gj] <= 20:
                # Within 1 m of the frontier and as close as the drivable
                # space allows. Turn on the spot once; if the frontier is
                # still there next time, it cannot be cleared: give up on it.
                if not self._looked_already(target):
                    self.turn_counter = 50
                    return None, "looking"
                self._give_up(target)
                return self._approach_goal(clusters, pose_ij, grid_codes, x, y)
            # Boxed in at this margin: try the next, smaller one.
        return None, None

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
            "done_reason": self.done_reason,
        }


def silence_prints():
    """GoToGoal prints on every call; shadow print in those modules only
    (their source is untouched) to keep the browser console readable."""
    import control.path_planner as pp
    import control.waypoint_follow as wf
    wf.print = _quiet
    pp.print = _quiet
