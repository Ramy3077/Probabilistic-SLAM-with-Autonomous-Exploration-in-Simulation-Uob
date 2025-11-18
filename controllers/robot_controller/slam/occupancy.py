from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


Pose = np.ndarray  # shape: (3,) -> [x, y, theta]


@dataclass
class LaserScan:
    """
    Attributes:
        ranges: np.ndarray shape (N,), distance per beam (meters)
        angle_min: starting angle relative to sensor frame (radians)
        angle_inc: angular increment per beam (radians)
        range_min: minimum valid range (meters)
        range_max: maximum valid range (meters)
    """

    ranges: np.ndarray
    angle_min: float
    angle_inc: float
    range_min: float
    range_max: float


@dataclass
class GridSpec:
    
    # Defines geometry and log-odds parameters of the occupancy grid.
    # The grid is indexed row-major as (i, j) with shape (height, width).
    # Origin is the world-frame coordinates of cell (0, 0) center.
    

    resolution: float  # meters per cell
    width: int
    height: int
    origin_x: float
    origin_y: float

    # Log-odds parameters
    l_occ: float = 0.85
    l_free: float = -0.4
    l_min: float = -4.0
    l_max: float = 4.0
    l0: float = 0.0  # prior


class OccupancyGrid:
    
    # 2D occupancy grid in log-odds form with basic utilities for
    # coordinate transforms and clamped log-odds updates.
    

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        self.log_odds = np.full(
            (spec.height, spec.width), fill_value=spec.l0, dtype=np.float32
        )

    def clear(self) -> None:
        self.log_odds.fill(self.spec.l0)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.spec.height and 0 <= j < self.spec.width

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        
        # Converts world coordinates (meters) to integer grid indices (i, j).
        # Returns the nearest cell indices.
        
        j = int(np.floor((x - self.spec.origin_x) / self.spec.resolution))
        i = int(np.floor((y - self.spec.origin_y) / self.spec.resolution))
        return i, j

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        
        # Returns center of cell (i, j) in world coordinates
        
        x = self.spec.origin_x + (j + 0.5) * self.spec.resolution
        y = self.spec.origin_y + (i + 0.5) * self.spec.resolution
        return x, y

    def _apply_delta(self, i: int, j: int, delta: float) -> None:
        if not self.in_bounds(i, j):
            return
        self.log_odds[i, j] = float(
            np.clip(self.log_odds[i, j] + delta, self.spec.l_min, self.spec.l_max)
        )

    def mark_free(self, i: int, j: int) -> None:
        self._apply_delta(i, j, self.spec.l_free)

    def mark_occupied(self, i: int, j: int) -> None:
        self._apply_delta(i, j, self.spec.l_occ)

    def probabilities(self) -> np.ndarray:
        
        # Returns occupancy probabilities via logistic transform p = 1 / (1 + exp(-l)).
        return 1.0 / (1.0 + np.exp(-self.log_odds))

def get_trinary_map(self) -> np.ndarray:
        """
        Converts the log-odds map into the {-1, 0, 1} format 
        for the frontier planner (Bassel).
        """
        # Define thresholds. You can tune these.
        occupied_thresh = self.spec.l_occ  # e.g., 0.85
        free_thresh = self.spec.l_free    # e.g., -0.4

        trinary_map = np.full(self.log_odds.shape, 0, dtype=np.int8)
        
        # Mark occupied cells as 1
        trinary_map[self.log_odds > occupied_thresh] = 1
        
        # Mark free cells as -1
        trinary_map[self.log_odds < free_thresh] = -1
        
        return trinary_map

        
def update_map(
    grid: OccupancyGrid,
    pose: Pose,
    scan: LaserScan,
    *,
    beam_subsample: int = 1,
    apply_free_and_occ: bool = False,
) -> None:
    
    # Update occupancy grid from a single scan at robot pose.
    """
    Template behavior:
        - If apply_free_and_occ is False (default), this is a no-op to keep the
          pipeline runnable during early scaffolding.
        - If True, applies a simple, approximate ray-carving update without
          performance optimizations or sensor error handling.

    Args:
        grid: OccupancyGrid to update in-place.
        pose: np.ndarray shape (3,) [x, y, theta] in world frame.
        scan: LaserScan with angles in robot frame, 0 pointing forward.
        beam_subsample: Use every k-th beam for speed.
        apply_free_and_occ: Enable naive free-space carving and hit marking.
    """
    if not apply_free_and_occ:
        return

    x_r, y_r, th_r = float(pose[0]), float(pose[1]), float(pose[2])

    ranges = np.asarray(scan.ranges, dtype=float)
    N = ranges.shape[0]
    if N == 0:
        return

    k_indices = range(0, N, max(1, int(beam_subsample)))
    res = grid.spec.resolution
    for k in k_indices:
        r = ranges[k]
        if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
            continue

        ang = th_r + (scan.angle_min + k * scan.angle_inc)
        x_end = x_r + r * np.cos(ang)
        y_end = y_r + r * np.sin(ang)

        i0, j0 = grid.world_to_grid(x_r, y_r)
        i1, j1 = grid.world_to_grid(x_end, y_end)

        # Discretize line using simple DDA-like stepping
        di = i1 - i0
        dj = j1 - j0
        steps = int(max(abs(di), abs(dj), 1))
        if steps == 0:
            continue
        for s in range(steps):
            ti = i0 + int(np.round(s * di / steps))
            tj = j0 + int(np.round(s * dj / steps))
            if (ti, tj) != (i1, j1):
                grid.mark_free(ti, tj)

        if r < (scan.range_max - 1e-6):
            grid.mark_occupied(i1, j1)



