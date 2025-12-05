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
        
        # DIAGNOSTIC: Track updates to specific cells to debug corruption
        DEBUG_WALL_CORRUPTION = False  # Set to True to enable logging
        if DEBUG_WALL_CORRUPTION:
            # Monitor cells in a specific range (adjust based on where corruption occurs)
            if 140 <= i <= 160 and 140 <= j <= 160:
                old_val = self.log_odds[i, j]
                new_val = np.clip(old_val + delta, self.spec.l_min, self.spec.l_max)
                if abs(delta) > 0.01:  # Only log significant changes
                    import traceback
                    caller = traceback.extract_stack()[-2]
                    action = "FREE" if delta < 0 else "OCC"
                    print(f"  [{action}] Cell ({i},{j}): {old_val:.2f} → {new_val:.2f} (Δ={delta:.2f}) from {caller.filename}:{caller.lineno}")
        
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
        
        Returns:
            np.ndarray: Trinary map where:
                -1 = FREE (navigable space)
                 0 = UNKNOWN (unexplored)
                 1 = OCCUPIED (obstacles)
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

        
# --- Helper functions ---

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm from toolbuddy/2D-Grid-SLAM.
    Returns a list of (x, y) tuples from (x0, y0) towards (x1, y1).
    """
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec


def update_map(
    grid: OccupancyGrid,
    pose: Pose,
    scan: LaserScan,
    beam_subsample: int = 1,
    apply_free_and_occ: bool = True,
) -> None:
    """
    Updates the map using toolbuddy's "Bresenham + Tip Thickening" logic.
    
    Logic:
    - Trace line from Robot to Hit using Bresenham.
    - Mark the path (except last 2 pixels) as Free (-0.7).
    - Mark the last 2 pixels as Occupied (0.9).
    - This creates a "Thick Wall" effect.
    """
    if not apply_free_and_occ:
        return

    x_r, y_r, th_r = pose
    ranges = np.asarray(scan.ranges)
    
    # toolbuddy parameters
    lo_free = -0.7
    lo_occ = 0.9
    
    # Pre-calculate beam angles
    angles = th_r + (scan.angle_min + np.arange(len(ranges)) * scan.angle_inc)
    
    # Iterate through beams
    for k in range(0, len(ranges), max(1, beam_subsample)):
        r = ranges[k]
        
        # Skip invalid ranges
        if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
            continue
            
        # Calculate endpoint in world coordinates
        x_end = x_r + r * np.cos(angles[k])
        y_end = y_r + r * np.sin(angles[k])
        
        # Convert to grid coordinates
        i_start, j_start = grid.world_to_grid(x_r, y_r)
        i_end, j_end = grid.world_to_grid(x_end, y_end)
        
        # Trace line using toolbuddy's Bresenham
        # Note: toolbuddy's Bresenham excludes the final point (x1, y1)
        # So 'rec' contains the path up to the endpoint
        rec = bresenham_line(i_start, j_start, i_end, j_end)
        
        # FIX: toolbuddy's Bresenham excludes the endpoint, so we must add it manually
        # to ensure the actual hit is marked.
        rec.append((i_end, j_end))
        
        # Apply toolbuddy's update rule
        for idx, (i, j) in enumerate(rec):
            if not grid.in_bounds(i, j):
                continue
                
            # Logic: Last 2 pixels are Occupied, rest are Free
            if idx < len(rec) - 2:
                change = lo_free
                
                # STICKY WALLS: Do not overwrite occupied cells with free space
                # If cell is already occupied (log_odds > 0), skip free update
                if grid.log_odds[i, j] > 0.0:
                    continue
            else:
                change = lo_occ
                
            grid.log_odds[i, j] += change
            
            # Clamp log-odds (toolbuddy uses -5.0 to 5.0)
            grid.log_odds[i, j] = max(-5.0, min(5.0, grid.log_odds[i, j]))
