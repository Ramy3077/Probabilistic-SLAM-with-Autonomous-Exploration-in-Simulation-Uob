# controllers/robot_controller/explore/planner.py
from typing import List, Tuple, Optional
import numpy as np

def choose_frontier(
    frontiers: List[Tuple[int, int]],
    pose_ij: Tuple[float, float],
) -> Optional[Tuple[int, int]]:
    """
    Args:
        frontiers: list of cells (i, j) returned by detect_frontiers().
        pose_ij: robot's current cell coordinates in grid index space.

    Returns:
        (i, j) of the selected frontier, or None if no frontiers exist.
    """
    if not frontiers:
        return None
    pi, pj = pose_ij
    return min(frontiers, key=lambda f: (f[0]-pi)**2 + (f[1]-pj)**2)
