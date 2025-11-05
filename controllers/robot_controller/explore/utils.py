# controllers/robot_controller/explore/utils.py
#for sahib to get world-meter goals
from typing import Tuple

def ij_to_xy(i: float, j: float, origin_xy: Tuple[float, float], resolution: float) -> Tuple[float, float]:
    """
    Convert grid index (i,j) to world meters (x,y).
    origin_xy: world coords (x0, y0) of grid cell (0,0).
    resolution: meters per cell.
    NOTE: Assumes i increases DOWN the grid. Confirm with SLAM.
    """
    x0, y0 = origin_xy
    x = x0 + j * resolution
    y = y0 + i * resolution
    return (x, y)
