# controllers/robot_controller/explore/utils.py
#for sahib to get world-meter goals
from typing import Tuple

def ij_to_xy(i: float, j: float, origin_xy: Tuple[float, float], resolution: float) -> Tuple[float, float]:
    """
    Convert grid index (i,j) to world meters (x,y).
    origin_xy: world coords (x0, y0) of cell (0,0) *corner*.
    resolution: meters per cell.
    Returns the CENTER of cell (i,j), consistent with OccupancyGrid.grid_to_world.
    """
    x0, y0 = origin_xy
    x = x0 + (j + 0.5) * resolution
    y = y0 + (i + 0.5) * resolution
    return (x, y)