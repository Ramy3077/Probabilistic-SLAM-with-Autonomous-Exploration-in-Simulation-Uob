from typing import Tuple, List
import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1  # agree with Ramy later regarding encoding

def detect_frontiers(grid: np.ndarray, unknown_val: int = UNKNOWN, free_val: int = FREE, connectivity: int = 4,) -> List[Tuple[int,int]]:
    """
    Args:
        grid: HxW int array with cells in {-1: unknown, 0: free, 1: occupied}.
        unknown_val, free_val: allow override if SLAM changes codes.
        connectivity: 4 or 8-neighbor frontier rule (default 4).
    Return list of (i,j) cells that are FREE and have at least one UNKNOWN 4-neighbor.
    """
    H, W = grid.shape
    F: List[Tuple[int, int]] = []
    di4 = [(-1,0),(1,0),(0,-1),(0,1)]
    di8 = di4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    nbrs = di4 if connectivity == 4 else di8

    for i in range(1, H-1):
        for j in range(1, W-1):
            if grid[i, j] == free_val:
                if any(grid[i+di, j+dj] == unknown_val for (di, dj) in nbrs):
                    F.append((i, j))
    return F