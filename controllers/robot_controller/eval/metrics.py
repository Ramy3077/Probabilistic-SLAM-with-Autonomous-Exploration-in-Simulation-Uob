import numpy as np

def coverage_percent(grid: np.ndarray, unknown_val=-1) -> float:
    known = (grid != unknown_val).sum()
    total = grid.size
    return 100.0 * known / total

def entropy_proxy(grid: np.ndarray, unknown_val=-1) -> float:
    # Lower is better; simple proxy: fraction of UNKNOWN cells
    return (grid == unknown_val).mean()
