import numpy as np

def explored_area_m2(grid: np.ndarray, resolution: float, unknown_val=-1) -> float:
    """Calculates absolute explored area in square meters."""
    known_cells = (grid != unknown_val).sum()
    return known_cells * (resolution ** 2)

def coverage_percent(grid: np.ndarray, resolution: float, navigable_area_m2: float = 0, unknown_val=-1) -> float:
    """
    Calculates coverage percentage.
    If navigable_area_m2 is provided (>0), calculates % of that area.
    Otherwise, falls back to % of total grid (legacy behavior).
    """
    explored = explored_area_m2(grid, resolution, unknown_val)
    
    if navigable_area_m2 > 0:
        return min(100.0, 100.0 * explored / navigable_area_m2)
    else:
        # Fallback: percentage of total grid size (not very useful for large empty grids)
        total_m2 = grid.size * (resolution ** 2)
        return 100.0 * explored / total_m2

def entropy_proxy(grid: np.ndarray, unknown_val=-1) -> float:
    # Lower is better; simple proxy: fraction of UNKNOWN cells
    return (grid == unknown_val).mean()
