from typing import Tuple, List, Set
import numpy as np
from dataclasses import dataclass

UNKNOWN, FREE, OCCUPIED = -1, 0, 1

@dataclass
class Frontier:
    """Represents a connected cluster of frontier cells."""
    cells: List[Tuple[int, int]]  # List of (row, col) coordinates
    centroid: Tuple[float, float] # (row_center, col_center)
    size: int                     # Number of cells

def detect_frontiers(
    grid: np.ndarray, 
    unknown_val: int = UNKNOWN, 
    free_val: int = FREE, 
    connectivity: int = 4
) -> List[Tuple[int, int]]:
    """
    detect_frontiers returns a list of all raw frontier pixels.
    Use get_frontier_clusters() for grouped frontiers.
    
    Args:
        grid: HxW int array with cells in {-1: unknown, 0: free, 1: occupied}.
        connectivity: 4 or 8-neighbor frontier rule (default 4).
    Return list of (i,j) cells that are FREE and have at least one UNKNOWN neighbor.
    """
    H, W = grid.shape
    F: List[Tuple[int, int]] = []
    
    # Pre-compute neighbor deltas
    di4 = [(-1,0), (1,0), (0,-1), (0,1)]
    di8 = di4 + [(-1,-1), (-1,1), (1,-1), (1,1)]
    nbrs = di4 if connectivity == 4 else di8
    
    # We can optimize this with vectorization if needed, but Python loops are fine for small maps.
    # To be safe against borders, we iterate 1..H-1, 1..W-1.
    for i in range(1, H-1):
        for j in range(1, W-1):
            if grid[i, j] == free_val:
                # Check neighbors
                if any(grid[i+di, j+dj] == unknown_val for (di, dj) in nbrs):
                    F.append((i, j))
    return F

def cluster_frontiers(
    frontier_cells: List[Tuple[int, int]], 
    min_size: int = 3,
    dist_threshold: int = 2
) -> List[Frontier]:
    """
    Group adjacent frontier cells into clusters.
    
    Args:
        frontier_cells: List of (i,j) frontier pixels.
        min_size: Discard clusters smaller than this.
        dist_threshold: Max Manhattan distance to consider cells connected (1 or 2).
    
    Returns:
        List of Frontier objects sorted by size (descending).
    """
    if not frontier_cells:
        return []
        
    # Convert to set for O(1) existence checks
    unvisited = set(frontier_cells)
    clusters: List[Frontier] = []
    
    # 8-connected neighbors for clustering
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), 
                   (0, -1),           (0, 1), 
                   (1, -1),  (1, 0),  (1, 1)]

    while unvisited:
        # Start a new cluster
        start = unvisited.pop()
        cluster_pixels = [start]
        queue = [start]
        
        while queue:
            curr_i, curr_j = queue.pop(0)
            
            # Check neighbors
            for di, dj in neighbors_8:
                nb_i, nb_j = curr_i + di, curr_j + dj
                if (nb_i, nb_j) in unvisited:
                    # Found a neighbor in the frontier set
                    if abs(di) + abs(dj) <= dist_threshold: 
                        unvisited.remove((nb_i, nb_j))
                        cluster_pixels.append((nb_i, nb_j))
                        queue.append((nb_i, nb_j))
        
        # Filter noise
        if len(cluster_pixels) >= min_size:
            # Calculate centroid
            ci = sum(p[0] for p in cluster_pixels) / len(cluster_pixels)
            cj = sum(p[1] for p in cluster_pixels) / len(cluster_pixels)
            
            f = Frontier(
                cells=cluster_pixels,
                centroid=(ci, cj),
                size=len(cluster_pixels)
            )
            clusters.append(f)
            
    # Sort by size (largest first)
    clusters.sort(key=lambda x: x.size, reverse=True)
    return clusters