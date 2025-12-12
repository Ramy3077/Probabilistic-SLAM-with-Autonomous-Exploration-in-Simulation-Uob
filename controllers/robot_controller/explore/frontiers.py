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
    
    Vectorized implementation for performance.
    """
    # Create a boolean mask for free cells
    is_free = (grid == free_val)
    
    # Create a boolean mask for unknown cells
    is_unknown = (grid == unknown_val)
    
    # We want to find free cells that are adjacent to at least one unknown cell.
    # We can do this by shifting the unknown mask and checking overlap.
    
    H, W = grid.shape
    
    # Shifts for 4-connectivity: UP, DOWN, LEFT, RIGHT
    # We padded with False so we don't wrap around
    padded_unknown = np.pad(is_unknown, pad_width=1, mode='constant', constant_values=False)
    
    # Check neighbors by slicing the padded array
    # Center is [1:-1, 1:-1]
    # Neighbors are shifted relative to center
    
    # 4-neighbors
    # up:    [0:-2, 1:-1]
    # down:  [2:  , 1:-1]
    # left:  [1:-1, 0:-2]
    # right: [1:-1, 2:  ]
    
    has_unknown_neighbor = (
        padded_unknown[0:-2, 1:-1] | # Up
        padded_unknown[2:  , 1:-1] | # Down
        padded_unknown[1:-1, 0:-2] | # Left
        padded_unknown[1:-1, 2:  ]   # Right
    )
    
    if connectivity == 8:
        # diagonals
        has_unknown_neighbor |= (
            padded_unknown[0:-2, 0:-2] | # Top-Left
            padded_unknown[0:-2, 2:  ] | # Top-Right
            padded_unknown[2:  , 0:-2] | # Bot-Left
            padded_unknown[2:  , 2:  ]   # Bot-Right
        )

    # A frontier cell is FREE and HAS_UNKNOWN_NEIGHBOR
    is_frontier = is_free & has_unknown_neighbor
    
    # Convert boolean mask to indices coords
    # np.argwhere returns (N, 2) array, we want list of tuples
    frontier_indices = np.argwhere(is_frontier)
    
    # Convert to list of tuples as expected by rest of code
    # (Though optimizing the rest to use arrays would be better eventually)
    return [tuple(x) for x in frontier_indices]

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