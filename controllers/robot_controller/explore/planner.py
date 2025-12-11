# controllers/robot_controller/explore/planner.py
from typing import List, Tuple, Optional
import numpy as np
from .frontiers import Frontier
from control.path_planner import AStarPlanner

def choose_frontier(
    frontiers: List[Frontier],
    pose_ij: Tuple[int, int],
    grid: np.ndarray,
    resolution: float,
    origin_xy: Tuple[float, float],
    planner: AStarPlanner,
    alpha: float = 1.0,  # weight for distance cost
    beta: float = 0.5,   # weight for size (utility)
    current_target_xy: Optional[Tuple[float, float]] = None,
    hysteresis_bonus: float = 50.0,
    blacklist: Optional[List[Tuple[float, float]]] = None,
    blacklist_radius: float = 0.5,
) -> Tuple[Optional[Frontier], Optional[List[Tuple[float, float]]]]:
    """
    Select the best frontier to explore and return it along with the path to reach it.
    
    Score = (beta * Size) - (alpha * PathLength) + (Hysteresis if applicable)
    
    Args:
        frontiers: List of Frontier clusters.
        pose_ij: Robot's current grid cell (i, j).
        grid: Occupancy grid (safe for A*).
        resolution: Grid resolution [m].
        origin_xy: World coordinates of grid (0,0).
        planner: initialized AStarPlanner instance.
        alpha: Weight for cost (path length).
        beta: Weight for utility (size).
        current_target_xy: Centroid of the currently selected frontier (if any).
        hysteresis_bonus: Score bonus if a frontier is close to current target.
        
    Returns:
        (Selected Frontier, Path to Centroid)
        Path is List[(x,y)] in world coords.
        Returns (None, None) if no reachable frontier found.
    """
    if not frontiers:
        return None, None

    best_frontier = None
    best_path_xy = None
    best_score = -float('inf')
    
    start_node = pose_ij

    # HEURISTIC OPTIMIZATION:
    # 1. Sort frontiers by simple Euclidean distance first.
    # 2. Only run expensive A* on the top N closest frontiers.
    # This prevents checking 40+ unreachable frontiers (which triggers full-map searches) every step.
    
    # Pre-calculate Euclidean distance for sorting
    candidates = []
    for f in frontiers:
        # Check blacklist first
        target_i, target_j = int(f.centroid[0]), int(f.centroid[1])
        cx = origin_xy[0] + f.centroid[1] * resolution
        cy = origin_xy[1] + f.centroid[0] * resolution
        
        is_blacklisted = False
        if blacklist:
            for bx, by in blacklist:
                if np.hypot(cx - bx, cy - by) < blacklist_radius:
                    is_blacklisted = True
                    break
        if is_blacklisted:
            continue

        # SAFETY CHECK: Ignore targets too close to walls
        # Robot radius ~0.225m. With buffer 0.35m => 7 pixels.
        # Check a box around the centroid.
        # target_i, target_j are integer grid coords of centroid.
        safe_radius = 7 
        r_min = max(0, target_i - safe_radius)
        r_max = min(grid.shape[0], target_i + safe_radius + 1)
        c_min = max(0, target_j - safe_radius)
        c_max = min(grid.shape[1], target_j + safe_radius + 1)
        
        # grid: 1=occupied, 0=free, -1=unknown
        # Check if any cell in window is 1 (occupied)
        window = grid[r_min:r_max, c_min:c_max]
        if np.any(window == 1):
            continue # Too close to wall

        dist_sq = (start_node[0] - target_i)**2 + (start_node[1] - target_j)**2
        candidates.append((dist_sq, f, (target_i, target_j), cx, cy))

    # Sort by distance (closest first)
    candidates.sort(key=lambda x: x[0])
    
    # Priority Selection:
    # Always include the current target if it exists in the candidates (hysteresis).
    # Then fill the rest with the closest frontiers.
    top_candidates = []
    
    # 1. Try to find the current target in candidates
    if current_target_xy is not None:
        for item in candidates:
            # item = (dist_sq, f, target_node, cx, cy)
            icx, icy = item[3], item[4]
            # Check if this candidate is close to current_target_xy
            if np.hypot(icx - current_target_xy[0], icy - current_target_xy[1]) < 1.0:
                 top_candidates.append(item)
                 # Remove it from candidates so we don't add it twice (optional but clean)
                 # Actually, simpler to just start top_candidates with it and rely on uniqueness or set.
                 break
    
    # 2. Add top N closest ones, avoiding duplicates
    TOP_N = 10
    
    # Create a set of frontier IDs or just use the object identity to avoid dupes
    added_frontiers = {id(item[1]) for item in top_candidates}
    
    for item in candidates:
        if len(top_candidates) >= TOP_N and (current_target_xy is None or len(top_candidates) > 1):
             # Ensure we have at least N items, or N+1 if we included the target
             break
        if id(item[1]) not in added_frontiers:
            top_candidates.append(item)
            added_frontiers.add(id(item[1]))
            
    # Iterate through the selected best candidates
    for _, f, target_node, cx, cy in top_candidates:
        
        # If centroid is not valid, try to find a valid point in the cluster
        if not planner._is_valid(target_node, grid):
            found_valid = False
            for cell in f.cells:
                if planner._is_valid(cell, grid):
                    target_node = cell
                    found_valid = True
                    break
            if not found_valid:
                continue # Skip this frontier if no reachable cells

        # Plan path on grid
        path_ij = planner._search(start_node, target_node, grid, resolution)
        
        if path_ij is None:
            continue # Unreachable
            
        # Calculate path length (cost)
        cost = 0.0
        for k in range(len(path_ij)-1):
            p1 = path_ij[k]
            p2 = path_ij[k+1]
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * resolution
            cost += dist
            
        score = (beta * f.size) - (alpha * cost)
        
        # Hysteresis: If this frontier is close to our current target, boost it
        if current_target_xy is not None:
            # Convert grid centroid back to world for comparison? 
            # Actually easier to convert current_target_xy to grid, or f.centroid to world.
            # f.centroid is (r, c) float.
            # Let's verify frontier.centroid format. typically (r, c).
            # Let's convert f.centroid to world to compare with current_target_xy
            
            dist_to_curr = np.hypot(cx - current_target_xy[0], cy - current_target_xy[1])
            if dist_to_curr < 1.0: # 1 meter tolerance
                score += hysteresis_bonus
        
        if score > best_score:
            best_score = score
            best_frontier = f
            # Convert grid path to world path
            best_path_xy = [
                planner._grid_to_world(ij, origin_xy, resolution)
                for ij in path_ij
            ]

    return best_frontier, best_path_xy
