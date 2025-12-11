"""
A* Path Planning for Grid-Based Navigation

This module implements A* path planning algorithm for navigating a differential
drive robot through an occupancy grid, avoiding obstacles while reaching frontier goals.

Author: Sahib (Control & Navigation)
Integrated with: Ramy's SLAM (occupancy grid), Bassel's Exploration (goal selection)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import heapq


@dataclass(order=True)
class Node:
    """
    A node in the A* search tree.
    
    Attributes:
        f: Total cost (g + h)
        g: Cost from start to this node
        h: Heuristic cost from this node to goal
        pos: Grid position (i, j)
        parent: Parent node for path reconstruction
    """
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    pos: Tuple[int, int] = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False, repr=False)


class AStarPlanner:
    """
    A* path planner for grid-based navigation.
    
    Features:
    - Obstacle avoidance using occupancy grid
    - Configurable robot safety margin
    - 4-connected or 8-connected movement
    - Smoothing option for curved paths
    """
    
    def __init__(
        self,
        safety_distance: float = 0.3,  # meters
        allow_diagonal: bool = True,
        heuristic: str = "euclidean"  # or "manhattan"
    ):
        """
        Initialize A* path planner.
        
        Args:
            safety_distance: Minimum distance from obstacles (meters)
            allow_diagonal: Allow 8-connected movement if True, else 4-connected
            heuristic: Distance metric ("euclidean" or "manhattan")
        """
        self.safety_distance = safety_distance
        self.allow_diagonal = allow_diagonal
        self.heuristic_type = heuristic
        
        # Motion primitives (4-connected)
        self.motions_4 = [
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1),  # Left
            (-1, 0),  # Up
        ]
        
        # Additional diagonal motions (8-connected)
        self.motions_diag = [
            (1, 1),   # Down-Right
            (1, -1),  # Down-Left
            (-1, 1),  # Up-Right
            (-1, -1), # Up-Left
        ]
        
    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        grid: np.ndarray,
        resolution: float,
        origin_xy: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using A*.
        
        Args:
            start_xy: Start position in world coordinates (x, y) [meters]
            goal_xy: Goal position in world coordinates (x, y) [meters]
            grid: Trinary occupancy grid {-1: free, 0: unknown, 1: occupied}
            resolution: Grid resolution [meters/cell]
            origin_xy: World coordinates of grid cell (0,0)
            
        Returns:
            List of waypoints [(x1,y1), (x2,y2), ...] in world coordinates,
            or None if no path exists.
        """
        # Convert world coordinates to grid indices
        start_ij = self._world_to_grid(start_xy, origin_xy, resolution)
        goal_ij = self._world_to_grid(goal_xy, origin_xy, resolution)
        
        # Validate start and goal
        # If start is invalid, BFS to find nearest valid cell
        if not self._is_valid(start_ij, grid):
            print(f"[A*] Start {start_ij} is invalid. Searching for nearest valid...")
            found_valid = False
            queue = [start_ij]
            visited = {start_ij}
            limit = 10
            
            while queue:
                curr = queue.pop(0)
                if self._is_valid(curr, grid):
                    start_ij = curr
                    found_valid = True
                    # Add path segment from original start to valid start?
                    # No, simply plan from valid start. Controller handles local deviation.
                    break
                
                # Limit BFS depth approx
                if abs(curr[0] - start_ij[0]) > limit or abs(curr[1] - start_ij[1]) > limit:
                    continue

                for di, dj in self.motions_4 + self.motions_diag:
                    ni, nj = curr[0] + di, curr[1] + dj
                    if (ni, nj) not in visited:
                        visited.add((ni, nj))
                        queue.append((ni, nj))
            
            if not found_valid:
                print(f"[A*] Could not find valid start near {start_ij}")
                return None

        if not self._is_valid(goal_ij, grid):
            print(f"[A*] Invalid goal position: {goal_ij}")
            return None
        
        # Run A* search
        path_ij = self._search(start_ij, goal_ij, grid, resolution)
        
        if path_ij is None:
            return None
        
        # Convert grid path to world coordinates
        path_xy = [
            self._grid_to_world(ij, origin_xy, resolution)
            for ij in path_ij
        ]
        
        # Optional: Smooth path for differential drive
        # path_xy = self._smooth_path(path_xy)
        
        return path_xy
    
    def _search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: np.ndarray,
        resolution: float
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* search algorithm in grid space.
        
        Returns:
            List of grid cells [(i1,j1), (i2,j2), ...] from start to goal,
            or None if no path found.
        """
        # Initialize
        open_set = []
        closed_set = set()
        
        # Create start node
        h_start = self._heuristic(start, goal, resolution)
        start_node = Node(f=h_start, g=0.0, h=h_start, pos=start, parent=None)
        heapq.heappush(open_set, start_node)
        
        # Track best costs
        g_score = {start: 0.0}
        
        # Motion primitives
        motions = self.motions_4 + (self.motions_diag if self.allow_diagonal else [])
        
        while open_set:
            # Get node with lowest f-score
            current = heapq.heappop(open_set)
            
            # Goal check
            if current.pos == goal:
                return self._reconstruct_path(current)
            
            # Skip if already processed
            if current.pos in closed_set:
                continue
            
            closed_set.add(current.pos)
            
            # Explore neighbors
            for di, dj in motions:
                neighbor_pos = (current.pos[0] + di, current.pos[1] + dj)
                
                # Check validity
                if not self._is_valid(neighbor_pos, grid):
                    continue
                
                if neighbor_pos in closed_set:
                    continue
                
                # Compute cost
                if abs(di) + abs(dj) == 2:  # Diagonal
                    move_cost = resolution * np.sqrt(2)
                else:  # Cardinal
                    move_cost = resolution
                
                tentative_g = current.g + move_cost
                
                # Check if this path is better
                if neighbor_pos in g_score and tentative_g >= g_score[neighbor_pos]:
                    continue
                
                # Update best path to neighbor
                g_score[neighbor_pos] = tentative_g
                h = self._heuristic(neighbor_pos, goal, resolution)
                f = tentative_g + h
                
                neighbor_node = Node(
                    f=f,
                    g=tentative_g,
                    h=h,
                    pos=neighbor_pos,
                    parent=current
                )
                heapq.heappush(open_set, neighbor_node)
        
        # No path found
        return None
    
    def _is_valid(self, pos: Tuple[int, int], grid: np.ndarray) -> bool:
        """
        Check if a grid cell is valid for navigation.
        
        Args:
            pos: Grid position (i, j)
            grid: Trinary grid {-1: free, 0: unknown, 1: occupied}
            
        Returns:
            True if cell is within bounds and navigable
        """
        i, j = pos
        rows, cols = grid.shape
        
        # Bounds check
        if not (0 <= i < rows and 0 <= j < cols):
            return False
        
        # Obstacle check (only navigate in FREE space)
        # -1 = free, 0 = unknown, 1 = occupied
        if grid[i, j] != -1:  # Not free
            return False
        
        # Safety margin check: ensure no obstacles nearby
        # This prevents robot from getting too close to walls
        # (Disabled for now - can be enabled for extra safety)
        # if not self._check_safety_margin(pos, grid):
        #     return False
        
        return True
    
    def _check_safety_margin(
        self,
        pos: Tuple[int, int],
        grid: np.ndarray,
        margin_cells: int = 1
    ) -> bool:
        """
        Check if there's enough clearance around a position.
        
        Args:
            pos: Grid position (i, j)
            grid: Trinary grid
            margin_cells: Number of cells to check around position
            
        Returns:
            True if safe margin exists
        """
        i, j = pos
        rows, cols = grid.shape
        
        for di in range(-margin_cells, margin_cells + 1):
            for dj in range(-margin_cells, margin_cells + 1):
                ni, nj = i + di, j + dj
                
                # Skip out of bounds
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                
                # Check if occupied
                if grid[ni, nj] == 1:
                    return False
        
        return True
    
    def _heuristic(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int],
        resolution: float
    ) -> float:
        """
        Heuristic function for A*.
        
        Args:
            pos: Current position (i, j)
            goal: Goal position (i, j)
            resolution: Grid resolution [m/cell]
            
        Returns:
            Estimated cost to goal
        """
        di = abs(pos[0] - goal[0])
        dj = abs(pos[1] - goal[1])
        
        if self.heuristic_type == "manhattan":
            return resolution * (di + dj)
        else:  # euclidean
            return resolution * np.sqrt(di**2 + dj**2)
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal node to start by following parent pointers.
        
        Args:
            node: Goal node with parent chain
            
        Returns:
            List of grid cells from start to goal
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.pos)
            current = current.parent
        
        # Reverse to get start→goal order
        path.reverse()
        
        return path
    
    def _world_to_grid(
        self,
        xy: Tuple[float, float],
        origin_xy: Tuple[float, float],
        resolution: float
    ) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        x, y = xy
        x0, y0 = origin_xy
        
        j = int(np.floor((x - x0) / resolution))
        i = int(np.floor((y - y0) / resolution))
        
        return (i, j)
    
    def _grid_to_world(
        self,
        ij: Tuple[int, int],
        origin_xy: Tuple[float, float],
        resolution: float
    ) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        i, j = ij
        x0, y0 = origin_xy
        
        x = x0 + (j + 0.5) * resolution
        y = y0 + (i + 0.5) * resolution
        
        return (x, y)
    
    def _smooth_path(
        self,
        path: List[Tuple[float, float]],
        max_iterations: int = 100,
        weight_data: float = 0.5,
        weight_smooth: float = 0.3
    ) -> List[Tuple[float, float]]:
        """
        Smooth path using gradient descent.
        
        This reduces sharp turns for differential drive robots.
        
        Args:
            path: Original path waypoints
            max_iterations: Maximum smoothing iterations
            weight_data: Weight for staying close to original path
            weight_smooth: Weight for smoothness
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Convert to numpy array for easier manipulation
        smooth = np.array(path, dtype=float)
        original = smooth.copy()
        
        tolerance = 0.001
        
        for _ in range(max_iterations):
            change = 0.0
            
            # Don't modify start and goal
            for i in range(1, len(smooth) - 1):
                for d in range(2):  # x and y
                    old_val = smooth[i, d]
                    
                    # Gradient descent update
                    smooth[i, d] += weight_data * (original[i, d] - smooth[i, d])
                    smooth[i, d] += weight_smooth * (
                        smooth[i-1, d] + smooth[i+1, d] - 2.0 * smooth[i, d]
                    )
                    
                    change += abs(old_val - smooth[i, d])
            
            if change < tolerance:
                break
        
        return [tuple(point) for point in smooth]


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """
    Compute total path length in meters.
    
    Args:
        path: List of waypoints [(x1,y1), (x2,y2), ...]
        
    Returns:
        Total Euclidean distance
    """
    if not path or len(path) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += np.sqrt(dx**2 + dy**2)
    
    return length


def simplify_path(
    path: List[Tuple[float, float]],
    tolerance: float = 0.05
) -> List[Tuple[float, float]]:
    """
    Simplify path by removing collinear waypoints.
    
    Uses Ramer-Douglas-Peucker algorithm.
    
    Args:
        path: Original path
        tolerance: Maximum deviation [meters]
        
    Returns:
        Simplified path
    """
    if len(path) < 3:
        return path
    
    # Find point with maximum distance from line start→end
    start = np.array(path[0])
    end = np.array(path[-1])
    
    max_dist = 0.0
    max_index = 0
    
    for i in range(1, len(path) - 1):
        point = np.array(path[i])
        
        # Distance from point to line segment
        dist = np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)
        
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    # If max distance exceeds tolerance, recurse
    if max_dist > tolerance:
        # Recursively simplify the two segments
        left = simplify_path(path[:max_index+1], tolerance)
        right = simplify_path(path[max_index:], tolerance)
        
        # Combine (remove duplicate at junction)
        return left[:-1] + right
    else:
        # All points are close to line, return just endpoints
        return [path[0], path[-1]]
