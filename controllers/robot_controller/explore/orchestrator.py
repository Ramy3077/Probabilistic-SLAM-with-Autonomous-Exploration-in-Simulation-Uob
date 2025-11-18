# explore/orchestrator.py

import numpy as np
from explore.frontiers import detect_frontiers
from explore.planner import choose_frontier
from explore.utils import ij_to_xy
from eval.metrics import coverage_percent, entropy_proxy


class ExplorationOrchestrator:
    """
    Glue between:
      - SLAM grid + pose
      - Frontier detection / planner
      - Evaluation metrics

    Input: grid (H,W) with {-1,0,1}, pose_xytheta = (x,y,theta)
    Output:
      - current_goal_xy: (x,y) in world coords or None
      - coverage_pct: float
      - entropy: float
      - frontiers: list of (i,j) indices
    """

    def __init__(self, origin_xy=(0.0, 0.0), resolution=0.05):
        self.origin_xy = origin_xy
        self.resolution = resolution
        self.current_goal_xy = None

    def update(self, grid: np.ndarray, pose_xytheta):
        """
        grid: occupancy grid, values {-1,0,1}
        pose_xytheta: (x, y, theta) in world coordinates (meters, radians)
        """
        x, y, theta = pose_xytheta

        # Convert pose (x,y) to grid indices (i,j)
        pose_i = int(y / self.resolution)
        pose_j = int(x / self.resolution)
        pose_ij = (pose_i, pose_j)

        # --- Frontier detection + planning ---
        frontiers = detect_frontiers(grid)
        goal_ij = choose_frontier(frontiers, pose_ij)

        if goal_ij is not None:
            self.current_goal_xy = ij_to_xy(
                goal_ij[0], goal_ij[1], self.origin_xy, self.resolution
            )
        else:
            self.current_goal_xy = None

        # --- Evaluation metrics ---
        cov = coverage_percent(grid)
        ent = entropy_proxy(grid)

        return self.current_goal_xy, cov, ent, frontiers
