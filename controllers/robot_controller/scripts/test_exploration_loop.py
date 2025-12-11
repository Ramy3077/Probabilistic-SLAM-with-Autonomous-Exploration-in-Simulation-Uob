# scripts/test_exploration_loop.py

import numpy as np
from explore.orchestrator import ExplorationOrchestrator


def main():
    orch = ExplorationOrchestrator(origin_xy=(0.0, 0.0), resolution=0.05)

    for step in range(10):
        # --- Mock SLAM grid: same pattern as in live_frontiers ---
        grid = np.full((30, 30), -1, dtype=int)   # unknown
        grid[5:25, 5:25] = 0                      # free
        grid[10:12, 10:20] = 1                    # occupied band

        # --- Mock pose in world coordinates (x,y,theta) ---
        pose_xytheta = (0.75, 0.75, 0.0)  # roughly center

        goal_xy, cov, ent, frontiers = orch.update(grid, pose_xytheta)

        print(
            f"step={step} "
            f"goal_xy={goal_xy} "
            f"coverage={cov:.2f}% "
            f"entropy={ent:.2f} "
            f"num_frontiers={len(frontiers)}"
        )


if __name__ == "__main__":
    main()
