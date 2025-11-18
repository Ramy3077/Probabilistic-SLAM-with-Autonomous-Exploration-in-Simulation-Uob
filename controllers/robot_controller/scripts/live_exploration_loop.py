# scripts/live_exploration_loop.py
"""
Week 3: simple exploration loop using ExplorationOrchestrator + CsvLogger.

For now this uses a MOCK grid and pose (same as previous demos).
Later, Ramy will feed a real grid + pose from SLAM,
and Sahib will use goal_xy for motion control.
"""

import os
import numpy as np

from explore.orchestrator import ExplorationOrchestrator
from eval.logger import CsvLogger


def main():
    # Make sure logs folder exists
    os.makedirs("eval_logs", exist_ok=True)

    # Logger for this run
    logger = CsvLogger("eval_logs/week3_frontier_mock.csv")

    # Orchestrator for frontier + metrics
    orch = ExplorationOrchestrator(origin_xy=(0.0, 0.0), resolution=0.05)

    PRINT_EVERY = 10

    for step in range(100):
        # --- MOCK SLAM GRID (same structure as before) ---
        grid = np.full((30, 30), -1, dtype=int)   # unknown
        grid[5:25, 5:25] = 0                      # free
        grid[10:12, 10:20] = 1                    # occupied band

        # --- MOCK pose (x, y, theta) in world coordinates ---
        pose_xytheta = (0.75, 0.75, 0.0)

        # --- Run exploration logic ---
        goal_xy, cov, ent, frontiers = orch.update(grid, pose_xytheta)

        # --- Log to CSV ---
        logger.log(
            pose_x=pose_xytheta[0],
            pose_y=pose_xytheta[1],
            pose_theta=pose_xytheta[2],
            goal_x=goal_xy[0] if goal_xy else None,
            goal_y=goal_xy[1] if goal_xy else None,
            num_frontiers=len(frontiers),
            coverage_pct=cov,
            entropy_proxy=ent,
            strategy="frontier",  # Week 3: later we'll compare with "random"
        )

        # --- Print occasionally for sanity ---
        if step % PRINT_EVERY == 0:
            print(
                f"[step {step}] "
                f"goal_xy={goal_xy} "
                f"frontiers={len(frontiers)} "
                f"coverage={cov:.2f}% "
                f"entropy={ent:.2f}"
            )

    logger.close()
    print("Logging complete → eval_logs/week3_frontier_mock.csv")


if __name__ == "__main__":
    main()
