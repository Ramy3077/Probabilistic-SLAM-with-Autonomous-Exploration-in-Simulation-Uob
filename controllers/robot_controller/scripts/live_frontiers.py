# controllers/robot_controller/scripts/live_frontiers.py
import os
import numpy as np
from explore.frontiers import detect_frontiers
from explore.planner import choose_frontier
from explore.utils import ij_to_xy
from eval.logger import CsvLogger
from eval.metrics import coverage_percent, entropy_proxy


def process_live_grid(grid, pose_ij, origin_xy, resolution, logger):
    """
    Runs frontier detection + evaluation on a live SLAM grid.
    Logs coverage/entropy and returns the next goal (x, y) in world coords.
    """
    # --- detect and plan ---
    frontiers = detect_frontiers(grid)
    goal_ij = choose_frontier(frontiers, pose_ij)
    goal_xy = None

    if goal_ij is not None:
        goal_xy = ij_to_xy(*goal_ij, origin_xy, resolution)
        # print only when caller requests; keep function quiet for reuse

    # --- evaluation metrics ---
    cov = coverage_percent(grid)
    ent = entropy_proxy(grid)

    # --- logging ---
    logger.log(
        pose_x=pose_ij[0],
        pose_y=pose_ij[1],
        pose_theta=0.0,  # placeholder until SLAM provides theta
        chosen_frontier_i=goal_ij[0] if goal_ij else None,
        chosen_frontier_j=goal_ij[1] if goal_ij else None,
        coverage_pct=cov,
        entropy_proxy=ent,
    )

    return goal_xy, len(frontiers), cov, ent


def main():
    """
    Simulated integration loop.
    Replace the mock grid and pose with Ramy's SLAM outputs.
    """
    os.makedirs("eval_logs", exist_ok=True)
    logger = CsvLogger("eval_logs/week2_live.csv")

    origin_xy = (0.0, 0.0)
    resolution = 0.05
    PRINT_EVERY = 10

    # ---- Mock loop (100 frames) ----
    for step in range(100):
        # --- mock SLAM grid (replace with real one later) ---
        grid = np.full((30, 30), -1, dtype=int)
        grid[5:25, 5:25] = 0
        grid[10:12, 10:20] = 1

        # --- mock pose in grid coordinates ---
        pose_ij = (15 + np.random.randn() * 0.5, 15 + np.random.randn() * 0.5)

        goal_xy, n_frontiers, cov, ent = process_live_grid(
            grid, pose_ij, origin_xy, resolution, logger
        )

        if step % PRINT_EVERY == 0:
            print(
                f"[step {step}] goal_xy={goal_xy}  "
                f"frontiers={n_frontiers}  coverage={cov:.2f}%  entropy={ent:.2f}"
            )

    logger.close()
    print("Logging complete → eval_logs/week2_live.csv")


if __name__ == "__main__":
    main()
