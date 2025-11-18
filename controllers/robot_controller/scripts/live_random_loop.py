# scripts/live_random_loop.py
"""
Week 3: random-walk baseline loop using CsvLogger.

For now:
- Uses the same MOCK grid as frontier script.
- Pose does a fake random walk in (x,y).
Later, SLAM will provide real pose and map.
"""

import os
import math
import random
import numpy as np

from eval.logger import CsvLogger
from eval.metrics import coverage_percent, entropy_proxy


def main():
    os.makedirs("eval_logs", exist_ok=True)
    logger = CsvLogger("eval_logs/week3_random_mock.csv")

    # Initial fake pose (x, y, theta)
    x, y, theta = 0.75, 0.75, 0.0

    PRINT_EVERY = 10
    DT = 0.1  # fake time step

    for step in range(100):
        # --- MOCK grid: same as live_exploration_loop ---
        grid = np.full((30, 30), -1, dtype=int)
        grid[5:25, 5:25] = 0
        grid[10:12, 10:20] = 1

        # --- RANDOM-WALK CONTROL ---
        # Small forward speed, random turn rate
        v = random.uniform(0.0, 0.15)        # m/s
        w = random.uniform(-1.0, 1.0)        # rad/s

        # Update fake pose (very rough, just to produce some variation)
        theta += w * DT
        dx = v * math.cos(theta) * DT
        dy = v * math.sin(theta) * DT
        x += dx
        y += dy

        pose_xytheta = (x, y, theta)

        # --- Metrics from grid ---
        cov = coverage_percent(grid)
        ent = entropy_proxy(grid)

        # --- Logging (no frontier goal) ---
        logger.log(
            pose_x=pose_xytheta[0],
            pose_y=pose_xytheta[1],
            pose_theta=pose_xytheta[2],
            goal_x=None,
            goal_y=None,
            num_frontiers=0,      # or len(detect_frontiers(grid)) if you prefer
            coverage_pct=cov,
            entropy_proxy=ent,
            strategy="random",
        )

        if step % PRINT_EVERY == 0:
            print(
                f"[step {step}] "
                f"pose=({x:.2f},{y:.2f}) "
                f"coverage={cov:.2f}% "
                f"entropy={ent:.2f}"
            )

    logger.close()
    print("Logging complete → eval_logs/week3_random_mock.csv")


if __name__ == "__main__":
    main()
