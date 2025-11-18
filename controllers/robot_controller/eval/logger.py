import csv
import time
from pathlib import Path


class CsvLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.f = self.path.open("w", newline="")

        self.w = csv.DictWriter(
            self.f,
            fieldnames=[
                "t",
                "pose_x",
                "pose_y",
                "pose_theta",
                "chosen_frontier_i",
                "chosen_frontier_j",
                "goal_x",
                "goal_y",
                "num_frontiers",
                "coverage_pct",
                "entropy_proxy",
                "strategy",
            ],
        )
        self.w.writeheader()

    def log(self, **kwargs):
        row = {"t": time.time(), **kwargs}
        self.w.writerow(row)

    def close(self):
        self.f.close()
