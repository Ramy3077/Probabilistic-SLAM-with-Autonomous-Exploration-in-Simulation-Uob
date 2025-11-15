# slam/interface.py
"""
Temporary SLAMInterface stub so exploration/control code can import it safely.
This does NOT change or use Ramy's FastSLAM logic. He can wire this later if needed.
"""

import numpy as np


class SLAMInterface:
    def __init__(self, grid_size=(100, 100)):
        # Simple placeholder grid: all unknown = -1
        self._grid = np.full(grid_size, -1, dtype=int)
        # Placeholder pose: (x, y, theta)
        self._pose = (0.0, 0.0, 0.0)

    def predict(self, odom):
        """Placeholder motion update. Does nothing for now."""
        pass

    def update(self, scan):
        """Placeholder measurement update. Does nothing for now."""
        pass

    def get_map_and_pose(self):
        """Return placeholder grid and pose."""
        return self._grid, self._pose
