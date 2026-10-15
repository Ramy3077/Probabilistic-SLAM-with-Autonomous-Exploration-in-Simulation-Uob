"""2D stand-in for the Webots world, written for the browser demo.

Provides what the Webots robot gave the SLAM loop: wheel-encoder speeds
(v_l, v_r in m/s) and a 360 degree LiDAR scan. The true pose is kept here and
is never shown to the filter. Robot constants match robots/robot.py.
"""
from __future__ import annotations

import math

import numpy as np

ROBOT_RADIUS = 0.225
WHEEL_RADIUS = 0.125
WHEEL_SEPARATION = 0.45
MAX_SPEED = 6.28  # rad/s at the wheel

LIDAR_BEAMS = 180
LIDAR_MAX_RANGE = 4.0  # same maxRange as the Webots Lidar node


class World:
    def __init__(self, spec: dict):
        self.res = float(spec["resolution"])
        self.origin_x = float(spec["origin_x"])
        self.origin_y = float(spec["origin_y"])
        rows = spec["rows"]
        self.occ = np.array([[c == "1" for c in row] for row in rows], dtype=bool)
        self.height, self.width = self.occ.shape
        self.start = np.array(spec["start"], dtype=float)

        r = int(math.ceil(ROBOT_RADIUS / self.res))
        di, dj = np.mgrid[-r : r + 1, -r : r + 1]
        keep = (di * self.res) ** 2 + (dj * self.res) ** 2 <= ROBOT_RADIUS ** 2
        self._disk_i = di[keep]
        self._disk_j = dj[keep]

    def cell(self, x, y):
        j = np.floor((x - self.origin_x) / self.res).astype(int)
        i = np.floor((y - self.origin_y) / self.res).astype(int)
        return i, j

    def occupied(self, i, j):
        inside = (i >= 0) & (i < self.height) & (j >= 0) & (j < self.width)
        out = np.ones(np.shape(i), dtype=bool)
        out[inside] = self.occ[i[inside], j[inside]]
        return out

    def collides(self, x: float, y: float) -> bool:
        i0, j0 = self.cell(np.array(x), np.array(y))
        return bool(self.occupied(self._disk_i + i0, self._disk_j + j0).any())

    def raycast(self, x, y, angles, max_range):
        step = self.res * 0.5
        d = np.arange(step, max_range + step, step)
        px = x + np.cos(angles)[:, None] * d[None, :]
        py = y + np.sin(angles)[:, None] * d[None, :]
        i, j = self.cell(px, py)
        hit = self.occupied(i, j)
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        ranges = np.where(any_hit, d[first], np.inf)
        return ranges


class SimRobot:
    """Differential-drive robot with wheel slip and a noisy range sensor."""

    def __init__(self, world: World, rng: np.random.Generator, noise: float = 1.0):
        self.world = world
        self.rng = rng
        self.noise = noise
        self.pose = world.start.copy()
        self.wheel_l = 0.0  # commanded wheel speeds, m/s
        self.wheel_r = 0.0
        self.bumped = False

    def set_velocity_vw(self, v: float, w: float) -> None:
        # Same conversion and clipping as Waypoint.set_velocity_vw in robots/robot.py
        wl = (2 * v - w * WHEEL_SEPARATION) / (2 * WHEEL_RADIUS)
        wr = (2 * v + w * WHEEL_SEPARATION) / (2 * WHEEL_RADIUS)
        wl = max(min(wl, MAX_SPEED), -MAX_SPEED)
        wr = max(min(wr, MAX_SPEED), -MAX_SPEED)
        self.wheel_l = wl * WHEEL_RADIUS
        self.wheel_r = wr * WHEEL_RADIUS

    def advance(self, dt: float):
        """Move the true robot one tick. Returns the encoder reading (v_l, v_r)."""
        k = self.noise
        # The encoders see the wheels turn; the body moves with slip on top.
        vl = self.wheel_l * (1 + self.rng.normal(0, 0.04 * k)) + self.rng.normal(0, 0.005 * k)
        vr = self.wheel_r * (1 + self.rng.normal(0, 0.04 * k)) + self.rng.normal(0, 0.005 * k)
        v = (vl + vr) / 2
        w = (vr - vl) / WHEEL_SEPARATION
        x, y, th = self.pose
        nx = x + v * dt * math.cos(th + w * dt / 2)
        ny = y + v * dt * math.sin(th + w * dt / 2)
        nth = (th + w * dt + math.pi) % (2 * math.pi) - math.pi
        if self.world.collides(nx, ny):
            # Blocked: allow the rotation, refuse the translation.
            self.bumped = True
            self.pose = np.array([x, y, nth])
        else:
            self.bumped = False
            self.pose = np.array([nx, ny, nth])
        return self.wheel_l, self.wheel_r

    def scan(self):
        """LiDAR in the robot frame: index 0 points backwards, the middle index forwards,
        like MyRobot.read_lidar (angle_min = -fov/2 for a 360 degree sensor)."""
        angle_min = -math.pi
        angle_inc = 2 * math.pi / LIDAR_BEAMS
        rel = angle_min + np.arange(LIDAR_BEAMS) * angle_inc
        x, y, th = self.pose
        ranges = self.world.raycast(x, y, th + rel, LIDAR_MAX_RANGE)
        finite = np.isfinite(ranges)
        ranges[finite] += self.rng.normal(0, 0.01 * self.noise, finite.sum())
        ranges[finite] = np.clip(ranges[finite], 0.05, LIDAR_MAX_RANGE)
        return ranges, angle_min, angle_inc
