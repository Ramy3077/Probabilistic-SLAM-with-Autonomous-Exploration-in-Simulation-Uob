from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np


Pose = np.ndarray  # shape: (3,) -> [x, y, theta]
Control = Tuple[float, float]  # (v_left, v_right) wheel speeds or generic 2-dof control


@dataclass
class Particle:
    """
    Simple data container for a particle.
    """
    x: float
    y: float
    theta: float
    weight: float = 1.0

    @property
    def pose(self) -> np.ndarray:
        """Helper to get pose as array for compatibility"""
        return np.array([self.x, self.y, self.theta])

    @pose.setter
    def pose(self, val: np.ndarray):
        self.x, self.y, self.theta = val


class ParticleSet:
    
    # Container for a set of particles, providing common SLAM utilities 
    
    def __init__(self, particles: Iterable[Particle]) -> None:
        self.particles: list[Particle] = list(particles)
        if len(self.particles) == 0:
            raise ValueError("ParticleSet must be initialized with at least one particle.")

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        poses = np.stack([p.pose for p in self.particles], axis=0)  # (N, 3)
        weights = np.asarray([p.weight for p in self.particles], dtype=float)  # (N,)
        return poses, weights

    def set_from_arrays(self, poses: np.ndarray, weights: np.ndarray) -> None:
        if poses.shape[0] != weights.shape[0]:
            raise ValueError("Poses and weights must have the same length.")
        self.particles = [
            Particle(x=poses[i, 0], y=poses[i, 1], theta=poses[i, 2], weight=float(weights[i])) 
            for i in range(poses.shape[0])
        ]

    def predict_move(self, forward_move: float, angular_move: float, noise_std: Tuple[float, float]) -> None:
        
        # Moves all particles based on control input + noise
        # noise_std: (sigma_xy, sigma_theta)
        
        poses, weights = self.as_arrays()
        N = len(self.particles)
        
        
        noise_x = np.random.normal(0, noise_std[0], N)
        noise_y = np.random.normal(0, noise_std[0], N)
        noise_th = np.random.normal(0, noise_std[1], N)

        # Update poses (Simplified motion model for Week 1)
        # x' = x + d * cos(theta)
        # y' = y + d * sin(theta)
        # th' = th + d_theta
        
        theta = poses[:, 2]
        poses[:, 0] += forward_move * np.cos(theta) + noise_x
        poses[:, 1] += forward_move * np.sin(theta) + noise_y
        poses[:, 2] += angular_move + noise_th
        
        # Normalize angles to [-pi, pi]
        poses[:, 2] = (poses[:, 2] + np.pi) % (2 * np.pi) - np.pi
        self.set_from_arrays(poses, weights)

    def normalize_weights(self, eps: float = 1e-12) -> None:
        _, w = self.as_arrays()
        w_sum = float(np.sum(w))
        if w_sum < eps:
            # Avoid divide-by-zero: fallback to uniform
            uniform = 1.0 / len(w)
            for p in self.particles:
                p.weight = uniform
            return
        inv_sum = 1.0 / w_sum
        for p in self.particles:
            p.weight *= inv_sum

    def effective_sample_size(self, eps: float = 1e-12) -> float:
        
        # Returns N_eff = 1 / sum_i w_i^2 for normalized weights
        # If weights are not normalized, the formula still works but magnitude differs
        
        _, w = self.as_arrays()
        denom = float(np.sum(np.square(w))) + eps
        return 1.0 / denom

    def resample_if_needed(self, threshold_ratio: float = 0.5) -> bool:
        # resample if N_eff drops below N * ratio.
        if self.effective_sample_size() < (len(self.particles) * threshold_ratio):
            self.resample_low_variance()
            return True
        return False

    def resample_low_variance(
        self,
        rng: Optional[np.random.Generator] = None,
        jitter_fn: Optional[Callable[[Pose, int], Pose]] = None,
    ) -> None:

        
        if rng is None:
            rng = np.random.default_rng()

        poses, weights = self.as_arrays()
        self.normalize_weights()
        N = len(self.particles)

        # Compute cumulative distribution
        cdf = np.cumsum([p.weight for p in self.particles])
        cdf[-1] = 1.0  # enforce exact 1 due to numerical precision

        # Systematic samples
        r0 = rng.uniform(0.0, 1.0 / N)
        positions = r0 + (np.arange(N, dtype=float) / N)

        indexes = np.empty(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cdf[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        # Resample
        poses_resampled = poses[indexes]
        weights_resampled = np.full(N, 1.0 / N, dtype=float)

        # Optional jitter
        if jitter_fn is not None:
            for k in range(N):
                poses_resampled[k] = jitter_fn(poses_resampled[k], k)

        self.set_from_arrays(poses_resampled, weights_resampled)


def initialize_particles_gaussian(
    num_particles: int,
    mean_pose: Pose,
    std: Tuple[float, float, float],
    rng: Optional[np.random.Generator] = None,
) -> ParticleSet:
    
    # Convenience initializer: draws particles from a diagonal Gaussian around mean_pose.
    # std: (sigma_x, sigma_y, sigma_theta)
    if rng is None:
        rng = np.random.default_rng()
    cov = np.diag(np.asarray(std, dtype=float) ** 2)
    poses = rng.multivariate_normal(mean=mean_pose.astype(float), cov=cov, size=num_particles)
    weights = np.full(num_particles, 1.0 / num_particles, dtype=float)
    return ParticleSet(
        Particle(x=poses[i, 0], y=poses[i, 1], theta=poses[i, 2], weight=weights[i]) 
        for i in range(num_particles)
    )


def init_particles_random(N: int, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> ParticleSet:
    # Initialize particles uniformly in a box
    ps = []
    for _ in range(N):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        th = np.random.uniform(-np.pi, np.pi)
        ps.append(Particle(x=x, y=y, theta=th, weight=1.0 / N))
    return ParticleSet(ps)
