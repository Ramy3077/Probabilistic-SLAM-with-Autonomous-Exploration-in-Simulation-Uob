from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from .occupancy import LaserScan, OccupancyGrid, update_map
from .particles import ParticleSet, Pose, initialize_particles_gaussian

# format: (pose, control, dt, rng) -> new_pose
MotionModelFn = Callable[[Pose, Tuple[float, float], float, Optional[np.random.Generator]], Pose]

@dataclass
class FastSLAMConfig:
    num_particles: int = 100
    # Resample when effective particles < 50% of total
    resample_threshold_ratio: float = 0.5 
    # Initial spread of particles
    init_pose_std: Tuple[float, float, float] = (0.05, 0.05, 0.05) 
    # Skip every Nth beam to make mapping faster
    beam_subsample: int = 2 

class ParticleFilterSLAM:
    """
    Clean Design: Particle Filter Localization + Single Global Map.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        init_pose: Pose,
        motion_model: MotionModelFn,
        config: Optional[FastSLAMConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.grid = grid
        self.motion_model = motion_model
        self.config = config or FastSLAMConfig()
        self.rng = rng or np.random.default_rng()

        # Initialize particles
        self.particles = initialize_particles_gaussian(
            num_particles=self.config.num_particles,
            mean_pose=init_pose.astype(float),
            std=self.config.init_pose_std,
            rng=self.rng,
        )
        self.particles.normalize_weights()

    def step(self, control: Tuple[float, float], dt: float, scan: Optional[LaserScan]) -> Pose:
        """
        Main SLAM loop:
        1. Motion update
        2. Measurement update
        3. Normalize + Resample
        4. Choose best pose
        5. Update global map
        """
        
        # 1. Motion update (Predict)
        for p in self.particles.particles:
            # Motion model returns new pose [x, y, theta]
            new_pose = self.motion_model(p.pose, control, dt, self.rng)
            p.x, p.y, p.theta = new_pose

        # If no scan, we just return the best guess based on motion
        if scan is None or len(scan.ranges) == 0:
            return self.best_pose()

        # 2. Measurement update (Weights)
        # Pre-calculate beam indices for efficiency
        ranges = np.asarray(scan.ranges, dtype=float)
        beam_indices = range(0, len(ranges), max(1, self.config.beam_subsample))
        prob_map = self.grid.probabilities()

        for p in self.particles.particles:
            p.weight = self._compute_likelihood(p, scan, ranges, beam_indices, prob_map)

        # 3. Normalize + Resample
        self.particles.normalize_weights()
        self._maybe_resample()

        # 4. Choose mapping pose (best particle)
        best_pose = self.best_pose()

        # 5. Update global map using ONLY this pose
        update_map(
            grid=self.grid,
            pose=best_pose,
            scan=scan,
            beam_subsample=self.config.beam_subsample,
            apply_free_and_occ=True,
        )

        return best_pose

    def _compute_likelihood(self, particle, scan, ranges, beam_indices, prob_map) -> float:
        """Helper to compute likelihood for a single particle"""
        x_r, y_r, th_r = particle.x, particle.y, particle.theta
        log_likelihood = 0.0
        
        for k in beam_indices:
            r = ranges[k]
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            
            # Beam endpoint
            ang = th_r + (scan.angle_min + k * scan.angle_inc)
            x_end = x_r + r * np.cos(ang)
            y_end = y_r + r * np.sin(ang)
            
            # Grid lookup
            i, j = self.grid.world_to_grid(x_end, y_end)
            
            if self.grid.in_bounds(i, j):
                p_occ = prob_map[i, j]
                # Simple sensor model
                if r < (scan.range_max - 0.1): # Hit
                    likelihood = 0.95 * p_occ + 0.05 * (1 - p_occ)
                else: # Miss (max range)
                    likelihood = 0.05 * p_occ + 0.95 * (1 - p_occ)
            else:
                likelihood = 0.01
            
            log_likelihood += np.log(max(likelihood, 1e-10))
            
        return np.exp(log_likelihood)

    def _maybe_resample(self) -> None:
        N = len(self.particles.particles)
        neff = self.particles.effective_sample_size()
        
        if neff < (N * self.config.resample_threshold_ratio):
            # Jitter function for diversity
            def add_jitter(pose: Pose, idx: int) -> Pose:
                noise_xy = 0.05
                noise_th = np.deg2rad(5)
                jittered = pose.copy()
                jittered[0] += np.random.normal(0, noise_xy)
                jittered[1] += np.random.normal(0, noise_xy)
                jittered[2] += np.random.normal(0, noise_th)
                jittered[2] = (jittered[2] + np.pi) % (2 * np.pi) - np.pi
                return jittered
            
            self.particles.resample_low_variance(rng=self.rng, jitter_fn=add_jitter)

    def best_pose(self) -> Pose:
        # Find particle with max weight
        # Note: After resampling, weights are uniform, so this might pick any.
        # But before resampling (or if not resampled), it picks the best.
        # Ideally we track the best *before* resampling if we want the absolute best of the previous step.
        # But for mapping, using the resampled set is also fine (they are all "good").
        # To be safe and consistent with "Clean Design", we pick max weight.
        poses, weights = self.particles.as_arrays()
        best_idx = int(np.argmax(weights))
        return poses[best_idx]