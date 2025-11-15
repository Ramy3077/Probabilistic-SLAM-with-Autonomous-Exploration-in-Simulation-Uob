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

class FastSLAM:

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

        # Initialize particles clustered around the start pose
        self.particles = initialize_particles_gaussian(
            num_particles=self.config.num_particles,
            mean_pose=init_pose.astype(float),
            std=self.config.init_pose_std,
            rng=self.rng,
        )
        self.particles.normalize_weights()

    def predict(self, control: Tuple[float, float], dt: float) -> None:
        
        # Moves every particle using Sahib's motion model.
        # Note: This loop is slower than vectorization, but allows using the flexible motion model function agrred on by the team.
        for p in self.particles.particles:
            p.pose = self.motion_model(p.pose, control, dt, self.rng)
        

    def measurement_update(self, scan: LaserScan) -> None:
        """
        Week 2 Task: Update particle weights based on how well 
        the scan matches the map.
        """
        pass

    def maybe_resample(self) -> None:
        #Checks if particles are degenerating and resamples if needed.
        N = len(self.particles.particles)
        neff = self.particles.effective_sample_size()
        
        if neff < (N * self.config.resample_threshold_ratio):
            self.particles.resample_low_variance(rng=self.rng)

    def update_map_with_best(self, scan: LaserScan) -> None:
        
        # Takes the single best particle and updates the global map.
        poses, weights = self.particles.as_arrays()
        best_idx = int(np.argmax(weights))
        best_pose = poses[best_idx]

        update_map(
            grid=self.grid,
            pose=best_pose,
            scan=scan,
            beam_subsample=self.config.beam_subsample,
            apply_free_and_occ=True,
        )

    def step(self, control: Tuple[float, float], dt: float, scan: Optional[LaserScan]) -> Pose:
        
        # 1. Move (Predict)
        self.predict(control=control, dt=dt)
        
        # 2. See (Update)
        if scan is not None:
            self.measurement_update(scan)
            self.particles.normalize_weights()
            self.maybe_resample()
            self.update_map_with_best(scan)
            
        return self.best_pose()

    def best_pose(self) -> Pose:
        poses, weights = self.particles.as_arrays()
        best_idx = int(np.argmax(weights))
        return poses[best_idx]