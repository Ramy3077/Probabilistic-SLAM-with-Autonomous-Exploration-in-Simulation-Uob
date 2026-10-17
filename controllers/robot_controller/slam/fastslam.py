from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from .occupancy import LaserScan, OccupancyGrid, distance_field, update_map
from .particles import ParticleSet, Pose, initialize_particles_gaussian

# format: (pose, control, dt, rng) -> new_pose
MotionModelFn = Callable[[Pose, Tuple[float, float], float, Optional[np.random.Generator]], Pose]

@dataclass
class FastSLAMConfig:
    num_particles: int = 100
    # Resample when effective particles < 50% of total
    resample_threshold_ratio: float = 0.5 
    # Initial spread of particles
    # Initial spread of particles. Zero: in SLAM the start pose defines the map
    # frame, and a spread here only rotates the first scan by a random amount.
    init_pose_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Use every Nth beam (scoring and mapping). 1 = all beams; scoring every
    # 3rd beam left comb-like gaps in the map and doubled the pose error.
    beam_subsample: int = 1
    # Measurement model: "field" = likelihood field (Thrun et al., Probabilistic
    # Robotics, sec. 6.4); "endpoint" = the original lookup of the occupancy
    # probability in the single cell under each beam endpoint.
    sensor_model: str = "field"
    sigma_hit: float = 0.05      # m, std of the endpoint-to-wall distance (one cell)
    z_hit: float = 0.8           # weight of the Gaussian term
    z_rand: float = 0.2          # floor for spurious / unmodelled readings
    p_unknown: float = 0.3       # likelihood of an endpoint in unexplored space
    # Scale on the summed log-likelihood. Neighbouring beams are not
    # independent, so the raw product is far too confident (one scan moves the
    # log weights by ~80 nats and the particle set collapses every step).
    likelihood_temperature: float = 1.0
    # Roughening added to every particle after resampling (std, m and rad).
    # Was 5 cm / 5 deg after every resample: far wider than one step of real
    # motion error, it acted as the filter's real process noise and capped
    # accuracy at ~5 cm even on a perfect map. Off now that the motion model
    # (models.motion.sample_motion_diffdrive) covers slip and stalls.
    roughen_xy: float = 0.0
    roughen_theta: float = 0.0
    # Pose reported and used for mapping: "best" = highest-weight particle,
    # "mean" = weighted mean (circular mean for heading).
    pose_estimate: str = "mean"
    # Map update (see occupancy.update_map)
    sticky_walls: bool = False
    wall_cells: int = 1
    # Cells with log-odds above this count as obstacles for the likelihood field
    field_occ_threshold: float = 0.0

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
        self.last_diag: dict = {}

    def step(self, control: Tuple[float, float], dt: float, scan: Optional[LaserScan]) -> Pose:
        """
        Main SLAM loop:
        1. Motion update
        2. Measurement update
        3. Normalize + choose best pose
        4. Resample
        5. Update global map
        """
        
        # 1. Motion update (Predict)
        for p in self.particles.particles:
            # Motion model returns new pose [x, y, theta]
            new_pose = self.motion_model(p.pose, control, dt, self.rng)
            p.x, p.y, p.theta = new_pose

        # If no scan, we just return the best guess based on motion
        if scan is None or len(scan.ranges) == 0:
            return self.estimate()

        # 2. Measurement update (Weights)
        # Pre-calculate beam indices for efficiency
        ranges = np.asarray(scan.ranges, dtype=float)
        beam_indices = range(0, len(ranges), max(1, self.config.beam_subsample))
        if self.config.sensor_model == "endpoint":
            prob_map = self.grid.probabilities()
            loglik = np.array([
                self._compute_likelihood(p, scan, ranges, beam_indices, prob_map)
                for p in self.particles.particles
            ])
        else:
            loglik = self._likelihood_field_loglik(scan, ranges)
        loglik = loglik * self.config.likelihood_temperature
        # Bayes update in log space: w_t = w_{t-1} * p(z_t | x_t).
        # Two bugs lived here. (a) The old code set w = exp(loglik) directly;
        # with tens of beams loglik is often below -28 for every particle, the
        # sum of weights fell under normalize_weights' 1e-12 guard and the
        # weights were reset to uniform, silently discarding the scan (77% of
        # steps against a perfect map). Subtracting the max before exp keeps
        # the best particle at weight 1. (b) It replaced w_{t-1} instead of
        # multiplying it, so evidence from steps without resampling was lost.
        _, prev_w = self.particles.as_arrays()
        logw = np.log(np.maximum(prev_w, 1e-300)) + loglik
        logw -= logw.max()
        for p, lw in zip(self.particles.particles, logw):
            p.weight = float(np.exp(lw))

        # 3. Normalize, then choose the mapping pose (best particle) while the
        #    weights still mean something. After resampling they are uniform
        #    and argmax would just return a jittered copy at index 0.
        self.particles.normalize_weights()
        best_pose = self.estimate()
        self._record_diagnostics(loglik)

        # 4. Resample
        self.last_diag["resampled"] = self._maybe_resample()

        # 5. Update global map using ONLY this pose
        update_map(
            grid=self.grid,
            pose=best_pose,
            scan=scan,
            beam_subsample=self.config.beam_subsample,
            apply_free_and_occ=True,
            sticky_walls=self.config.sticky_walls,
            wall_cells=self.config.wall_cells,
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
            
        return log_likelihood

    def _likelihood_field_loglik(self, scan: LaserScan, ranges: np.ndarray) -> np.ndarray:
        """Likelihood-field model, vectorised over particles and beams.

        Each valid beam endpoint is scored by its distance d to the nearest
        occupied cell: p = z_hit * exp(-d^2 / 2 sigma^2) + z_rand. Endpoints in
        cells never observed get p_unknown (no evidence either way), so a
        half-built map does not punish particles for seeing new walls.
        Max-range / non-finite readings carry no endpoint and are skipped, as
        in the original model's hit branch.
        """
        cfg = self.config
        g = self.grid
        res = g.spec.resolution
        k = np.arange(0, len(ranges), max(1, cfg.beam_subsample))
        r = ranges[k]
        ok = np.isfinite(r) & (r >= scan.range_min) & (r < scan.range_max - 0.1)
        k, r = k[ok], r[ok]
        poses, _ = self.particles.as_arrays()
        if r.size == 0:
            return np.zeros(len(poses))

        dist = distance_field(g.log_odds > cfg.field_occ_threshold, res, max_dist=3.0 * cfg.sigma_hit)
        known = g.log_odds != 0.0

        ang = poses[:, 2:3] + (scan.angle_min + k * scan.angle_inc)[None, :]
        xe = poses[:, 0:1] + r[None, :] * np.cos(ang)
        ye = poses[:, 1:2] + r[None, :] * np.sin(ang)
        j = np.floor((xe - g.spec.origin_x) / res).astype(int)
        i = np.floor((ye - g.spec.origin_y) / res).astype(int)
        inside = (i >= 0) & (i < g.spec.height) & (j >= 0) & (j < g.spec.width)
        ii, jj = np.where(inside, i, 0), np.where(inside, j, 0)
        d = dist[ii, jj]
        p = cfg.z_hit * np.exp(-0.5 * (d / cfg.sigma_hit) ** 2) + cfg.z_rand
        p = np.where(known[ii, jj], p, cfg.p_unknown)
        p = np.where(inside, p, cfg.z_rand)
        return np.log(p).sum(axis=1)

    def _record_diagnostics(self, loglik: np.ndarray) -> None:
        """Store per-step filter health numbers (read by scripts/accuracy_harness.py)."""
        _, w = self.particles.as_arrays()
        nz = w[w > 0]
        self.last_diag = {
            "ess": float(self.particles.effective_sample_size()),
            "entropy": float(-(nz * np.log(nz)).sum()),
            "loglik_spread": float(loglik.max() - loglik.min()),
            # True when the raw likelihoods summed below normalize_weights' eps,
            # i.e. the update was replaced by uniform weights.
            "discarded": bool(np.exp(loglik).sum() < 1e-12),
        }

    def _maybe_resample(self) -> bool:
        N = len(self.particles.particles)
        neff = self.particles.effective_sample_size()
        
        if neff < (N * self.config.resample_threshold_ratio):
            # Roughening for diversity. Drawn from self.rng (it used NumPy's
            # global RNG, so seeded runs were not reproducible).
            sxy, sth = self.config.roughen_xy, self.config.roughen_theta
            add_jitter = None
            if sxy > 0 or sth > 0:
                def add_jitter(pose: Pose, idx: int) -> Pose:
                    jittered = pose.copy()
                    jittered[0] += self.rng.normal(0, sxy)
                    jittered[1] += self.rng.normal(0, sxy)
                    jittered[2] += self.rng.normal(0, sth)
                    jittered[2] = (jittered[2] + np.pi) % (2 * np.pi) - np.pi
                    return jittered

            self.particles.resample_low_variance(rng=self.rng, jitter_fn=add_jitter)
            return True
        return False

    def estimate(self) -> Pose:
        """Pose estimate from the current (pre-resampling) weights."""
        if self.config.pose_estimate == "mean":
            poses, w = self.particles.as_arrays()
            w = w / w.sum()
            th = np.arctan2((w * np.sin(poses[:, 2])).sum(), (w * np.cos(poses[:, 2])).sum())
            return np.array([(w * poses[:, 0]).sum(), (w * poses[:, 1]).sum(), th])
        return self.best_pose()

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