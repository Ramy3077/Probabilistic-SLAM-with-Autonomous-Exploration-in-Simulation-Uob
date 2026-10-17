"""Regression tests for the accuracy bugs found in docs/ACCURACY_INVESTIGATION.md."""
import math

import numpy as np

from models.motion import sample_motion_diffdrive
from slam.fastslam import FastSLAMConfig, ParticleFilterSLAM
from slam.occupancy import GridSpec, LaserScan, OccupancyGrid, distance_field, update_map
from slam.particles import Particle, ParticleSet


def _box_world():
    """4 m x 4 m room, 5 cm cells, walls on the border; returns (grid, occ)."""
    spec = GridSpec(resolution=0.05, width=80, height=80, origin_x=-2.0, origin_y=-2.0)
    occ = np.zeros((80, 80), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    g = OccupancyGrid(spec)
    g.log_odds[:] = np.where(occ, 5.0, -5.0)
    return g, occ


def _scan_from(pose, occ, res=0.05, origin=-2.0, n=360):
    angle_min, inc = -math.pi, 2 * math.pi / n
    ang = pose[2] + angle_min + np.arange(n) * inc
    ranges = np.full(n, np.inf)
    for k, a in enumerate(ang):
        for d in np.arange(0.01, 4.0, res / 4):
            j = int(np.floor((pose[0] + d * math.cos(a) - origin) / res))
            i = int(np.floor((pose[1] + d * math.sin(a) - origin) / res))
            if occ[i, j]:
                ranges[k] = d
                break
    return LaserScan(ranges=ranges, angle_min=angle_min, angle_inc=inc, range_min=0.1, range_max=4.0)


def _static(pose, control, dt, rng=None):
    return np.asarray(pose, dtype=float)


def _filter_with(poses, grid, **cfg):
    slam = ParticleFilterSLAM(grid, np.zeros(3), _static,
                              FastSLAMConfig(num_particles=len(poses), **cfg),
                              rng=np.random.default_rng(0))
    slam.particles = ParticleSet(Particle(*p, weight=1.0 / len(poses)) for p in poses)
    return slam


def test_scan_is_not_discarded_when_likelihoods_underflow():
    """With 360 beams the raw likelihood of every particle is far below 1e-12.
    The old code then reset all weights to uniform and dropped the scan."""
    grid, occ = _box_world()
    truth = np.array([0.3, -0.2, 0.4])
    scan = _scan_from(truth, occ)
    poses = [truth, truth + [0.15, 0, 0], truth + [0, 0, 0.1]]
    for model in ("endpoint", "field"):
        slam = _filter_with(poses, grid, sensor_model=model, beam_subsample=1,
                            resample_threshold_ratio=0.0)  # never resample here
        # run only the weighting part of step(): no map update, no resampling
        import slam.fastslam as fs
        saved = fs.update_map
        fs.update_map = lambda **k: None
        try:
            slam.step((0.0, 0.0), 0.064, scan)
        finally:
            fs.update_map = saved
        _, w = slam.particles.as_arrays()
        assert np.argmax(w) == 0, model
        assert w[0] > 0.9, (model, w)


def test_weights_accumulate_between_resamples():
    grid, occ = _box_world()
    truth = np.array([0.0, 0.0, 0.0])
    scan = _scan_from(truth, occ, n=90)
    slam = _filter_with([truth, truth + [0.05, 0, 0]], grid, resample_threshold_ratio=0.0,
                        likelihood_temperature=0.05)
    import slam.fastslam as fs
    saved = fs.update_map
    fs.update_map = lambda **k: None
    try:
        slam.step((0.0, 0.0), 0.064, scan)
        w1 = slam.particles.as_arrays()[1].copy()
        slam.step((0.0, 0.0), 0.064, scan)
        w2 = slam.particles.as_arrays()[1]
    finally:
        fs.update_map = saved
    ratio1 = w1[1] / w1[0]
    ratio2 = w2[1] / w2[0]
    assert ratio1 < 1.0
    assert math.isclose(ratio2, ratio1 ** 2, rel_tol=1e-6)


def test_resampling_is_reproducible_from_the_filter_rng():
    grid, _ = _box_world()
    out = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)  # must not matter any more
        slam = ParticleFilterSLAM(grid, np.zeros(3), _static,
                                  FastSLAMConfig(num_particles=20, roughen_xy=0.05),
                                  rng=np.random.default_rng(7))
        for i, p in enumerate(slam.particles.particles):
            p.weight = 1.0 if i == 0 else 1e-9
        slam.particles.normalize_weights()
        slam._maybe_resample()
        out.append(slam.particles.as_arrays()[0])
    assert np.array_equal(out[0], out[1])


def test_free_space_can_clear_a_wrong_wall():
    spec = GridSpec(resolution=0.05, width=60, height=60, origin_x=-1.5, origin_y=-1.5)
    scan = LaserScan(ranges=np.array([1.0]), angle_min=0.0, angle_inc=0.1, range_min=0.1, range_max=4.0)
    g = OccupancyGrid(spec)
    i, j = g.world_to_grid(0.5, 0.0)
    g.log_odds[i, j] = 0.9  # a wall marked from a bad pose
    for _ in range(5):
        update_map(g, np.zeros(3), scan, sticky_walls=False, wall_cells=1)
    assert g.log_odds[i, j] < 0.0
    g2 = OccupancyGrid(spec)
    g2.log_odds[i, j] = 0.9
    for _ in range(5):
        update_map(g2, np.zeros(3), scan, sticky_walls=True, wall_cells=1)
    assert g2.log_odds[i, j] > 0.0  # the old behaviour keeps it forever


def test_thin_wall_marks_only_the_endpoint_cell():
    spec = GridSpec(resolution=0.05, width=60, height=60, origin_x=-1.5, origin_y=-1.5)
    scan = LaserScan(ranges=np.array([1.0]), angle_min=0.0, angle_inc=0.1, range_min=0.1, range_max=4.0)
    g = OccupancyGrid(spec)
    update_map(g, np.zeros(3), scan, sticky_walls=False, wall_cells=1)
    assert (g.log_odds > 0).sum() == 1
    i, j = np.argwhere(g.log_odds > 0)[0]
    assert (i, j) == g.world_to_grid(1.0, 0.0)


def test_distance_field():
    occ = np.zeros((20, 20), dtype=bool)
    occ[10, 10] = True
    d = distance_field(occ, 0.05, max_dist=0.3)
    assert d[10, 10] == 0.0
    assert math.isclose(d[10, 13], 0.15)
    assert math.isclose(d[13, 14], 0.25)
    assert d[0, 0] == 0.3


def test_diffdrive_noise_scales_with_motion_and_allows_stall():
    rng = np.random.default_rng(0)
    still = np.array([sample_motion_diffdrive(np.zeros(3), (0.0, 0.0), 0.064, rng) for _ in range(2000)])
    moving = np.array([sample_motion_diffdrive(np.zeros(3), (0.5, 0.5), 0.064, rng) for _ in range(2000)])
    assert np.std(still[:, 0]) < np.std(moving[:, 0])
    # some particles hold the "wheels slipped, robot did not move" hypothesis
    assert (np.abs(moving[:, 0]) < 1e-3).mean() > 0.02
