import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from controllers.robot_controller.slam.particles import ParticleSet, Particle, init_particles_random

def test_initialization():

    N = 100
    # Create particles in a 10x10m room
    pset = init_particles_random(N, x_range=(0, 10), y_range=(0, 10))
    
    poses, weights = pset.as_arrays()
    
    assert len(pset.particles) == N
    assert np.all(weights == 1.0 / N)
    
    # Checking bounds
    assert np.all(poses[:, 0] >= 0) and np.all(poses[:, 0] <= 10)
    assert np.all(poses[:, 1] >= 0) and np.all(poses[:, 1] <= 10)
    print("\nInitialization checks out.")

def test_motion_statistics():

    N = 1000 
    # Starting all at (0,0,0) to measure drift easily
    start_particles = [Particle(np.zeros(3), 1.0/N) for _ in range(N)]
    pset = ParticleSet(start_particles)
    
    # Move: 1.0 meter forward, 0.5 radians turn
    # Noise: 0.05m xy error, 0.02 rad theta error
    move_dist = 1.0
    move_turn = 0.5
    sigma_xy = 0.05
    sigma_th = 0.02
    
    pset.predict_move(
        forward_move=move_dist, 
        angular_move=move_turn, 
        noise_std=(sigma_xy, sigma_th)
    )
    
    poses, _ = pset.as_arrays()
    
    
    # We moved 1m at 0 deg, so X should be ~1.0, Y ~0.0
    # (Note: Simple motion model x += d * cos(th))
    avg_x = np.mean(poses[:, 0])
    avg_th = np.mean(poses[:, 2])
    
    print(f"\n Mean X: {avg_x:.4f} (Target: 1.0)")
    print(f" Mean Theta: {avg_th:.4f} (Target: 0.5)")
    
    assert abs(avg_x - move_dist) < 0.1, "Drifted too far from target distance"
    assert abs(avg_th - move_turn) < 0.1, "Drifted too far from target angle"

    
    # The spread of particles should match the noise injected
    std_x = np.std(poses[:, 0])
    std_th = np.std(poses[:, 2])
    
    print(f" Std Dev X: {std_x:.4f} (Target: ~{sigma_xy})")
    
    # Allow some margin 
    assert sigma_xy * 0.8 < std_x < sigma_xy * 1.5, "Noise injection is too small or too huge"
    print(" Motion model is valid.")

def test_resampling_logic():

    N = 10
    # create particles
    start_particles = [Particle(np.array([i, 0.0, 0.0]), 1.0/N) for i in range(N)]
    pset = ParticleSet(start_particles)
    
    #  Forcing particle #5 to be the "Correct" one
    pset.particles[5].weight = 0.99
    # Give others garbage weight
    for i in range(N):
        if i != 5:
            pset.particles[i].weight = 0.0011
            
    
    # Since one particle dominates, N_eff should be very low (approx 1.0)
    n_eff = pset.effective_sample_size()
    print(f"\n N_eff before: {n_eff:.2f} (Should be small)")
    assert n_eff < 2.0, "Effective sample size calc is wrong"

    #  Trigger Resampling
    did_resample = pset.resample_if_needed(threshold_ratio=0.5)
    assert did_resample is True, "Should have triggered resampling!"
    
    # Checking Aftermath
    poses, weights = pset.as_arrays()
    # Count how many particles are now at x=5.0
    survivors = np.sum(poses[:, 0] == 5.0)
    print(f"Particle #5 clones: {survivors}/{N}")
    
    assert survivors > N * 0.8, "The strong particle did not take over the population"
    
    # Weights should be reset to uniform (1/N)
    assert np.allclose(weights, 1.0/N), "Weights were not reset after resampling"
    print(" Resampling logic works.")

if __name__ == "__main__":

    try:
        test_initialization()
        test_motion_statistics()
        test_resampling_logic()
        print("\nall systems Good")
    except AssertionError as e:
        print(f"\nfail: {e}")