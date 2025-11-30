# Motion models for robot odometry prediction with noise

from typing import Optional, Tuple
import numpy as np

Pose = np.ndarray  
Control = Tuple[float, float]  


def sample_motion_simple(
    pose: Pose,
    control: Control,
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> Pose:
    # Differential drive motion model
    if rng is None:
        rng = np.random.default_rng()
    
    x, y, theta = pose
    v_l, v_r = control
    
    # Robot parameters 
    # wheel_radius = 0.025  # REMOVED: Inputs are already linear velocity (m/s)
    axle_length = 0.45    # Corrected from 0.09 to match robot.py
    
    # Convert wheel velocities to linear and angular velocity
    # Inputs v_l, v_r are already in m/s
    v = (v_l + v_r) / 2.0  # Linear velocity
    omega = (v_r - v_l) / axle_length  # Angular velocity 
    
    # Add noise to control inputs 
    # Increased noise to account for slip during sharp turns
    noise_v = rng.normal(0, 0.05)  
    noise_omega = rng.normal(0, 0.2)  
    
    v_noisy = v + noise_v
    omega_noisy = omega + noise_omega
    
    # Update pose using differential drive kinematics
    if abs(omega_noisy) < 1e-6:  # Straight line motion
        x_new = x + v_noisy * dt * np.cos(theta)
        y_new = y + v_noisy * dt * np.sin(theta)
        theta_new = theta
    else:  # Curved motion
        radius = v_noisy / omega_noisy
        x_new = x + radius * (np.sin(theta + omega_noisy * dt) - np.sin(theta))
        y_new = y - radius * (np.cos(theta + omega_noisy * dt) - np.cos(theta))
        theta_new = theta + omega_noisy * dt
    
    # Normalize angle to [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([x_new, y_new, theta_new], dtype=float)


def sample_motion_velocity(
    pose: Pose,
    control: Control,
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> Pose:
    # Velocity motion model
    if rng is None:
        rng = np.random.default_rng()
    
    x, y, theta = pose
    v, omega = control
    
    # Adding noise
    v_noisy = v + rng.normal(0, 0.01)
    omega_noisy = omega + rng.normal(0, 0.02)
    
    # Update pose
    if abs(omega_noisy) < 1e-6:
        x_new = x + v_noisy * dt * np.cos(theta)
        y_new = y + v_noisy * dt * np.sin(theta)
        theta_new = theta
    else:
        radius = v_noisy / omega_noisy
        x_new = x + radius * (np.sin(theta + omega_noisy * dt) - np.sin(theta))
        y_new = y - radius * (np.cos(theta + omega_noisy * dt) - np.cos(theta))
        theta_new = theta + omega_noisy * dt
    
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([x_new, y_new, theta_new], dtype=float)
