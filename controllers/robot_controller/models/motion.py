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
    noise_v = rng.normal(0, 0.02)  
    noise_omega = rng.normal(0, 0.05)  
    
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


def compute_motion_odometry(
    pose: Pose,
    control: Control,
    dt: float,
) -> Pose:
    # Deterministic motion model for pure odometry mapping (No Noise)
    
    x, y, theta = pose
    v_l, v_r = control
    
    # Robot parameters 
    axle_length = 0.45    # Corrected from 0.09 to match robot.py
    
    # Convert wheel velocities to linear and angular velocity
    # Inputs v_l, v_r are already in m/s
    v = (v_l + v_r) / 2.0  # Linear velocity
    omega = (v_r - v_l) / axle_length  # Angular velocity 
    
    # Update pose using differential drive kinematics (Deterministic)
    if abs(omega) < 1e-6:  # Straight line motion
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta
    else:  # Curved motion
        radius = v / omega
        x_new = x + radius * (np.sin(theta + omega * dt) - np.sin(theta))
        y_new = y - radius * (np.cos(theta + omega * dt) - np.cos(theta))
        theta_new = theta + omega * dt
    
    # Normalize angle to [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([x_new, y_new, theta_new], dtype=float)


def sample_motion_diffdrive(
    pose: Pose,
    control: Control,
    dt: float,
    rng: Optional[np.random.Generator] = None,
    alphas: Tuple[float, float, float, float] = (0.05, 0.01, 0.10, 0.05),
    floor: Tuple[float, float] = (0.005, 0.01),
    p_slip: float = 0.2,
    axle_length: float = 0.45,
) -> Pose:
    """Differential-drive velocity model with motion-proportional noise.

    Follows the alpha parameterisation of Thrun et al. (Probabilistic Robotics,
    table 5.3): the spread grows with how fast the robot moves and turns,
    instead of sample_motion_simple's fixed 0.02 m/s / 0.05 rad/s added at every
    step whatever the motion.

        sigma_v = a1*|v| + a2*|w| + floor_v
        sigma_w = a3*|v| + a4*|w| + floor_w

    p_slip: chance per particle per step that the wheels slip or the robot is
    blocked. Half of those draws are a full stall (no translation: a robot
    pushed into a wall while the encoders still count), the other half keep a
    random fraction of the encoder distance. This keeps a few particles on the
    "did not move" hypothesis every step, so a lineage can follow a stall that
    lasts many steps; the scan then picks those particles out. Without it the
    filter needed large roughening after every resample to survive collisions.
    """
    if rng is None:
        rng = np.random.default_rng()
    x, y, theta = pose
    v_l, v_r = control
    v = (v_l + v_r) / 2.0
    w = (v_r - v_l) / axle_length
    a1, a2, a3, a4 = alphas
    v_n = v + rng.normal(0.0, a1 * abs(v) + a2 * abs(w) + floor[0])
    w_n = w + rng.normal(0.0, a3 * abs(v) + a4 * abs(w) + floor[1])
    if p_slip > 0 and rng.random() < p_slip:
        v_n *= 0.0 if rng.random() < 0.5 else rng.random()
    if abs(w_n) < 1e-6:
        x_new = x + v_n * dt * np.cos(theta)
        y_new = y + v_n * dt * np.sin(theta)
        theta_new = theta
    else:
        radius = v_n / w_n
        x_new = x + radius * (np.sin(theta + w_n * dt) - np.sin(theta))
        y_new = y - radius * (np.cos(theta + w_n * dt) - np.cos(theta))
        theta_new = theta + w_n * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, theta_new], dtype=float)
