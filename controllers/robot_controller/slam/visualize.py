# Visualization utilities for SLAM debugging and validation

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")

from .occupancy import OccupancyGrid, LaserScan
from .particles import ParticleSet, Pose


def plot_map(
    grid: OccupancyGrid,
    pose: Optional[Pose] = None,
    particles: Optional[ParticleSet] = None,
    scan: Optional[LaserScan] = None,
    title: str = "SLAM Map",
    output_path: Optional[str] = None,
    show_particles: bool = True,
    show_scan: bool = True,
) -> None:
    # Visualize occupancy grid with robot pose, particles, and laser scan
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get probability map for visualization
    prob_map = grid.probabilities()
    
    # Create RGB image: unknown=gray, free=white, occupied=black
    height, width = prob_map.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Unknown cells (near 0.5 probability) - gray
    unknown_mask = np.abs(prob_map - 0.5) < 0.1
    img[unknown_mask] = [128, 128, 128]
    
    # Free cells (low probability) - white
    free_mask = prob_map < 0.4
    img[free_mask] = [255, 255, 255]
    
    # Occupied cells (high probability) - black
    occupied_mask = prob_map > 0.6
    img[occupied_mask] = [0, 0, 0]
    
    # Display map (flip vertically for correct orientation)
    extent = [
        grid.spec.origin_x,
        grid.spec.origin_x + grid.spec.width * grid.spec.resolution,
        grid.spec.origin_y,
        grid.spec.origin_y + grid.spec.height * grid.spec.resolution,
    ]
    ax.imshow(np.flipud(img), extent=extent, origin='lower', interpolation='nearest')
    
    # Draw particles
    if show_particles and particles is not None:
        poses, weights = particles.as_arrays()
        # Scale particle sizes by weight
        sizes = 20 + 100 * (weights / np.max(weights)) if len(weights) > 0 else 20
        ax.scatter(poses[:, 0], poses[:, 1], s=sizes, c='blue', alpha=0.3, label='Particles')
    
    # Draw robot pose
    if pose is not None:
        x, y, theta = pose
        # Draw robot as a circle with orientation wedge
        robot_circle = Circle((x, y), 0.1, color='red', fill=False, linewidth=2, label='Robot')
        ax.add_patch(robot_circle)
        
        # Orientation indicator
        arrow_len = 0.15
        ax.arrow(x, y, arrow_len * np.cos(theta), arrow_len * np.sin(theta),
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        # Draw laser scan if provided
        if show_scan and scan is not None:
            ranges = np.asarray(scan.ranges, dtype=float)
            N = len(ranges)
            
            for k in range(0, N, max(1, 10)):  # Subsample for clarity
                r = ranges[k]
                if np.isfinite(r) and scan.range_min < r < scan.range_max:
                    beam_angle = theta + (scan.angle_min + k * scan.angle_inc)
                    x_end = x + r * np.cos(beam_angle)
                    y_end = y + r * np.sin(beam_angle)
                    ax.plot([x, x_end], [y, y_end], 'g-', alpha=0.2, linewidth=0.5)
                    ax.plot(x_end, y_end, 'go', markersize=2)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def save_map_image(
    grid: OccupancyGrid,
    pose: Pose,
    output_path: str,
    title: str = "Occupancy Map"
) -> None:
    # Save map and robot pose to PNG file
    plot_map(grid, pose=pose, particles=None, scan=None, 
             title=title, output_path=output_path, show_particles=False, show_scan=False)
