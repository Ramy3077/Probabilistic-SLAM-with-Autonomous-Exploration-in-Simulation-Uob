# Visualization utilities for SLAM debugging and validation using Matplotlib

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

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
    scale: int = 10,  # Unused in matplotlib version but kept for API compatibility
) -> None:
    """
    Visualize occupancy grid with robot pose, particles, and laser scan using Matplotlib.
    Style matches Swepz/LidarBasedGridMapping:
    - Map: gray_r colormap (Dark=Occupied, Light=Free)
    - Robot: Red dot
    - Laser: Blue lines
    """
    
    # Get probability map
    prob_map = grid.probabilities()
    height, width = prob_map.shape
    resolution = grid.spec.resolution
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Map
    # origin='lower' places (0,0) at bottom-left.
    # extent defines the world coordinates of the image [xmin, xmax, ymin, ymax]
    # We assume grid starts at origin_x, origin_y
    extent = [
        grid.spec.origin_x,
        grid.spec.origin_x + width * resolution,
        grid.spec.origin_y,
        grid.spec.origin_y + height * resolution
    ]
    
    # cmap='gray_r': 0 (Free) -> White, 1 (Occupied) -> Black
    # vmin=0, vmax=1 ensures consistent scaling
    ax.imshow(prob_map, cmap='gray_r', origin='lower', extent=extent, vmin=0.0, vmax=1.0)
    
    # Plot Particles
    if show_particles and particles is not None:
        poses, weights = particles.as_arrays()
        # Plot as small blue dots with transparency
        ax.scatter(poses[:, 0], poses[:, 1], c='blue', s=2, alpha=0.3, label='Particles')

    # Plot Laser Scan
    if show_scan and scan is not None and pose is not None:
        x_r, y_r, theta = pose
        ranges = np.asarray(scan.ranges, dtype=float)
        N = len(ranges)
        
        # Collect beam endpoints for efficient plotting
        beam_x = []
        beam_y = []
        
        for k in range(0, N, max(1, 5)):  # Subsample beams
            r = ranges[k]
            if np.isfinite(r) and scan.range_min < r < scan.range_max:
                beam_angle = theta + (scan.angle_min + k * scan.angle_inc)
                x_end = x_r + r * np.cos(beam_angle)
                y_end = y_r + r * np.sin(beam_angle)
                
                # Plot individual beam lines (can be slow for many beams, but looks good)
                ax.plot([x_r, x_end], [y_r, y_end], 'b-', linewidth=0.1, alpha=0.2)
                
                beam_x.append(x_end)
                beam_y.append(y_end)
        
        # Plot endpoints
        if beam_x:
            ax.scatter(beam_x, beam_y, c='blue', s=1, alpha=0.5, label='Scan')

    # Plot Robot Pose
    if pose is not None:
        x, y, theta = pose
        # Robot position (Red dot)
        ax.scatter(x, y, c='red', s=20, zorder=10, label='Robot')
        
        # Orientation arrow
        arrow_len = 0.3  # meters
        ax.arrow(x, y, arrow_len * np.cos(theta), arrow_len * np.sin(theta), 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=10)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    ax.grid(False) # Grid lines might clutter the map
    
    # Save or Show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.close(fig)  # Close to free memory


def save_map_image(
    grid: OccupancyGrid,
    pose: Pose,
    output_path: str,
    title: str = "Occupancy Map",
    scale: int = 10,
) -> None:
    # Save map and robot pose to PNG file
    plot_map(grid, pose=pose, particles=None, scan=None, 
             output_path=output_path, show_particles=False, show_scan=False, scale=scale)
