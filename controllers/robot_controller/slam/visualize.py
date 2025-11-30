# Visualization utilities for SLAM debugging and validation using PIL

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available, visualization disabled")

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
    scale: int = 10,  # pixels per grid cell
) -> None:
    # Visualize occupancy grid with robot pose, particles, and laser scan
    if not PIL_AVAILABLE:
        print("Pillow not available, skipping visualization")
        return
    
    # Get probability map
    prob_map = grid.probabilities()
    height, width = prob_map.shape
    
    # Create RGB image: unknown=gray, free=white, occupied=black
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
    
    # Flip vertically for correct orientation (origin at bottom-left)
    img = np.flipud(img)
    
    # Create PIL image and scale up
    pil_img = Image.fromarray(img, mode='RGB')
    scaled_width = width * scale
    scaled_height = height * scale
    pil_img = pil_img.resize((scaled_width, scaled_height), Image.NEAREST)
    
    draw = ImageDraw.Draw(pil_img)
    
    # Helper function to convert world coords to image coords
    def world_to_image(x: float, y: float) -> Tuple[int, int]:
        i, j = grid.world_to_grid(x, y)
        # Flip i for image coordinates (top-left origin)
        img_x = j * scale
        img_y = (height - 1 - i) * scale
        return int(img_x), int(img_y)
    
    # Draw particles
    if show_particles and particles is not None:
        poses, weights = particles.as_arrays()
        max_weight = np.max(weights) if len(weights) > 0 else 1.0
        
        for p_pose, weight in zip(poses, weights):
            px, py = world_to_image(p_pose[0], p_pose[1])
            # Radius based on weight
            radius = int(2 + 6 * (weight / max_weight))
            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=(100, 100, 255, 128),
                outline=(0, 0, 255)
            )
    
    # Draw laser scan
    if show_scan and scan is not None and pose is not None:
        x_r, y_r, theta = pose
        ranges = np.asarray(scan.ranges, dtype=float)
        N = len(ranges)
        
        for k in range(0, N, max(1, 10)):  # Subsample beams
            r = ranges[k]
            if np.isfinite(r) and scan.range_min < r < scan.range_max:
                beam_angle = theta + (scan.angle_min + k * scan.angle_inc)
                x_end = x_r + r * np.cos(beam_angle)
                y_end = y_r + r * np.sin(beam_angle)
                
                px_start, py_start = world_to_image(x_r, y_r)
                px_end, py_end = world_to_image(x_end, y_end)
                
                # Draw beam
                draw.line([(px_start, py_start), (px_end, py_end)], 
                         fill=(0, 255, 0, 50), width=1)
                # Draw endpoint
                draw.ellipse([(px_end - 2, py_end - 2), (px_end + 2, py_end + 2)],
                            fill=(0, 255, 0))
    
    # Draw robot pose
    if pose is not None:
        x, y, theta = pose
        px, py = world_to_image(x, y)
        
        # Robot circle (radius ~10cm in world = ~2 cells)
        robot_radius = int(2 * scale)
        draw.ellipse(
            [(px - robot_radius, py - robot_radius), 
             (px + robot_radius, py + robot_radius)],
            outline=(255, 0, 0),
            width=3
        )
        
        # Orientation arrow
        arrow_len = int(3 * scale)
        px_end = px + arrow_len * np.cos(theta)
        py_end = py - arrow_len * np.sin(theta)  # Negative because image y is flipped
        draw.line([(px, py), (int(px_end), int(py_end))], 
                 fill=(255, 0, 0), width=3)
        
        # Arrowhead
        draw.ellipse([(int(px_end) - 3, int(py_end) - 3), 
                     (int(px_end) + 3, int(py_end) + 3)],
                    fill=(255, 0, 0))
    
    # Save image
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"Saved visualization to {output_path}")
    
    return pil_img


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
