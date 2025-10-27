"""
Performance optimizations and utilities for the depth blur filter.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import gc
import psutil
import os


class PerformanceMonitor:
    """Monitor system performance and memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def cleanup_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()


def optimize_image_for_processing(image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """
    Optimize image size for processing while maintaining quality.
    
    Args:
        image: Input image array
        max_dimension: Maximum dimension for processing
        
    Returns:
        Optimized image array
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_dimension / height, max_dimension / width, 1.0)
    
    if scale < 1.0:
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Use high-quality resampling
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    return image


def create_depth_visualization(depth_map: np.ndarray) -> np.ndarray:
    """
    Create a colorized visualization of the depth map.
    
    Args:
        depth_map: Normalized depth map (0-1)
        
    Returns:
        Colorized depth visualization
    """
    # Normalize to 0-255
    depth_vis = (depth_map * 255).astype(np.uint8)
    
    # Apply colormap (jet colormap for depth visualization)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    return depth_colored


def estimate_processing_time(image_size: Tuple[int, int], blur_strength: float) -> float:
    """
    Estimate processing time based on image size and parameters.
    
    Args:
        image_size: (height, width) of image
        blur_strength: Blur strength parameter
        
    Returns:
        Estimated processing time in seconds
    """
    height, width = image_size
    pixels = height * width
    
    # Base time estimates (rough approximations)
    depth_time = pixels / 1000000 * 2.0  # 2 seconds per megapixel for depth
    blur_time = pixels / 1000000 * blur_strength * 0.5  # Blur time scales with strength
    
    return depth_time + blur_time


def validate_image_requirements(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that image meets processing requirements.
    
    Args:
        image: Image array to validate
        
    Returns:
        (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False, "Image must be RGB format"
    
    height, width = image.shape[:2]
    
    if height < 100 or width < 100:
        return False, "Image too small (minimum 100x100 pixels)"
    
    if height > 10000 or width > 10000:
        return False, "Image too large (maximum 10000x10000 pixels)"
    
    # Check memory requirements (rough estimate)
    memory_needed = height * width * 3 * 4 * 3  # RGB + depth + blurred (4 bytes per float)
    memory_mb = memory_needed / 1024 / 1024
    
    if memory_mb > 8000:  # 8GB limit
        return False, f"Image too large for available memory (estimated {memory_mb:.1f}MB needed)"
    
    return True, ""


def create_focal_point_overlay(image: np.ndarray, focal_point: Tuple[int, int], 
                              radius: int = 20) -> np.ndarray:
    """
    Create a visual overlay showing the focal point.
    
    Args:
        image: Base image
        focal_point: (x, y) coordinates
        radius: Radius of the focal point indicator
        
    Returns:
        Image with focal point overlay
    """
    overlay = image.copy()
    x, y = focal_point
    
    # Draw circle
    cv2.circle(overlay, (x, y), radius, (0, 255, 0), 2)
    
    # Draw crosshairs
    cv2.line(overlay, (x - radius, y), (x + radius, y), (0, 255, 0), 2)
    cv2.line(overlay, (x, y - radius), (x, y + radius), (0, 255, 0), 2)
    
    return overlay
