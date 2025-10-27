"""
Focal blur algorithm implementation.
Applies variable Gaussian blur based on depth distance from focal point.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor


class BlurProcessor:
    """Handles focal blur application with variable blur strength."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the blur processor.
        
        Args:
            max_workers: Maximum number of threads for parallel processing
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def _calculate_blur_kernel_size(self, depth_diff: float, blur_strength: float, 
                                 max_blur: int = 50) -> int:
        """
        Calculate blur kernel size based on depth difference and blur strength.
        
        Args:
            depth_diff: Absolute difference between focal depth and current depth (0-1)
            blur_strength: User-controlled blur strength (0-10)
            max_blur: Maximum blur kernel size
            
        Returns:
            Blur kernel size (odd number)
        """
        # Scale blur strength from 0-10 to 0-1
        strength_factor = blur_strength / 10.0
        
        # Calculate base blur size
        # Use exponential scaling for more realistic depth-of-field
        blur_size = depth_diff * strength_factor * max_blur
        
        # Ensure odd number for Gaussian blur
        blur_size = int(blur_size)
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Clamp to reasonable range
        blur_size = max(1, min(blur_size, max_blur))
        
        return blur_size
    
    def _process_tile(self, image_tile: np.ndarray, depth_tile: np.ndarray, 
                     focal_depth: float, blur_strength: float) -> np.ndarray:
        """
        Process a single tile of the image.
        
        Args:
            image_tile: Image tile (H, W, 3)
            depth_tile: Depth tile (H, W)
            focal_depth: Focal depth value (0-1)
            blur_strength: Blur strength (0-10)
            
        Returns:
            Blurred image tile
        """
        height, width = image_tile.shape[:2]
        result = image_tile.copy()
        
        # Calculate depth differences
        depth_diff = np.abs(depth_tile - focal_depth)
        
        # Find unique blur kernel sizes needed
        unique_blur_sizes = set()
        for i in range(height):
            for j in range(width):
                blur_size = self._calculate_blur_kernel_size(
                    depth_diff[i, j], blur_strength
                )
                unique_blur_sizes.add(blur_size)
        
        # Apply blur for each unique kernel size
        for blur_size in unique_blur_sizes:
            if blur_size == 1:
                continue  # No blur needed
            
            # Create mask for pixels that need this blur size
            mask = np.zeros((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    if self._calculate_blur_kernel_size(
                        depth_diff[i, j], blur_strength
                    ) == blur_size:
                        mask[i, j] = 255
            
            if np.any(mask):
                # Apply blur to the masked region
                blurred_tile = cv2.GaussianBlur(image_tile, (blur_size, blur_size), 0)
                
                # Blend original and blurred using mask
                mask_3d = np.stack([mask] * 3, axis=2) / 255.0
                result = result * (1 - mask_3d) + blurred_tile * mask_3d
        
        return result.astype(np.uint8)
    
    def apply_focal_blur(self, image: np.ndarray, depth_map: np.ndarray, 
                        focal_point: Tuple[int, int], blur_strength: float,
                        tile_size: int = 256, progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Apply focal blur to the entire image.
        
        Args:
            image: Input RGB image (H, W, 3)
            depth_map: Normalized depth map (H, W, 0-1)
            focal_point: (x, y) coordinates of focal point
            blur_strength: Blur strength (0-10)
            tile_size: Size of tiles for parallel processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Blurred image
        """
        height, width = image.shape[:2]
        focal_x, focal_y = focal_point
        
        # Get focal depth at the clicked point
        focal_depth = depth_map[focal_y, focal_x]
        
        # Initialize result
        result = image.copy()
        
        # Calculate number of tiles
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y
        
        # Process tiles in parallel
        futures = []
        tile_results = {}
        
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Calculate tile boundaries
                start_y = tile_y * tile_size
                end_y = min(start_y + tile_size, height)
                start_x = tile_x * tile_size
                end_x = min(start_x + tile_size, width)
                
                # Extract tiles
                image_tile = image[start_y:end_y, start_x:end_x]
                depth_tile = depth_map[start_y:end_y, start_x:end_x]
                
                # Submit tile processing
                future = self.executor.submit(
                    self._process_tile, image_tile, depth_tile, 
                    focal_depth, blur_strength
                )
                futures.append((future, tile_y, tile_x, start_y, end_y, start_x, end_x))
        
        # Collect results
        completed_tiles = 0
        for future, tile_y, tile_x, start_y, end_y, start_x, end_x in futures:
            try:
                blurred_tile = future.result()
                result[start_y:end_y, start_x:end_x] = blurred_tile
                
                completed_tiles += 1
                if progress_callback:
                    progress_callback(completed_tiles, total_tiles)
                    
            except Exception as e:
                print(f"Error processing tile ({tile_x}, {tile_y}): {e}")
                # Keep original tile if processing fails
                continue
        
        return result
    
    def apply_focal_blur_simple(self, image: np.ndarray, depth_map: np.ndarray,
                              focal_point: Tuple[int, int], blur_strength: float) -> np.ndarray:
        """
        Simplified focal blur implementation for smaller images or preview.
        Uses a single-pass approach with less memory usage.
        
        Args:
            image: Input RGB image (H, W, 3)
            depth_map: Normalized depth map (H, W, 0-1)
            focal_point: (x, y) coordinates of focal point
            blur_strength: Blur strength (0-10)
            
        Returns:
            Blurred image
        """
        height, width = image.shape[:2]
        focal_x, focal_y = focal_point
        
        # Get focal depth
        focal_depth = depth_map[focal_y, focal_x]
        
        # Calculate depth differences
        depth_diff = np.abs(depth_map - focal_depth)
        
        # Create blur map
        blur_map = np.zeros_like(depth_diff)
        for i in range(height):
            for j in range(width):
                blur_map[i, j] = self._calculate_blur_kernel_size(
                    depth_diff[i, j], blur_strength
                )
        
        # Apply blur using OpenCV's bilateral filter for better quality
        result = image.copy()
        
        # Find unique blur sizes
        unique_sizes = np.unique(blur_map)
        
        for blur_size in unique_sizes:
            if blur_size == 1:
                continue
            
            # Create mask for this blur size
            mask = (blur_map == blur_size)
            
            if np.any(mask):
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(image, (int(blur_size), int(blur_size)), 0)
                
                # Blend using mask
                mask_3d = np.stack([mask] * 3, axis=2)
                result = np.where(mask_3d, blurred, result)
        
        return result.astype(np.uint8)
    
    def cleanup(self):
        """Clean up thread pool executor."""
        self.executor.shutdown(wait=True)
