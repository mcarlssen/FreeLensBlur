"""
Focal blur algorithm implementation.
Applies variable Gaussian blur based on depth distance from focal point.
Supports both CPU and GPU acceleration with automatic device detection.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

# Try to import PyTorch for GPU acceleration
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available for GPU blur acceleration ({e}), using CPU fallback")


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
        self.device = self._detect_device()
        self.force_cpu_mode = False  # Flag to force CPU processing
        print(f"BlurProcessor initialized on device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect best available device (CUDA, MPS, or CPU)."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
        
    def _calculate_blur_kernel_size(self, depth_diff: float, blur_strength: float, 
                                 max_blur: int = 100) -> int:
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
        
        # Calculate base blur size with more aggressive scaling
        # Use cubic scaling for much more dramatic effect
        blur_size = (depth_diff ** 0.3) * strength_factor * max_blur
        
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
                              focal_point: Tuple[int, int], blur_strength: float, 
                              focal_length: float = 50.0) -> np.ndarray:
        """
        Simplified focal blur implementation with proper depth of field simulation.
        Uses realistic depth of field physics for portrait-friendly adjustments.
        
        Args:
            image: Input RGB image (H, W, 3)
            depth_map: Normalized depth map (H, W, 0-1)
            focal_point: (x, y) coordinates of focal point
            blur_strength: Blur strength (0-10)
            focal_length: Simulated focal length in mm (affects depth of field)
            
        Returns:
            Blurred image
        """
        try:
            print(f"apply_focal_blur_simple called with focal_point={focal_point}, blur_strength={blur_strength}")
            
            # Get focal depth
            focal_x, focal_y = focal_point
            focal_depth = depth_map[focal_y, focal_x]
            print(f"Focal depth at ({focal_x}, {focal_y}): {focal_depth}")
            
            # Calculate depth differences
            depth_diff = np.abs(depth_map - focal_depth)
            print(f"Depth difference range: {depth_diff.min():.3f} to {depth_diff.max():.3f}")
            
            # Apply proper depth of field physics
            # Depth of field is inversely proportional to focal length
            # Longer lenses = shallower depth of field = more blur for same depth difference
            dof_factor = self._calculate_depth_of_field_factor(focal_length)
            print(f"Focal length: {focal_length}mm, DOF factor: {dof_factor:.3f}")
            
            # Apply depth of field physics to create realistic blur falloff
            # This creates the "breathing room" effect you want for portraits
            adjusted_depth_diff = self._apply_depth_of_field_physics(depth_diff, focal_length, focal_depth)
            
            print(f"Adjusted depth range: {adjusted_depth_diff.min():.3f} to {adjusted_depth_diff.max():.3f}")
            
            # Apply smooth, creamy bokeh blur
            blurred_image = image.copy()
            height, width = image.shape[:2]
            
            # Create smooth depth-based blur using Gaussian falloff
            max_depth_diff = adjusted_depth_diff.max()
            
            if max_depth_diff > 0.001:  # Lowered threshold for more sensitive blur detection
                print(f"✅ Applying blur - max depth difference: {max_depth_diff:.6f}")
                # Create a range of blur levels for smooth interpolation
                # Use more levels for creamier bokeh
                base_blur_levels = [1, 3, 7, 15, 25, 41, 61, 85, 115]  # More levels, smoother transitions
                blur_levels = []
                
                # Scale blur levels based on blur_strength (1-10)
                strength_factor = blur_strength / 10.0
                
                for base_size in base_blur_levels:
                    scaled_size = int(base_size * strength_factor)
                    if scaled_size % 2 == 0:
                        scaled_size += 1
                    if scaled_size < 1:
                        scaled_size = 1
                    blur_levels.append(scaled_size)
                
                # Pre-compute blurred versions for smooth interpolation
                blurred_levels = []
                for blur_size in blur_levels:
                    if blur_size <= 150:  # Reasonable max blur
                        # Use sigma proportional to kernel size for natural bokeh
                        sigma = blur_size / 3.0
                        blurred_level = cv2.GaussianBlur(image, (blur_size, blur_size), sigma)
                        blurred_levels.append(blurred_level)
                
                print(f"Created {len(blurred_levels)} blur levels for smooth bokeh")
                
                # Apply smooth pixel-level blur with Gaussian falloff
                for y in range(height):
                    for x in range(width):
                        depth_val = adjusted_depth_diff[y, x]
                        
                        if depth_val <= 0.001:  # Lowered threshold for more sensitive blur detection
                            continue  # No blur for in-focus areas
                        
                        # Use Gaussian falloff instead of linear for creamier bokeh
                        # This creates more natural depth-of-field transitions
                        normalized_depth = depth_val / max_depth_diff
                        
                        # Apply Gaussian curve for smoother falloff
                        # This mimics how real lenses transition from sharp to blurry
                        gaussian_factor = np.exp(-0.5 * (normalized_depth * 3) ** 2)
                        blur_strength_norm = 1.0 - gaussian_factor
                        
                        # Map to blur level index with smooth interpolation
                        level_index = blur_strength_norm * (len(blurred_levels) - 1)
                        
                        if level_index <= 0:
                            blurred_image[y, x] = blurred_levels[0][y, x]
                        elif level_index >= len(blurred_levels) - 1:
                            blurred_image[y, x] = blurred_levels[-1][y, x]
                        else:
                            # Smooth interpolation between blur levels
                            lower_idx = int(level_index)
                            upper_idx = lower_idx + 1
                            weight = level_index - lower_idx
                            
                            # Blend the two blur levels smoothly
                            blurred_image[y, x] = (
                                blurred_levels[lower_idx][y, x] * (1 - weight) + 
                                blurred_levels[upper_idx][y, x] * weight
                            )
            
                # Apply edge-aware processing to reduce artifacts around subjects
                blurred_image = self._apply_edge_aware_blur(blurred_image, image, adjusted_depth_diff, focal_depth)
                
                print(f"Applied smooth Gaussian-based bokeh with {len(blurred_levels)} levels")
                return blurred_image
            else:
                print(f"⚠️  SKIPPING BLUR - max depth difference too small: {max_depth_diff:.6f}")
                print("   This usually means the image has insufficient depth variation.")
                print("   Try using an image with clear foreground/background separation.")
                return image
            
        except Exception as e:
            print(f"Error in apply_focal_blur_simple: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def _apply_edge_aware_blur(self, blurred_image: np.ndarray, original_image: np.ndarray, 
                              depth_diff: np.ndarray, focal_depth: float) -> np.ndarray:
        """
        Apply edge-aware processing to reduce artifacts around subject boundaries.
        
        Args:
            blurred_image: Current blurred result
            original_image: Original sharp image
            depth_diff: Depth difference map
            focal_depth: Focal depth value
            
        Returns:
            Edge-enhanced blurred image
        """
        try:
            # Detect edges in the original image
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create edge proximity mask
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_mask = edges_dilated.astype(np.float32) / 255.0
            
            # Apply Gaussian blur to edge mask for smooth transitions
            edge_mask_smooth = cv2.GaussianBlur(edge_mask, (7, 7), 2.0)
            
            # Create transition zones around edges
            # Areas very close to edges should be less blurred
            edge_proximity = 1.0 - edge_mask_smooth
            
            # Adjust blur strength based on edge proximity
            # Closer to edges = less blur to preserve detail
            height, width = blurred_image.shape[:2]
            
            for y in range(height):
                for x in range(width):
                    edge_factor = edge_proximity[y, x]
                    depth_val = depth_diff[y, x]
                    
                    # If we're close to an edge and the depth difference is small,
                    # reduce the blur to preserve edge detail
                    if edge_factor > 0.7 and depth_val < 0.1:
                        # Blend with original image to preserve edge detail
                        blend_factor = edge_factor * 0.5
                        blurred_image[y, x] = (
                            blurred_image[y, x] * (1 - blend_factor) + 
                            original_image[y, x] * blend_factor
                        )
            
            return blurred_image
            
        except Exception as e:
            print(f"Error in edge-aware processing: {e}")
            return blurred_image
    
    def _calculate_depth_of_field_factor(self, focal_length: float) -> float:
        """
        Calculate depth of field factor based on focal length.
        
        Real depth of field physics:
        - DOF ∝ 1/focal_length² (quadratic relationship)
        - Longer lenses have much shallower depth of field
        - This creates the "breathing room" effect for portraits
        
        Args:
            focal_length: Focal length in mm
            
        Returns:
            DOF factor (higher = more blur for same depth difference)
        """
        # Normalize to 50mm baseline (standard portrait lens)
        baseline_focal_length = 50.0
        
        # Quadratic relationship: DOF ∝ 1/f²
        # This means 100mm lens has 4x shallower DOF than 50mm lens
        dof_factor = (focal_length / baseline_focal_length) ** 2
        
        # Clamp to reasonable range
        dof_factor = max(0.1, min(dof_factor, 10.0))
        
        return dof_factor
    
    def _apply_depth_of_field_physics(self, depth_diff: np.ndarray, focal_length: float, focal_depth: float) -> np.ndarray:
        """
        Apply realistic depth of field physics to depth differences.
        
        This creates the "breathing room" effect you want:
        - Wider lenses (24-35mm) = deeper DOF = more area stays sharp
        - Longer lenses (85-200mm) = shallower DOF = less area stays sharp
        
        Args:
            depth_diff: Raw depth differences from focal point
            focal_length: Focal length in mm
            focal_depth: Focal depth value
            
        Returns:
            Adjusted depth differences with DOF physics applied
        """
        # Calculate DOF factor
        dof_factor = self._calculate_depth_of_field_factor(focal_length)
        
        # Apply focus tolerance - this creates the "breathing room"
        # Wider lenses have more tolerance for depth differences
        if focal_length <= 35:  # Wide angle
            focus_tolerance = 0.15  # More forgiving
        elif focal_length <= 85:  # Standard to portrait
            focus_tolerance = 0.08  # Moderate
        else:  # Telephoto
            focus_tolerance = 0.03  # Very strict
        
        print(f"🎯 Focus tolerance: {focus_tolerance:.3f} (focal_length={focal_length}mm)")
        
        # Apply DOF physics with focus tolerance
        # This creates the realistic depth of field behavior
        adjusted_diff = depth_diff.copy()
        
        # Create focus tolerance zone - areas within this zone stay sharp
        focus_mask = depth_diff <= focus_tolerance
        adjusted_diff[focus_mask] = 0.0  # Keep these areas sharp
        
        # Apply DOF scaling to areas outside focus tolerance
        out_of_focus_mask = depth_diff > focus_tolerance
        if np.any(out_of_focus_mask):
            # Scale depth differences by DOF factor
            # Longer lenses make the same depth difference create more blur
            adjusted_diff[out_of_focus_mask] = depth_diff[out_of_focus_mask] * dof_factor
        
        # Apply smooth transition at focus tolerance boundary
        # This prevents harsh transitions between sharp and blurry areas
        transition_width = focus_tolerance * 0.3  # 30% of focus tolerance
        transition_mask = (depth_diff > focus_tolerance - transition_width) & (depth_diff <= focus_tolerance + transition_width)
        
        if np.any(transition_mask):
            # Create smooth transition using sigmoid function
            transition_values = depth_diff[transition_mask]
            normalized_transition = (transition_values - (focus_tolerance - transition_width)) / (2 * transition_width)
            normalized_transition = np.clip(normalized_transition, 0, 1)
            
            # Sigmoid transition for smooth falloff
            sigmoid_transition = 1 / (1 + np.exp(-10 * (normalized_transition - 0.5)))
            adjusted_diff[transition_mask] = transition_values * sigmoid_transition * dof_factor
        
        return adjusted_diff
    
    def apply_focal_blur_gpu(self, image: np.ndarray, depth_map: np.ndarray,
                           focal_point: Tuple[int, int], blur_strength: float, 
                           focal_length: float = 50.0) -> np.ndarray:
        """
        GPU-accelerated focal blur implementation using PyTorch.
        Much faster than CPU version for large images.
        
        Args:
            image: Input RGB image (H, W, 3)
            depth_map: Normalized depth map (H, W, 0-1)
            focal_point: (x, y) coordinates of focal point
            blur_strength: Blur strength (0-10)
            focal_length: Simulated focal length in mm
            
        Returns:
            Blurred image
        """
        if not TORCH_AVAILABLE or self.device == "cpu" or self.force_cpu_mode:
            if self.force_cpu_mode:
                print("🖥️ CPU mode forced - using CPU blur")
            else:
                print("GPU not available, falling back to CPU blur")
            return self.apply_focal_blur_simple(image, depth_map, focal_point, blur_strength, focal_length)
        
        try:
            print(f"apply_focal_blur_gpu called with focal_point={focal_point}, blur_strength={blur_strength}")
            
            # Convert to PyTorch tensors
            image_tensor = torch.from_numpy(image).float().to(self.device)
            depth_tensor = torch.from_numpy(depth_map).float().to(self.device)
            
            # Get focal depth
            focal_x, focal_y = focal_point
            focal_depth = depth_tensor[focal_y, focal_x]
            print(f"Focal depth at ({focal_x}, {focal_y}): {focal_depth.item()}")
            
            # Calculate depth differences
            depth_diff = torch.abs(depth_tensor - focal_depth)
            print(f"Depth difference range: {depth_diff.min().item():.3f} to {depth_diff.max().item():.3f}")
            
            # Apply proper depth of field physics (same as CPU version)
            dof_factor = self._calculate_depth_of_field_factor(focal_length)
            print(f"Focal length: {focal_length}mm, DOF factor: {dof_factor:.3f}")
            
            # Convert depth differences to numpy for DOF physics calculation
            depth_diff_np = depth_diff.cpu().numpy()
            adjusted_depth_diff_np = self._apply_depth_of_field_physics(depth_diff_np, focal_length, focal_depth.item())
            adjusted_depth_diff = torch.from_numpy(adjusted_depth_diff_np).to(self.device)
            
            print(f"Adjusted depth range: {adjusted_depth_diff.min().item():.3f} to {adjusted_depth_diff.max().item():.3f}")
            
            # GPU-accelerated blur processing
            blurred_image = image_tensor.clone()
            height, width = image_tensor.shape[:2]
            
            max_depth_diff = adjusted_depth_diff.max()
            
            if max_depth_diff > 0.001:  # Lowered threshold for more sensitive blur detection
                print(f"✅ Applying GPU blur - max depth difference: {max_depth_diff:.6f}")
                # Create blur levels
                base_blur_levels = [1, 3, 7, 15, 25, 41, 61, 85, 115]
                blur_levels = []
                
                strength_factor = blur_strength / 10.0
                for base_size in base_blur_levels:
                    scaled_size = int(base_size * strength_factor)
                    if scaled_size % 2 == 0:
                        scaled_size += 1
                    if scaled_size < 1:
                        scaled_size = 1
                    blur_levels.append(scaled_size)
                
                # Pre-compute blurred versions on GPU
                blurred_levels = []
                for blur_size in blur_levels:
                    if blur_size <= 150:
                        sigma = blur_size / 3.0
                        # Use PyTorch's Gaussian blur (much faster on GPU)
                        # Convert to CHW format and add batch dimension
                        image_chw = image_tensor.permute(2, 0, 1).unsqueeze(0)
                        
                        # Apply Gaussian blur using conv2d with Gaussian kernel
                        blurred_level = self._gaussian_blur_pytorch(image_chw, blur_size, sigma)
                        blurred_level = blurred_level.squeeze(0).permute(1, 2, 0)
                        blurred_levels.append(blurred_level)
                
                print(f"Created {len(blurred_levels)} GPU blur levels")
                
                # Vectorized GPU processing - much faster than pixel loops
                normalized_depth = adjusted_depth_diff / max_depth_diff
                
                # Apply Gaussian falloff curve
                gaussian_factor = torch.exp(-0.5 * (normalized_depth * 3) ** 2)
                blur_strength_norm = 1.0 - gaussian_factor
                
                # Map to blur level indices
                level_indices = blur_strength_norm * (len(blurred_levels) - 1)
                
                # Create masks for each blur level
                for i in range(len(blurred_levels)):
                    if i == len(blurred_levels) - 1:
                        # Last level - everything above threshold
                        mask = level_indices >= i
                    else:
                        # Intermediate levels - between thresholds
                        mask = (level_indices >= i) & (level_indices < i + 1)
                    
                    if mask.any():
                        # Apply this blur level to masked pixels
                        blurred_image[mask] = blurred_levels[i][mask]
                
                # Handle interpolation between levels for smooth transitions
                for i in range(len(blurred_levels) - 1):
                    # Find pixels that need interpolation between levels i and i+1
                    interp_mask = (level_indices >= i) & (level_indices < i + 1)
                    
                    if interp_mask.any():
                        # Calculate interpolation weights
                        weights = level_indices[interp_mask] - i
                        weights = weights.unsqueeze(-1)  # Add channel dimension
                        
                        # Interpolate between the two blur levels
                        blurred_image[interp_mask] = (
                            blurred_levels[i][interp_mask] * (1 - weights) + 
                            blurred_levels[i + 1][interp_mask] * weights
                        )
            
                # Convert back to numpy
                result = blurred_image.cpu().numpy().astype(np.uint8)
                print(f"Applied GPU-accelerated Gaussian bokeh with {len(blurred_levels)} levels")
                return result
            else:
                print(f"⚠️  SKIPPING GPU BLUR - max depth difference too small: {max_depth_diff:.6f}")
                print("   This usually means the image has insufficient depth variation.")
                print("   Try using an image with clear foreground/background separation.")
                return image
            
        except Exception as e:
            print(f"Error in apply_focal_blur_gpu: {e}")
            print("Falling back to CPU implementation...")
            return self.apply_focal_blur_simple(image, depth_map, focal_point, blur_strength, focal_length)
    
    def _gaussian_blur_pytorch(self, image_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian blur using PyTorch conv2d operations.
        
        Args:
            image_tensor: Input tensor in CHW format with batch dimension (B, C, H, W)
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel
            
        Returns:
            Blurred tensor in same format
        """
        if kernel_size <= 1:
            return image_tensor
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma, image_tensor.device)
        
        # Apply blur to each channel separately
        blurred_channels = []
        for c in range(image_tensor.shape[1]):
            channel = image_tensor[:, c:c+1, :, :]  # Keep channel dimension
            blurred_channel = F.conv2d(channel, kernel, padding=kernel_size//2)
            blurred_channels.append(blurred_channel)
        
        # Concatenate channels back together
        blurred = torch.cat(blurred_channels, dim=1)
        return blurred
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """
        Create a Gaussian kernel for convolution.
        
        Args:
            kernel_size: Size of the kernel (should be odd)
            sigma: Standard deviation
            device: Device to create tensor on
            
        Returns:
            Gaussian kernel tensor (1, 1, kernel_size, kernel_size)
        """
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - kernel_size // 2
        
        # Create 2D coordinate grids
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        
        # Calculate Gaussian values
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize
        gaussian = gaussian / gaussian.sum()
        
        # Reshape for conv2d: (out_channels, in_channels, height, width)
        kernel = gaussian.unsqueeze(0).unsqueeze(0)
        
        return kernel
    
    def set_cpu_mode(self, cpu_mode: bool):
        """Set CPU mode flag."""
        self.force_cpu_mode = cpu_mode
        if cpu_mode:
            print("🖥️ BlurProcessor: CPU mode enabled")
        else:
            print("🚀 BlurProcessor: GPU mode enabled")
    
    def cleanup(self):
        """Clean up thread pool executor."""
        self.executor.shutdown(wait=True)
