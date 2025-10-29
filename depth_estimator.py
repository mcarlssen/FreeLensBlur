"""
Depth map generation using MiDaS v3.0 model.
Supports both CPU and GPU inference with automatic device detection.
Falls back to OpenCV-based depth estimation if PyTorch is not available.
"""

# Try to import PyTorch, but don't fail if it's not available
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available ({e}), using OpenCV fallback for depth estimation")

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import os


class DepthEstimator:
    """MiDaS depth estimation model wrapper."""
    
    def __init__(self, model_name: str = "DPT_Large", detail_level: str = "high", 
                 hole_filling_enabled: bool = True, background_correction_enabled: bool = True,
                 smoothing_strength: str = "medium"):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: MiDaS model to use ("DPT_Large" or "DPT_Hybrid")
            detail_level: Detail level ("high", "medium", "low")
            hole_filling_enabled: Whether to enable hole filling
            background_correction_enabled: Whether to enable background correction
            smoothing_strength: Smoothing strength ("low", "medium", "high")
        """
        self.model_name = model_name
        self.detail_level = detail_level
        self.hole_filling_enabled = hole_filling_enabled
        self.background_correction_enabled = background_correction_enabled
        self.smoothing_strength = smoothing_strength
        self.model = None
        self.transform = None
        self.device = None
        self.force_cpu_mode = False  # Flag to force CPU processing
        self.depth_cache = {}  # Cache depth maps by image hash
        
    def _detect_device(self) -> str:
        """Detect best available device (CUDA, MPS, or CPU)."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        # Force CPU mode if requested
        if self.force_cpu_mode:
            print("🖥️ DepthEstimator: Forcing CPU mode")
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def set_cpu_mode(self, cpu_mode: bool):
        """Set CPU mode flag and clear cache if mode changes."""
        if self.force_cpu_mode != cpu_mode:
            print(f"🔄 DepthEstimator: CPU mode changed from {self.force_cpu_mode} to {cpu_mode}")
            self.force_cpu_mode = cpu_mode
            # Clear cache since device change affects results
            self.depth_cache.clear()
            print(f"🗑️ Cleared depth cache due to device mode change")
            
            # If model is already loaded, we need to reload it on the new device
            if self.model is not None:
                print(f"🔄 Reloading model for {'CPU' if cpu_mode else 'GPU'} processing...")
                self.model = None
                self.transform = None
                self.device = None
        else:
            print(f"ℹ️ DepthEstimator: CPU mode already set to {cpu_mode}")
    
    def _load_model(self):
        """Load the MiDaS model and preprocessing transform."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using OpenCV fallback for depth estimation")
            self.model = None
            self.transform = None
            return
            
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            # Load model and processor
            model_name = "Intel/dpt-large"
            self.model = DPTForDepthEstimation.from_pretrained(model_name)
            self.transform = DPTImageProcessor.from_pretrained(model_name)
            
            # Move to device
            self.device = self._detect_device()
            self.model.to(self.device)
            self.model.eval()
            
            print(f"MiDaS model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            print("Falling back to OpenCV depth estimation...")
            self.model = None
            self.transform = None
    
    def _image_hash(self, image_array: np.ndarray) -> str:
        """Generate a hash for the image to use as cache key."""
        # Include detail_level and CPU mode in cache key so different settings get separate cache entries
        image_bytes = image_array.tobytes()
        detail_bytes = self.detail_level.encode('utf-8')
        cpu_mode_bytes = str(self.force_cpu_mode).encode('utf-8')
        combined_bytes = image_bytes + detail_bytes + cpu_mode_bytes
        return str(hash(combined_bytes))
    
    def _preprocess_image(self, image_array: np.ndarray) -> dict:
        """
        Preprocess image for MiDaS model with high resolution support.
        
        Args:
            image_array: RGB image array (H, W, 3)
            
        Returns:
            Preprocessed inputs ready for model
        """
        if not TORCH_AVAILABLE:
            return {}
            
        if self.transform is not None:
            # Use DPTImageProcessor with higher resolution
            pil_image = Image.fromarray(image_array)
            
            # Calculate optimal input size for better depth resolution
            # Use larger input size for better detail preservation
            height, width = image_array.shape[:2]
            
            # Calculate scale to get closer to original resolution
            # Use different target resolutions based on detail level
            max_dim = max(height, width)
            
            if self.detail_level == "high":
                # For high detail, use much higher resolution to preserve fine details
                target_max_dim = min(max_dim, 2048)  # Up to 2048px for maximum detail
            elif self.detail_level == "medium":
                # For medium detail, use moderate resolution
                target_max_dim = min(max_dim, 1536)  # Up to 1536px
            else:  # low detail
                # For low detail, use standard resolution
                target_max_dim = min(max_dim, 1024)  # Up to 1024px
            
            if max_dim > target_max_dim:
                scale = target_max_dim / max_dim
                target_height = int(height * scale)
                target_width = int(width * scale)
            else:
                target_height = height
                target_width = width
            
            # Ensure dimensions are multiples of 32 (model requirement)
            target_height = ((target_height + 31) // 32) * 32
            target_width = ((target_width + 31) // 32) * 32
            
            print(f"Using {self.detail_level}-res depth input: {target_width}x{target_height} (original: {width}x{height})")
            
            # Resize image for processing
            resized_image = cv2.resize(image_array, (target_width, target_height))
            pil_resized = Image.fromarray(resized_image)
            
            inputs = self.transform(pil_resized, return_tensors="pt")
            return {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Fallback preprocessing with detail-level-based resolution
            height, width = image_array.shape[:2]
            
            # Use different target sizes based on detail level
            max_dim = max(height, width)
            
            if self.detail_level == "high":
                target_max_dim = min(max_dim, 1024)  # Higher resolution for fallback
            elif self.detail_level == "medium":
                target_max_dim = min(max_dim, 768)   # Medium resolution
            else:  # low detail
                target_max_dim = min(max_dim, 512)   # Standard resolution
            
            if max_dim > target_max_dim:
                scale = target_max_dim / max_dim
                target_size = int(target_max_dim)
            else:
                target_size = max_dim
            
            # Ensure multiple of 32
            target_size = ((target_size + 31) // 32) * 32
            
            resized = cv2.resize(image_array, (target_size, target_size))
            
            # Normalize to [0, 1] then to ImageNet stats
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            return {"pixel_values": tensor.to(self.device)}
    
    def _postprocess_depth(self, depth_tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth tensor to normalized depth map with improved hole reduction.
        
        Args:
            depth_tensor: Raw depth output from model
            original_size: (height, width) of original image
            
        Returns:
            Normalized depth map (0-1, where 1 is closest)
        """
        if not TORCH_AVAILABLE:
            return np.ones(original_size) * 0.5
            
        # Remove batch dimension and move to CPU
        depth = depth_tensor.squeeze().cpu().numpy()
        
        # Use high-quality interpolation for resizing
        depth_resized = cv2.resize(depth, (original_size[1], original_size[0]), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Apply improved depth smoothing to reduce holes and artifacts
        print(f"🎨 Applying {self.detail_level} detail level smoothing with hole reduction...")
        depth_smoothed = self._apply_depth_smoothing(depth_resized)
        
        # Normalize to 0-1 range (invert so closer objects have higher values)
        depth_min = depth_smoothed.min()
        depth_max = depth_smoothed.max()
        
        if depth_max > depth_min:
            depth_normalized = (depth_smoothed - depth_min) / (depth_max - depth_min)
            # Invert so closer = higher values
            depth_normalized = 1.0 - depth_normalized
        else:
            depth_normalized = np.ones_like(depth_smoothed) * 0.5
        
        # Apply hole-filling and background correction if enabled
        if self.hole_filling_enabled or self.background_correction_enabled:
            depth_enhanced = self._fill_depth_holes(depth_normalized)
        else:
            depth_enhanced = depth_normalized
        
        return depth_enhanced
    
    def _apply_depth_smoothing(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply depth smoothing with hole reduction based on detail level and smoothing strength.
        
        Args:
            depth_map: Raw depth map from model
            
        Returns:
            Smoothed depth map with reduced holes
        """
        depth_float = depth_map.astype(np.float32)
        
        # Determine smoothing parameters based on detail level and smoothing strength
        if self.smoothing_strength == "low":
            # Minimal smoothing - preserve maximum detail
            if self.detail_level == "high":
                print(f"   High detail + Low smoothing: Minimal smoothing with hole reduction")
                depth_smoothed = cv2.GaussianBlur(depth_float, (3, 3), 0.3)
                kernel = np.ones((3, 3), np.uint8)
            elif self.detail_level == "medium":
                print(f"   Medium detail + Low smoothing: Light smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=3, sigmaColor=0.005, sigmaSpace=10)
                kernel = np.ones((3, 3), np.uint8)
            else:  # low detail
                print(f"   Low detail + Low smoothing: Light smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=5, sigmaColor=0.01, sigmaSpace=15)
                kernel = np.ones((5, 5), np.uint8)
                
        elif self.smoothing_strength == "high":
            # Strong smoothing - maximum hole reduction
            if self.detail_level == "high":
                print(f"   High detail + High smoothing: Moderate smoothing with strong hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=7, sigmaColor=0.02, sigmaSpace=20)
                kernel = np.ones((7, 7), np.uint8)
            elif self.detail_level == "medium":
                print(f"   Medium detail + High smoothing: Strong smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=9, sigmaColor=0.03, sigmaSpace=25)
                kernel = np.ones((7, 7), np.uint8)
            else:  # low detail
                print(f"   Low detail + High smoothing: Very strong smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=11, sigmaColor=0.05, sigmaSpace=30)
                kernel = np.ones((9, 9), np.uint8)
                
        else:  # medium smoothing (default)
            # Moderate smoothing - balance between detail and hole reduction
            if self.detail_level == "high":
                print(f"   High detail + Medium smoothing: Light smoothing with hole reduction")
                depth_smoothed = cv2.GaussianBlur(depth_float, (3, 3), 0.5)
                kernel = np.ones((3, 3), np.uint8)
            elif self.detail_level == "medium":
                print(f"   Medium detail + Medium smoothing: Moderate smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=5, sigmaColor=0.01, sigmaSpace=15)
                kernel = np.ones((5, 5), np.uint8)
            else:  # low detail
                print(f"   Low detail + Medium smoothing: Moderate smoothing with hole reduction")
                depth_smoothed = cv2.bilateralFilter(depth_float, d=7, sigmaColor=0.03, sigmaSpace=25)
                kernel = np.ones((7, 7), np.uint8)
        
        # Apply morphological operations to fill holes
        depth_smoothed = cv2.morphologyEx(depth_smoothed, cv2.MORPH_CLOSE, kernel)
        
        return depth_smoothed
    
    def _fill_depth_holes(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Fill holes and correct background depth assignments.
        
        This addresses the specific issue where background areas between
        foreground objects are incorrectly assigned close depth values.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            Depth map with holes filled and background corrected
        """
        print(f"🔧 Applying hole-filling and background correction...")
        
        # Create a copy to work with
        depth_corrected = depth_map.copy()
        
        # Apply hole filling if enabled
        if self.hole_filling_enabled:
            print(f"   Hole filling: ENABLED")
            depth_corrected = self._apply_hole_filling(depth_corrected)
        else:
            print(f"   Hole filling: DISABLED")
        
        # Apply background correction if enabled
        if self.background_correction_enabled:
            print(f"   Background correction: ENABLED")
            depth_corrected = self._apply_background_correction(depth_corrected)
        else:
            print(f"   Background correction: DISABLED")
        
        print(f"✅ Hole-filling and background correction completed")
        return depth_corrected
    
    def _apply_hole_filling(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply hole filling to correct depth assignments in small regions.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            Depth map with holes filled
        """
        depth_corrected = depth_map.copy()
        
        # Detect potential holes using depth gradient analysis
        grad_x = cv2.Sobel(depth_corrected, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_corrected, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Identify potential hole regions (high gradients in small areas)
        hole_threshold = 0.3
        potential_holes = gradient_magnitude > hole_threshold
        
        # Apply morphological operations to identify connected hole regions
        kernel = np.ones((5, 5), np.uint8)
        potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # For each potential hole region, check if it should be filled
        contours, _ = cv2.findContours(potential_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Only process small to medium sized regions (likely holes, not major objects)
            area = cv2.contourArea(contour)
            if area < 1000:  # Adjust threshold based on image size
                # Create mask for this region
                mask = np.zeros_like(depth_corrected, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Get depth values around the hole
                mask_dilated = cv2.dilate(mask, kernel, iterations=2)
                surrounding_mask = mask_dilated - mask
                
                if np.any(surrounding_mask):
                    # Calculate average depth of surrounding area
                    surrounding_depths = depth_corrected[surrounding_mask > 0]
                    if len(surrounding_depths) > 0:
                        avg_surrounding_depth = np.mean(surrounding_depths)
                        
                        # If the hole is significantly different from surrounding area,
                        # fill it with the surrounding depth
                        hole_depths = depth_corrected[mask > 0]
                        if len(hole_depths) > 0:
                            avg_hole_depth = np.mean(hole_depths)
                            depth_diff = abs(avg_hole_depth - avg_surrounding_depth)
                            
                            # If difference is significant, fill the hole
                            if depth_diff > 0.1:  # Adjust threshold
                                depth_corrected[mask > 0] = avg_surrounding_depth
        
        return depth_corrected
    
    def _apply_background_correction(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply background correction to fix incorrectly assigned close depth values.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            Depth map with background corrected
        """
        depth_corrected = depth_map.copy()
        
        # Calculate gradient magnitude for isolation detection
        grad_x = cv2.Sobel(depth_corrected, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_corrected, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Apply background correction
        # Areas with very high depth values that are isolated should be corrected
        high_depth_threshold = 0.8  # Areas that are very close
        isolated_high_depth = (depth_corrected > high_depth_threshold) & (gradient_magnitude < 0.1)
        
        if np.any(isolated_high_depth):
            # These are likely background areas incorrectly assigned close depth
            # Correct them to be further away
            depth_corrected[isolated_high_depth] = 0.3  # Set to background depth
        
        return depth_corrected
    
    def _enhance_depth_edges(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Enhance depth map edges for better subject boundary detection.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            Edge-enhanced depth map
        """
        # Detect edges in the depth map
        edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
        
        # Dilate edges slightly to create transition zones
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create smooth transition zones around edges
        edge_mask = edges_dilated.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to edge mask for smooth transitions
        edge_mask_smooth = cv2.GaussianBlur(edge_mask, (5, 5), 1.0)
        
        # Enhance depth transitions near edges
        # This helps create more natural depth boundaries
        depth_enhanced = depth_map.copy()
        
        # Apply slight smoothing in edge regions
        depth_smooth = cv2.GaussianBlur(depth_map, (3, 3), 0.5)
        
        # Blend original and smoothed depth based on edge proximity
        depth_enhanced = depth_enhanced * (1 - edge_mask_smooth * 0.3) + \
                        depth_smooth * (edge_mask_smooth * 0.3)
        
        return depth_enhanced
    
    def _fallback_depth_estimation(self, image_array: np.ndarray) -> np.ndarray:
        """
        Fallback depth estimation using OpenCV stereo matching.
        This is much less accurate than MiDaS but doesn't require the model.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Simple depth estimation based on edge detection and blur
        # This is a very basic approximation
        edges = cv2.Canny(gray, 50, 150)
        
        # Use distance transform to create depth-like effect
        # Closer to edges = shallower depth
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Normalize
        depth_map = dist_transform / dist_transform.max()
        
        # Smooth the result
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        
        return depth_map
    
    def estimate_depth(self, image_array: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Estimate depth map for the given image.
        
        Args:
            image_array: RGB image array (H, W, 3)
            use_cache: Whether to use cached results
            
        Returns:
            Normalized depth map (0-1, where 1 is closest)
        """
        print(f"🔍 Depth estimation called with detail_level='{self.detail_level}', use_cache={use_cache}")
        
        # Check cache first
        if use_cache:
            image_hash = self._image_hash(image_array)
            if image_hash in self.depth_cache:
                print(f"📦 Using cached depth map for detail_level='{self.detail_level}'")
                return self.depth_cache[image_hash]
            else:
                print(f"🆕 No cached depth map found for detail_level='{self.detail_level}', generating new one")
        
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        original_size = image_array.shape[:2]
        
        try:
            if self.model is not None and TORCH_AVAILABLE:
                # Use MiDaS model
                with torch.no_grad():
                    # Preprocess
                    inputs = self._preprocess_image(image_array)
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    depth_tensor = outputs.predicted_depth
                    
                    # Postprocess
                    depth_map = self._postprocess_depth(depth_tensor, original_size)
            else:
                # Use fallback method
                depth_map = self._fallback_depth_estimation(image_array)
            
            # Cache the result
            if use_cache:
                self.depth_cache[image_hash] = depth_map
                print(f"💾 Cached depth map for detail_level='{self.detail_level}'")
            
            return depth_map
            
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            print("Using fallback depth estimation...")
            return self._fallback_depth_estimation(image_array)
    
    def clear_cache(self):
        """Clear the depth map cache."""
        self.depth_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached depth maps."""
        return len(self.depth_cache)
