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
    
    def __init__(self, model_name: str = "DPT_Large"):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: MiDaS model to use ("DPT_Large" or "DPT_Hybrid")
        """
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.device = None
        self.depth_cache = {}  # Cache depth maps by image hash
        
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
        return str(hash(image_array.tobytes()))
    
    def _preprocess_image(self, image_array: np.ndarray) -> dict:
        """
        Preprocess image for MiDaS model.
        
        Args:
            image_array: RGB image array (H, W, 3)
            
        Returns:
            Preprocessed inputs ready for model
        """
        if not TORCH_AVAILABLE:
            return {}
            
        if self.transform is not None:
            # Use DPTImageProcessor
            pil_image = Image.fromarray(image_array)
            inputs = self.transform(pil_image, return_tensors="pt")
            return {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Fallback preprocessing
            # Resize to model input size (typically 384x384)
            resized = cv2.resize(image_array, (384, 384))
            
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
        Postprocess depth tensor to normalized depth map.
        
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
        
        # Resize to original image size
        depth_resized = cv2.resize(depth, (original_size[1], original_size[0]))
        
        # Normalize to 0-1 range (invert so closer objects have higher values)
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
            # Invert so closer = higher values
            depth_normalized = 1.0 - depth_normalized
        else:
            depth_normalized = np.ones_like(depth_resized) * 0.5
        
        return depth_normalized
    
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
        # Check cache first
        if use_cache:
            image_hash = self._image_hash(image_array)
            if image_hash in self.depth_cache:
                return self.depth_cache[image_hash]
        
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
