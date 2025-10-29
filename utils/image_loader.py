"""
Image loading and validation utilities for the depth blur filter.
"""

import os
from PIL import Image, ExifTags
import numpy as np
from typing import Tuple, Optional


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to the image if present.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image with correct orientation applied
    """
    try:
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(274)  # Orientation tag
            if orientation:
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # No EXIF data or error reading it
        pass
    
    return image


def load_jpeg_image(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Load and validate a JPEG image file.
    
    Args:
        file_path: Path to the JPEG file
        
    Returns:
        Tuple of (image_array, error_message)
        image_array: RGB numpy array if successful, None if failed
        error_message: Empty string if successful, error description if failed
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    
    if not file_path.lower().endswith(('.jpg', '.jpeg')):
        return None, "Only JPEG files are supported"
    
    try:
        # Load image with PIL
        pil_image = Image.open(file_path)
        
        # Apply EXIF orientation
        pil_image = apply_exif_orientation(pil_image)
        
        # Convert to RGB if necessary (handles RGBA, L, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Validate image dimensions
        height, width = image_array.shape[:2]
        if height < 100 or width < 100:
            return None, "Image too small (minimum 100x100 pixels)"
        
        if height > 10000 or width > 10000:
            return None, "Image too large (maximum 10000x10000 pixels)"
        
        return image_array, ""
        
    except Exception as e:
        return None, f"Error loading image: {str(e)}"


def save_jpeg_image(image_array: np.ndarray, file_path: str, quality: int = 95) -> str:
    """
    Save a numpy array as a JPEG image.
    
    Args:
        image_array: RGB numpy array
        file_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        Error message (empty string if successful)
    """
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        # Save as JPEG
        pil_image.save(file_path, 'JPEG', quality=quality, optimize=True)
        
        return ""
        
    except Exception as e:
        return f"Error saving image: {str(e)}"


def resize_for_preview(image_array: np.ndarray, max_size: Tuple[int, int] = (2560, 1440)) -> np.ndarray:
    """
    Resize image for preview display while maintaining aspect ratio.
    
    Args:
        image_array: Input image array
        max_size: Maximum (width, height) for preview
        
    Returns:
        Resized image array
    """
    try:
        height, width = image_array.shape[:2]
        max_width, max_height = max_size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale == 1.0:
            return image_array
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize using PIL for better quality
        pil_image = Image.fromarray(image_array)
        resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        result = np.array(resized)
        return result
        
    except Exception as e:
        print(f"Error in resize_for_preview: {e}")
        return image_array
