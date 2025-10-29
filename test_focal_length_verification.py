#!/usr/bin/env python3
"""
Test focal length effect to verify it's working properly.
"""

import numpy as np
import cv2
from blur_processor import BlurProcessor
from depth_estimator import DepthEstimator
from utils.image_loader import load_jpeg_image, resize_for_preview

def test_focal_length_effect():
    """Test if focal length actually affects the blur visually."""
    print("Testing focal length effect...")
    
    # Load test image
    image_path = "winston_blur01.jpg"
    try:
        image, error = load_jpeg_image(image_path)
        if error:
            print(f"Error loading image: {error}")
            return
        print(f"Loaded image: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Resize for testing
    image = resize_for_preview(image, (800, 600))
    print(f"Resized image: {image.shape}")
    
    # Estimate depth
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_depth(image)
    print(f"Generated depth map: {depth_map.shape}")
    
    # Set focal point
    height, width = image.shape[:2]
    focal_point = (width // 2, height // 2)
    blur_strength = 5.0
    
    blur_processor = BlurProcessor()
    
    # Test different focal lengths
    focal_lengths = [24, 50, 85, 135]  # Wide, normal, portrait, telephoto
    
    for focal_length in focal_lengths:
        print(f"\n=== Testing focal length: {focal_length}mm ===")
        
        # Apply blur with this focal length
        blurred = blur_processor.apply_focal_blur_simple(
            image, depth_map, focal_point, blur_strength, focal_length
        )
        
        # Save result
        output_path = f"test_focal_{focal_length}mm.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")
    
    print("\n=== Analysis ===")
    print("If focal length is working correctly:")
    print("- 24mm should have DEEPER depth of field (less blur)")
    print("- 135mm should have SHALLOWER depth of field (more blur)")
    print("- The difference should be visually noticeable")
    print("\nCheck the saved images to see if there's a visible difference!")

if __name__ == "__main__":
    test_focal_length_effect()
