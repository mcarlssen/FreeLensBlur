#!/usr/bin/env python3
"""
Simple test script for the depth blur filter.
Tests basic functionality without GUI.
"""

import sys
import os
import numpy as np
import cv2
from depth_estimator import DepthEstimator
from blur_processor import BlurProcessor
from utils.image_loader import load_jpeg_image, save_jpeg_image
from utils.performance import PerformanceMonitor, estimate_processing_time


def test_depth_estimation():
    """Test depth estimation functionality."""
    print("Testing depth estimation...")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize depth estimator
    estimator = DepthEstimator()
    
    # Estimate depth
    depth_map = estimator.estimate_depth(test_image)
    
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    
    return depth_map


def test_blur_processing():
    """Test blur processing functionality."""
    print("Testing blur processing...")
    
    # Create test image and depth map
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    depth_map = np.random.rand(512, 512)
    
    # Initialize blur processor
    processor = BlurProcessor()
    
    # Apply blur
    focal_point = (256, 256)  # Center of image
    blur_strength = 5.0
    
    blurred = processor.apply_focal_blur_simple(
        test_image, depth_map, focal_point, blur_strength
    )
    
    print(f"Blurred image shape: {blurred.shape}")
    
    return blurred


def test_performance():
    """Test performance monitoring."""
    print("Testing performance monitoring...")
    
    monitor = PerformanceMonitor()
    
    print(f"Memory usage: {monitor.get_memory_usage():.1f} MB")
    print(f"CPU usage: {monitor.get_cpu_usage():.1f}%")
    
    # Test processing time estimation
    estimated_time = estimate_processing_time((1024, 1024), 5.0)
    print(f"Estimated processing time for 1MP image: {estimated_time:.1f}s")


def main():
    """Run all tests."""
    print("Depth Blur Filter - Test Suite")
    print("=" * 40)
    
    try:
        # Test depth estimation
        depth_map = test_depth_estimation()
        print("✓ Depth estimation test passed\n")
        
        # Test blur processing
        blurred = test_blur_processing()
        print("✓ Blur processing test passed\n")
        
        # Test performance monitoring
        test_performance()
        print("✓ Performance monitoring test passed\n")
        
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
