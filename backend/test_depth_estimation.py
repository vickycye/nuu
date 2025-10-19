#!/usr/bin/env python3
"""
Test script for depth estimation pipeline
Run this to test the depth estimation functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.utils.depth_estimator import DepthEstimator
from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_depth_estimation():
    """Test the depth estimation pipeline"""
    
    # Check if we have extracted frames from previous test
    frames_dir = settings.FRAMES_DIR
    if not frames_dir.exists():
        logger.error(f"Frames directory {frames_dir} does not exist")
        logger.info("Please run test_video_processing.py first to extract frames")
        return
    
    # Find test frames - try multiple patterns
    test_frames = []
    patterns = [
        "test_job_001_frame_*.jpg",
        "test_job_001_frame_*.png", 
        "output_frame_*.jpg",
        "output_frame_*.png",
        "*_frame_*.jpg",
        "*_frame_*.png",
        "*.jpg",
        "*.png"
    ]
    
    for pattern in patterns:
        test_frames = list(frames_dir.glob(pattern))
        if test_frames:
            logger.info(f"Found frames using pattern: {pattern}")
            break
    
    if not test_frames:
        logger.error("No test frames found")
        logger.info("Available patterns tried:")
        for pattern in patterns:
            logger.info(f"  - {pattern}")
        logger.info("Please run test_video_processing.py first to extract frames")
        return
    
    logger.info(f"Found {len(test_frames)} test frames")
    
    # Initialize depth estimator
    try:
        depth_estimator = DepthEstimator(settings.DEPTH_MAPS_DIR)
        logger.info("✅ Depth estimator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize depth estimator: {str(e)}")
        logger.info("This might be due to missing MiDaS dependencies")
        logger.info("Try installing: pip install torch torchvision timm")
        return
    
    # Test depth estimation on all frames (or limit for testing)
    # test_frames = test_frames[:3]  # Uncomment to limit to first 3 frames for quick testing
    logger.info(f"Testing depth estimation on {len(test_frames)} frames...")
    
    try:
        # Test single frame depth estimation
        logger.info("Testing single frame depth estimation...")
        result = depth_estimator.estimate_depth(str(test_frames[0]), "test_depth", 0)
        logger.info(f"✅ Single frame depth estimation successful")
        logger.info(f"Depth range: {result['depth_range']}")
        
        # Test batch depth estimation
        logger.info("Testing batch depth estimation...")
        frame_paths = [str(frame) for frame in test_frames]
        batch_results = depth_estimator.estimate_depth_batch(frame_paths, "test_batch")
        
        logger.info(f"✅ Batch depth estimation successful: {len(batch_results)} depth maps generated")
        
        # Get depth statistics
        stats = depth_estimator.get_depth_statistics("test_batch")
        logger.info(f"Depth statistics: {stats}")
        
        # List generated depth maps
        depth_maps = list(settings.DEPTH_MAPS_DIR.glob("test_*_depth_*.png"))
        logger.info(f"Generated depth map files: {len(depth_maps)}")
        
        for depth_map in depth_maps[:5]:  # Show first 5
            logger.info(f"  - {depth_map.name}")
        
        if len(depth_maps) > 5:
            logger.info(f"  ... and {len(depth_maps) - 5} more depth maps")
        
        logger.info("✅ Depth estimation test completed successfully!")
        logger.info("🎯 Ready for 3D reconstruction implementation!")
        
    except Exception as e:
        logger.error(f"❌ Depth estimation test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_depth_estimation())
