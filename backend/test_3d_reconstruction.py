#!/usr/bin/env python3
"""
Test script for 3D reconstruction pipeline
Run this to test the complete 3D reconstruction functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.utils.reconstruction import Reconstruction3D
from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_3d_reconstruction():
    """Test the 3D reconstruction pipeline"""
    
    # Check if we have test data from previous steps
    frames_dir = settings.FRAMES_DIR
    depth_maps_dir = settings.DEPTH_MAPS_DIR
    
    if not frames_dir.exists() or not depth_maps_dir.exists():
        logger.error("Missing test data directories")
        logger.info("Please run test_video_processing.py and test_depth_estimation.py first")
        return
    
    # Find test frames and depth maps
    test_frames = list(frames_dir.glob("test_job_001_frame_*.jpg"))
    test_depth_maps = list(depth_maps_dir.glob("test_batch_depth_raw_*.npy"))
    
    if not test_frames or not test_depth_maps:
        logger.error("No test frames or depth maps found")
        logger.info("Please run test_video_processing.py and test_depth_estimation.py first")
        return
    
    logger.info(f"Found {len(test_frames)} test frames and {len(test_depth_maps)} depth maps")
    
    # Initialize 3D reconstruction
    try:
        reconstruction_3d = Reconstruction3D(settings.MODELS_DIR)
        logger.info("✅ 3D reconstruction engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize 3D reconstruction: {str(e)}")
        return
    
    # Prepare test data
    job_id = "test_reconstruction"
    
    # Create frame data structure
    frame_data = []
    for i, frame_path in enumerate(test_frames):  # Use all frames
        frame_data.append({
            'frame_number': i,
            'file_path': str(frame_path),
            'timestamp': i * 0.5
        })
    
    # Create depth data structure
    depth_data = []
    for i, depth_path in enumerate(test_depth_maps):  # Use all depth maps
        depth_data.append({
            'frame_number': i,
            'raw_depth_path': str(depth_path)
        })
    
    # Test 3D reconstruction
    try:
        logger.info("Testing 3D reconstruction...")
        result = reconstruction_3d.reconstruct_3d_model(job_id, frame_data, depth_data)
        
        logger.info(f"✅ 3D reconstruction completed successfully!")
        logger.info(f"Point cloud size: {result['point_cloud_size']} points")
        logger.info(f"Mesh faces: {result['mesh_faces']}")
        logger.info(f"Camera poses: {result['camera_poses']}")
        
        # List generated model files
        model_files = list(settings.MODELS_DIR.glob(f"{job_id}_*"))
        logger.info(f"Generated model files: {len(model_files)}")
        
        for model_file in model_files:
            file_size = model_file.stat().st_size / 1024  # KB
            logger.info(f"  - {model_file.name} ({file_size:.1f} KB)")
        
        # Check specific file types
        model_paths = result['model_paths']
        logger.info(f"✅ Generated model files: {len(model_paths)}")
        for model_type in model_paths:
            logger.info(f"  - {model_type}")
        
        logger.info("🎯 3D reconstruction test completed successfully!")
        logger.info("🚀 Ready for frontend integration!")
        
    except Exception as e:
        logger.error(f"❌ 3D reconstruction test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_3d_reconstruction())
