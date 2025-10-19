#!/usr/bin/env python3
"""
Test script for video processing pipeline
Run this to test the video processing functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.utils.video_processor import VideoProcessor
from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_video_processing():
    """Test the video processing pipeline"""
    
    # Check if sample directory exists and has videos
    sample_dir = settings.SAMPLE_DIR
    if not sample_dir.exists():
        logger.error(f"Sample directory {sample_dir} does not exist")
        return
    
    # Find sample videos
    sample_videos = []
    for ext in settings.SUPPORTED_VIDEO_FORMATS:
        sample_videos.extend(sample_dir.glob(f"*{ext}"))
    
    if not sample_videos:
        logger.warning(f"No sample videos found in {sample_dir}")
        logger.info("Please add a sample video to test the processing pipeline")
        return
    
    # Test with first sample video
    test_video = sample_videos[0]
    logger.info(f"Testing with sample video: {test_video}")
    
    # Initialize video processor
    video_processor = VideoProcessor(settings.FRAMES_DIR)
    
    try:
        # Test video validation
        logger.info("Testing video validation...")
        is_valid = video_processor.validate_video(str(test_video))
        logger.info(f"Video validation result: {is_valid}")
        
        if not is_valid:
            logger.error("Video validation failed")
            return
        
        # Test video info extraction
        logger.info("Testing video info extraction...")
        video_info = video_processor.get_video_info(str(test_video))
        logger.info(f"Video info: {video_info}")
        
        # Test frame extraction
        logger.info("Testing frame extraction...")
        job_id = "test_job_001"
        frame_data = video_processor.extract_frames(str(test_video), job_id, 0.5)
        
        logger.info(f"Extracted {frame_data['total_frames_extracted']} frames")
        logger.info(f"Video info: {frame_data['video_info']}")
        
        # List extracted frames
        frames_dir = settings.FRAMES_DIR
        extracted_frames = list(frames_dir.glob(f"{job_id}_frame_*.jpg"))
        logger.info(f"Frame files created: {len(extracted_frames)}")
        
        for frame_file in extracted_frames[:5]:  # Show first 5 frames
            logger.info(f"  - {frame_file.name}")
        
        if len(extracted_frames) > 5:
            logger.info(f"  ... and {len(extracted_frames) - 5} more frames")
        
        logger.info("✅ Video processing test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_video_processing())
