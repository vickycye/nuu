#!/usr/bin/env python3
"""
Script to rename frame files to match expected naming convention
"""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rename_frames():
    """Rename frames to match expected naming convention"""
    
    frames_dir = settings.FRAMES_DIR
    if not frames_dir.exists():
        logger.error(f"Frames directory {frames_dir} does not exist")
        return
    
    # Find current frames with different naming patterns
    current_patterns = [
        "output_frame_*.png",
        "output_frame_*.jpg",
        "*_frame_*.png",
        "*_frame_*.jpg"
    ]
    
    frames_to_rename = []
    for pattern in current_patterns:
        frames = list(frames_dir.glob(pattern))
        frames_to_rename.extend(frames)
    
    if not frames_to_rename:
        logger.info("No frames found to rename")
        return
    
    # Remove duplicates and sort
    frames_to_rename = sorted(list(set(frames_to_rename)))
    
    logger.info(f"Found {len(frames_to_rename)} frames to rename")
    
    # Rename frames to test_job_001_frame_XXXX.jpg format
    renamed_count = 0
    for i, frame_path in enumerate(frames_to_rename):
        try:
            # Create new name with proper format
            new_name = f"test_job_001_frame_{i:04d}.jpg"
            new_path = frames_dir / new_name
            
            # Rename the file
            frame_path.rename(new_path)
            renamed_count += 1
            
            if renamed_count <= 5:  # Show first 5 renames
                logger.info(f"Renamed: {frame_path.name} -> {new_name}")
            
        except Exception as e:
            logger.error(f"Error renaming {frame_path.name}: {str(e)}")
    
    if renamed_count > 5:
        logger.info(f"... and {renamed_count - 5} more frames renamed")
    
    logger.info(f"✅ Successfully renamed {renamed_count} frames")
    logger.info("Now you can run test_depth_estimation.py")

if __name__ == "__main__":
    rename_frames()
