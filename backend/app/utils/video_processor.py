import cv2
import os
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# this class extracts frames from a video at specified intervals defined in the config.py file
class VideoProcessor:
    def __init__(self, frames_dir: Path):
        self.frames_dir = frames_dir
        self.frames_dir.mkdir(exist_ok=True)
    
    def extract_frames(self, video_path: str, job_id: str, interval: float = 0.5) -> Dict[str, Any]:
        """
        Extract frames from video at specified intervals
        Returns metadata about extracted frames
        """
        try:
            # open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video info - FPS: {fps}, Duration: {duration:.2f}s, Resolution: {width}x{height}")
            
            # calculate frame extraction points
            frame_interval = int(fps * interval)  # frames between extractions
            frame_numbers = list(range(0, total_frames, frame_interval))
            
            extracted_frames = []
            frame_count = 0
            
            for frame_num in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # apply rotation if needed (for phone videos)
                    frame = self._apply_rotation(frame, cap)
                    
                    # save frame
                    frame_filename = f"{job_id}_frame_{frame_count:04d}.jpg"
                    frame_path = self.frames_dir / frame_filename
                    
                    # resize frame if too large (max 1920x1080)
                    if width > 1920 or height > 1080:
                        scale = min(1920/width, 1080/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append({
                        'frame_number': frame_count,
                        'original_frame': frame_num,
                        'timestamp': frame_num / fps,
                        'file_path': str(frame_path),
                        'filename': frame_filename
                    })
                    frame_count += 1
                else:
                    logger.warning(f"Could not read frame {frame_num}")
            
            cap.release()
            
            logger.info(f"Extracted {len(extracted_frames)} frames from video")
            
            return {
                'total_frames_extracted': len(extracted_frames),
                'frames': extracted_frames,
                'video_info': {
                    'fps': fps,
                    'duration': duration,
                    'total_frames': total_frames,
                    'width': width,
                    'height': height,
                    'interval': interval
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    # get video metadata (duration, fps, resolution)
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata (duration, fps, resolution)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'file_size': os.path.getsize(video_path)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    # make sure video file format and properties are valid
    def validate_video(self, video_path: str) -> bool:
        """
        Validate video file format and basic properties
        """
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                return False
            
            # Check file size (max 100MB)
            file_size = os.path.getsize(video_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                return False
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
            
        except Exception as e:
            logger.error(f"Error validating video: {str(e)}")
            return False
    
    def cleanup_frames(self, job_id: str):
        """
        Clean up extracted frames for a specific job
        """
        try:
            frame_files = list(self.frames_dir.glob(f"{job_id}_frame_*.jpg"))
            for frame_file in frame_files:
                frame_file.unlink()
            logger.info(f"Cleaned up {len(frame_files)} frames for job {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up frames: {str(e)}")
    
    def _apply_rotation(self, frame, cap):
        """
        Apply rotation to frame based on video metadata (for phone videos)
        """
        try:
            # Get rotation metadata from video
            rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            
            # Log rotation info for debugging
            if rotation != 0:
                logger.info(f"Detected rotation metadata: {rotation} degrees")
            
            if rotation == 90:
                # Rotate 90 degrees clockwise
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                logger.info("Applied 90° clockwise rotation")
            elif rotation == 180:
                # Rotate 180 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                logger.info("Applied 180° rotation")
            elif rotation == 270:
                # Rotate 90 degrees counter-clockwise
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                logger.info("Applied 90° counter-clockwise rotation")
            
            return frame
            
        except Exception as e:
            logger.warning(f"Could not apply rotation: {str(e)}")
            return frame
