import os
import shutil
from fastapi import UploadFile
from typing import Optional
from pathlib import Path
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# manages teh file system operations for the application
class FileManager:
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        settings.UPLOAD_DIR.mkdir(exist_ok=True)
        settings.FRAMES_DIR.mkdir(exist_ok=True)
        settings.DEPTH_MAPS_DIR.mkdir(exist_ok=True)
        settings.MODELS_DIR.mkdir(exist_ok=True)
    
    async def save_upload(self, file: UploadFile, job_id: str) -> str:
        """Save uploaded file and return path"""
        file_path = settings.UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
    
    async def copy_sample_file(self, sample_path: Path, job_id: str) -> Path:
        """Copy sample file to uploads directory"""
        file_path = settings.UPLOAD_DIR / f"{job_id}_{sample_path.name}"
        shutil.copy2(sample_path, file_path)
        logger.info(f"Copied sample file: {sample_path} -> {file_path}")
        return file_path
    
    def get_frame_path(self, job_id: str, frame_number: int) -> Path:
        """Get path for extracted frame"""
        return settings.FRAMES_DIR / f"{job_id}_frame_{frame_number:04d}.jpg"
    
    def get_model_path(self, job_id: str) -> Path:
        """Get path for completed 3D model"""
        return settings.MODELS_DIR / f"{job_id}_model.glb"
    
    def cleanup_job_files(self, job_id: str):
        """Clean up all files associated with a job"""
        try:
            # Clean up uploaded file
            for upload_file in settings.UPLOAD_DIR.glob(f"{job_id}_*"):
                upload_file.unlink()
            
            # Clean up frames
            for frame_file in settings.FRAMES_DIR.glob(f"{job_id}_frame_*.jpg"):
                frame_file.unlink()
            
            # Clean up depth maps
            for depth_file in settings.DEPTH_MAPS_DIR.glob(f"{job_id}_*"):
                depth_file.unlink()
            
            # Clean up model
            model_file = self.get_model_path(job_id)
            if model_file.exists():
                model_file.unlink()
            
            logger.info(f"Cleaned up files for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up files for job {job_id}: {str(e)}")
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size
        except Exception:
            return 0