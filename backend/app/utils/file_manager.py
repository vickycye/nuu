import os
import shutil
from fastapi import UploadFile
from typing import Optional

# file manager class to handle file operations
class FileManager:
    def __init__(self):
        self.base_path = "temp"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ["uploads", "frames", "depth_maps", "models"]
        for dir_name in dirs:
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)
    
    async def save_upload(self, file: UploadFile, job_id: str) -> str:
        """Save uploaded file and return path"""
        file_path = os.path.join(self.base_path, "uploads", f"{job_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path
    
    def get_frame_path(self, job_id: str, frame_number: int) -> str:
        """Get path for extracted frame"""
        return os.path.join(self.base_path, "frames", f"{job_id}_frame_{frame_number:04d}.jpg")
    
    def get_model_path(self, job_id: str) -> str:
        """Get path for completed 3D model"""
        return os.path.join(self.base_path, "models", f"{job_id}_model.glb")