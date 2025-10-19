import os
from pathlib import Path

class Settings:
    # base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    TEMP_DIR = BASE_DIR / "temp"
    SAMPLE_DIR = BASE_DIR / "sample_input"
    
    # file size limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB so our backend doesn't explode
    
    # video processing settings
    FRAME_EXTRACTION_INTERVAL = 0.1  # seconds, this can be tweaked later
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    # processing directories
    UPLOAD_DIR = TEMP_DIR / "uploads"
    FRAMES_DIR = TEMP_DIR / "frames"
    DEPTH_MAPS_DIR = TEMP_DIR / "depth_maps"
    MODELS_DIR = TEMP_DIR / "models"
    
    def __init__(self):
        # create directories if they don't exist
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.SAMPLE_DIR.mkdir(exist_ok=True)
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.FRAMES_DIR.mkdir(exist_ok=True)
        self.DEPTH_MAPS_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)

settings = Settings()
