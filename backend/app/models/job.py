from pydantic import BaseModel # python library used  for data validation and settings management
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime

# to be sent to the frontend to clearly communicate the status of the job with the user
class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING_FRAMES = "extracting_frames"
    ESTIMATING_DEPTH = "estimating_depth"
    RECONSTRUCTING_3D = "reconstructing_3d"
    COMPLETED = "completed"
    FAILED = "failed" # can fail at any time and send this back to the frontend

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: int  # 0-100 percentage
    message: str
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None #used to send any additional information to the frontend

# used to update the job status and progress
class JobUpdate(BaseModel):
    status: JobStatus
    progress: int
    message: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None