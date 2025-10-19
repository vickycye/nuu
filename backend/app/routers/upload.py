from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
# API router allows us to organize the API into separate route groups
# uploadfile supports large files because it doesn't load the whole file into memory at once
# file is used to declare a file input parameter in an API endpoint
# BackgroundTasks is used to run tasks int eh background after sending a response to the client
from pydantic import BaseModel
from app.models.job import JobInfo, JobStatus
from app.utils.file_manager import FileManager
from app.core.database import JobDatabase
from app.utils.processing_pipeline import ProcessingPipeline
from app.core.config import settings
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
file_manager = FileManager()
job_db = JobDatabase()
processing_pipeline = ProcessingPipeline()

# Response model for upload endpoint
class UploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload video file and start processing pipeline
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = file.filename.lower().split('.')[-1]
        if f'.{file_extension}' not in settings.SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.SUPPORTED_VIDEO_FORMATS)}"
            )
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = await file_manager.save_upload(file, job_id)
        
        # Create job record
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.UPLOADED,
            progress=0,
            message="Video uploaded successfully",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'filename': file.filename,
                'file_size': file.size,
                'file_path': file_path
            }
        )
        job_db.create_job(job_info)
        
        # Start background processing
        background_tasks.add_task(processing_pipeline.process_video, job_id, file_path)
        
        logger.info(f"Video uploaded successfully for job {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            status="uploaded",
            message="Video uploaded successfully. Processing started."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")
