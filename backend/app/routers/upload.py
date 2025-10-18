from fastapi import APIRouter, UploadFile, File, BackgroundTasks
# API router allows us to organize the API into separate route groups
# uploadfile supports large files because it doesn't load the whole file into memory at once
# file is used to declare a file input parameter in an API endpoint
# BackgroundTasks is used to run tasks int eh background after sending a response to the client
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

@router.post("/upload-sample/{sample_name}")
async def upload_sample_video(
    background_tasks: BackgroundTasks,
    sample_name: str
):
    """
    Upload a sample video from the sample_input directory for testing
    """
    try:
        sample_path = settings.SAMPLE_DIR / sample_name
        
        if not sample_path.exists():
            available_samples = [f.name for f in settings.SAMPLE_DIR.iterdir() if f.is_file()]
            raise HTTPException(
                status_code=404, 
                detail=f"Sample video '{sample_name}' not found. Available samples: {available_samples}"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Copy sample to uploads directory
        file_path = await file_manager.copy_sample_file(sample_path, job_id)
        
        # Create job record
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.UPLOADED,
            progress=0,
            message=f"Sample video '{sample_name}' uploaded successfully",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'filename': sample_name,
                'file_size': sample_path.stat().st_size,
                'file_path': str(file_path),
                'is_sample': True
            }
        )
        job_db.create_job(job_info)
        
        # Start background processing
        background_tasks.add_task(processing_pipeline.process_video, job_id, str(file_path))
        
        logger.info(f"Sample video uploaded successfully for job {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            status="uploaded",
            message=f"Sample video '{sample_name}' uploaded successfully. Processing started."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading sample video: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during sample upload")

@router.get("/samples")
async def list_sample_videos():
    """
    List available sample videos for testing
    """
    try:
        sample_files = []
        for file_path in settings.SAMPLE_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_VIDEO_FORMATS:
                sample_files.append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
                })
        
        return {
            'samples': sample_files,
            'count': len(sample_files)
        }
        
    except Exception as e:
        logger.error(f"Error listing sample videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")