from fastapi import APIRouter, UploadFile, File, BackgroundTasks
# API router allows us to organize the API into separate route groups
# uploadfile supports large files because it doesn't load the whole file into memory at once
# file is used to declare a file input parameter in an API endpoint
# BackgroundTasks is used to run tasks int eh background after sending a response to the client
from app.models.job import JobInfo, JobStatus
from app.utils.file_manager import FileManager
from app.core.database import JobDatabase
import uuid
from datetime import datetime

router = APIRouter()
file_manager = FileManager()
job_db = JobDatabase()

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # generate job ID
    job_id = str(uuid.uuid4())
    
    # save uploaded file
    file_path = await file_manager.save_upload(file, job_id)
    
    # create job record
    job_info = JobInfo(
        job_id=job_id,
        status=JobStatus.UPLOADED,
        progress=0,
        message="Video uploaded successfully",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    job_db.create_job(job_info)
    
    # start background processing
    background_tasks.add_task(process_video_pipeline, job_id, file_path) # implement later
    
    return {"job_id": job_id, "status": "uploaded"}

async def process_video_pipeline(job_id: str, video_path: str):
    # This will be implemented in the video processor
    # Updates job status throughout processing
    pass
