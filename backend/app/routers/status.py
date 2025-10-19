from fastapi import APIRouter, HTTPException
from app.models.job import JobInfo
from app.core.database import JobDatabase

router = APIRouter()
job_db = JobDatabase()

# get the status of a job
@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job_info = job_db.get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job_info.status.value if hasattr(job_info.status, 'value') else job_info.status,
        "progress": job_info.progress,
        "message": job_info.message,
        "error": job_info.error,
        "updated_at": job_info.updated_at.isoformat() if job_info.updated_at else None
    }