from typing import Dict, Optional, List
from app.models.job import JobInfo, JobUpdate
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# handles job operations
class JobDatabase:
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
    
    def create_job(self, job_info: JobInfo):
        self.jobs[job_info.job_id] = job_info
        logger.info(f"Created job {job_info.job_id}")
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[JobInfo]:
        return list(self.jobs.values())
    
    def update_job(self, job_id: str, update: JobUpdate):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = update.status
            job.progress = update.progress
            job.message = update.message
            job.error = update.error
            job.updated_at = datetime.now()
            if update.metadata:
                job.metadata = update.metadata
            logger.info(f"Updated job {job_id}: {update.status} - {update.message}")
        else:
            logger.warning(f"Attempted to update non-existent job {job_id}")
    
    def delete_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Deleted job {job_id}")
            return True
        return False
    
    def get_jobs_by_status(self, status: str) -> List[JobInfo]:
        return [job for job in self.jobs.values() if job.status == status]