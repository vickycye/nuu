from typing import Dict, Optional
from app.models.job import JobInfo, JobUpdate
from datetime import datetime

# job database class to handle job operations
class JobDatabase:
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
    
    def create_job(self, job_info: JobInfo):
        self.jobs[job_info.job_id] = job_info
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self.jobs.get(job_id)
    
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