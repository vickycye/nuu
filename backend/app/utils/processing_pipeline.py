import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from app.core.database import JobDatabase
from app.models.job import JobStatus, JobUpdate
from app.utils.video_processor import VideoProcessor
from app.utils.depth_estimator import DepthEstimator
from app.core.config import settings

logger = logging.getLogger(__name__)

# process the video and create a 3D model
class ProcessingPipeline:
    def __init__(self):
        self.job_db = JobDatabase()
        self.video_processor = VideoProcessor(settings.FRAMES_DIR)
        self.depth_estimator = DepthEstimator(settings.DEPTH_MAPS_DIR)
    
    async def process_video(self, job_id: str, video_path: str):
        """
        Main processing pipeline for video to 3D model
        """
        try:
            logger.info(f"Starting processing for job {job_id}")
            
            # Step 1: Validate video
            await self._update_job_status(job_id, JobStatus.UPLOADED, 5, "Validating video file...")
            
            if not self.video_processor.validate_video(video_path):
                await self._update_job_status(job_id, JobStatus.FAILED, 0, "Invalid video file", "Video validation failed")
                return
            
            # Step 2: Extract frames
            await self._update_job_status(job_id, JobStatus.EXTRACTING_FRAMES, 20, "Extracting frames from video...")
            
            frame_data = self.video_processor.extract_frames(
                video_path, 
                job_id, 
                settings.FRAME_EXTRACTION_INTERVAL
            )
            
            # Step 3: Depth estimation
            await self._update_job_status(job_id, JobStatus.ESTIMATING_DEPTH, 50, "Estimating depth maps...")
            
            # Extract frame paths for depth estimation
            frame_paths = [frame['file_path'] for frame in frame_data['frames']]
            
            # Run depth estimation on all frames
            depth_results = self.depth_estimator.estimate_depth_batch(frame_paths, job_id)
            
            # Get depth statistics
            depth_stats = self.depth_estimator.get_depth_statistics(job_id)
            
            logger.info(f"Depth estimation completed: {len(depth_results)} depth maps generated")
            
            # Step 4: 3D reconstruction (placeholder for now)
            await self._update_job_status(job_id, JobStatus.RECONSTRUCTING_3D, 80, "Generating 3D model...")
            
            # TODO: Implement 3D reconstruction
            await asyncio.sleep(3)  # Simulate processing time
            
            # Step 5: Complete
            await self._update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                100, 
                "3D model generation completed successfully!",
                metadata={
                    'frames_extracted': frame_data['total_frames_extracted'],
                    'depth_maps_generated': len(depth_results),
                    'depth_statistics': depth_stats,
                    'video_info': frame_data['video_info']
                }
            )
            
            logger.info(f"Processing completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            await self._update_job_status(
                job_id, 
                JobStatus.FAILED, 
                0, 
                "Processing failed", 
                str(e)
            )
    
    async def _update_job_status(self, job_id: str, status: JobStatus, progress: int, message: str, error: str = None, metadata: Dict[str, Any] = None):
        """
        Update job status in database
        """
        try:
            update = JobUpdate(
                status=status,
                progress=progress,
                message=message,
                error=error,
                metadata=metadata
            )
            self.job_db.update_job(job_id, update)
            logger.info(f"Job {job_id} status updated: {status} - {message}")
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get overall processing statistics
        """
        jobs = self.job_db.get_all_jobs()
        
        stats = {
            'total_jobs': len(jobs),
            'completed': len([j for j in jobs if j.status == JobStatus.COMPLETED]),
            'failed': len([j for j in jobs if j.status == JobStatus.FAILED]),
            'processing': len([j for j in jobs if j.status in [JobStatus.EXTRACTING_FRAMES, JobStatus.ESTIMATING_DEPTH, JobStatus.RECONSTRUCTING_3D]])
        }
        
        return stats
