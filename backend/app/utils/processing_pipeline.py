import asyncio
import logging
import random
from pathlib import Path
from typing import Dict, Any

from app.core.database import JobDatabase
from app.models.job import JobStatus, JobUpdate
from app.utils.video_processor import VideoProcessor
from app.utils.depth_estimator import DepthEstimator
from app.utils.reconstruction import Reconstruction3D
from app.core.config import settings

logger = logging.getLogger(__name__)

# HARDCODED TEST MODEL PATH (for demo purposes)
HARDCODED_MODEL_URL = "/models/test_reconstruct_model.glb"
USE_DEMO_MODE = True  # Set to False to use real reconstruction

# process the video and create a 3D model
class ProcessingPipeline:
    def __init__(self):
        self.job_db = JobDatabase()
        self.video_processor = VideoProcessor(settings.FRAMES_DIR)
        
        # Only initialize heavy ML components if not in demo mode
        if not USE_DEMO_MODE:
            from app.utils.depth_estimator import DepthEstimator
            from app.utils.reconstruction import Reconstruction3D
            self.depth_estimator = DepthEstimator(settings.DEPTH_MAPS_DIR)
            self.reconstruction_3d = Reconstruction3D(settings.MODELS_DIR)
        else:
            self.depth_estimator = None
            self.reconstruction_3d = None
    
    async def process_video(self, job_id: str, video_path: str):
        """
        Main processing pipeline for video to 3D model
        """
        try:
            logger.info(f"Starting processing for job {job_id} (Demo mode: {USE_DEMO_MODE})")
            
            if USE_DEMO_MODE:
                # DEMO MODE: Simulate processing with delays and return hardcoded model
                
                # Step 1: Validate video
                await self._update_job_status(job_id, JobStatus.UPLOADED, 5, "Validating video file...")
                await asyncio.sleep(0.5)
                
                # Step 2: Simulate frame extraction
                await self._update_job_status(job_id, JobStatus.EXTRACTING_FRAMES, 20, "Extracting frames from video...")
                await asyncio.sleep(1.5)
                
                # Step 3: Simulate depth estimation
                await self._update_job_status(job_id, JobStatus.ESTIMATING_DEPTH, 50, "Estimating depth maps...")
                await asyncio.sleep(2.0)
                
                # Step 4: Simulate 3D reconstruction
                await self._update_job_status(job_id, JobStatus.RECONSTRUCTING_3D, 80, "Generating 3D model...")
                delay = random.uniform(2.0, 4.0)
                await asyncio.sleep(delay)
                
                logger.info(f"Using hardcoded test model after simulated processing")
                
                # Return hardcoded model
                reconstruction_result = {
                    'point_cloud_size': 10000,  # Fake stats for demo
                    'mesh_faces': 5000,
                    'model_paths': {
                        'glb': HARDCODED_MODEL_URL,
                        'obj': None
                    }
                }
                
                # Step 5: Complete
                await self._update_job_status(
                    job_id, 
                    JobStatus.COMPLETED, 
                    100, 
                    "3D model generation completed successfully!",
                    metadata={
                        'frames_extracted': 30,  # Fake data
                        'depth_maps_generated': 30,
                        'depth_statistics': {'min': 0.1, 'max': 10.0, 'mean': 3.5},
                        'reconstruction_result': reconstruction_result,
                        'model_paths': reconstruction_result['model_paths'],
                        'video_info': {'duration': 10, 'fps': 30}
                    }
                )
                
                logger.info(f"Demo processing completed for job {job_id}")
                
            else:
                # REAL MODE: Full processing pipeline
                
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
                
                # extract frame paths for depth estimation
                frame_paths = [frame['file_path'] for frame in frame_data['frames']]
                
                # run depth estimation on all frames
                depth_results = self.depth_estimator.estimate_depth_batch(frame_paths, job_id)
                
                # get depth statistics
                depth_stats = self.depth_estimator.get_depth_statistics(job_id)
                
                logger.info(f"Depth estimation completed: {len(depth_results)} depth maps generated")
                
                # Step 4: 3D reconstruction
                await self._update_job_status(job_id, JobStatus.RECONSTRUCTING_3D, 80, "Generating 3D model...")
                
                # run 3D reconstruction
                reconstruction_result = self.reconstruction_3d.reconstruct_3d_model(
                    job_id, frame_data['frames'], depth_results
                )
                
                logger.info(f"3D reconstruction completed: {reconstruction_result['point_cloud_size']} points, {reconstruction_result['mesh_faces']} faces")
                
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
                        'reconstruction_result': reconstruction_result,
                        'model_paths': reconstruction_result['model_paths'],
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
