from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from app.core.database import JobDatabase
from app.models.job import JobStatus
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
job_db = JobDatabase()

# download the model
@router.get("/model/{job_id}")
async def download_model(job_id: str, format: str = "glb"):
    """
    Download completed 3D model
    """
    try:
        # get job info
        job_info = job_db.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_info.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job_info.status}"
            )
        
        # get model paths from metadata
        model_paths = job_info.metadata.get('model_paths', {})
        
        # determine file path based on format
        if format == "glb" and "glb" in model_paths:
            file_path = Path(model_paths["glb"])
        elif format == "obj" and "obj" in model_paths:
            file_path = Path(model_paths["obj"])
        elif format == "ply" and "pointcloud" in model_paths:
            file_path = Path(model_paths["pointcloud"])
        else:
            # default to GLB, fallback to OBJ, then PLY
            if "glb" in model_paths:
                file_path = Path(model_paths["glb"])
                format = "glb"
            elif "obj" in model_paths:
                file_path = Path(model_paths["obj"])
                format = "obj"
            elif "pointcloud" in model_paths:
                file_path = Path(model_paths["pointcloud"])
                format = "ply"
            else:
                raise HTTPException(status_code=404, detail="No model files found")
        
        # check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # set appropriate media type
        media_types = {
            "glb": "model/gltf-binary",
            "obj": "application/obj",
            "ply": "application/ply"
        }
        
        media_type = media_types.get(format, "application/octet-stream")
        
        logger.info(f"Serving {format} model for job {job_id}: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=f"room_model_{job_id}.{format}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# get information about the model
@router.get("/model/{job_id}/info")
async def get_model_info(job_id: str):
    """
    Get information about the 3D model
    """
    try:
        # get job info
        job_info = job_db.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_info.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job_info.status}"
            )
        
        # extract model information
        metadata = job_info.metadata or {}
        reconstruction_result = metadata.get('reconstruction_result', {})
        model_paths = metadata.get('model_paths', {})
        
        # get file sizes
        file_sizes = {}
        for format_name, path in model_paths.items():
            file_path = Path(path)
            if file_path.exists():
                file_sizes[format_name] = file_path.stat().st_size
        
        return {
            "job_id": job_id,
            "status": job_info.status,
            "model_info": {
                "point_cloud_size": reconstruction_result.get('point_cloud_size', 0),
                "mesh_faces": reconstruction_result.get('mesh_faces', 0),
                "camera_poses": reconstruction_result.get('camera_poses', 0),
                "available_formats": list(model_paths.keys()),
                "file_sizes": file_sizes
            },
            "processing_info": {
                "frames_extracted": metadata.get('frames_extracted', 0),
                "depth_maps_generated": metadata.get('depth_maps_generated', 0),
                "video_info": metadata.get('video_info', {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# download the camrea poses
@router.get("/model/{job_id}/poses")
async def download_camera_poses(job_id: str):
    """
    Download camera poses data
    """
    try:
        # get job info
        job_info = job_db.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_info.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job_info.status}"
            )
        
        # get poses file path
        model_paths = job_info.metadata.get('model_paths', {})
        if "poses" not in model_paths:
            raise HTTPException(status_code=404, detail="Camera poses not found")
        
        poses_path = Path(model_paths["poses"])
        if not poses_path.exists():
            raise HTTPException(status_code=404, detail="Camera poses file not found")
        
        logger.info(f"Serving camera poses for job {job_id}: {poses_path}")
        
        return FileResponse(
            path=str(poses_path),
            media_type="application/json",
            filename=f"camera_poses_{job_id}.json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading camera poses: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
