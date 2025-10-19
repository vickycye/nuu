import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from PIL import Image
import requests
import os

logger = logging.getLogger(__name__)

class DepthEstimator:
    """
    Depth estimation using MiDaS pretrained model
    """
    
    def __init__(self, depth_maps_dir: Path, model_name: str = "MiDaS_small"):
        self.depth_maps_dir = depth_maps_dir
        self.depth_maps_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing depth estimator with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """
        Load MiDaS model and preprocessing transform
        """
        try:
            # Use a simpler approach with torch.hub
            logger.info("Loading MiDaS model from torch.hub...")
            
            # Load MiDaS model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Set up transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_name == "MiDaS_small":
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.default_transform
            
            logger.info(f"Loaded MiDaS model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading MiDaS model: {str(e)}")
            logger.info("Falling back to simple depth estimation...")
            self._setup_fallback_model()
    
    def _setup_fallback_model(self):
        """
        Setup a simple fallback depth estimation model
        """
        logger.info("Setting up fallback depth estimation...")
        self.model = None
        self.transform = None
        self.use_fallback = True
    
    def _install_midas(self):
        """
        Install MiDaS dependencies
        """
        try:
            import subprocess
            import sys
            
            # Install MiDaS
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "opencv-python", "pillow"
            ])
            
            # Clone and install MiDaS
            midas_path = Path("models/midas")
            midas_path.mkdir(parents=True, exist_ok=True)
            
            if not (midas_path / "midas").exists():
                subprocess.check_call([
                    "git", "clone", 
                    "https://github.com/isl-org/MiDaS.git",
                    str(midas_path / "midas")
                ])
            
            # Add to Python path
            sys.path.append(str(midas_path))
            
            logger.info("MiDaS installation completed")
            
        except Exception as e:
            logger.error(f"Error installing MiDaS: {str(e)}")
            raise
    
    def estimate_depth(self, frame_path: str, job_id: str, frame_number: int) -> Dict[str, Any]:
        """
        Estimate depth for a single frame
        """
        try:
            if hasattr(self, 'use_fallback') and self.use_fallback:
                return self._estimate_depth_fallback(frame_path, job_id, frame_number)
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(frame_path)
            
            # Run depth estimation
            with torch.no_grad():
                depth = self.model(image)
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy and normalize
            depth_np = depth.cpu().numpy()
            depth_normalized = self._normalize_depth(depth_np)
            
            # Save depth map
            depth_filename = f"{job_id}_depth_{frame_number:04d}.png"
            depth_path = self.depth_maps_dir / depth_filename
            
            # Save as both PNG (visualization) and NPY (raw data)
            self._save_depth_map(depth_normalized, depth_path)
            
            # Also save raw depth data
            raw_depth_path = self.depth_maps_dir / f"{job_id}_depth_raw_{frame_number:04d}.npy"
            np.save(raw_depth_path, depth_np)
            
            logger.info(f"Generated depth map for frame {frame_number}: {depth_filename}")
            
            return {
                'frame_number': frame_number,
                'depth_map_path': str(depth_path),
                'raw_depth_path': str(raw_depth_path),
                'depth_filename': depth_filename,
                'depth_range': {
                    'min': float(np.min(depth_np)),
                    'max': float(np.max(depth_np)),
                    'mean': float(np.mean(depth_np))
                }
            }
            
        except Exception as e:
            logger.error(f"Error estimating depth for frame {frame_number}: {str(e)}")
            raise
    
    def estimate_depth_batch(self, frame_paths: List[str], job_id: str) -> List[Dict[str, Any]]:
        """
        Estimate depth for multiple frames in batch
        """
        results = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                result = self.estimate_depth(frame_path, job_id, i)
                results.append(result)
                
                # Update progress
                progress = int((i + 1) / len(frame_paths) * 100)
                logger.info(f"Depth estimation progress: {progress}% ({i+1}/{len(frame_paths)})")
                
            except Exception as e:
                logger.error(f"Failed to process frame {i}: {str(e)}")
                continue
        
        logger.info(f"Completed depth estimation for {len(results)} frames")
        return results
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image for MiDaS
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback preprocessing
            image = cv2.resize(image, (384, 384))  # MiDaS small default size
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def _estimate_depth_fallback(self, frame_path: str, job_id: str, frame_number: int) -> Dict[str, Any]:
        """
        Fallback depth estimation using simple computer vision techniques
        """
        try:
            # Load image
            image = cv2.imread(frame_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple depth estimation using edge detection and distance transform
            edges = cv2.Canny(gray, 50, 150)
            
            # Distance transform for depth approximation
            dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
            
            # Normalize to 0-255 range
            depth_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Save depth map
            depth_filename = f"{job_id}_depth_{frame_number:04d}.png"
            depth_path = self.depth_maps_dir / depth_filename
            
            # Save as both PNG (visualization) and NPY (raw data)
            self._save_depth_map(depth_normalized, depth_path)
            
            # Also save raw depth data
            raw_depth_path = self.depth_maps_dir / f"{job_id}_depth_raw_{frame_number:04d}.npy"
            np.save(raw_depth_path, dist_transform)
            
            logger.info(f"Generated fallback depth map for frame {frame_number}: {depth_filename}")
            
            return {
                'frame_number': frame_number,
                'depth_map_path': str(depth_path),
                'raw_depth_path': str(raw_depth_path),
                'depth_filename': depth_filename,
                'depth_range': {
                    'min': float(np.min(dist_transform)),
                    'max': float(np.max(dist_transform)),
                    'mean': float(np.mean(dist_transform))
                },
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback depth estimation: {str(e)}")
            raise
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth map for visualization
        """
        # Normalize to 0-255 range
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        
        if depth_max > depth_min:
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)
        
        return depth_normalized
    
    def _save_depth_map(self, depth_map: np.ndarray, output_path: Path):
        """
        Save depth map as image
        """
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path), depth_colored)
    
    def get_depth_statistics(self, job_id: str) -> Dict[str, Any]:
        """
        Get statistics for all depth maps of a job
        """
        try:
            depth_files = list(self.depth_maps_dir.glob(f"{job_id}_depth_raw_*.npy"))
            
            if not depth_files:
                return {"error": "No depth maps found"}
            
            all_depths = []
            for depth_file in depth_files:
                depth_data = np.load(depth_file)
                all_depths.append(depth_data)
            
            # Combine all depth data
            combined_depth = np.concatenate([d.flatten() for d in all_depths])
            
            return {
                "total_frames": len(depth_files),
                "depth_statistics": {
                    "min": float(np.min(combined_depth)),
                    "max": float(np.max(combined_depth)),
                    "mean": float(np.mean(combined_depth)),
                    "std": float(np.std(combined_depth)),
                    "median": float(np.median(combined_depth))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting depth statistics: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_depth_maps(self, job_id: str):
        """
        Clean up depth maps for a specific job
        """
        try:
            depth_files = list(self.depth_maps_dir.glob(f"{job_id}_depth_*"))
            for depth_file in depth_files:
                depth_file.unlink()
            logger.info(f"Cleaned up {len(depth_files)} depth maps for job {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up depth maps: {str(e)}")
