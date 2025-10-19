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
    
    def __init__(self, depth_maps_dir: Path, model_name: str = "midas_v21_small_256"):
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
            # Add local MiDaS to Python path
            import sys
            local_midas_path = Path(__file__).parent.parent.parent / "models" / "midas"
            if local_midas_path.exists():
                sys.path.insert(0, str(local_midas_path))
                logger.info(f"Added local MiDaS path: {local_midas_path}")
            
            # Import MiDaS utilities
            from midas.model_loader import load_model
            from midas.transforms import Resize, NormalizeImage, PrepareForNet
            
            # Load the model
            # Get the path to the model weights file
            weights_dir = Path(__file__).parent.parent.parent / "models" / "midas" / "weights"
            model_path = weights_dir / f"{self.model_name}.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found: {model_path}")
            
            self.model, self.transform, self.net_w, self.net_h = load_model(
                self.device, 
                str(model_path),
                model_type=self.model_name,
                optimize=True
            )
            
            logger.info(f"Loaded MiDaS model: {self.model_name}")
            logger.info(f"Model input size: {self.net_w}x{self.net_h}")
            
        except ImportError as e:
            logger.error(f"MiDaS import failed: {str(e)}")
            logger.info("Trying to install MiDaS...")
            self._install_midas()
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading MiDaS model: {str(e)}")
            raise
    
    def _install_midas(self):
        """
        Install MiDaS dependencies
        """
        try:
            import subprocess
            import sys
            
            # Install required packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "opencv-python", "pillow", "timm"
            ])
            
            # Check if local MiDaS already exists
            local_midas_path = Path(__file__).parent.parent.parent / "models" / "midas"
            if local_midas_path.exists():
                logger.info("Local MiDaS installation found, using existing installation")
                sys.path.insert(0, str(local_midas_path))
            else:
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
            image = self.transform({"image": image})["image"]
            # Convert to tensor if it's still a numpy array
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            
            # Ensure 4D tensor (batch_size, channels, height, width)
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        else:
            # Fallback preprocessing
            image = cv2.resize(image, (self.net_w, self.net_h))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Ensure it's a tensor and move to device
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            if image.dim() == 3:
                image = image.unsqueeze(0)
        
        return image.to(self.device)
    
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
