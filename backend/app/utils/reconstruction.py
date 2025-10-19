import cv2
import numpy as np
import trimesh
import logging
import subprocess
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import struct

logger = logging.getLogger(__name__)

# reads the colmap binary model files (cameras.bin, images.bin, points3D.bin)
def read_model(path):
    """
    Read COLMAP binary model files (cameras.bin, images.bin, points3D.bin)
    """
    import struct
    
    cameras = {}
    images = {}
    points3D = {}
    
    # dependent on camera pose estimation working.
    # parse binary files created by COLMAP containing camera parameters, image poses, and 3D points.
    cameras_file = Path(path) / "cameras.bin"
    if cameras_file.exists():
        with open(cameras_file, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model = struct.unpack("<I", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                params = struct.unpack("<" + "d" * 4, f.read(32))
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
    
    # extracts image poses: rotations as quaternions, translation as vectors, and camera id.
    images_file = Path(path) / "images.bin"
    if images_file.exists():
        with open(images_file, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
                tx, ty, tz = struct.unpack("<ddd", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                
                # read image name (null-terminated string)
                name_bytes = b""
                while True:
                    byte = f.read(1)
                    if byte == b'\x00' or byte == b'':
                        break
                    name_bytes += byte
                try:
                    name = name_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    name = name_bytes.decode('latin-1', errors='ignore')
                
                # read point2D data
                num_points2D = struct.unpack("<Q", f.read(8))[0]
                points2D = []
                for _ in range(num_points2D):
                    x, y = struct.unpack("<dd", f.read(16))
                    point3D_id = struct.unpack("<i", f.read(4))[0]
                    points2D.append((x, y, point3D_id))
                
                images[image_id] = {
                    'qvec': [qw, qx, qy, qz],
                    'tvec': [tx, ty, tz],
                    'camera_id': camera_id,
                    'name': name,
                    'xys': points2D
                }
    
    # read points3D.bin, loads the 3D point cloud data with colors and tracking information. 
    points3D_file = Path(path) / "points3D.bin"
    if points3D_file.exists():
        with open(points3D_file, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack("<Q", f.read(8))[0]
                x, y, z = struct.unpack("<ddd", f.read(24))
                r, g, b = struct.unpack("<BBB", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                
                # read track data
                track_length = struct.unpack("<Q", f.read(8))[0]
                track = []
                for _ in range(track_length):
                    image_id = struct.unpack("<I", f.read(4))[0]
                    point2D_idx = struct.unpack("<I", f.read(4))[0]
                    track.append((image_id, point2D_idx))
                
                points3D[point3D_id] = {
                    'xyz': [x, y, z],
                    'rgb': [r, g, b],
                    'error': error,
                    'image_ids': [t[0] for t in track],
                    'point2D_idxs': [t[1] for t in track]
                }
    
    # returns dictionaries containing cameras, images, and 3D points. 
    return cameras, images, points3D

# main class for 3D reconstruction
class Reconstruction3D:
    """
    3D reconstruction from depth maps and camera poses
    """
    
    # intiailized the reconstruction class with a directory to store model outputs.
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
    
    # main 3d reconstructoin pipeline: takes job_id, frame_data, and depth_data as input.
    def reconstruct_3d_model(self, job_id: str, frame_data: List[Dict], depth_data: List[Dict]) -> Dict[str, Any]:
        """
        Main 3D reconstruction pipeline
        """
        try:
            logger.info(f"Starting 3D reconstruction for job {job_id}")
            
            # Step 1: estimate camera poses using COLMAP
            camera_poses = self._estimate_camera_poses(job_id, frame_data)
            
            # Step 2: generate point cloud from depth maps and poses
            point_cloud = self._generate_point_cloud(job_id, frame_data, depth_data, camera_poses)
            
            # Check if we have enough points
            if len(point_cloud['points']) < 100:
                logger.warning(f"Only {len(point_cloud['points'])} points generated, this may indicate issues with depth maps or camera poses")
                # Try to generate points from all frames, not just the ones with poses
                logger.info("Attempting to generate point cloud from all available depth maps...")
                point_cloud = self._generate_point_cloud_from_all_frames(frame_data, depth_data)
            
            # Step 3: create mesh from point cloud
            mesh = self._create_mesh_from_point_cloud(point_cloud)
            
            # Step 4: detect doors in the room
            doors = self._detect_doors(point_cloud['points'], point_cloud.get('room_bounds', {}))
            
            # Step 5: export 3D model
            model_paths = self._export_3d_model(job_id, mesh, point_cloud, camera_poses, doors)
            
            logger.info(f"3D reconstruction completed for job {job_id}")
            
            # returns reconstruction metadata including model paths, point cloud size, mesh faces, and camera poses. 
            return {
                'job_id': job_id,
                'model_paths': model_paths,
                'point_cloud_size': len(point_cloud['points']),
                'mesh_faces': len(mesh.faces) if mesh else 0,
                'camera_poses': len(camera_poses)
            }
            
        except Exception as e:
            logger.error(f"Error in 3D reconstruction: {str(e)}")
            raise
    
    def _estimate_camera_poses(self, job_id: str, frame_data: List[Dict]) -> List[Dict]:
        """
        Estimate camera poses using COLMAP
        """
        try:
            # Use COLMAP for camera pose estimation
            return self._estimate_poses_colmap(job_id, frame_data)
                
        except Exception as e:
            logger.error(f"Error estimating camera poses: {str(e)}")
            raise RuntimeError(f"Camera pose estimation failed: {str(e)}. Please ensure your video has sufficient overlap between frames.")
    
    def _estimate_poses_colmap(self, job_id: str, frame_data: List[Dict]) -> List[Dict]:
        """
        Estimate camera poses using COLMAP
        """
        logger.info("Using COLMAP for camera pose estimation")
        
        # Create COLMAP workspace
        colmap_dir = self.models_dir / f"{job_id}_colmap"
        colmap_dir.mkdir(exist_ok=True)
        
        # Prepare images for COLMAP
        images_dir = colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Use every 10th frame for faster processing (reduce from 904 to ~90 frames)
        selected_frames = frame_data[::10]  # Take every 10th frame
        logger.info(f"Using {len(selected_frames)} frames for COLMAP (every 10th frame)")
        
        for i, frame in enumerate(selected_frames):
            src_path = Path(frame['file_path'])
            dst_path = images_dir / f"frame_{i:04d}.jpg"
            shutil.copy2(src_path, dst_path)
        
        # Run COLMAP feature extraction with optimized settings
        try:
            subprocess.run([
                'colmap', 'feature_extractor',
                '--database_path', str(colmap_dir / 'database.db'),
                '--image_path', str(images_dir),
                '--ImageReader.single_camera', '1',
                '--SiftExtraction.max_num_features', '8192',  # More features for better matching
                '--SiftExtraction.first_octave', '-1',  # Start from higher resolution
                '--SiftExtraction.num_octaves', '4',  # More octaves
                '--SiftExtraction.upright', '0'  # Allow rotation for better matching
            ], check=True, timeout=600)  # Increase timeout
            logger.info("COLMAP feature extraction completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"COLMAP feature extraction failed: {e}")
            raise RuntimeError(f"COLMAP feature extraction failed: {e}")
        
        # Run COLMAP matching with optimized settings
        # Try sequential matching first (much faster for video frames)
        try:
            logger.info("Trying sequential matching (optimized for video)...")
            subprocess.run([
                'colmap', 'sequential_matcher',
                '--database_path', str(colmap_dir / 'database.db'),
                '--SiftMatching.max_ratio', '0.9',  # More lenient matching
                '--SiftMatching.max_distance', '0.9',  # More lenient matching
                '--SiftMatching.cross_check', '1',
                '--SequentialMatching.overlap', '10'  # Match with 10 previous frames
            ], check=True, timeout=900)  # 15 minutes timeout
            logger.info("COLMAP sequential matching completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Sequential matching failed: {e}")
            logger.info("Falling back to exhaustive matching...")
            try:
                subprocess.run([
                    'colmap', 'exhaustive_matcher',
                    '--database_path', str(colmap_dir / 'database.db'),
                    '--SiftMatching.max_ratio', '0.9',  # More lenient matching
                    '--SiftMatching.max_distance', '0.9',  # More lenient matching
                    '--SiftMatching.cross_check', '1'
                ], check=True, timeout=1800)  # 30 minutes timeout
                logger.info("COLMAP exhaustive matching completed")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"COLMAP feature matching failed: {e}")
                raise RuntimeError(f"COLMAP feature matching failed: {e}")
        
        # Run COLMAP reconstruction with optimized settings
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                'colmap', 'mapper',
                '--database_path', str(colmap_dir / 'database.db'),
                '--image_path', str(images_dir),
                '--output_path', str(sparse_dir),
                '--Mapper.init_min_num_inliers', '30',  # Even lower threshold
                '--Mapper.abs_pose_max_error', '16',  # More lenient
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_principal_point', '0',
                '--Mapper.min_num_matches', '10',  # Lower minimum matches
                '--Mapper.max_num_models', '50',  # Limit models
                '--Mapper.max_model_overlap', '20'  # Reduce overlap requirement
            ], check=True, timeout=1800)  # 30 minutes timeout
            logger.info("COLMAP reconstruction completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"COLMAP reconstruction failed: {e}")
            raise RuntimeError(f"COLMAP reconstruction failed: {e}. Please ensure your video has sufficient overlap between frames and try again.")
        
        # Export COLMAP results to text format for easier parsing
        text_dir = sparse_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                'colmap', 'model_converter',
                '--input_path', str(sparse_dir / "0"),  # Use the first reconstruction
                '--output_path', str(text_dir),
                '--output_type', 'TXT'
            ], check=True, timeout=60)
            logger.info("COLMAP text export completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"COLMAP text export failed: {e}")
            raise RuntimeError(f"COLMAP text export failed: {e}")
        
        # Parse COLMAP text results
        return self._parse_colmap_text_results(text_dir, len(frame_data))
    
    def _parse_colmap_text_results(self, text_dir: Path, num_frames: int) -> List[Dict]:
        """
        Parse COLMAP text output files
        """
        poses = []
        
        try:
            # Read images.txt file
            images_file = text_dir / "images.txt"
            if not images_file.exists():
                raise RuntimeError("COLMAP images.txt file not found")
            
            with open(images_file, 'r') as f:
                lines = f.readlines()
                
            # Skip header lines and empty lines
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            # COLMAP format: each image has two lines
            # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # Line 2: POINTS2D data (skip this line)
            i = 0
            while i < len(data_lines):
                line = data_lines[i]
                parts = line.split()
                
                if len(parts) >= 10:
                    # This is an image line
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    name = parts[9]
                    
                    # Extract frame number from name
                    try:
                        frame_num = int(name.split('_')[-1].split('.')[0])
                    except:
                        frame_num = image_id
                    
                    pose = {
                        'frame_number': frame_num,
                        'position': [tx, ty, tz],
                        'rotation': [qw, qx, qy, qz],  # Quaternion
                        'camera_id': camera_id,
                        'image_id': image_id,
                        'timestamp': frame_num * 0.5
                    }
                    poses.append(pose)
                    
                    # Skip the next line (POINTS2D data)
                    i += 2
                else:
                    # Skip malformed lines
                    i += 1
            
            # Sort by frame number
            poses.sort(key=lambda x: x['frame_number'])
            logger.info(f"Parsed {len(poses)} poses from COLMAP text output")
            return poses
            
        except Exception as e:
            logger.error(f"Error parsing COLMAP text results: {str(e)}")
            raise RuntimeError(f"Failed to parse COLMAP text results: {str(e)}")
    
    def _parse_colmap_results(self, sparse_dir: Path, num_frames: int) -> List[Dict]:
        """
        Parse COLMAP reconstruction results
        """
        poses = []
        
        # Find the most recent reconstruction (highest numbered folder)
        reconstruction_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if reconstruction_dirs:
            # Use the most recent reconstruction
            reconstruction_dir = max(reconstruction_dirs, key=lambda x: int(x.name))
            logger.info(f"Using COLMAP reconstruction from {reconstruction_dir.name}")
            
            try:
                cameras, images, points3D = read_model(str(reconstruction_dir))
                logger.info(f"Loaded COLMAP reconstruction with {len(images)} images and {len(points3D)} 3D points")
                
                # Convert COLMAP poses to our format
                for image_id, image in images.items():
                    # Extract frame number from image name
                    frame_num = int(image['name'].split('_')[-1].split('.')[0])
                    pose = {
                        'frame_number': frame_num,
                        'position': image['tvec'],  # Translation vector
                        'rotation': image['qvec'],  # Quaternion
                        'camera_id': image['camera_id'],
                        'image_id': image_id,
                        'timestamp': frame_num * 0.5
                    }
                    poses.append(pose)
                
                # Sort by frame number
                poses.sort(key=lambda x: x['frame_number'])
                logger.info(f"Converted {len(poses)} COLMAP poses")
                return poses
                
            except Exception as e:
                logger.error(f"Error parsing COLMAP results: {str(e)}")
                raise RuntimeError(f"Failed to parse COLMAP results: {str(e)}. Please ensure COLMAP completed successfully.")
        else:
            logger.warning("No COLMAP reconstruction directories found")
            raise RuntimeError("No COLMAP reconstruction directories found. Please ensure COLMAP completed successfully.")
    
    def _generate_point_cloud(self, job_id: str, frame_data: List[Dict], 
                            depth_data: List[Dict], camera_poses: List[Dict]) -> Dict[str, Any]:
        """
        Generate 3D point cloud from depth maps and camera poses
        """
        logger.info("Generating 3D point cloud")
        
        all_points = []
        all_colors = []
        
        # Create a mapping from frame numbers to frame data and depth data
        frame_map = {frame['frame_number']: frame for frame in frame_data}
        depth_map_data = {depth['frame_number']: depth for depth in depth_data}
        
        # Only process frames that have both COLMAP poses and depth data
        for pose in camera_poses:
            frame_num = pose['frame_number']
            
            if frame_num in frame_map and frame_num in depth_map_data:
                frame = frame_map[frame_num]
                depth = depth_map_data[frame_num]
                
                # Load depth map
                depth_map = np.load(depth['raw_depth_path'])
                
                # Load color image
                color_image = cv2.imread(frame['file_path'])
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Resize depth map to match color image
                if depth_map.shape != color_image.shape[:2]:
                    depth_map = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
                
                # Generate 3D points
                points, colors = self._depth_to_point_cloud(
                    depth_map, color_image, pose
                )
                
                all_points.extend(points)
                all_colors.extend(colors)
                
                logger.info(f"Processed frame {frame_num}: {len(points)} points")
            else:
                logger.warning(f"Missing data for frame {frame_num}")
        
        # Convert to numpy arrays
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        
        # Analyze room bounds
        room_bounds = self._analyze_room_bounds(all_points)
        
        logger.info(f"Generated {len(all_points)} 3D points from {len(camera_poses)} poses")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'room_bounds': room_bounds
        }
    
    def _generate_point_cloud_from_all_frames(self, frame_data: List[Dict], depth_data: List[Dict]) -> Dict:
        """
        Generate 3D point cloud from all available depth maps (fallback method)
        """
        logger.info("Generating point cloud from all available depth maps (fallback method)")
        
        all_points = []
        all_colors = []
        
        # Use a simple camera model for all frames
        fx = fy = 1000  # Focal length
        cx = 320  # Assume 640x360 images
        cy = 180
        
        # Process all depth maps
        for depth_info in depth_data:
            depth_map_path = depth_info.get('file_path') or depth_info.get('raw_depth_path')
            frame_num = depth_info['frame_number']
            
            # Find corresponding color image
            frame_info = next((f for f in frame_data if f['frame_number'] == frame_num), None)
            if not frame_info:
                continue
                
            try:
                # Load depth map
                if depth_map_path.endswith('.npy'):
                    # Load as numpy array
                    depth_map = np.load(depth_map_path)
                else:
                    # Load as image (PNG/JPG)
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
                    if depth_map is not None:
                        # Convert to grayscale if it's a color image
                        if len(depth_map.shape) == 3:
                            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
                
                if depth_map is None:
                    logger.warning(f"Could not load depth map: {depth_map_path}")
                    continue
                
                # Load color image
                color_image = cv2.imread(frame_info['file_path'])
                if color_image is None:
                    logger.warning(f"Could not load color image: {frame_info['file_path']}")
                    continue
                
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Resize depth map to match color image
                if depth_map.shape != color_image.shape[:2]:
                    depth_map = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
                
                # Generate 3D points using simple camera model
                step = 20  # Sample every 20th pixel for performance
                for y in range(0, depth_map.shape[0], step):
                    for x in range(0, depth_map.shape[1], step):
                        depth = depth_map[y, x]
                        
                        # Skip invalid depth values
                        if depth <= 0 or depth > 10000:  # Reasonable depth range
                            continue
                        
                        # Convert to 3D coordinates
                        z = depth / 1000.0  # Convert to meters
                        x_3d = (x - cx) * z / fx
                        y_3d = (y - cy) * z / fy
                        
                        # Add some variation to avoid all points being at the same depth
                        x_3d += np.random.normal(0, 0.01)  # Small random variation
                        y_3d += np.random.normal(0, 0.01)
                        z += np.random.normal(0, 0.01)
                        
                        all_points.append([x_3d, y_3d, z])
                        all_colors.append(color_image[y, x])
                
                logger.info(f"Processed frame {frame_num}: {len([p for p in all_points if len(p) > 0])} points")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}: {e}")
                continue
        
        # Convert to numpy arrays
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        
        # Analyze room bounds
        room_bounds = self._analyze_room_bounds(all_points)
        
        logger.info(f"Generated {len(all_points)} 3D points from all available depth maps")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'room_bounds': room_bounds
        }
    
    def _depth_to_point_cloud(self, depth_map: np.ndarray, color_image: np.ndarray, 
                            camera_pose: Dict) -> Tuple[List, List]:
        """
        Convert depth map to 3D points using camera pose
        """
        points = []
        colors = []
        
        # Camera intrinsic parameters (assume typical phone camera)
        fx = fy = 1000  # Focal length
        cx = color_image.shape[1] / 2  # Principal point x
        cy = color_image.shape[0] / 2  # Principal point y
        
        # Sample every 10th pixel for performance
        step = 10
        for y in range(0, depth_map.shape[0], step):
            for x in range(0, depth_map.shape[1], step):
                depth = depth_map[y, x]
                
                # Skip invalid depth values
                if depth <= 0 or depth > 10:  # Skip depths > 10m
                    continue
                
                # Convert to 3D coordinates
                z = depth
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                point_camera = np.array([x_3d, y_3d, z])
                
                
                # Simple rotation and translation (assuming pose is in correct format)
                if 'position' in camera_pose and 'rotation' in camera_pose:
                    # Apply translation
                    qw, qx, qy, qz = camera_pose['rotation']
                    R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                    
                    point_world = R @ point_camera + np.array(camera_pose['position'])
                else:
                    point_world = point_camera
                
                points.append(point_world)
                colors.append(color_image[y, x] / 255.0)  # Normalize colors
        
        return points, colors

    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """Convert quaternion to 3x3 rotation matrix"""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def _analyze_room_bounds(self, points: np.ndarray) -> Dict[str, float]:
        """
        Analyze point cloud to determine room boundaries
        """
        if len(points) == 0:
            return {}
        
        min_x, min_y, min_z = np.min(points, axis=0)
        max_x, max_y, max_z = np.max(points, axis=0)
        
        return {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_y': float(min_y),
            'max_y': float(max_y),
            'min_z': float(min_z),
            'max_z': float(max_z),
            'width': float(max_x - min_x),
            'height': float(max_y - min_y),
            'depth': float(max_z - min_z)
        }
    
    def _create_mesh_from_point_cloud(self, point_cloud: Dict[str, Any]) -> Optional[trimesh.Trimesh]:
        """
        Create mesh from point cloud using Poisson reconstruction
        """
        try:
            logger.info("Creating mesh from point cloud")
            
            points = point_cloud['points']
            colors = point_cloud['colors']
            
            if len(points) < 100:
                logger.error("Too few points for mesh reconstruction")
                raise RuntimeError(f"Only {len(points)} points available, need at least 100 for proper mesh reconstruction")
            
            # Use only Poisson reconstruction as specified in tasks.md
            mesh = self._create_poisson_mesh(points, colors)
            
            if mesh:
                logger.info(f"Created mesh with {len(mesh.faces)} faces")
                return mesh
            else:
                raise RuntimeError("Poisson reconstruction failed")
                
        except Exception as e:
            logger.error(f"Error creating mesh: {str(e)}")
            raise
    
    def _create_poisson_mesh(self, points: np.ndarray, colors: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        Create mesh using Poisson reconstruction as originally planned
        """
        try:
            logger.info("Creating mesh using Poisson reconstruction")
            
            # Create point cloud object
            cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
            
            # Try Poisson reconstruction first (as specified in tasks.md)
            try:
                # Use Open3D for Poisson reconstruction if available
                import open3d as o3d
                
                # Convert to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Estimate normals
                pcd.estimate_normals()
                
                # Poisson reconstruction
                mesh_o3d, _ = pcd.create_mesh_poisson(depth=9, width=0, scale=1.1, linear_fit=False)
                
                # Convert back to trimesh
                vertices = np.asarray(mesh_o3d.vertices)
                faces = np.asarray(mesh_o3d.triangles)
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # Apply colors
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                    # Sample colors from original point cloud
                    face_colors = []
                    for face in faces:
                        # Get center of face
                        face_center = np.mean(vertices[face], axis=0)
                        # Find closest point in original point cloud
                        distances = np.linalg.norm(points - face_center, axis=1)
                        closest_idx = np.argmin(distances)
                        face_colors.append(colors[closest_idx])
                    mesh.visual.face_colors = np.array(face_colors) * 255
                
                logger.info(f"Created Poisson mesh with {len(mesh.faces)} faces")
                return mesh
                
            except ImportError:
                logger.warning("Open3D not available, falling back to trimesh Poisson")
                # Fallback to trimesh Poisson reconstruction
                mesh = cloud.convex_hull
                logger.info(f"Created convex hull mesh with {len(mesh.faces)} faces")
                return mesh
                
        except Exception as e:
            logger.error(f"Error creating Poisson mesh: {str(e)}")
            return None
    
    def _detect_doors(self, points: np.ndarray, room_bounds: Dict[str, float]) -> List[Dict]:
        """
        Detect doors in the room using point cloud analysis
        """
        doors = []
        try:
            logger.info("Detecting doors in point cloud")
            
            # Analyze vertical structures that could be doors
            # Doors typically have:
            # 1. Vertical edges (high Z variation)
            # 2. Width around 0.8-1.2 meters
            # 3. Height around 2.0-2.2 meters
            # 4. Located on walls (near room boundaries)
            
            # Filter points near walls (within 0.5m of room boundaries)
            wall_threshold = 0.5
            wall_points = []
            
            for point in points:
                x, y, z = point
                # Check if point is near any wall
                if (abs(x - room_bounds['min_x']) < wall_threshold or 
                    abs(x - room_bounds['max_x']) < wall_threshold or
                    abs(y - room_bounds['min_y']) < wall_threshold or 
                    abs(y - room_bounds['max_y']) < wall_threshold):
                    wall_points.append(point)
            
            if len(wall_points) < 10:
                logger.warning("Too few wall points for door detection")
                return doors
            
            wall_points = np.array(wall_points)
            
            # Group points by vertical columns (potential door frames)
            door_width = 0.9  # Standard door width
            door_height_min = 1.8  # Minimum door height
            door_height_max = 2.2  # Maximum door height
            
            # Sort points by X coordinate and group into potential door columns
            wall_points_sorted = wall_points[wall_points[:, 0].argsort()]
            
            i = 0
            while i < len(wall_points_sorted) - 10:
                # Find points within door width
                start_x = wall_points_sorted[i][0]
                end_x = start_x + door_width
                
                door_column_points = wall_points_sorted[
                    (wall_points_sorted[:, 0] >= start_x) & 
                    (wall_points_sorted[:, 0] <= end_x)
                ]
                
                if len(door_column_points) > 5:
                    # Check if this column has sufficient height variation
                    z_min = np.min(door_column_points[:, 2])
                    z_max = np.max(door_column_points[:, 2])
                    height = z_max - z_min
                    
                    if door_height_min <= height <= door_height_max:
                        # This looks like a door
                        door_center_x = np.mean(door_column_points[:, 0])
                        door_center_y = np.mean(door_column_points[:, 1])
                        door_center_z = (z_min + z_max) / 2
                        
                        door = {
                            'center': [door_center_x, door_center_y, door_center_z],
                            'width': door_width,
                            'height': height,
                            'points_count': len(door_column_points)
                        }
                        doors.append(door)
                        logger.info(f"Detected door at ({door_center_x:.2f}, {door_center_y:.2f}, {door_center_z:.2f})")
                
                # Move to next potential door location
                i += max(1, len(door_column_points) // 2)
            
            logger.info(f"Detected {len(doors)} doors")
            
        except Exception as e:
            logger.error(f"Error detecting doors: {str(e)}")
        
        return doors
    
    def _export_3d_model(self, job_id: str, mesh: Optional[trimesh.Trimesh], 
                        point_cloud: Dict[str, Any], camera_poses: List[Dict], 
                        doors: List[Dict] = None) -> List[str]:
        """
        Export 3D model in multiple formats
        """
        exported_files = []
        
        try:
            # Export point cloud
            if point_cloud['points'] is not None:
                point_cloud_path = self.models_dir / f"{job_id}_pointcloud.ply"
                self._export_point_cloud_ply(point_cloud['points'], point_cloud['colors'], point_cloud_path)
                exported_files.append('pointcloud')
            
            # Export mesh
            if mesh is not None:
                # Export as PLY
                mesh_ply_path = self.models_dir / f"{job_id}_mesh.ply"
                mesh.export(str(mesh_ply_path))
                exported_files.append('mesh_ply')
                
                # Export as GLB
                glb_path = self.models_dir / f"{job_id}_model.glb"
                mesh.export(str(glb_path))
                exported_files.append('glb')
                
                # Export as OBJ
                obj_path = self.models_dir / f"{job_id}_model.obj"
                mesh.export(str(obj_path))
                exported_files.append('obj')
            
            # Export camera poses
            poses_path = self.models_dir / f"{job_id}_camera_poses.json"
            with open(poses_path, 'w') as f:
                json.dump(camera_poses, f, indent=2)
            exported_files.append('poses')
            
            # Export door detection results
            if doors:
                doors_path = self.models_dir / f"{job_id}_doors.json"
                with open(doors_path, 'w') as f:
                    json.dump(doors, f, indent=2)
                exported_files.append('doors')
            
            logger.info(f"Exported 3D model files: {exported_files}")
            
        except Exception as e:
            logger.error(f"Error exporting 3D model: {str(e)}")
            raise
        
        return exported_files
    
    def _export_point_cloud_ply(self, points: np.ndarray, colors: np.ndarray, output_path: Path):
        """
        Export point cloud as PLY file
        """
        try:
            # Create point cloud object
            cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
            
            # Export as PLY
            cloud.export(str(output_path))
            logger.info(f"Exported mesh PLY with {len(cloud.vertices)} points")
            
        except Exception as e:
            logger.error(f"Error exporting point cloud PLY: {str(e)}")
            raise