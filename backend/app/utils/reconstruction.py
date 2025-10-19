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
            
            # check if we have enough points
            if len(point_cloud['points']) < 100:
                logger.warning(f"Only {len(point_cloud['points'])} points generated, this may indicate issues with depth maps or camera poses")
                # Try to generate points from all frames, not just the ones with poses
                logger.info("Attempting to generate point cloud from all available depth maps...")
                point_cloud = self._generate_point_cloud_from_all_frames(job_id, frame_data, depth_data)
            
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
    
    # colmap camera pose estimation
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
    
    # actual function that estimates camera pose.
    def _estimate_poses_colmap(self, job_id: str, frame_data: List[Dict]) -> List[Dict]:
        """
        Estimate camera poses using COLMAP
        """
        logger.info("Using COLMAP for camera pose estimation")
        
        # create COLMAP workspace
        colmap_dir = self.models_dir / f"{job_id}_colmap"
        colmap_dir.mkdir(exist_ok=True)
        
        # prepare images for COLMAP
        images_dir = colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # use all frames for maximum coverage
        selected_frames = frame_data
        logger.info(f"Using {len(selected_frames)} frames for COLMAP (all frames)")
        
        # copy the frames over to the workspace
        for i, frame in enumerate(selected_frames):
            src_path = Path(frame['file_path'])
            dst_path = images_dir / f"frame_{i:04d}.jpg"
            shutil.copy2(src_path, dst_path)
        
        # run COLMAP feature extraction with optimized settings
        try:
            ### THIS IS THE PART THAT REQUIRES THE MOST TWEAKING
            subprocess.run([
                'colmap', 'feature_extractor',
                '--database_path', str(colmap_dir / 'database.db'),
                '--image_path', str(images_dir),
                '--ImageReader.single_camera', '1',
                '--SiftExtraction.max_num_features', '4096',  # many more features for better matching
                '--SiftExtraction.first_octave', '-1',  # start from higher resolution
                '--SiftExtraction.num_octaves', '4',  # more octaves
                '--SiftExtraction.upright', '0'  # allow rotation for better matching
            ], check=True, timeout=5400)  # 90 minutes timeout
            logger.info("COLMAP feature extraction completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"COLMAP feature extraction failed: {e}")
            raise RuntimeError(f"COLMAP feature extraction failed: {e}")
        
        # try to run COLMAP matching with optimized settings
        # rry sequential matching first (better for video sequences)
        try:
            logger.info("Trying sequential matching (optimized for video sequences)...")
            subprocess.run([
                'colmap', 'sequential_matcher',
                '--database_path', str(colmap_dir / 'database.db'),
                '--SiftMatching.max_ratio', '0.85',  # balanced matching for quality
                '--SiftMatching.max_distance', '0.85',  # balanced matching for quality
                '--SiftMatching.cross_check', '1',  # enable cross check for quality
                '--SequentialMatching.overlap', '20'  # match with 20 previous frames # TWEAKABLE
            ], check=True, timeout=5400)  # 90 minutes timeout
            logger.info("COLMAP sequential matching completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Sequential matching failed: {e}")
            logger.info("Falling back to exhaustive matching...") # this takes a LONG time
            try:
                subprocess.run([
                    'colmap', 'exhaustive_matcher',
                    '--database_path', str(colmap_dir / 'database.db'),
                '--SiftMatching.max_ratio', '0.85',  # balanced matching for quality
                '--SiftMatching.max_distance', '0.85',  # balanced matching for quality
                '--SiftMatching.cross_check', '1',  # enable cross check for quality
                ], check=True, timeout=5400)  # 90 minutes timeout
                logger.info("COLMAP exhaustive matching completed")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"COLMAP feature matching failed: {e}")
                raise RuntimeError(f"COLMAP feature matching failed: {e}")
        
        # run COLMAP reconstruction with optimized settings
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        try:
            ### MORE TWEAKING
            subprocess.run([
                'colmap', 'mapper',
                '--database_path', str(colmap_dir / 'database.db'),
                '--image_path', str(images_dir),
                '--output_path', str(sparse_dir),
                '--Mapper.init_min_num_inliers', '20',  # balanced threshold
                '--Mapper.abs_pose_max_error', '12',  # balanced error tolerance
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_principal_point', '0',
                '--Mapper.min_num_matches', '10',  # balanced minimum matches
                '--Mapper.max_num_models', '50',  # reasonable number of models
                '--Mapper.max_model_overlap', '20'  # balanced overlap requirement
            ], check=True, timeout=5400)  # 90 minutes timeout
            logger.info("COLMAP reconstruction completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"COLMAP reconstruction failed: {e}")
            raise RuntimeError(f"COLMAP reconstruction failed: {e}. Please ensure your video has sufficient overlap between frames and try again.")
        
        # export COLMAP results to text format for easier parsing
        text_dir = sparse_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        # export the COLMAP results to text format for easier parsing
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
        
        # parse COLMAP text results
        return self._parse_colmap_text_results(text_dir, len(frame_data))
    
    def _parse_colmap_text_results(self, text_dir: Path, num_frames: int) -> List[Dict]:
        """
        Parse COLMAP text output files
        """
        poses = []
        
        try:
            # read images.txt file
            images_file = text_dir / "images.txt"
            if not images_file.exists():
                raise RuntimeError("COLMAP images.txt file not found")
            
            with open(images_file, 'r') as f:
                lines = f.readlines()
                
            # skip header lines and empty lines
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            # COLMAP format: each image has two lines
            # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # Line 2: POINTS2D data (skip this line)
            i = 0
            while i < len(data_lines):
                line = data_lines[i]
                parts = line.split()
                
                if len(parts) >= 10:
                    # this is an image line
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    name = parts[9]
                    
                    # extract frame number from name
                    try:
                        frame_num = int(name.split('_')[-1].split('.')[0])
                    except:
                        frame_num = image_id
                    
                    pose = {
                        'frame_number': frame_num,
                        'position': [tx, ty, tz],
                        'rotation': [qw, qx, qy, qz],  # quaternion
                        'camera_id': camera_id,
                        'image_id': image_id,
                        'timestamp': frame_num * 0.5
                    }
                    poses.append(pose)
                    
                    # skip the next line (POINTS2D data)
                    i += 2
                else:
                    # skip malformed lines
                    i += 1
            
            # ssort by frame number
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
        
        # find the most recent reconstruction (highest numbered folder)
        reconstruction_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if reconstruction_dirs:
            # use the most recent reconstruction
            reconstruction_dir = max(reconstruction_dirs, key=lambda x: int(x.name))
            logger.info(f"Using COLMAP reconstruction from {reconstruction_dir.name}")
            
            try:
                cameras, images, points3D = read_model(str(reconstruction_dir))
                logger.info(f"Loaded COLMAP reconstruction with {len(images)} images and {len(points3D)} 3D points")
                
                # convert COLMAP poses to our format
                for image_id, image in images.items():
                    # extract frame number from image name
                    frame_num = int(image['name'].split('_')[-1].split('.')[0])
                    pose = {
                        'frame_number': frame_num,
                        'position': image['tvec'],  # translation vector
                        'rotation': image['qvec'],  # quaternion
                        'camera_id': image['camera_id'],
                        'image_id': image_id,
                        'timestamp': frame_num * 0.5
                    }
                    poses.append(pose)
                
                # sort by frame number
                poses.sort(key=lambda x: x['frame_number'])
                logger.info(f"Converted {len(poses)} COLMAP poses")
                return poses
                
            except Exception as e:
                logger.error(f"Error parsing COLMAP results: {str(e)}")
                raise RuntimeError(f"Failed to parse COLMAP results: {str(e)}. Please ensure COLMAP completed successfully.")
        else:
            logger.warning("No COLMAP reconstruction directories found")
            raise RuntimeError("No COLMAP reconstruction directories found. Please ensure COLMAP completed successfully.")
    
    # generates the 3D point cloud from the depth map and camera psopes
    def _generate_point_cloud(self, job_id: str, frame_data: List[Dict], 
                            depth_data: List[Dict], camera_poses: List[Dict]) -> Dict[str, Any]:
        """
        Generate 3D point cloud from depth maps and camera poses
        """
        logger.info("Generating 3D point cloud")
        
        all_points = []
        all_colors = []
        
        # load camera intrinsics from COLMAP
        colmap_dir = self.models_dir / f"{job_id}_colmap"
        camera_intrinsics = self._load_camera_intrinsics(colmap_dir)
        
        # create a mapping from frame numbers to frame data and depth data
        frame_map = {frame['frame_number']: frame for frame in frame_data}
        depth_map_data = {depth['frame_number']: depth for depth in depth_data}
        
        # only process frames that have both COLMAP poses and depth data
        for pose in camera_poses:
            frame_num = pose['frame_number']
            
            if frame_num in frame_map and frame_num in depth_map_data:
                frame = frame_map[frame_num]
                depth = depth_map_data[frame_num]
                
                # load depth map
                depth_map = np.load(depth['raw_depth_path'])
                
                # load color image
                color_image = cv2.imread(frame['file_path'])
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # resize depth map to match color image
                if depth_map.shape != color_image.shape[:2]:
                    depth_map = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
                
                # generate 3D points
                points, colors = self._depth_to_point_cloud(
                    depth_map, color_image, pose, camera_intrinsics
                )
                
                all_points.extend(points)
                all_colors.extend(colors)
                
                logger.info(f"Processed frame {frame_num}: {len(points)} points")
            else:
                logger.warning(f"Missing data for frame {frame_num}")
        
        # convert to numpy arrays
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        
        # analyze room bounds
        room_bounds = self._analyze_room_bounds(all_points)
        
        logger.info(f"Generated {len(all_points)} 3D points from {len(camera_poses)} poses")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'room_bounds': room_bounds
        }
    
    def _generate_point_cloud_from_all_frames(self, job_id: str, frame_data: List[Dict], depth_data: List[Dict]) -> Dict:
        """
        Generate 3D point cloud from all available depth maps (fallback method)
        """
        logger.info("Generating point cloud from all available depth maps (fallback method)")
        
        all_points = []
        all_colors = []
        
        # load camera intrinsics from COLMAP if available
        colmap_dir = self.models_dir / f"{job_id}_colmap"
        camera_intrinsics = self._load_camera_intrinsics(colmap_dir)
        
        # use loaded intrinsics or fallback to defaults
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # process all depth maps
        for depth_info in depth_data:
            depth_map_path = depth_info.get('file_path') or depth_info.get('raw_depth_path')
            frame_num = depth_info['frame_number']
            
            # find corresponding color image
            frame_info = next((f for f in frame_data if f['frame_number'] == frame_num), None)
            if not frame_info:
                continue
                
            try:
                # load depth map
                if depth_map_path.endswith('.npy'):
                    # load as numpy array
                    depth_map = np.load(depth_map_path)
                else:
                    # load as image (PNG/JPG)
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
                    if depth_map is not None:
                        # convert to grayscale if it's a color image
                        if len(depth_map.shape) == 3:
                            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
                
                if depth_map is None:
                    logger.warning(f"Could not load depth map: {depth_map_path}")
                    continue
                
                # load color image
                color_image = cv2.imread(frame_info['file_path'])
                if color_image is None:
                    logger.warning(f"Could not load color image: {frame_info['file_path']}")
                    continue
                
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # resize depth map to match color image
                if depth_map.shape != color_image.shape[:2]:
                    depth_map = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
                
                # generate 3D points using simple camera model
                step = 10  # sample every 10th pixel for better quality
                for y in range(0, depth_map.shape[0], step):
                    for x in range(0, depth_map.shape[1], step):
                        depth = depth_map[y, x]
                        
                        # skip invalid depth values
                        # MiDaS produces relative depth values, not absolute meters
                        if depth <= 0 or depth > 10000:  # Skip invalid depths
                            continue
                        
                        # convert to 3D coordinates
                        # MiDaS depth values are relative and need to be scaled
                        # scale depth to reasonable range (0.1m to 10m)
                        z = (depth / 1000.0) * 10.0  # Scale to meters
                        x_3d = (x - cx) * z / fx
                        y_3d = (y - cy) * z / fy
                        
                        # add some variation to avoid all points being at the same depth
                        x_3d += np.random.normal(0, 0.01)  # Small random variation
                        y_3d += np.random.normal(0, 0.01)
                        z += np.random.normal(0, 0.01)
                        
                        all_points.append([x_3d, y_3d, z])
                        all_colors.append(color_image[y, x])
                
                logger.info(f"Processed frame {frame_num}: {len([p for p in all_points if len(p) > 0])} points")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}: {e}")
                continue
        
        # convert to numpy arrays
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        
        # analyze room bounds
        room_bounds = self._analyze_room_bounds(all_points)
        
        logger.info(f"Generated {len(all_points)} 3D points from all available depth maps")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'room_bounds': room_bounds
        }
    
    def _depth_to_point_cloud(self, depth_map: np.ndarray, color_image: np.ndarray, 
                            camera_pose: Dict, camera_intrinsics: Dict = None) -> Tuple[List, List]:
        """
        Convert depth map to 3D points using camera pose
        """
        points = []
        colors = []
        
        # use provided camera intrinsics or fallback to defaults
        if camera_intrinsics:
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']
        else:
            # fallback to generic camera model
            fx = fy = 1000  # focal length
            cx = color_image.shape[1] / 2  # principal point x
            cy = color_image.shape[0] / 2  # principal point y
        
        # sample every 5th pixel for better quality
        step = 5
        for y in range(0, depth_map.shape[0], step):
            for x in range(0, depth_map.shape[1], step):
                depth = depth_map[y, x]
                
                # skip invalid depth values
                # MiDaS produces relative depth values, not absolute meters
                if depth <= 0 or depth > 10000:  # Skip invalid depths
                    continue
                
                # convert to 3D coordinates
                # MiDaS depth values are relative and need to be scaled
                # scale depth to reasonable range (0.1m to 10m)
                z = (depth / 1000.0) * 10.0  # scale to meters
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                point_camera = np.array([x_3d, y_3d, z])
                
                
                # simple rotation and translation (assuming pose is in correct format)
                if 'position' in camera_pose and 'rotation' in camera_pose:
                    # apply translation
                    qw, qx, qy, qz = camera_pose['rotation']
                    R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                    
                    point_world = R @ point_camera + np.array(camera_pose['position'])
                else:
                    point_world = point_camera
                
                points.append(point_world)
                colors.append(color_image[y, x] / 255.0)  # Normalize colors
        
        return points, colors
    
    def _load_camera_intrinsics(self, colmap_dir: Path, image_name: str = None) -> Dict:
        """
        Load camera intrinsics from COLMAP cameras.bin or images.txt
        """
        try:
            # try to load from cameras.bin first
            cameras_bin = colmap_dir / "sparse" / "cameras.bin"
            if cameras_bin.exists():
                # for now, return default intrinsics: would need COLMAP Python bindings for full implementation
                logger.info("Found cameras.bin but using default intrinsics (COLMAP Python bindings not available)")
                return {
                    'fx': 1000.0,
                    'fy': 1000.0,
                    'cx': 1080.0 / 2,  # correct for 1080x1920 images, i think
                    'cy': 1920.0 / 2,
                    'width': 1080,
                    'height': 1920
                }
            
            # try to load from images.txt
            images_txt = colmap_dir / "sparse" / "images.txt"
            if images_txt.exists():
                # for now, return default intrinsics, would need to parse text file for full implemettation
                logger.info("Found images.txt but using default intrinsics (text parsing not implemented)")
                return {
                    'fx': 1000.0,
                    'fy': 1000.0,
                    'cx': 1080.0 / 2,  # correct for 1080x1920 images, i think
                    'cy': 1920.0 / 2,
                    'width': 1080,
                    'height': 1920
                }
                
        except Exception as e:
            logger.warning(f"Could not load camera intrinsics: {e}")
        
        # fallback to default intrinsics
        logger.warning("Using default camera intrinsics - this may cause poor reconstruction quality")
        return {
            'fx': 1000.0,
            'fy': 1000.0,
            'cx': 1080.0 / 2,  # correct for 1080x1920 images, I think
            'cy': 1920.0 / 2,
            'width': 1080,
            'height': 1920
        }

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
        Create high-quality mesh from point cloud using Open3D Poisson surface reconstruction
        """
        try:
            logger.info("Creating high-quality mesh from point cloud using Open3D")
            
            points = point_cloud['points']
            colors = point_cloud['colors']
            
            if len(points) < 100:
                logger.error("Too few points for mesh reconstruction")
                raise RuntimeError(f"Only {len(points)} points available, need at least 100 for proper mesh reconstruction")
            
            # use Open3D Poisson reconstruction for high-quality results
            mesh = self._create_poisson_mesh(points, colors)
            
            if mesh is None:
                raise RuntimeError("Open3D Poisson reconstruction failed")
            
            logger.info(f"Successfully created high-quality mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")
            return mesh
                
        except Exception as e:
            logger.error(f"Error creating mesh: {str(e)}")
            raise RuntimeError(f"Mesh creation failed: {str(e)}")
    
    def _create_poisson_mesh(self, points: np.ndarray, colors: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        Create mesh using Open3D Ball Pivoting Algorithm (BPA) - more robust than Poisson
        """
        try:
            logger.info("Creating mesh using Open3D Ball Pivoting Algorithm")
            
            # import Open3D, didn't work before hopefuly now
            import open3d as o3d
            
            # validate input
            if len(points) < 100:
                raise RuntimeError(f"Too few points for mesh reconstruction: {len(points)} (need at least 100)")
            
            # convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # remove outliers for better reconstruction
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            logger.info(f"Point cloud after outlier removal: {len(pcd.points)} points")
            
            # estimate normals
            pcd.estimate_normals()
            
            # Ball Pivoting Algorithm - better than Poisson supposedly>/
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2])
            )
            
            # clean mesh
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()
            
            # Aaply room structure detection and enhancement
            mesh_o3d = self._enhance_room_structure(mesh_o3d, points)
            
            # convert to trimesh
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            
            if len(vertices) == 0 or len(faces) == 0:
                raise RuntimeError("Ball Pivoting reconstruction produced empty mesh")
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # apply room-appropriate colors
            self._apply_room_colors(mesh, points, colors)
            
            logger.info(f"Created BPA mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")
            return mesh
                
        except ImportError as e:
            logger.error(f"Open3D is required for mesh reconstruction but not available: {e}")
            raise RuntimeError("Open3D is required for mesh reconstruction. Please install it with: conda install -c conda-forge open3d")
        except Exception as e:
            logger.error(f"Error creating BPA mesh: {str(e)}")
            raise RuntimeError(f"Ball Pivoting mesh reconstruction failed: {str(e)}")
    
    def _enhance_room_structure(self, mesh_o3d, points: np.ndarray):
        """
        Enhance mesh to look more like recognizable room structures
        """
        try:
            logger.info("Enhancing room structure detection")
            
            # get mesh vertices and faces
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            
            if len(vertices) == 0 or len(faces) == 0:
                return mesh_o3d
            
            # detect room boundaries from point cloud
            room_bounds = self._detect_room_boundaries(points)
            
            # identify structural elements
            floor_faces, wall_faces, ceiling_faces = self._identify_structural_elements(
                vertices, faces, room_bounds
            )
            
            # apply structural enhancements
            mesh_o3d = self._apply_structural_enhancements(
                mesh_o3d, floor_faces, wall_faces, ceiling_faces, room_bounds
            )
            
            logger.info(f"Enhanced room structure: {len(floor_faces)} floor, {len(wall_faces)} wall, {len(ceiling_faces)} ceiling faces")
            return mesh_o3d
            
        except Exception as e:
            logger.warning(f"Room structure enhancement failed: {e}, using original mesh")
            return mesh_o3d
    
    def _detect_room_boundaries(self, points: np.ndarray) -> Dict[str, float]:
        """
        Detect room boundaries (floor, ceiling, walls) from point cloud
        """
        try:
            # calculate room bounds
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            # identify floor (lowest Y), ceiling (highest Y)
            floor_level = min_coords[1]
            ceiling_level = max_coords[1]
            
            # identify wall boundaries (X and Z extents)
            wall_bounds = {
                'min_x': min_coords[0],
                'max_x': max_coords[0],
                'min_z': min_coords[2],
                'max_z': max_coords[2]
            }
            
            room_bounds = {
                'floor_level': floor_level,
                'ceiling_level': ceiling_level,
                'room_height': ceiling_level - floor_level,
                **wall_bounds
            }
            
            logger.info(f"Detected room bounds: {room_bounds}")
            return room_bounds
            
        except Exception as e:
            logger.warning(f"Room boundary detection failed: {e}")
            return {}
    
    def _identify_structural_elements(self, vertices: np.ndarray, faces: np.ndarray, room_bounds: Dict[str, float]) -> Tuple[List, List, List]:
        """
        Identify which faces belong to floor, walls, and ceiling
        """
        try:
            floor_faces = []
            wall_faces = []
            ceiling_faces = []
            
            if not room_bounds:
                return floor_faces, wall_faces, ceiling_faces
            
            floor_level = room_bounds.get('floor_level', 0)
            ceiling_level = room_bounds.get('ceiling_level', 0)
            room_height = room_bounds.get('room_height', 0)
            
            # thresholds for classification
            floor_threshold = floor_level + room_height * 0.1  # Bottom 10%
            ceiling_threshold = ceiling_level - room_height * 0.1  # Top 10%
            wall_threshold = room_height * 0.2  # Middle 60% for walls
            
            for i, face in enumerate(faces):
                # get face center
                face_center = np.mean(vertices[face], axis=0)
                y_coord = face_center[1]
                
                # classify based on Y position
                if y_coord <= floor_threshold:
                    floor_faces.append(i)
                elif y_coord >= ceiling_threshold:
                    ceiling_faces.append(i)
                elif floor_threshold < y_coord < ceiling_threshold:
                    wall_faces.append(i)
            
            return floor_faces, wall_faces, ceiling_faces
            
        except Exception as e:
            logger.warning(f"Structural element identification failed: {e}")
            return [], [], []
    
    def _apply_structural_enhancements(self, mesh_o3d, floor_faces: List, wall_faces: List, ceiling_faces: List, room_bounds: Dict[str, float]):
        """
        Apply enhancements to make structural elements more recognizable
        """
        try:
            # light smoothing for better appearance (reduced iterations for speed)
            mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=1)
            
            logger.info("Applied structural enhancements to mesh")
            return mesh_o3d
            
        except Exception as e:
            logger.warning(f"Structural enhancement failed: {e}")
            return mesh_o3d
    
    def _apply_room_colors(self, mesh: trimesh.Trimesh, points: np.ndarray, colors: np.ndarray):
        """
        Apply room-appropriate colors to mesh faces (optimized for speed)
        """
        try:
            # room-appropriate colors
            floor_color = [139, 69, 19, 255]    # brown for floor
            wall_color = [255, 255, 255, 255]   # white for walls
            ceiling_color = [240, 240, 240, 255] # light gray for ceiling
            
            # pre-calculate room bounds for efficiency
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            room_height = max_y - min_y
            floor_threshold = min_y + room_height * 0.1
            ceiling_threshold = max_y - room_height * 0.1
            
            # vectorized color assignment for speed
            face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
            y_coords = face_centers[:, 1]
            
            # create color array
            face_colors = np.full((len(mesh.faces), 4), wall_color, dtype=np.uint8)
            face_colors[y_coords <= floor_threshold] = floor_color
            face_colors[y_coords >= ceiling_threshold] = ceiling_color
            
            # apply colors to mesh
            mesh.visual.face_colors = face_colors
            logger.info("Applied room-appropriate colors to mesh faces")
            
        except Exception as e:
            logger.warning(f"Error applying room colors to mesh: {e}, using default colors")
            mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray
    
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
            
            # filter points near walls (within 0.5m of room boundaries)
            wall_threshold = 0.5
            wall_points = []
            
            for point in points:
                x, y, z = point
                # check if point is near any wall
                if (abs(x - room_bounds['min_x']) < wall_threshold or 
                    abs(x - room_bounds['max_x']) < wall_threshold or
                    abs(y - room_bounds['min_y']) < wall_threshold or 
                    abs(y - room_bounds['max_y']) < wall_threshold):
                    wall_points.append(point)
            
            if len(wall_points) < 10:
                logger.warning("Too few wall points for door detection")
                return doors
            
            wall_points = np.array(wall_points)
            
            # group points by vertical columns (potential door frames)
            door_width = 0.9  # standard door width
            door_height_min = 1.8  # minimum door height
            door_height_max = 2.2  # maximum door height
            
            # sort points by X coordinate and group into potential door columns
            wall_points_sorted = wall_points[wall_points[:, 0].argsort()]
            
            i = 0
            while i < len(wall_points_sorted) - 10:
                # find points within door width
                start_x = wall_points_sorted[i][0]
                end_x = start_x + door_width
                
                door_column_points = wall_points_sorted[
                    (wall_points_sorted[:, 0] >= start_x) & 
                    (wall_points_sorted[:, 0] <= end_x)
                ]
                
                if len(door_column_points) > 5:
                    # check if this column has sufficient height variation
                    z_min = np.min(door_column_points[:, 2])
                    z_max = np.max(door_column_points[:, 2])
                    height = z_max - z_min
                    
                    if door_height_min <= height <= door_height_max:
                        # this looks like a door
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
                
                # move to next potential door location
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
            # export point cloud
            if point_cloud['points'] is not None:
                point_cloud_path = self.models_dir / f"{job_id}_pointcloud.ply"
                self._export_point_cloud_ply(point_cloud['points'], point_cloud['colors'], point_cloud_path)
                exported_files.append('pointcloud')
            
            # export mesh
            if mesh is not None:
                # export as PLY
                mesh_ply_path = self.models_dir / f"{job_id}_mesh.ply"
                mesh.export(str(mesh_ply_path))
                exported_files.append('mesh_ply')
                
                # export as GLB
                glb_path = self.models_dir / f"{job_id}_model.glb"
                mesh.export(str(glb_path))
                exported_files.append('glb')
                
                # export as OBJ
                obj_path = self.models_dir / f"{job_id}_model.obj"
                mesh.export(str(obj_path))
                exported_files.append('obj')
            
            # export camera poses
            poses_path = self.models_dir / f"{job_id}_camera_poses.json"
            with open(poses_path, 'w') as f:
                json.dump(camera_poses, f, indent=2)
            exported_files.append('poses')
            
            # export door detection results
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
            # create point cloud object
            cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
            
            # export as PLY
            cloud.export(str(output_path))
            logger.info(f"Exported mesh PLY with {len(cloud.vertices)} points")
            
        except Exception as e:
            logger.error(f"Error exporting point cloud PLY: {str(e)}")
            raise