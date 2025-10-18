# Kat - Frontend + Web Interface + Everything Else
Goal: Build the user-facing experience

### Core Tasks (Must complete)
1. Upload interface
    - create a simple HTML page with drag-and-drop video upload
    - loading indicator while backend processes
    - display status messages ("Processing frames...", "generating 3d model") [depends on how vicky is doing]
    - send video to backend via POST request
    - doesn't have to look pretty yet

2. 3D viewer with three.js
    - Create three.js scene with lighting and camera
    - load .glb or .obj model from backend
    - implement WASD + mouse controls for first-person navigation
    - add basic UI controls (reset camera, toggle grid)

3. Integrate + polish
    - connect upload -> processing -> viewer flow
    - handle error gracefully (video too large, processing failed)
    - make it look presentable (basic CSS, responsive layout)

### More goals:
- 2D floor plan toggle view
- Download button for 3D model
- mobile-responsive camera controls

# Vicky - Computer Vision + Backend
Goal: Turn video into 3D model

### Core Tasks (Must complete)
1. Video Processing Pipeline:
    - Flask/FastAPI endpoint to receive upload video
    - extract frames at 0.5s intervals using OpenCV
    - save frames to temporary directory
    - return processing status to frontend

2. Depth estimation 
    - integrate MiDaS (pretrained model)
    - run depth estimation on each frame
    - save depth maps as images or numpy arrays

3. 3D reconstruction
    - use COLMAP or openSFM to estimate camera poses from frames
    - generate point cloud from depth maps + camera poses
    - convert point cloud to mesh (poisson reconstructoin)
    - exports as .glb or .obj file.

4. Backend API
    - endpoint to check processing status
    - endpoint to download completed 3d model
    - basic error handling

### Stretch goals
- detect major objects (furniture) 
- identify walls/ceiling/floor planes
- improve mesh quality with smoothing.


# Sync Point 1 - Define API Contract
We both have to agree on the upload endpoint `POST /api/upload` (video file), status endpoint `GET /api/status/{job_id}` (returns processing state), download endpoint `GET /api/model/{job_id}` (returns .glb file). 

# Sync Point 2 - Test with mock data 
Kat uses a sample .glb file to build the viewer first (don't wait for vicky bc she might choke)
Vicky tests pipeline with a short test video, output a model file
Together we can integrate and test the full flow

# Sync Point 3 - End-to-End Demo
At this point we will test with a real room video, fix bugs together, and prepare demo presentation + brief each other on everything we did. 