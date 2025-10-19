# Hardcoded Test Model Setup

## Overview
The system is now configured to return a hardcoded test GLB model after video processing, instead of actually performing 3D reconstruction. This allows for demo/testing purposes while the reconstruction pipeline is being developed.

## Setup Instructions

### 1. Place Your Test Model
Put your `test_reconstruct_model.glb` file in:
```
frontend/public/models/test_reconstruct_model.glb
```

### 2. How It Works

#### Backend Processing Flow:
1. **Video Upload** → User uploads a video
2. **Validation** → Video is validated (5% progress)
3. **Frame Extraction** → Frames are extracted from the video (20% progress)
4. **Depth Estimation** → Depth maps are generated (50% progress)
5. **3D Reconstruction** → **HARDCODED**: 2-4 second delay, then returns test model (80% progress)
6. **Complete** → Frontend receives model URL and displays it (100% progress)

#### Key Changes Made:

**Backend (`backend/app/utils/processing_pipeline.py`):**
- Added `HARDCODED_MODEL_URL = "/models/test_reconstruct_model.glb"`
- Modified Step 4 (3D Reconstruction) to:
  - Add a random 2-4 second delay with `await asyncio.sleep(delay)`
  - Skip actual reconstruction
  - Return hardcoded model URL in the response

**Backend (`backend/app/routers/status.py`):**
- Added `metadata` field to the status response
- Frontend can now access `metadata.model_paths.glb` to get the model URL

**Frontend (`frontend/src/App.jsx`):**
- Modified `pollJobStatus` to extract model URL from `data.metadata?.model_paths?.glb`
- Model is automatically loaded into Three.js viewer when processing completes

### 3. User Experience

When a user uploads a video:
1. ✅ "Uploading video..." (instant)
2. ✅ "Extracting frames from video..." (~2-5 seconds, depending on video)
3. ✅ "Estimating depth information..." (~3-10 seconds, depending on frame count)
4. ✅ "Reconstructing 3D model..." (2-4 seconds **hardcoded delay**)
5. ✅ "3D model ready!" → Three.js viewer displays `test_reconstruct_model.glb`

### 4. Testing

1. Place your GLB file in `frontend/public/models/test_reconstruct_model.glb`
2. Restart the backend: `cd backend && python3.11 run.py`
3. Upload any video through the frontend
4. After processing, your test model will be displayed

### 5. Reverting to Real Reconstruction

When ready to use actual reconstruction:
1. Open `backend/app/utils/processing_pipeline.py`
2. Replace the hardcoded Step 4 with the original reconstruction call
3. Remove the `HARDCODED_MODEL_URL` constant
4. Remove `import random`

## File Locations

- **Test Model**: `frontend/public/models/test_reconstruct_model.glb`
- **Backend Logic**: `backend/app/utils/processing_pipeline.py` (lines 64-83)
- **Status Response**: `backend/app/routers/status.py` (line 21)
- **Frontend Polling**: `frontend/src/App.jsx` (lines 50-57)

## Notes

- The hardcoded delay (2-4 seconds) makes the demo feel more realistic
- All other processing steps (frame extraction, depth estimation) still run normally
- The test model will be served from the frontend's public directory, not from backend
- Model URL uses absolute path `/models/test_reconstruct_model.glb` which Vite serves automatically

