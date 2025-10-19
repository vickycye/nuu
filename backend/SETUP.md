# Nuu 3D Room Scanner - Backend Setup

## Quick Start

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install MiDaS for depth estimation:**
   ```bash
   python install_midas.py
   ```

3. **Add sample video:**
   - Place a room video in `sample_input/` directory
   - Supported formats: .mp4, .avi, .mov, .mkv
   - Recommended: 10-30 seconds, 720p-1080p, good lighting

4. **Test the pipeline:**
   ```bash
   # Test video processing
   python test_video_processing.py
   
   # Test depth estimation
   python test_depth_estimation.py
   
   # Test 3D reconstruction
   python test_3d_reconstruction.py
   ```

5. **Run the backend:**
   ```bash
   python run.py
   ```

6. **Test the API:**
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health
   - List samples: http://localhost:8000/api/samples

## Testing Video Processing

1. **Test with sample video:**
   ```bash
   python test_video_processing.py
   ```

2. **Upload sample via API:**
   ```bash
   curl -X POST http://localhost:8000/api/upload-sample/your_video.mp4
   ```

3. **Check processing status:**
   ```bash
   curl http://localhost:8000/api/status/{job_id}
   ```

## API Endpoints

- `POST /api/upload` - Upload video file
- `POST /api/upload-sample/{filename}` - Process sample video
- `GET /api/status/{job_id}` - Check processing status
- `GET /api/model/{job_id}` - Download 3D model (.glb, .obj, .ply)
- `GET /api/model/{job_id}/info` - Get model information
- `GET /api/model/{job_id}/poses` - Download camera poses
- `GET /api/samples` - List available sample videos

## Directory Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models
│   ├── routers/             # API endpoints
│   ├── utils/               # Processing utilities
│   └── core/                # Configuration
├── sample_input/            # Test videos
├── temp/                    # Temporary files
│   ├── uploads/            # Uploaded videos
│   ├── frames/             # Extracted frames
│   ├── depth_maps/         # Depth estimation results
│   └── models/             # Generated 3D models
├── requirements.txt
├── run.py                  # Start server
└── test_video_processing.py # Test script
```

## Next Steps

1. Add your room video to `sample_input/`
2. Run `python test_video_processing.py` to test frame extraction
3. Run `python test_depth_estimation.py` to test depth estimation
4. Start the server with `python run.py`
5. Test the API endpoints
6. Ready for frontend integration!
