# Sample Input Videos

This directory contains sample room videos for testing the Nuu 3D room scanner.

## How to use sample videos:

1. Place your test room videos in this directory
2. Supported formats: .mp4, .avi, .mov, .mkv
3. Use the API endpoint: `POST /api/upload-sample/{filename}` to process a sample video
4. List available samples: `GET /api/samples`

## Recommended video characteristics:
- Duration: 10-30 seconds
- Resolution: 720p or 1080p
- Frame rate: 24-30 fps
- Content: Slow, steady movement around a room
- Lighting: Good, even lighting
- File size: Under 100MB

## Example usage:
```bash
# List available samples
curl http://localhost:8000/api/samples

# Process a sample video
curl -X POST http://localhost:8000/api/upload-sample/room_tour.mp4
```

## Adding your own sample videos:
1. Record a short video of a room (10-30 seconds)
2. Move slowly and steadily around the room
3. Ensure good lighting
4. Save the video in this directory
5. Use the upload-sample endpoint to test

