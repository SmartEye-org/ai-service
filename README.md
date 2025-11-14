# Smart eyes AI Service

AI service for person detection and face recognition using YOLOv8 and MediaPipe with real-time WebSocket streaming support.

## Features

- ✅ Person detection with YOLOv8
- ✅ Face detection with MediaPipe  
- ✅ NGSI-LD compliant output
- ✅ RESTful API with FastAPI
- ✅ **Real-time WebSocket streaming**
- ✅ **Batch processing support**
- ✅ Async detection processing

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### REST Endpoints

#### Health Check
```bash
GET /health
```

#### Single Image Detection
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "image=@test.jpg" \
  -F "camera_id=camera-01"
```

#### Batch Detection
```bash
curl -X POST http://localhost:8000/detect/batch \
  -F "images=@frame1.jpg" \
  -F "images=@frame2.jpg" \
  -F "camera_ids=camera-01" \
  -F "camera_ids=camera-02"
```

#### Stream Control

**Start Stream**
```bash
POST /stream/start?camera_id=camera-01
```

**Stop Stream**
```bash
POST /stream/stop?camera_id=camera-01
```

**Stream Status**
```bash
GET /stream/status
```

Response:
```json
{
  "active_cameras": ["camera-01", "camera-02"],
  "total_connections": 5,
  "camera_connections": {
    "camera-01": 2,
    "camera-02": 3
  },
  "statistics": {
    "total_connections": 10,
    "active_connections": 5,
    "messages_sent": 1234,
    "errors": 0,
    "cameras_monitored": 2
  }
}
```

### WebSocket Endpoints

#### Real-time Detection Streaming

Connect to camera-specific stream:
```bash
wscat -c ws://localhost:8000/ws/detect/camera-01
```

Then send binary frame data. Server responds with:
```json
{
  "type": "detection",
  "camera_id": "camera-01",
  "data": {
    "detections": [
      {
        "person_id": 0,
        "bbox": [100, 200, 300, 400],
        "confidence": 0.95,
        "face_detected": true,
        "timestamp": "2025-11-13T10:30:00"
      }
    ],
    "total_persons": 1,
    "camera_id": "camera-01",
    "timestamp": "2025-11-13T10:30:00"
  }
}
```

#### Monitor All Cameras

```bash
wscat -c ws://localhost:8000/ws/monitor
```

Send "ping" to receive "pong". Server sends heartbeat every 30 seconds.

## Configuration

Create `.env` file (see `.env.example`):

```env
# Application
APP_NAME="Smart Residential AI Service"
VERSION="1.0.0"
DEBUG=False

# Streaming configs
STREAM_FPS=5                    # Process 5 frames per second
BATCH_SIZE=4                    # Process 4 frames at once
FRAME_SKIP=2                    # Process every 2nd frame
MAX_CONCURRENT_STREAMS=10       # Max concurrent streams
DETECTION_TIMEOUT=2.0           # Detection timeout in seconds

# WebSocket configs
WS_HEARTBEAT_INTERVAL=30        # Heartbeat interval in seconds
WS_MAX_MESSAGE_SIZE=10485760    # Max message size (10MB)
```

## Testing

### Test REST API
```bash
python scripts/test_api.py
```

### Test WebSocket
```bash
python scripts/test_websocket.py
```

**Note:** Add test image to `tests/test_image.jpg` before running tests.

## Architecture

```
ai-service/
├── app/
│   ├── main.py                 # FastAPI app with WebSocket support
│   ├── config.py               # Configuration settings
│   └── services/
│       ├── detection.py        # Detection service (async + batch)
│       └── websocket_manager.py # WebSocket connection manager
├── models/
│   └── yolov8n.pt             # YOLO model (auto-downloads)
├── scripts/
│   ├── test_api.py            # REST API tests
│   └── test_websocket.py      # WebSocket tests
└── requirements.txt
```

## Development

### Run with auto-reload
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run tests
```bash
pytest tests/
```

### Format code
```bash
black app/
isort app/
```

## Performance

- **Async processing**: Non-blocking detection with asyncio
- **Batch processing**: Process multiple frames simultaneously
- **Frame skipping**: Reduce load by processing every Nth frame
- **WebSocket streaming**: Real-time communication with low latency

## License

MIT License

## Support

For issues and questions, please create an issue on GitHub.
