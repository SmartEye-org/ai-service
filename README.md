# AI Service - Smart Residential Monitoring

## Overview
AI detection service cho hệ thống giám sát chung cư sử dụng YOLOv8 và MediaPipe.

## Features
- Person detection (YOLOv8)
- Face detection (MediaPipe)
- Action recognition
- NGSI-LD compliant output

## Tech Stack
- Python 3.9+
- FastAPI
- YOLOv8 (Ultralytics)
- MediaPipe
- OpenCV

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

### API Endpoints
```
POST /api/v1/detect
- Upload image
- Returns: detections + NGSI-LD entities
```

## Development
```bash
# Run demo script
python scripts/demo.py

# Run tests
pytest
```

## Team
- [Tên thành viên 1] - AI Lead
- [Tên thành viên 2] - Backend
- [Tên thành viên 3] - DevOps

## License
MIT
