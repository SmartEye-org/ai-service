"""
FastAPI Application - Smart Residential AI Service
MVP Version with WebSocket Support
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import cv2
import numpy as np
import asyncio
import logging

from app.config import settings
from app.services.detection import get_detection_service
from app.services.websocket_manager import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Detection(BaseModel):
    person_id: int
    bbox: List[int]
    confidence: float
    face_detected: bool
    timestamp: str


class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_persons: int
    timestamp: str
    ngsi_ld_entities: Optional[List[Dict]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize detection service on startup"""
    logger.info("==> Starting AI Service <==")
    try:
        # This will initialize the detection service
        service = get_detection_service()
        logger.info("==> AI Service ready!")
    except Exception as e:
        logger.error(f"==> Failed to start service: {e}")
        raise


# Health check
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now().isoformat()
    }


# Main detection endpoint
@app.post(f"{settings.API_V1_PREFIX}/detect", response_model=DetectionResponse)
async def detect_persons(
    image: UploadFile = File(...),
    camera_id: Optional[str] = "camera-01"
):
    """
    Detect persons in uploaded image
    
    Args:
        image: Image file (jpg, png)
        camera_id: Camera identifier
        
    Returns:
        Detection results with NGSI-LD entities
    """
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        service = get_detection_service()
        results = service.process_image(img)
        
        # Create NGSI-LD entities (simplified for MVP)
        entities = []
        for det in results['detections']:
            entity = {
                "id": f"urn:ngsi-ld:Person:{camera_id}-person-{det['person_id']}-{int(datetime.now().timestamp())}",
                "type": "Person",
                "@context": [
                    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
                ],
                "detectionConfidence": {
                    "type": "Property",
                    "value": det['confidence']
                },
                "faceDetected": {
                    "type": "Property",
                    "value": det['face_detected']
                },
                "bbox": {
                    "type": "Property",
                    "value": det['bbox']
                },
                "observedAt": {
                    "type": "Property",
                    "value": det['timestamp']
                }
            }
            entities.append(entity)
        
        return {
            **results,
            "ngsi_ld_entities": entities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ===== NEW: WebSocket Endpoints =====

@app.websocket("/ws/detect/{camera_id}")
async def websocket_detect(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for real-time detection streaming
    
    Client sends: binary frame data
    Server responds: JSON detection results
    """
    await manager.connect(websocket, camera_id)
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Process detection
            try:
                service = get_detection_service()
                result = await service.detect_from_bytes_async(data, camera_id)
                
                # Send result back
                await websocket.send_json({
                    "type": "detection",
                    "camera_id": camera_id,
                    "data": result,
                })
                
            except Exception as e:
                logger.error(f"Detection error for camera {camera_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "camera_id": camera_id,
                    "error": str(e),
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for camera {camera_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for monitoring all cameras
    Broadcasts detection results from all cameras
    """
    await manager.connect(websocket, "monitor")
    
    try:
        # Send initial stats
        await websocket.send_json({
            "type": "stats",
            "data": manager.get_stats(),
        })
        
        # Keep connection alive
        while True:
            # Wait for ping/pong
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.WS_HEARTBEAT_INTERVAL
                )
                
                if message == "ping":
                    await websocket.send_text("pong")
            
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")
        manager.disconnect(websocket)


# ===== NEW: REST Streaming Endpoints =====

@app.post("/stream/start")
async def start_stream(camera_id: str):
    """Start streaming for a camera (REST endpoint)"""
    connections = manager.get_camera_connections(camera_id)
    
    return {
        "status": "started" if connections > 0 else "no_clients",
        "camera_id": camera_id,
        "active_connections": connections,
        "message": f"Streaming started for camera {camera_id}" if connections > 0 
                   else "No active clients connected",
    }


@app.post("/stream/stop")
async def stop_stream(camera_id: str):
    """Stop streaming for a camera (REST endpoint)"""
    # Notify all clients
    await manager.send_to_camera(camera_id, {
        "type": "stream_stopped",
        "camera_id": camera_id,
        "message": "Stream stopped by server",
    })
    
    return {
        "status": "stopped",
        "camera_id": camera_id,
        "message": f"Streaming stopped for camera {camera_id}",
    }


@app.get("/stream/status")
async def stream_status():
    """Get status of all active streams"""
    stats = manager.get_stats()
    
    camera_details = {
        camera_id: manager.get_camera_connections(camera_id)
        for camera_id in manager.active_connections.keys()
    }
    
    return {
        "active_cameras": list(manager.active_connections.keys()),
        "total_connections": stats['active_connections'],
        "camera_connections": camera_details,
        "statistics": stats,
    }


# ===== NEW: Batch Detection Endpoint =====

@app.post("/detect/batch")
async def detect_batch_endpoint(
    images: List[UploadFile] = File(...),
    camera_ids: Optional[List[str]] = Form(None)
):
    """
    Batch detection endpoint
    Upload multiple images for batch processing
    """
    if not camera_ids:
        camera_ids = [f"camera-{i}" for i in range(len(images))]
    
    if len(images) != len(camera_ids):
        return {
            "error": "Number of images must match number of camera_ids",
            "images_count": len(images),
            "camera_ids_count": len(camera_ids),
        }
    
    # Read all images
    frames = []
    for img, cam_id in zip(images, camera_ids):
        content = await img.read()
        frames.append((content, cam_id))
    
    # Process batch
    service = get_detection_service()
    results = await service.detect_batch(frames)
    
    return {
        "batch_size": len(results),
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(f"{settings.API_V1_PREFIX}/stats")
async def get_stats():
    """Get service statistics"""
    ws_stats = manager.get_stats()
    
    return {
        "total_detections": 0,
        "uptime": "0h 0m",
        "model_loaded": True,
        "websocket": ws_stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
