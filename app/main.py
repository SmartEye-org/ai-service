"""
FastAPI Application - Smart Residential AI Service
MVP Version
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import cv2
import numpy as np

from app.config import settings
from app.services.detection import get_detection_service

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
    print("==> Starting AI Service... <==")
    try:
        # This will initialize the detection service
        service = get_detection_service()
        print("==> AI Service ready!")
    except Exception as e:
        print(f"==> Failed to start service: {e}")
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


@app.get(f"{settings.API_V1_PREFIX}/stats")
async def get_stats():
    """Get service statistics (MVP - mock data)"""
    return {
        "total_detections": 0,
        "uptime": "0h 0m",
        "model_loaded": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
