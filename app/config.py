from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Smart Residential AI Service"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # Model paths
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"
    
    # Detection thresholds
    PERSON_CONFIDENCE: float = 0.5
    FACE_CONFIDENCE: float = 0.5
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()