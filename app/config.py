"""
Configuration Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "Smart Residential AI Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Models
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"
    
    # Thresholds
    PERSON_CONFIDENCE: float = 0.5
    FACE_CONFIDENCE: float = 0.5
    
    # CORS
    ALLOWED_ORIGINS: str = "*"
    
    # ===== NEW: Streaming configs =====
    STREAM_FPS: int = 5  # Process 5 frames per second
    BATCH_SIZE: int = 4   # Process 4 frames at once
    FRAME_SKIP: int = 2   # Process every 2nd frame
    MAX_CONCURRENT_STREAMS: int = 10
    DETECTION_TIMEOUT: float = 2.0  # seconds
    
    # WebSocket configs
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds
    WS_MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    @property
    def cors_origins(self) -> List[str]:
        """Convert ALLOWED_ORIGINS string to list"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

# Create settings instance
settings = Settings()
