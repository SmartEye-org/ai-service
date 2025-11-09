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
    
    # CORS - Đọc như string
    ALLOWED_ORIGINS: str = "*"
    
    @property
    def cors_origins(self) -> List[str]:
        """Convert ALLOWED_ORIGINS string to list"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

# Create settings instance
settings = Settings()
