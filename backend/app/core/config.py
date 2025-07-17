# Backend Core Configuration
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # App Info
    app_name: str = "ASLense API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./aslens.db"
    
    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]
    
    # File Storage
    upload_dir: str = "uploads"
    thumbnail_dir: str = "public/thumbnails"
    video_dir: str = "data/videos"
    
    # Path objects for file operations
    @property
    def UPLOAD_DIR(self) -> Path:
        return Path(self.upload_dir)
    
    @property
    def THUMBNAIL_DIR(self) -> Path:
        return Path(self.thumbnail_dir)
    
    @property
    def VIDEO_DIR(self) -> Path:
        return Path(self.video_dir)
    
    # AI Models
    model_path: str = "models"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
