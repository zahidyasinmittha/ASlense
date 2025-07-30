# settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    """Application settings and configuration."""

    # App Info
    app_name: str
    app_version: str
    debug: bool

    # Database
    database_url: str

    # Email Configuration
    EMAIL_HOST_USER: str
    EMAIL_HOST_PASSWORD: str
    EMAIL_PORT: int
    EMAIL_HOST: str
    EMAIL_SEND: bool
    EMAIL_USE_TLS: bool
    EMAIL_USE_SSL: bool
    CONTACT_RECIPIENT_EMAIL: str

    # File Storage
    upload_dir: str
    thumbnail_dir: str
    video_dir: str

    # AI Models
    model_path: str

    # GROQ API Keys
    GROQ_API_KEY_MAIN: str
    GROQ_API_KEY_COMPOSER: str
    GROQ_API_KEY_VALIDATOR: str

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

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"

# instantiate once for app-wide use
settings = Settings()
