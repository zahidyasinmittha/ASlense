# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)  # "admin" or "user"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    profile_picture = Column(String)
    
    # Relationships
    predictions = relationship("PredictionHistory", back_populates="user")
    progress = relationship("UserProgress", back_populates="user")

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_signs_practiced = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    total_predictions = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    practice_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_practice_date = Column(DateTime)
    total_practice_time = Column(Integer, default=0)  # in seconds
    level = Column(String, default="Beginner")
    experience_points = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="progress")

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    target_word = Column(String, nullable=False)
    predicted_words = Column(Text)  # JSON string of top predictions
    is_correct = Column(Boolean, nullable=False)
    confidence_score = Column(Float)
    model_used = Column(String, nullable=False)
    practice_mode = Column(String, nullable=False)  # "camera" or "upload"
    timestamp = Column(DateTime, default=func.now())
    session_id = Column(String)  # To group predictions in same session
    
    # Relationships
    user = relationship("User", back_populates="predictions")

class Video(Base):
    __tablename__ = "videos"

    id          = Column(Integer, primary_key=True, index=True)
    word        = Column(String, index=True)
    title       = Column(String)
    description = Column(String)
    difficulty  = Column(String)
    duration    = Column(String)   # e.g. "42s"
    video_file  = Column(String)   # "A.mp4"
    thumbnail   = Column(String)   # "thumbnails/A.jpg"
    category    = Column(String, index=True)

class UserSettings(Base):
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String, unique=True, index=True)
    setting_value = Column(String)
    description = Column(String)

class AdminLog(Base):
    __tablename__ = "admin_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    target_user_id = Column(Integer, ForeignKey("users.id"))
    details = Column(Text)
    timestamp = Column(DateTime, default=func.now())
    ip_address = Column(String)

class TranslationSession(Base):
    __tablename__ = "translation_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_type = Column(String, nullable=False)  # "mediapipe", "openpose", "custom"
    input_mode = Column(String, nullable=False)  # "word", "sentence"
    started_at = Column(DateTime, default=func.now())
    ended_at = Column(DateTime)
    session_duration = Column(Integer)  # seconds
    translations_count = Column(Integer, default=0)
    correct_translations = Column(Integer, default=0)
    total_confidence = Column(Float, default=0.0)
    average_confidence = Column(Float)
    accuracy_percentage = Column(Float)
    
    # Relationships
    translations = relationship("TranslationHistory", back_populates="session")

class TranslationHistory(Base):
    __tablename__ = "translation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("translation_sessions.id"), nullable=False)
    predicted_text = Column(String, nullable=False)
    target_text = Column(String)  # Optional - for accuracy tracking
    confidence = Column(Float, nullable=False)
    is_correct = Column(Boolean)  # True if predicted_text matches target_text
    model_used = Column(String, nullable=False)
    input_mode = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    session = relationship("TranslationSession", back_populates="translations")