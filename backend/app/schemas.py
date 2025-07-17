# app/schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class PracticeMode(str, Enum):
    CAMERA = "camera"
    UPLOAD = "upload"

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.USER
    is_active: bool = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    last_login: Optional[datetime] = None
    profile_picture: Optional[str] = None

    class Config:
        from_attributes = True

class UserWithProgress(User):
    progress: Optional['UserProgressResponse'] = None

# Authentication Schemas
class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str
    user: User

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# User Progress Schemas
class UserProgressBase(BaseModel):
    total_signs_practiced: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    accuracy_rate: float = 0.0
    practice_streak: int = 0
    longest_streak: int = 0
    total_practice_time: int = 0
    level: str = "Beginner"
    experience_points: int = 0

class UserProgressResponse(UserProgressBase):
    id: int
    user_id: int
    last_practice_date: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserProgressUpdate(BaseModel):
    is_correct: bool
    practice_time: int = 0
    target_word: str

# Prediction History Schemas
class PredictionHistoryBase(BaseModel):
    target_word: str
    predicted_words: str  # JSON string
    is_correct: bool
    confidence_score: Optional[float] = None
    model_used: str
    practice_mode: PracticeMode
    session_id: Optional[str] = None

class PredictionHistoryCreate(PredictionHistoryBase):
    user_id: int

class PredictionHistoryResponse(PredictionHistoryBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Enhanced Prediction Response Schema
class EnhancedPredictionResult(BaseModel):
    target_word: str
    predictions: List[Dict[str, Any]]
    top_predictions: List[Dict[str, Any]]
    is_match: bool
    is_correct: bool  # True if target word is in top 4
    match_confidence: float
    model_used: str
    rank_in_top_4: Optional[int] = None  # Rank if in top 4

# Admin Schemas
class AdminLogBase(BaseModel):
    action: str
    target_user_id: Optional[int] = None
    details: Optional[str] = None
    ip_address: Optional[str] = None

class AdminLogCreate(AdminLogBase):
    admin_id: int

class AdminLogResponse(AdminLogBase):
    id: int
    admin_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Dashboard Schemas
class UserDashboard(BaseModel):
    user: User
    progress: UserProgressResponse
    recent_predictions: List[PredictionHistoryResponse]
    practice_stats: Dict[str, Any]

class AdminDashboard(BaseModel):
    total_users: int
    active_users: int
    total_predictions_today: int
    total_predictions_week: int
    top_performing_users: List[Dict[str, Any]]
    recent_user_activities: List[Dict[str, Any]]
    system_stats: Dict[str, Any]

# Analytics Schemas
class UserAnalytics(BaseModel):
    daily_practice: List[Dict[str, Any]]
    accuracy_trends: List[Dict[str, Any]]
    most_practiced_words: List[Dict[str, Any]]
    model_performance: Dict[str, Any]

class AdminAnalytics(BaseModel):
    user_growth: List[Dict[str, Any]]
    prediction_trends: List[Dict[str, Any]]
    model_usage: Dict[str, Any]
    popular_words: List[Dict[str, Any]]

# Video Schemas
class VideoBase(BaseModel):
    word: str
    title: str
    description: Optional[str] = None
    difficulty: Optional[str] = None
    duration: Optional[str] = None
    video_file: Optional[str] = None
    thumbnail: Optional[str] = None
    category: Optional[str] = None

class VideoCreate(VideoBase):
    pass

class VideoUpdate(BaseModel):
    word: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    difficulty: Optional[str] = None
    duration: Optional[str] = None
    video_file: Optional[str] = None
    thumbnail: Optional[str] = None
    category: Optional[str] = None

class Video(VideoBase):
    id: int
    
    class Config:
        from_attributes = True

# Update forward references
UserWithProgress.model_rebuild()
