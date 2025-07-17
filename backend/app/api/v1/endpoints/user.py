# app/api/v1/endpoints/user.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.auth import get_current_active_user
from app.schemas import (
    User, UserUpdate, UserDashboard, UserProgressResponse, 
    PredictionHistoryResponse, UserAnalytics
)
from app.services.user_service import UserService, ProgressService, PredictionService
from app.models import User as UserModel

router = APIRouter(tags=["user"])

@router.get("/profile", response_model=User)
async def get_user_profile(current_user: UserModel = Depends(get_current_active_user)):
    """Get current user's profile."""
    return current_user

@router.put("/profile", response_model=User)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile."""
    # Users can't change their own role
    if hasattr(user_update, 'role') and user_update.role is not None:
        if current_user.role != "admin":
            user_update.role = None
    
    user_service = UserService(db)
    updated_user = user_service.update_user(current_user.id, user_update)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update profile"
        )
    
    return updated_user

@router.get("/dashboard", response_model=UserDashboard)
async def get_user_dashboard(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user dashboard data."""
    progress_service = ProgressService(db)
    prediction_service = PredictionService(db)
    
    # Get user progress
    progress = progress_service.get_user_progress(current_user.id)
    if not progress:
        # Create initial progress if doesn't exist
        from app.schemas import UserProgressUpdate
        progress = progress_service.update_progress(
            current_user.id, 
            UserProgressUpdate(is_correct=False, practice_time=0, target_word="")
        )
    
    # Get recent predictions
    recent_predictions = prediction_service.get_user_predictions(current_user.id, limit=10)
    
    # Calculate practice stats
    practice_stats = {
        "accuracy_percentage": progress.accuracy_rate,
        "total_sessions": progress.total_predictions,
        "practice_streak": progress.practice_streak,
        "level": progress.level,
        "experience_points": progress.experience_points,
        "signs_mastered": progress.total_signs_practiced,
        "practice_time_hours": round(progress.total_practice_time / 3600, 1)
    }
    
    return UserDashboard(
        user=current_user,
        progress=progress,
        recent_predictions=recent_predictions,
        practice_stats=practice_stats
    )

@router.get("/progress", response_model=dict)
async def get_user_progress(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's progress."""
    progress_service = ProgressService(db)
    progress = progress_service.get_user_progress(current_user.id)
    
    if not progress:
        # Create default progress if none exists
        from app.schemas import UserProgressUpdate
        progress = progress_service.update_progress(
            current_user.id, 
            UserProgressUpdate(is_correct=False, practice_time=0, target_word="")
        )
    
    # Get prediction service to calculate signs learned vs mastered
    prediction_service = PredictionService(db)
    user_predictions = prediction_service.get_user_predictions(current_user.id, limit=1000)
    
    # Calculate unique signs practiced and mastered
    unique_signs_practiced = set()
    signs_mastered = set()
    
    for pred in user_predictions:
        unique_signs_practiced.add(pred.target_word)
        # Consider a sign "mastered" if user got it correct with confidence > 0.8 in recent attempts
        if pred.is_correct and pred.confidence_score > 0.8:
            signs_mastered.add(pred.target_word)
    
    # Get current user XP and level from user model
    current_xp = getattr(current_user, 'total_xp', None) or progress.experience_points
    current_level_str = getattr(current_user, 'level', None) or progress.level
    
    # Map level to XP thresholds
    level_xp_map = {
        "Beginner": (0, 100),
        "Intermediate": (100, 500), 
        "Advanced": (500, 1000),
        "Expert": (1000, float('inf'))
    }
    
    current_level_min, next_level_xp = level_xp_map.get(current_level_str, (0, 100))
    
    # Calculate level progress percentage
    if next_level_xp == float('inf'):
        level_progress = 100
    else:
        level_progress = min(100, ((current_xp - current_level_min) / (next_level_xp - current_level_min)) * 100)
    
    print(f"User {current_user.id} progress: {len(unique_signs_practiced)} signs practiced, {len(signs_mastered)} mastered, {current_xp} XP, {progress.accuracy_rate:.2f}% accuracy")
    
    # Return data in format expected by frontend
    return {
        "signs_practiced": len(unique_signs_practiced),
        "signs_learned": len(unique_signs_practiced),  # For now, same as practiced
        "signs_mastered": len(signs_mastered),
        "total_signs": 136,  # Total available signs
        "accuracy_rate": round(progress.accuracy_rate, 2),
        "current_level": current_level_str,
        "current_xp": current_xp,
        "next_level_xp": next_level_xp if next_level_xp != float('inf') else current_xp,
        "level_progress": round(level_progress),
        "practice_streak": progress.practice_streak,
        "total_practice_time": progress.total_practice_time,
        "total_predictions": progress.total_predictions,
        "correct_predictions": progress.correct_predictions,
    }

@router.get("/predictions", response_model=List[PredictionHistoryResponse])
async def get_user_predictions(
    skip: int = 0,
    limit: int = 50,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's prediction history."""
    prediction_service = PredictionService(db)
    predictions = prediction_service.get_user_predictions(current_user.id, skip=skip, limit=limit)
    return predictions

@router.get("/predictions/recent", response_model=List[PredictionHistoryResponse])
async def get_recent_predictions(
    limit: int = 10,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's recent prediction history."""
    prediction_service = PredictionService(db)
    predictions = prediction_service.get_user_predictions(current_user.id, skip=0, limit=limit)
    return predictions

@router.get("/analytics", response_model=UserAnalytics)
async def get_user_analytics(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user analytics data."""
    prediction_service = PredictionService(db)
    analytics_data = prediction_service.get_user_analytics(current_user.id)
    
    return UserAnalytics(
        daily_practice=analytics_data['daily_practice'],
        accuracy_trends=analytics_data['daily_practice'],  # Reuse for trends
        most_practiced_words=analytics_data['most_practiced_words'],
        model_performance=analytics_data['model_performance']
    )

@router.get("/achievements")
async def get_user_achievements(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user achievements and badges."""
    progress_service = ProgressService(db)
    progress = progress_service.get_user_progress(current_user.id)
    
    if not progress:
        return {"achievements": [], "badges": []}
    
    achievements = []
    badges = []
    
    # Define achievements based on progress
    if progress.total_predictions >= 1:
        achievements.append({
            "id": "first_prediction",
            "title": "First Steps",
            "description": "Made your first prediction",
            "icon": "🎯",
            "unlocked": True
        })
    
    if progress.practice_streak >= 3:
        achievements.append({
            "id": "streak_3",
            "title": "On a Roll",
            "description": "3-day practice streak",
            "icon": "🔥",
            "unlocked": True
        })
    
    if progress.practice_streak >= 7:
        achievements.append({
            "id": "streak_7",
            "title": "Week Warrior",
            "description": "7-day practice streak",
            "icon": "⚡",
            "unlocked": True
        })
    
    if progress.accuracy_rate >= 80:
        achievements.append({
            "id": "accuracy_80",
            "title": "Sharp Shooter",
            "description": "80% accuracy rate",
            "icon": "🎯",
            "unlocked": True
        })
    
    if progress.total_signs_practiced >= 10:
        achievements.append({
            "id": "signs_10",
            "title": "Sign Explorer",
            "description": "Practiced 10 different signs",
            "icon": "🗺️",
            "unlocked": True
        })
    
    # Define badges based on level and XP
    if progress.level == "Beginner":
        badges.append({
            "id": "beginner",
            "name": "Beginner",
            "color": "green",
            "icon": "🌱"
        })
    elif progress.level == "Intermediate":
        badges.append({
            "id": "intermediate", 
            "name": "Intermediate",
            "color": "blue",
            "icon": "📚"
        })
    elif progress.level == "Advanced":
        badges.append({
            "id": "advanced",
            "name": "Advanced", 
            "color": "purple",
            "icon": "🎓"
        })
    elif progress.level == "Expert":
        badges.append({
            "id": "expert",
            "name": "Expert",
            "color": "gold",
            "icon": "👑"
        })
    
    return {
        "achievements": achievements,
        "badges": badges,
        "total_xp": progress.experience_points,
        "level": progress.level
    }

@router.delete("/account")
async def delete_user_account(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete current user's account (soft delete)."""
    user_service = UserService(db)
    success = user_service.delete_user(current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete account"
        )
    
    return {"message": "Account deactivated successfully"}
