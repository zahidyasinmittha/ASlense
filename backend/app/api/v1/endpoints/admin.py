# app/api/v1/endpoints/admin.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import func, text, String
from typing import List
from app.core.database import get_db
from app.auth import require_admin
from app.schemas import (
    User, UserCreate, UserUpdate, AdminDashboard, 
    PredictionHistoryResponse, AdminLogResponse,
    Video, VideoCreate, VideoUpdate,
    PSLAlphabet, PSLAlphabetCreate, PSLAlphabetUpdate
)
from app.services.user_service import UserService, AdminService, PredictionService
from app.models import User as UserModel, Video as VideoModel

router = APIRouter()

@router.get("/dashboard", response_model=AdminDashboard)
async def get_admin_dashboard(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get admin dashboard data."""
    admin_service = AdminService(db)
    dashboard_data = admin_service.get_admin_dashboard_data()
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_DASHBOARD",
        details="Accessed admin dashboard"
    )
    
    return dashboard_data

@router.get("/stats")
async def get_admin_stats(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get admin dashboard statistics."""
    from app.models import PredictionHistory
    
    # User statistics
    total_users = db.query(func.count(UserModel.id)).scalar() or 0
    active_users = db.query(func.count(UserModel.id)).filter(UserModel.is_active == True).scalar() or 0
    
    # Video statistics
    total_videos = db.query(func.count(VideoModel.id)).scalar() or 0
    
    # Prediction statistics
    total_predictions = db.query(func.count(PredictionHistory.id)).scalar() or 0
    correct_predictions = db.query(func.count(PredictionHistory.id))\
        .filter(PredictionHistory.is_correct == True).scalar() or 0
    
    average_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    return {
        "totalUsers": total_users,
        "activeUsers": active_users,
        "totalVideos": total_videos,
        "totalPredictions": total_predictions,
        "averageAccuracy": round(average_accuracy, 2)
    }

@router.get("/metrics")
async def get_system_metrics(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system performance metrics."""
    from app.models import PredictionHistory
    from datetime import datetime, timedelta
    
    # Calculate metrics
    total_predictions = db.query(func.count(PredictionHistory.id)).scalar() or 0
    correct_predictions = db.query(func.count(PredictionHistory.id))\
        .filter(PredictionHistory.is_correct == True).scalar() or 0
    
    prediction_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Daily active users (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    daily_active_users = db.query(func.count(func.distinct(PredictionHistory.user_id)))\
        .filter(PredictionHistory.timestamp >= yesterday).scalar() or 0
    
    # Average session time (placeholder - would need session tracking)
    average_session_time = 15.5
    
    # Total sessions (unique user-day combinations)
    # Use SQLite compatible concatenation with explicit column references
    from app.models import PredictionHistory
    total_sessions = db.query(func.count(func.distinct(
        PredictionHistory.user_id.cast(String) + '-' + func.date(PredictionHistory.timestamp).cast(String)
    ))).scalar() or 0
    
    return {
        "predictionAccuracy": round(prediction_accuracy, 2),
        "averageSessionTime": average_session_time,
        "dailyActiveUsers": daily_active_users,
        "totalSessions": total_sessions
    }

@router.get("/users", response_model=List[User])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    print("dfjb")
    """Get all users (admin only)."""
    user_service = UserService(db)
    admin_service = AdminService(db)
    
    users = user_service.get_all_users(skip=skip, limit=limit)
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_ALL_USERS",
        details=f"Viewed users list (skip={skip}, limit={limit})"
    )
    print(users)
    return users

@router.post("/users", response_model=User)
async def create_user_admin(
    user_create: UserCreate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user (admin only)."""
    try:
        user_service = UserService(db)
        admin_service = AdminService(db)
        
        user = user_service.create_user(user_create)
        
        # Log admin action
        admin_service.log_admin_action(
            admin_id=admin_user.id,
            action="CREATE_USER",
            target_user_id=user.id,
            details=f"Created user: {user.username} with role: {user.role}"
        )
        
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/users/{user_id}", response_model=User)
async def update_user_admin(
    user_id: int,
    user_update: UserUpdate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user (admin only)."""
    user_service = UserService(db)
    admin_service = AdminService(db)
    
    user = user_service.update_user(user_id, user_update)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="UPDATE_USER",
        target_user_id=user_id,
        details=f"Updated user: {user.username}"
    )
    
    return user

@router.delete("/users/{user_id}")
async def delete_user_admin(
    user_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete (deactivate) user (admin only)."""
    user_service = UserService(db)
    admin_service = AdminService(db)
    
    # Get user info before deletion
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    success = user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete user"
        )
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="DELETE_USER",
        target_user_id=user_id,
        details=f"Deactivated user: {user.username}"
    )
    
    return {"message": "User deactivated successfully"}

@router.patch("/users/{user_id}/toggle-status")
async def toggle_user_status(
    user_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Toggle user active status (admin only)."""
    user_service = UserService(db)
    admin_service = AdminService(db)
    
    # Get user
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Toggle status
    new_status = not user.is_active
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    db_user.is_active = new_status
    db.commit()
    db.refresh(db_user)
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="TOGGLE_USER_STATUS",
        target_user_id=user_id,
        details=f"{'Activated' if new_status else 'Deactivated'} user: {user.username}"
    )
    
    return {
        "message": f"User {'activated' if new_status else 'deactivated'} successfully",
        "user_id": user_id,
        "is_active": new_status
    }

@router.get("/users/{user_id}/predictions", response_model=List[PredictionHistoryResponse])
async def get_user_predictions_admin(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get user's prediction history (admin only)."""
    prediction_service = PredictionService(db)
    admin_service = AdminService(db)
    
    predictions = prediction_service.get_user_predictions(user_id, skip=skip, limit=limit)
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_USER_PREDICTIONS",
        target_user_id=user_id,
        details=f"Viewed predictions for user ID: {user_id}"
    )
    
    return predictions

@router.post("/create-admin")
async def create_admin_user(
    user_create: UserCreate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new admin user (admin only)."""
    try:
        user_service = UserService(db)
        admin_service = AdminService(db)
        
        # Force role to admin
        user_create.role = "admin"
        admin = user_service.create_user(user_create)
        
        # Log admin action
        admin_service.log_admin_action(
            admin_id=admin_user.id,
            action="CREATE_ADMIN",
            target_user_id=admin.id,
            details=f"Created admin user: {admin.username}"
        )
        
        return {"message": f"Admin user {admin.username} created successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/logs", response_model=List[AdminLogResponse])
async def get_admin_logs(
    skip: int = 0,
    limit: int = 100,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get admin action logs (admin only)."""
    from app.models import AdminLog
    
    logs = db.query(AdminLog)\
        .order_by(AdminLog.timestamp.desc())\
        .offset(skip).limit(limit).all()
    
    return logs

@router.get("/analytics")
async def get_admin_analytics(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system analytics (admin only)."""
    from app.models import User, PredictionHistory
    from datetime import datetime, timedelta
    
    # User statistics
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    admin_users = db.query(func.count(User.id)).filter(User.role == "admin").scalar()
    
    # Prediction statistics
    total_predictions = db.query(func.count(PredictionHistory.id)).scalar()
    correct_predictions = db.query(func.count(PredictionHistory.id))\
        .filter(PredictionHistory.is_correct == True).scalar()
    
    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_predictions = db.query(func.count(PredictionHistory.id))\
        .filter(PredictionHistory.timestamp >= week_ago).scalar()
    
    # Model usage
    model_usage = db.query(
        PredictionHistory.model_used,
        func.count(PredictionHistory.id).label('count')
    ).group_by(PredictionHistory.model_used).all()
    
    return {
        "user_stats": {
            "total": total_users,
            "active": active_users,
            "admins": admin_users
        },
        "prediction_stats": {
            "total": total_predictions,
            "correct": correct_predictions,
            "accuracy": (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            "recent_week": recent_predictions
        },
        "model_usage": {
            usage[0]: usage[1] for usage in model_usage
        }
    }

# Video Management Endpoints
@router.get("/videos", response_model=List[Video])
async def get_all_videos(
    skip: int = 0,
    limit: int = 100,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all videos (admin only)."""
    admin_service = AdminService(db)
    videos = db.query(VideoModel).offset(skip).limit(limit).all()
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_ALL_VIDEOS",
        details=f"Viewed videos list (skip={skip}, limit={limit})"
    )
    
    return videos

@router.post("/videos", response_model=Video)
async def create_video(
    video_create: VideoCreate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new video (admin only)."""
    admin_service = AdminService(db)
    
    # Create video
    db_video = VideoModel(**video_create.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="CREATE_VIDEO",
        details=f"Created video: {db_video.title} for word: {db_video.word}"
    )
    
    return db_video

@router.put("/videos/{video_id}", response_model=Video)
async def update_video(
    video_id: int,
    video_update: VideoUpdate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update video (admin only)."""
    admin_service = AdminService(db)
    
    # Get video
    db_video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not db_video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Update video
    update_data = video_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_video, field, value)
    
    db.commit()
    db.refresh(db_video)
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="UPDATE_VIDEO",
        details=f"Updated video: {db_video.title} (ID: {video_id})"
    )
    
    return db_video

@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete video (admin only)."""
    admin_service = AdminService(db)
    
    # Get video
    db_video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not db_video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    video_title = db_video.title
    
    # Delete video
    db.delete(db_video)
    db.commit()
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="DELETE_VIDEO",
        details=f"Deleted video: {video_title} (ID: {video_id})"
    )
    
    return {"message": "Video deleted successfully"}

@router.get("/videos/count")
async def get_videos_count(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get total count of videos (admin only)."""
    admin_service = AdminService(db)
    
    total_videos = db.query(VideoModel).count()
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_VIDEO_COUNT",
        details="Viewed video count"
    )
    
    return {"count": total_videos}

@router.get("/videos/stats")
async def get_video_stats(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get video statistics (admin only)."""
    admin_service = AdminService(db)
    
    total_videos = db.query(VideoModel).count()
    videos_by_category = db.query(VideoModel.category, func.count(VideoModel.id)).group_by(VideoModel.category).all()
    videos_by_difficulty = db.query(VideoModel.difficulty, func.count(VideoModel.id)).group_by(VideoModel.difficulty).all()
    
    # Log admin action
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_VIDEO_STATS",
        details="Viewed video statistics"
    )
    
    return {
        "total_videos": total_videos,
        "by_category": {category: count for category, count in videos_by_category if category},
        "by_difficulty": {difficulty: count for difficulty, count in videos_by_difficulty if difficulty}
    }

# PSL Alphabet Admin Management
@router.get("/psl-alphabet/stats")
async def get_psl_alphabet_stats(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get PSL alphabet statistics (admin only)."""
    from app.models import PSLAlphabet
    
    total_entries = db.query(func.count(PSLAlphabet.id)).scalar() or 0
    active_entries = db.query(func.count(PSLAlphabet.id)).filter(PSLAlphabet.is_active == True).scalar() or 0
    
    # Group by difficulty
    entries_by_difficulty = db.query(
        PSLAlphabet.difficulty,
        func.count(PSLAlphabet.id)
    ).group_by(PSLAlphabet.difficulty).all()
    
    admin_service = AdminService(db)
    admin_service.log_admin_action(
        admin_id=admin_user.id,
        action="VIEW_PSL_ALPHABET_STATS",
        details="Viewed PSL alphabet statistics"
    )
    
    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "inactive_entries": total_entries - active_entries,
        "by_difficulty": {difficulty: count for difficulty, count in entries_by_difficulty if difficulty}
    }
