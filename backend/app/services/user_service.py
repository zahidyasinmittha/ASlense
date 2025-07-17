# app/services/user_service.py
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
from app.models import User, UserProgress, PredictionHistory, AdminLog
from app.schemas import UserCreate, UserUpdate, PredictionHistoryCreate, UserProgressUpdate
from app.auth import get_password_hash

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user with initial progress."""
        # Check if user already exists
        existing_user = self.db.query(User).filter(
            or_(User.username == user_create.username, User.email == user_create.email)
        ).first()
        if existing_user:
            raise ValueError("User with this username or email already exists")

        # Create new user
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            password_hash=get_password_hash(user_create.password),
            role=user_create.role
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        # Create initial progress record
        progress = UserProgress(user_id=db_user.id)
        self.db.add(progress)
        self.db.commit()

        return db_user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()

    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        return self.db.query(User).offset(skip).limit(limit).all()

    def update_user(self, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Update user information."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        update_data = user_update.dict(exclude_unset=True)
        if "password" in update_data:
            update_data["password_hash"] = get_password_hash(update_data.pop("password"))

        for field, value in update_data.items():
            setattr(user, field, value)

        self.db.commit()
        self.db.refresh(user)
        return user

    def delete_user(self, user_id: int) -> bool:
        """Soft delete user (set inactive)."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        self.db.commit()
        return True

    def update_last_login(self, user_id: int):
        """Update user's last login timestamp."""
        user = self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()

class ProgressService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_progress(self, user_id: int) -> Optional[UserProgress]:
        """Get user's progress."""
        return self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()

    def update_progress(self, user_id: int, progress_update: UserProgressUpdate) -> UserProgress:
        """Update user progress after practice session."""
        progress = self.get_user_progress(user_id)
        if not progress:
            progress = UserProgress(
                user_id=user_id,
                total_signs_practiced=0,
                correct_predictions=0,
                total_predictions=0,
                accuracy_rate=0.0,
                practice_streak=0,
                longest_streak=0,
                total_practice_time=0,
                level="Beginner",
                experience_points=0
            )
            self.db.add(progress)

        # Update basic stats (handle None values)
        progress.total_predictions = (progress.total_predictions or 0) + 1
        if progress_update.is_correct:
            progress.correct_predictions = (progress.correct_predictions or 0) + 1

        # Calculate accuracy
        progress.accuracy_rate = (progress.correct_predictions / progress.total_predictions) * 100

        # Update practice time
        progress.total_practice_time = (progress.total_practice_time or 0) + progress_update.practice_time

        # Update streak
        today = datetime.utcnow().date()
        if progress.last_practice_date:
            last_practice = progress.last_practice_date.date()
            if last_practice == today:
                pass  # Same day, don't update streak
            elif last_practice == today - timedelta(days=1):
                progress.practice_streak = (progress.practice_streak or 0) + 1
            else:
                progress.practice_streak = 1
        else:
            progress.practice_streak = 1

        progress.longest_streak = max(progress.longest_streak or 0, progress.practice_streak)
        progress.last_practice_date = datetime.utcnow()

        # Update experience points
        points = 10 if progress_update.is_correct else 2
        progress.experience_points += points

        # Update level based on XP
        if progress.experience_points >= 1000:
            progress.level = "Expert"
        elif progress.experience_points >= 500:
            progress.level = "Advanced"
        elif progress.experience_points >= 100:
            progress.level = "Intermediate"
        else:
            progress.level = "Beginner"

        # Track unique signs practiced
        existing_word = self.db.query(PredictionHistory).filter(
            and_(
                PredictionHistory.user_id == user_id,
                PredictionHistory.target_word == progress_update.target_word
            )
        ).first()
        
        if not existing_word:
            progress.total_signs_practiced += 1

        self.db.commit()
        self.db.refresh(progress)
        return progress

class PredictionService:
    def __init__(self, db: Session):
        self.db = db

    def save_prediction(self, prediction_data: PredictionHistoryCreate) -> PredictionHistory:
        """Save prediction to history."""
        prediction = PredictionHistory(**prediction_data.dict())
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def get_user_predictions(self, user_id: int, skip: int = 0, limit: int = 50) -> List[PredictionHistory]:
        """Get user's prediction history."""
        return self.db.query(PredictionHistory)\
            .filter(PredictionHistory.user_id == user_id)\
            .order_by(PredictionHistory.timestamp.desc())\
            .offset(skip).limit(limit).all()

    def get_user_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get user analytics data."""
        # Get predictions from last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        predictions = self.db.query(PredictionHistory)\
            .filter(
                and_(
                    PredictionHistory.user_id == user_id,
                    PredictionHistory.timestamp >= thirty_days_ago
                )
            ).all()

        # Daily practice data
        daily_practice = {}
        for pred in predictions:
            date_key = pred.timestamp.strftime('%Y-%m-%d')
            if date_key not in daily_practice:
                daily_practice[date_key] = {'total': 0, 'correct': 0}
            daily_practice[date_key]['total'] += 1
            if pred.is_correct:
                daily_practice[date_key]['correct'] += 1

        # Most practiced words
        word_stats = {}
        for pred in predictions:
            word = pred.target_word
            if word not in word_stats:
                word_stats[word] = {'total': 0, 'correct': 0}
            word_stats[word]['total'] += 1
            if pred.is_correct:
                word_stats[word]['correct'] += 1

        most_practiced = sorted(word_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:10]

        # Model performance
        model_stats = {}
        for pred in predictions:
            model = pred.model_used
            if model not in model_stats:
                model_stats[model] = {'total': 0, 'correct': 0}
            model_stats[model]['total'] += 1
            if pred.is_correct:
                model_stats[model]['correct'] += 1

        return {
            'daily_practice': [
                {
                    'date': date,
                    'total': data['total'],
                    'correct': data['correct'],
                    'accuracy': (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
                }
                for date, data in sorted(daily_practice.items())
            ],
            'most_practiced_words': [
                {
                    'word': word,
                    'total': data['total'],
                    'correct': data['correct'],
                    'accuracy': (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
                }
                for word, data in most_practiced
            ],
            'model_performance': {
                model: {
                    'total': data['total'],
                    'correct': data['correct'],
                    'accuracy': (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
                }
                for model, data in model_stats.items()
            }
        }

class AdminService:
    def __init__(self, db: Session):
        self.db = db

    def log_admin_action(self, admin_id: int, action: str, target_user_id: Optional[int] = None, 
                        details: Optional[str] = None, ip_address: Optional[str] = None):
        """Log admin action."""
        log_entry = AdminLog(
            admin_id=admin_id,
            action=action,
            target_user_id=target_user_id,
            details=details,
            ip_address=ip_address
        )
        self.db.add(log_entry)
        self.db.commit()

    def get_admin_dashboard_data(self) -> Dict[str, Any]:
        """Get admin dashboard data."""
        # Total users
        total_users = self.db.query(func.count(User.id)).scalar()
        active_users = self.db.query(func.count(User.id)).filter(User.is_active == True).scalar()

        # Today's predictions
        today = datetime.utcnow().date()
        today_predictions = self.db.query(func.count(PredictionHistory.id))\
            .filter(func.date(PredictionHistory.timestamp) == today).scalar()

        # This week's predictions
        week_ago = datetime.utcnow() - timedelta(days=7)
        week_predictions = self.db.query(func.count(PredictionHistory.id))\
            .filter(PredictionHistory.timestamp >= week_ago).scalar()

        # Top performing users
        top_users = self.db.query(User.username, UserProgress.accuracy_rate, UserProgress.total_predictions)\
            .join(UserProgress)\
            .filter(UserProgress.total_predictions >= 10)\
            .order_by(UserProgress.accuracy_rate.desc())\
            .limit(10).all()

        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_predictions_today': today_predictions,
            'total_predictions_week': week_predictions,
            'top_performing_users': [
                {
                    'username': user[0],
                    'accuracy': user[1],
                    'total_predictions': user[2]
                }
                for user in top_users
            ]
        }
