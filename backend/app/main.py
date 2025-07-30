# app/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqladmin import Admin
from sqladmin.models import ModelView

from app.core.database import engine, get_db
from app.core.config import settings
from app.models import Base, Video, User, UserProgress, PredictionHistory, TranslationSession
from app.api.v1.api import api_router
from app.auth import create_default_admin

app = FastAPI(
    title="ASLense API",
    description="ASL Video Learning Platform with Authentication",
    version="2.0.0"
)

#––– SQLAdmin Views ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class VideoAdmin(ModelView, model=Video):
    column_list = [Video.id, Video.title, Video.difficulty]
    column_searchable_list = [Video.title, Video.word]


class ProgressAdmin(ModelView, model=UserProgress):
    column_list = [UserProgress.id, UserProgress.user_id, UserProgress.accuracy_rate]
    column_searchable_list = [UserProgress.user_id]

class PredictionAdmin(ModelView, model=PredictionHistory):
    column_list = [PredictionHistory.id, PredictionHistory.user_id, PredictionHistory.target_word, PredictionHistory.is_correct]
    column_searchable_list = [PredictionHistory.target_word, PredictionHistory.user_id]

class TranslationSessionAdmin(ModelView, model=TranslationSession):
    column_list = [TranslationSession.id, TranslationSession.user_id, TranslationSession.average_confidence,TranslationSession.accuracy_percentage,TranslationSession.total_confidence  ]
    column_searchable_list = [TranslationSession.user_id]

admin = Admin(app, engine)
admin.add_view(VideoAdmin)
admin.add_view(ProgressAdmin)
admin.add_view(PredictionAdmin)
admin.add_view(TranslationSessionAdmin)


#––– Middleware & Static Files ––––––––––––––––––––––––––––––––––––––––––––––––––

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # You can log here if desired
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be cautious in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# Ensure thumbnail dir exists, then mount
thumbnail_dir = settings.THUMBNAIL_DIR
thumbnail_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/thumbnails",
    StaticFiles(directory=str(thumbnail_dir)),
    name="thumbnails"
)


#––– Startup & Tables ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Auto-create tables
Base.metadata.create_all(bind=engine)

# Create default admin user
@app.on_event("startup")
async def on_startup():
    db: Session = next(get_db())
    try:
        create_default_admin(db)
    except Exception:
        pass
    finally:
        db.close()


#––– Health & Root Endpoints ––––––––––––––––––––––––––––––––––––––––––––––––––––

@app.get("/")
async def root():
    return {
        "message": "ASLense API v2.0",
        "status": "active",
        "features": [
            "User Authentication",
            "Progress Tracking",
            "Admin Dashboard",
            "Enhanced Predictions",
            "Practice History"
        ]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


#––– API Routers ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

app.include_router(api_router, prefix="/api/v1")

# Legacy routes (to be deprecated)
from app.api.v1.endpoints import auth, user, videos, learn, practice, admin as legacy_admin, translate

app.include_router(auth.router, prefix="/auth", tags=["auth-legacy"])
app.include_router(user.router, prefix="/user", tags=["user-legacy"])
app.include_router(videos.router, prefix="/videos", tags=["videos-legacy"])
app.include_router(learn.router, prefix="/learn", tags=["learn-legacy"])
app.include_router(practice.router, prefix="/practice", tags=["practice-legacy"])
app.include_router(translate.router, prefix="/translate", tags=["translate-legacy"])
app.include_router(legacy_admin.router, prefix="/admin-api", tags=["admin-legacy"])
