# app/main.py
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import time
import os

from app.core.database import engine, get_db
from app.models import Base, Video
from app.api.v1.api import api_router
from app.core.config import settings
from app.auth import create_default_admin

from sqladmin import Admin
from sqladmin.models import ModelView

app = FastAPI(
    title="ASLense API",
    description="ASL Video Learning Platform with Authentication",
    version="2.0.0"
)

class VideoAdmin(ModelView, model=Video):
    column_list = [Video.id, Video.title, Video.category, Video.difficulty]
    column_searchable_list = [Video.title, Video.word, Video.category]
    column_filters = [Video.category, Video.difficulty]

admin = Admin(app, engine)
admin.add_view(VideoAdmin)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# Mount static files for thumbnails
thumbnail_dir = settings.THUMBNAIL_DIR
if not thumbnail_dir.exists():
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(thumbnail_dir)), name="thumbnails")

# Create tables
Base.metadata.create_all(bind=engine)

# Create default admin user on startup
@app.on_event("startup")
async def startup_event():
    db = next(get_db())
    try:
        create_default_admin(db)
    except Exception as e:
        # Admin creation error logged internally
        pass
    finally:
        db.close()

# Root endpoint
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

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

# Router mounting - Use the organized API structure
app.include_router(api_router, prefix="/api/v1")

# Legacy routes for backward compatibility (will be deprecated)
# These can be removed once frontend is updated to use /api/v1 prefix
from app.api.v1.endpoints import auth, user, videos, learn, practice, admin, translate

# Mount individual routers for backward compatibility
app.include_router(auth.router, prefix="/auth", tags=["auth-legacy"])
app.include_router(user.router, prefix="/user", tags=["user-legacy"])
app.include_router(videos.router, prefix="/videos", tags=["videos-legacy"])
app.include_router(learn.router, prefix="/learn", tags=["learn-legacy"])
app.include_router(practice.router, prefix="/practice", tags=["practice-legacy"])
app.include_router(translate.router, prefix="/translate", tags=["translate-legacy"])
app.include_router(admin.router, prefix="/admin-api", tags=["admin-legacy"])