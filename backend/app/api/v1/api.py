from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    user,
    videos,
    learn,
    practice,
    translate,
    admin
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(learn.router, prefix="/learn", tags=["learning"])
api_router.include_router(practice.router, prefix="/practice", tags=["practice"])
api_router.include_router(translate.router, prefix="/translate", tags=["translation"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
