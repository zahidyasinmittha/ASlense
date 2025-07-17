# app/api/v1/endpoints/videos.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.schemas import Video
from app.models import Video as VideoModel

router = APIRouter(tags=["videos"])

@router.get("/", response_model=List[Video])
async def get_videos(
    skip: int = 0,
    limit: int = 50,
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
    word: Optional[str] = Query(None, description="Search by word"),
    db: Session = Depends(get_db)
):
    """Get all videos with optional filters."""
    query = db.query(VideoModel)
    
    if category:
        query = query.filter(VideoModel.category == category)
    
    if difficulty:
        query = query.filter(VideoModel.difficulty == difficulty)
    
    if word:
        query = query.filter(VideoModel.word.ilike(f"%{word}%"))
    
    videos = query.offset(skip).limit(limit).all()
    return videos

@router.get("/count")
async def get_videos_count(db: Session = Depends(get_db)):
    """Get total count of videos."""
    total_videos = db.query(VideoModel).count()
    print(f"Total videos in database: {total_videos}")
    return {"count": total_videos}


@router.get("/{video_id}", response_model=Video)
async def get_video(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific video by ID."""
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    return video

@router.get("/search/{word}", response_model=List[Video])
async def search_videos_by_word(
    word: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Search videos by word (for learning/practice)."""
    videos = db.query(VideoModel).filter(
        VideoModel.word.ilike(f"%{word}%")
    ).limit(limit).all()
    return videos

@router.get("/categories/list")
async def get_video_categories(db: Session = Depends(get_db)):
    """Get all available video categories."""
    categories = db.query(VideoModel.category).distinct().filter(
        VideoModel.category.isnot(None)
    ).all()
    return [cat[0] for cat in categories if cat[0]]

@router.get("/difficulties/list")
async def get_video_difficulties(db: Session = Depends(get_db)):
    """Get all available difficulty levels."""
    difficulties = db.query(VideoModel.difficulty).distinct().filter(
        VideoModel.difficulty.isnot(None)
    ).all()
    return [diff[0] for diff in difficulties if diff[0]]
