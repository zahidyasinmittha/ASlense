from fastapi import APIRouter, Depends, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
import os
import tempfile
import shutil
import urllib.parse
from typing import Optional

from app.core.database import get_db
from app.models import Video
from app.core.config import settings
# from app.services import VideoService  # TODO: Implement VideoService
from app.settings_service import settings_service

# Temporary VideoService placeholder
class VideoService:
    def __init__(self, *args, **kwargs):
        pass

router = APIRouter()

# ---------------- Settings endpoints -------------
@router.get("/settings/paths")
async def get_path_settings(db: Session = Depends(get_db)):
    """Get current path settings"""
    custom_paths = settings_service.get_path_settings(db)
    return {
        "custom_video_dir": custom_paths["video_dir"],
        "custom_thumbnail_dir": custom_paths["thumbnail_dir"],
        "default_video_dir": settings._VIDEO_DIR,
        "default_thumbnail_dir": settings._THUMBNAIL_DIR
    }

@router.post("/settings/paths")
async def set_path_settings(
    video_dir: Optional[str] = None,
    thumbnail_dir: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Set custom path settings"""
    try:
        # Validate paths if provided
        if video_dir and not Path(video_dir).exists():
            raise HTTPException(status_code=400, detail=f"Video directory does not exist: {video_dir}")
        
        if thumbnail_dir:
            thumb_path = Path(thumbnail_dir)
            thumb_path.mkdir(parents=True, exist_ok=True)
        
        result = settings_service.set_path_settings(db, video_dir, thumbnail_dir)
        return {"message": "Path settings updated successfully", "settings": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/settings/paths")
async def clear_path_settings(db: Session = Depends(get_db)):
    """Clear custom path settings and use defaults"""
    success = settings_service.clear_path_settings(db)
    if success:
        return {"message": "Path settings cleared, using defaults"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear path settings")

# ---------------- CSV upload and import -------------
@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(..., description="CSV file to upload"),
    custom_video_dir: Optional[str] = None,
    custom_thumbnail_dir: Optional[str] = None,
    save_paths: bool = False,
    db: Session = Depends(get_db)
):
    """Upload and import videos from CSV file with optional custom paths"""
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.csv', '.tsv')):
        raise HTTPException(status_code=400, detail="File must be a CSV or TSV file")
    
    # Check file size (optional limit)
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File size ({file.size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
        )
    
    # Save custom paths if requested
    if save_paths and (custom_video_dir or custom_thumbnail_dir):
        settings_service.set_path_settings(db, custom_video_dir, custom_thumbnail_dir)
    
    # Get effective paths (custom or from settings or defaults)
    if not custom_video_dir:
        custom_video_dir = settings_service.get_setting(db, "custom_video_dir")
    if not custom_thumbnail_dir:
        custom_thumbnail_dir = settings_service.get_setting(db, "custom_thumbnail_dir")
    
    # Create video service with custom paths
    try:
        video_service = VideoService(custom_thumbnail_dir, custom_video_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Process the temporary file
        result = video_service.import_from_csv(temp_file_path, db)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if result["success"]:
            response_data = {
                "message": result["message"],
                "imported": result["imported"],
                "failed": result["failed"],
                "failed_videos": result["failed_videos"]
            }
            
            # Add path info to response
            if custom_video_dir or custom_thumbnail_dir:
                response_data["paths_used"] = {
                    "video_dir": custom_video_dir or settings._VIDEO_DIR,
                    "thumbnail_dir": custom_thumbnail_dir or settings._THUMBNAIL_DIR
                }
            
            return response_data
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

# ---------------- CSV importer (server path) -------------
@router.post("/import-csv")
async def import_csv(
    file_path: str = Query(..., description="Server-side CSV path"),
    custom_video_dir: Optional[str] = None,
    custom_thumbnail_dir: Optional[str] = None,
    save_paths: bool = False,
    db: Session = Depends(get_db)
):
    """Import videos from CSV file with optional custom paths"""
    
    # Save custom paths if requested
    if save_paths and (custom_video_dir or custom_thumbnail_dir):
        settings_service.set_path_settings(db, custom_video_dir, custom_thumbnail_dir)
    
    # Get effective paths (custom or from settings or defaults)
    if not custom_video_dir:
        custom_video_dir = settings_service.get_setting(db, "custom_video_dir")
    if not custom_thumbnail_dir:
        custom_thumbnail_dir = settings_service.get_setting(db, "custom_thumbnail_dir")
    
    # Create video service with custom paths
    try:
        video_service = VideoService(custom_thumbnail_dir, custom_video_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = video_service.import_from_csv(file_path, db)
    
    if result["success"]:
        response_data = {
            "message": result["message"],
            "imported": result["imported"],
            "failed": result["failed"],
            "failed_videos": result["failed_videos"]
        }
        
        # Add path info to response
        if custom_video_dir or custom_thumbnail_dir:
            response_data["paths_used"] = {
                "video_dir": custom_video_dir or settings._VIDEO_DIR,
                "thumbnail_dir": custom_thumbnail_dir or settings._THUMBNAIL_DIR
            }
        
        return response_data
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@router.post("/delete-all-videos")
async def delete_all_videos(db: Session = Depends(get_db)):
    """Delete all videos from database"""
    # Use default video service for deletion
    video_service = VideoService()
    result = video_service.delete_all_videos(db)
    
    if result["success"]:
        return {"message": result["message"]}
    else:
        raise HTTPException(status_code=500, detail=result["error"])
    
# ---------------- video streaming ---------------
@router.get("/stream/{video_file}")
async def stream_video(
    video_file: str,
    db: Session = Depends(get_db)
):
    """Stream video files from the configured VIDEO_DIR directory"""
    # Get custom video directory if set
    custom_video_dir = settings_service.get_setting(db, "custom_video_dir")
    
    if custom_video_dir:
        file_path = Path(custom_video_dir) / video_file
    else:
        file_path = settings.VIDEO_DIR / video_file
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{video_file}' not found")
    
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like
    
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Range": f"bytes 0-{file_size-1}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    
    return StreamingResponse(
        iterfile(), 
        media_type="video/mp4", 
        headers=headers
    )

# ---------------- paged list ---------------
@router.get("/videos")
async def get_videos(
    category: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1),
    db: Session = Depends(get_db)
):
    q = db.query(Video)
    if category:
        q = q.filter(Video.category == category)
    offset = (page - 1) * limit
    vids = q.offset(offset).limit(limit).all()
    return {
        "videos": [v.__dict__ for v in vids],
        "total":  q.count(),
        "page":   page,
    }

# ---------------- search -------------------
@router.get("/search")
async def search_videos(
    query: str,
    difficulty: str = Query("all"),
    page: int = Query(1),
    limit: int = Query(20),
    db: Session = Depends(get_db)
):
    term = f"%{query}%"
    q = db.query(Video).filter(
        (Video.title.ilike(term)) | (Video.description.ilike(term))
    )
    if difficulty != "all":
        q = q.filter(Video.difficulty.ilike(difficulty))
    vids = q.offset((page - 1) * limit).limit(limit).all()
    return {
        "videos": [v.__dict__ for v in vids],
        "total":  q.count(),
        "page":   page,
    }

# ---------------- thumbnail serving ---------------
@router.get("/thumbnails/{thumbnail_name}")
async def serve_thumbnail(
    thumbnail_name: str,
    db: Session = Depends(get_db)
):
    """Serve thumbnail images from the configured THUMBNAIL_DIR directory"""
    
    # URL decode the thumbnail name to handle special characters
    decoded_name = urllib.parse.unquote(thumbnail_name)
    
    # Get custom thumbnail directory if set
    custom_thumbnail_dir = settings_service.get_setting(db, "custom_thumbnail_dir")
    
    if custom_thumbnail_dir:
        file_path = Path(custom_thumbnail_dir) / decoded_name
    else:
        file_path = settings.THUMBNAIL_DIR / decoded_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail '{decoded_name}' not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=decoded_name
    )
