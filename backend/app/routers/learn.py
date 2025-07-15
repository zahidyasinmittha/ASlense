from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from pathlib import Path
import csv, os

from moviepy import VideoFileClip  # Correct import for VideoFileClip
from app.db import get_db
from app.models import Video
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

THUMB_DIR = Path(f'{os.getenv("THUMBNAIL_DIR")}')                   
VIDEO_DIR = Path(f'{os.getenv("VIDEO_DIR")}') 
THUMB_DIR.mkdir(exist_ok=True)

# ---------------- CSV importer -------------
@router.post("/import-csv")
async def import_csv(
    file_path: str = Query(..., description="Server-side CSV path"),
    db: Session = Depends(get_db)
):
    try:
        with open(file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                video_name = row["videos"].strip()            # e.g., "A.mp4"
                video_path = VIDEO_DIR / video_name

                # Probe duration & capture thumbnail
                try:
                    with VideoFileClip(str(video_path)) as clip:
                        secs = int(clip.duration)
                        duration = f"{secs}s"
                        thumb_name = Path(video_name).with_suffix(".jpg").name
                        thumb_path = THUMB_DIR / thumb_name
                        clip.save_frame(str(thumb_path), t=min(1, clip.duration / 2))
                except Exception as e:
                    print(f"⚠ {video_name} – {e}")
                    continue

                description = (
                    row.get("description") or
                    f"{row.get('word','').title()} – {row.get('Difficulty','').title()} "
                    f"({row.get('Category','')}) • {duration}"
                )

                db.add(
                    Video(
                        word=row.get("word", ""),
                        title=row.get("title") or row.get("word", ""),
                        description=description,
                        difficulty=row.get("Difficulty", ""),
                        duration=duration,
                        video_file=video_name,   # Save name only (as requested)
                        thumbnail=str(thumb_name),  # Save full relative path
                        category=row.get("Category", ""),
                    )
                )
                # Print update on terminal (as requested)
                print(f"✔ Imported {video_name:20} | Duration: {duration:>5} | Thumbnail: {thumb_path}")
        db.commit()
        return {"message": "CSV imported successfully"}

    except FileNotFoundError:
        return {"error": "CSV not found"}
    
# ---------------- video streaming ---------------
@router.get("/stream/{video_file}")
async def stream_video(video_file: str):
    """Stream video files from the VIDEO_DIR directory"""
    file_path = VIDEO_DIR / video_file
    
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


@router.post("/delete-all-videos")
async def delete_all_videos(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(Video).delete()
        db.commit()
        print(f"Deleted {deleted_count} videos from the database.")  # Terminal print as requested
        return {"message": f"Successfully deleted {deleted_count} videos"}
    except Exception as e:
        db.rollback()  # Undo if error occurs
        print(f"Error deleting videos: {str(e)}")  # Terminal print for errors
        return {"error": str(e)}