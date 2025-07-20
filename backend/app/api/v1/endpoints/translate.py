from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, timedelta
import json
import asyncio
import random

from app.core.database import get_db
from app.auth import get_current_user
from app.models import User, TranslationSession, TranslationHistory
from app.services.user_service import UserService
from app.schemas import UserProgressUpdate
from app.inference import run_inference

router = APIRouter()

# In-memory store for active translation sessions
active_sessions = {}

@router.get("/")
async def translate_info():
    """Get information about translation features."""
    return {
        "message": "Real-time ASL Translation Service",
        "status": "active",
        "features": ["ASL to text", "Real-time translation", "Session tracking"],
        "models": ["mediapipe", "openpose", "custom"]
    }

@router.post("/start-session")
async def start_translation_session(
    model_type: str = "mediapipe",
    input_mode: str = "word",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new translation session."""
    try:
        # Create session record
        session = TranslationSession(
            user_id=current_user.id,
            model_type=model_type,
            input_mode=input_mode,
            started_at=datetime.utcnow(),
            translations_count=0,
            correct_translations=0,
            total_confidence=0.0
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Store in active sessions
        session_id = f"session_{current_user.id}_{session.id}"
        active_sessions[session_id] = {
            "session_id": session.id,
            "user_id": current_user.id,
            "model_type": model_type,
            "input_mode": input_mode,
            "started_at": datetime.utcnow(),
            "translations": [],
            "accuracy_sum": 0,
            "translation_count": 0
        }
        
        return {
            "session_id": session_id,
            "message": "Translation session started",
            "model_type": model_type,
            "input_mode": input_mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.post("/translate")
async def translate_sign(
    session_id: str,
    video_file: UploadFile = File(...),
    target_text: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Translate ASL sign to text."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        
        # Run inference
        result = await run_inference(video_file, session_data["model_type"])
        
        # Calculate accuracy if target text provided
        confidence = result.get("confidence", 0.0)
        predicted_text = result.get("predicted_text", "")
        is_correct = False
        
        if target_text:
            is_correct = predicted_text.lower().strip() == target_text.lower().strip()
        
        # Update session data
        session_data["translations"].append({
            "predicted_text": predicted_text,
            "confidence": confidence,
            "target_text": target_text,
            "is_correct": is_correct,
            "timestamp": datetime.utcnow().isoformat()
        })
        session_data["translation_count"] += 1
        session_data["accuracy_sum"] += confidence
        
        # Save to database
        translation_record = TranslationHistory(
            user_id=current_user.id,
            session_id=session_data["session_id"],
            predicted_text=predicted_text,
            target_text=target_text,
            confidence=confidence,
            is_correct=is_correct,
            model_used=session_data["model_type"],
            input_mode=session_data["input_mode"]
        )
        db.add(translation_record)
        
        # Update session record
        session_record = db.query(TranslationSession).filter(
            TranslationSession.id == session_data["session_id"]
        ).first()
        if session_record:
            session_record.translations_count += 1
            session_record.total_confidence += confidence
            if is_correct:
                session_record.correct_translations += 1
        
        db.commit()
        
        # Calculate current session accuracy
        session_accuracy = (session_data["accuracy_sum"] / session_data["translation_count"]) * 100
        
        return {
            "predicted_text": predicted_text,
            "confidence": confidence,
            "is_correct": is_correct,
            "session_accuracy": round(session_accuracy, 2),
            "translations_count": session_data["translation_count"],
            "model_used": session_data["model_type"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@router.get("/session-stats/{session_id}")
async def get_session_stats(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current session statistics."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    if session_data["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access to session")
    
    # Calculate session stats
    translation_count = session_data["translation_count"]
    avg_accuracy = (session_data["accuracy_sum"] / translation_count) if translation_count > 0 else 0
    session_time = (datetime.utcnow() - session_data["started_at"]).total_seconds()
    
    correct_count = sum(1 for t in session_data["translations"] if t["is_correct"])
    accuracy_percentage = (correct_count / translation_count * 100) if translation_count > 0 else 0
    
    return {
        "translations": translation_count,
        "accuracy": round(avg_accuracy, 2),
        "session_time": int(session_time),
        "correct_translations": correct_count,
        "accuracy_percentage": round(accuracy_percentage, 2),
        "model_type": session_data["model_type"],
        "input_mode": session_data["input_mode"]
    }

@router.post("/end-session/{session_id}")
async def end_translation_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End translation session and update user progress."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        
        if session_data["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Unauthorized access to session")
        
        # Update session record
        session_record = db.query(TranslationSession).filter(
            TranslationSession.id == session_data["session_id"]
        ).first()
        
        if session_record:
            session_record.ended_at = datetime.utcnow()
            session_duration = (session_record.ended_at - session_record.started_at).total_seconds()
            session_record.session_duration = int(session_duration)
            
            # Calculate final accuracy
            if session_record.translations_count > 0:
                session_record.average_confidence = session_record.total_confidence / session_record.translations_count
                session_record.accuracy_percentage = (session_record.correct_translations / session_record.translations_count) * 100
        
        # Update user progress
        user_service = UserService(db)
        translation_count = session_data["translation_count"]
        correct_count = sum(1 for t in session_data["translations"] if t["is_correct"])
        is_accurate_session = (correct_count / translation_count) >= 0.7 if translation_count > 0 else False
        
        progress_update = UserProgressUpdate(
            is_correct=is_accurate_session,
            practice_time=int((datetime.utcnow() - session_data["started_at"]).total_seconds())
        )
        user_service.update_progress(current_user.id, progress_update)
        
        db.commit()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return {
            "message": "Session ended successfully",
            "translations_count": translation_count,
            "correct_translations": correct_count,
            "session_duration": int((datetime.utcnow() - session_data["started_at"]).total_seconds()),
            "final_accuracy": round((correct_count / translation_count * 100) if translation_count > 0 else 0, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@router.get("/recent-history")
async def get_recent_translations(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recent translation history."""
    try:
        translations = db.query(TranslationHistory).filter(
            TranslationHistory.user_id == current_user.id
        ).order_by(TranslationHistory.created_at.desc()).limit(limit).all()
        
        return [
            {
                "predicted_text": t.predicted_text,
                "target_text": t.target_text,
                "confidence": t.confidence,
                "is_correct": t.is_correct,
                "model_used": t.model_used,
                "input_mode": t.input_mode,
                "timestamp": t.created_at.isoformat()
            }
            for t in translations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

# Simulate translation for demo purposes
async def simulate_translation(text_input: str, model_type: str, input_mode: str):
    """Simulate ASL translation for demo purposes."""
    # Common ASL words and phrases
    asl_vocabulary = [
        "hello", "thank you", "please", "sorry", "yes", "no", "good", "bad",
        "happy", "sad", "love", "family", "friend", "help", "water", "food",
        "how are you", "nice to meet you", "good morning", "good night",
        "see you later", "take care", "i love you", "thank you very much"
    ]
    
    # Simulate processing delay
    await asyncio.sleep(random.uniform(0.5, 2.0))
    
    # Generate prediction
    if input_mode == "word":
        predicted = random.choice(asl_vocabulary[:16])  # Single words
    else:
        predicted = random.choice(asl_vocabulary[16:])  # Phrases
    
    # Simulate confidence based on model
    base_confidence = {
        "mediapipe": 0.92,
        "openpose": 0.87,
        "custom": 0.89
    }.get(model_type, 0.85)
    
    confidence = base_confidence + random.uniform(-0.15, 0.08)
    confidence = max(0.60, min(0.98, confidence))
    
    return {
        "predicted_text": predicted,
        "confidence": confidence
    }