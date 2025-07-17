from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
import tempfile
import os
import asyncio
import json
import base64
from typing import List, Dict, Optional
import threading
import time

from app.core.database import get_db
from app.models import Video, User, UserProgress
from app.real_model_integration import get_pro_model, get_mini_model, predict_video_with_model, predict_frame_with_model
from app.auth import get_current_active_user
from app.schemas import EnhancedPredictionResult, PredictionHistoryCreate, UserProgressUpdate
from app.services.user_service import PredictionService, ProgressService

router = APIRouter()

def enhance_prediction_result(predictions: List[Dict], target_word: str, model_used: str) -> EnhancedPredictionResult:
    """
    Enhance prediction result to check if target word is in top 4
    and mark as correct if it is.
    """
    # Ensure we have at least 4 predictions, pad with empty if needed
    while len(predictions) < 4:
        predictions.append({"word": "", "confidence": 0.0, "rank": len(predictions) + 1})
    
    top_4_predictions = predictions[:4]
    top_2_predictions = predictions[:2]
    
    # Check if target word is in top 4
    target_word_lower = target_word.lower()
    is_correct = False
    rank_in_top_4 = None
    match_confidence = 0.0
    
    for i, pred in enumerate(top_4_predictions):
        if pred["word"].lower() == target_word_lower:
            is_correct = True
            rank_in_top_4 = i + 1
            match_confidence = pred["confidence"]
            break
    
    # Traditional exact match check for backward compatibility
    is_match = any(pred["word"].lower() == target_word_lower for pred in top_2_predictions)
    
    return EnhancedPredictionResult(
        target_word=target_word,
        predictions=predictions,
        top_predictions=top_2_predictions,
        is_match=is_match,
        is_correct=is_correct,
        match_confidence=match_confidence,
        model_used=model_used,
        rank_in_top_4=rank_in_top_4
    )

@router.get("/ping")
async def practice_ping():
    return {"msg": "Practice endpoint works"}

@router.post("/predict-video")
async def predict_video(
    video_file: UploadFile = File(...),
    target_word: str = Form(...),
    model_type: str = Form(default="mini"),  # "pro" or "mini"
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict ASL sign from uploaded video with enhanced result tracking
    Returns top 4 predictions and checks if target word matches
    """
    
    # Validate inputs
    if not video_file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    if model_type not in ["pro", "mini"]:
        raise HTTPException(status_code=400, detail="Model type must be 'pro' or 'mini'")
    
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video_file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        # Process video and get predictions using real models
        predictions = predict_video_with_model(temp_video_path, model_type)
        
        # Enhance prediction result
        enhanced_result = enhance_prediction_result(predictions, target_word, model_type)
        
        # Save prediction to history if user is authenticated
        if current_user:
            prediction_service = PredictionService(db)
            progress_service = ProgressService(db)
            
            # Save prediction history
            prediction_data = PredictionHistoryCreate(
                user_id=current_user.id,
                target_word=target_word,
                predicted_words=json.dumps(predictions),
                is_correct=enhanced_result.is_correct,
                confidence_score=enhanced_result.match_confidence,
                model_used=model_type,
                practice_mode="upload"
            )
            prediction_service.save_prediction(prediction_data)
            
            # Update user progress
            progress_update = UserProgressUpdate(
                is_correct=enhanced_result.is_correct,
                practice_time=30,  # Estimated time for video upload
                target_word=target_word
            )
            progress_service.update_progress(current_user.id, progress_update)
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        return enhanced_result
    
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
        print(f"Error in video prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

async def process_frame_with_model(frame, model_type: str = "mini") -> List[Dict]:
    """
    Process a single frame with the model for live prediction
    """
    try:
        predictions = predict_frame_with_model(frame, model_type)
        return predictions
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return [{"word": "Error", "confidence": 0.0, "rank": 1}]

@router.get("/available-words")
async def get_available_words(db: Session = Depends(get_db)):
    """
    Get list of available words/signs that can be practiced
    """
    try:
        # Get unique words from the database
        words = db.query(Video.word).filter(Video.word.isnot(None)).distinct().all()
        word_list = [word[0] for word in words if word[0]]
        
        # Also include some common ASL words
        common_words = [
            "Hello", "Thank You", "Please", "Sorry", "Yes", "No", 
            "Good", "Bad", "Happy", "Sad", "Love", "Family", 
            "Friend", "Help", "Water", "Food", "Home", "Work"
        ]
        
        # Combine and remove duplicates
        all_words = list(set(word_list + common_words))
        all_words.sort()
        
        return {
            "words": all_words,
            "total": len(all_words),
            "database_words": len(word_list),
            "common_words": len(common_words)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available words: {str(e)}")

@router.get("/models/status")
async def get_models_status():
    """
    Check the status of available models
    """
    try:
        pro_model = get_pro_model()
        mini_model = get_mini_model()
    except Exception:
        pro_model = None
        mini_model = None
    
    status = {
        "pro_model": {
            "name": "AS Pro",
            "type": "ensemble",
            "loaded": pro_model is not None and pro_model.loaded,
            "accuracy": "90%+",
            "speed": "medium",
            "description": "High accuracy ensemble model for complex signs"
        },
        "mini_model": {
            "name": "AS Mini", 
            "type": "fast",
            "loaded": mini_model is not None and mini_model.loaded,
            "accuracy": "85%+", 
            "speed": "fast",
            "description": "Optimized fast model for real-time prediction"
        }
    }
    
    return status

@router.post("/predict-frames")
async def predict_frames(
    request_data: dict,
    db: Session = Depends(get_db)
):
    """
    Predict ASL sign from a sequence of frames (base64 encoded)
    Used for live camera capture
    """
    try:
        frames = request_data.get("frames", [])
        target_word = request_data.get("target_word", "")
        model_type = request_data.get("model_type", "mini")
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames provided")
        
        if model_type not in ["pro", "mini"]:
            raise HTTPException(status_code=400, detail="Model type must be 'pro' or 'mini'")
        
        # Convert base64 frames to OpenCV format
        cv_frames = []
        for frame_b64 in frames:
            # Decode base64 to image
            frame_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv_frames.append(frame)
        
        if not cv_frames:
            raise HTTPException(status_code=400, detail="No valid frames found")
        
        # Use the last frame for prediction (could be enhanced to use sequence)
        predictions = predict_frame_with_model(cv_frames[-1], model_type)
        
        # Enhance prediction result
        enhanced_result = enhance_prediction_result(predictions, target_word, model_type)
        
        return enhanced_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame prediction failed: {str(e)}")

@router.websocket("/live-predict")
async def websocket_live_predict(websocket: WebSocket):
    """
    WebSocket endpoint for live ASL sign prediction
    Accepts frames and returns real-time predictions
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "stop":
                break
            elif message.get("type") == "frame":
                # Extract frame data
                frame_b64 = message.get("frame", "")
                target_word = message.get("target_word", "")
                model_type = message.get("model_type", "mini")
                
                if frame_b64:
                    try:
                        # Decode base64 frame
                        frame_data = base64.b64decode(frame_b64)
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Get predictions
                            predictions = predict_frame_with_model(frame, model_type)
                            
                            # Send predictions back to frontend
                            response = {
                                "type": "predictions",
                                "predictions": predictions[:4],
                                "timestamp": time.time()
                            }
                            
                            await websocket.send_text(json.dumps(response))
                    
                    except Exception as e:
                        error_response = {
                            "type": "error",
                            "message": f"Frame processing error: {str(e)}",
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(error_response))
                        
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            error_response = {
                "type": "error", 
                "message": f"WebSocket error: {str(e)}",
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(error_response))
        except:
            pass


@router.post("/predict-with-user")
async def predict_with_user(
    prediction_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Save user prediction results and update progress/XP
    """
    try:
        # Extract data from request
        target_word = prediction_data.get("target_word", "")
        predicted_word = prediction_data.get("predicted_word", "")
        confidence = prediction_data.get("confidence", 0.0)
        is_correct = prediction_data.get("is_correct", False)
        model_used = prediction_data.get("model_used", "mini")
        practice_mode = prediction_data.get("practice_mode", "upload")  # Default to upload
        top_4_predictions = prediction_data.get("top_4_predictions", [])
        
        # Save prediction history
        prediction_service = PredictionService(db)
        progress_service = ProgressService(db)
        
        prediction_history = PredictionHistoryCreate(
            user_id=current_user.id,
            target_word=target_word,
            predicted_words=json.dumps(top_4_predictions),
            is_correct=is_correct,
            confidence_score=confidence,
            model_used=model_used,
            practice_mode=practice_mode
        )
        
        prediction_service.save_prediction(prediction_history)
        
        # Calculate XP and level progression
        xp_gained = 10 if is_correct else 5  # Base XP
        if confidence > 0.8:
            xp_gained += 5  # Bonus for high confidence
        
        # Update user progress
        progress_update = UserProgressUpdate(
            is_correct=is_correct,
            practice_time=30,  # Estimated time
            target_word=target_word
        )
        
        # Get or create user progress
        user_progress = db.query(UserProgress).filter(UserProgress.user_id == current_user.id).first()
        if not user_progress:
            # Create new progress record if it doesn't exist
            user_progress = UserProgress(
                user_id=current_user.id,
                level="Beginner",
                experience_points=0
            )
            db.add(user_progress)
            db.flush()  # Flush to get the ID
        
        old_level = user_progress.level
        old_xp = user_progress.experience_points or 0
        
        progress_service.update_progress(current_user.id, progress_update)
        
        # Add XP to user progress
        new_xp = old_xp + xp_gained
        user_progress.experience_points = new_xp
        
        # Calculate new level (100 XP per level)
        level_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
        level_index = min(new_xp // 100, len(level_names) - 1)
        new_level = level_names[level_index]
        level_up = new_level != old_level
        
        if level_up:
            user_progress.level = new_level
        
        db.commit()
        
        print(f"Progress updated for user {current_user.id}: XP gained={xp_gained}, Total XP={new_xp}, Correct={is_correct}, Target word={target_word}")
        
        return {
            "success": True,
            "xp_gained": xp_gained,
            "total_xp": new_xp,
            "new_level": new_level if level_up else None,
            "level_up": level_up
        }
        
    except Exception as e:
        print(f"Error saving user prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to save prediction")
