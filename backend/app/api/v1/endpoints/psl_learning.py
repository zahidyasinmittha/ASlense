# app/api/v1/endpoints/psl_learning.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import asyncio
import json
import base64
import cv2
import numpy as np
import binascii
import tempfile
import os
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from app.core.database import get_db
from app.auth import get_current_active_user
from app.models import User, PSLAlphabet, PredictionHistory, UserProgress
from app.schemas import PredictionHistoryCreate, UserProgressUpdate
from app.services.psl_service import get_psl_service
from app.services.user_service import PredictionService, ProgressService

router = APIRouter()

# PSL Learning WebSocket Configuration
PSL_FRAME_INTERVAL = 400  # 400ms between predictions (2.5 FPS)
PSL_PREDICTION_BUFFER_SIZE = 10  # Keep last 10 predictions for smoothing
PSL_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for reliable prediction

class PSLConnectionManager:
    """Manage PSL learning WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_sessions[session_id] = {
            "websocket": websocket,
            "frame_count": 0,
            "predictions": [],
            "start_time": datetime.now(),
            "model_type": "mini",
            "target_letter": None
        }
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
    
    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.user_sessions:
            websocket = self.user_sessions[session_id]["websocket"]
            try:
                await websocket.send_json(message)
            except:
                pass

manager = PSLConnectionManager()

@router.get("/status")
async def get_psl_status():
    """Get PSL learning module status"""
    psl_service = get_psl_service()
    return {
        "status": "active",
        "frame_interval": PSL_FRAME_INTERVAL,
        "prediction_frequency": f"{1000/PSL_FRAME_INTERVAL:.1f} FPS",
        "models": psl_service.get_model_status(),
        "confidence_threshold": PSL_CONFIDENCE_THRESHOLD,
        "buffer_size": PSL_PREDICTION_BUFFER_SIZE
    }

@router.get("/alphabet")
async def get_psl_alphabet(
    skip: int = 0,
    limit: int = 50,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get PSL alphabet data for learning"""
    query = db.query(PSLAlphabet).filter(PSLAlphabet.is_active == True)
    
    if difficulty:
        query = query.filter(PSLAlphabet.difficulty == difficulty.lower())
    
    alphabet_data = query.order_by(PSLAlphabet.letter).offset(skip).limit(limit).all()
    
    return {
        "alphabet": alphabet_data,
        "total": query.count(),
        "skip": skip,
        "limit": limit
    }

@router.get("/alphabet/{letter}")
async def get_psl_letter_info(
    letter: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific PSL letter"""
    alphabet_entry = db.query(PSLAlphabet).filter(
        PSLAlphabet.letter.ilike(letter.upper()),
        PSLAlphabet.is_active == True
    ).first()
    
    if not alphabet_entry:
        raise HTTPException(status_code=404, detail=f"PSL letter '{letter}' not found")
    
    return alphabet_entry

@router.post("/practice-image")
async def practice_psl_image(
    image_file: UploadFile = File(...),
    target_letter: str = Form(...),
    model_type: str = Form(default="mini"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Practice PSL recognition with uploaded image
    Supports both MobileNet (mini) and EfficientNet (pro) models
    """
    
    # Validate inputs
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model_type not in ["mini", "pro"]:
        raise HTTPException(status_code=400, detail="Model type must be 'mini' or 'pro'")
    
    if len(target_letter) != 1:
        raise HTTPException(status_code=400, detail="Target letter must be a single character")
    
    try:
        # Get PSL service
        psl_service = get_psl_service()
        
        if not psl_service.is_model_loaded(model_type):
            raise HTTPException(
                status_code=503, 
                detail=f"PSL {model_type} model not available"
            )
        
        # Read and process image
        image_content = await image_file.read()
        nparr = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Get predictions
        predictions = psl_service.predict_frame(image, model_type)
        
        # Check if target letter is in predictions
        target_letter_upper = target_letter.upper()
        is_correct = False
        match_confidence = 0.0
        rank_in_predictions = None
        
        for i, pred in enumerate(predictions):
            if pred["letter"].upper() == target_letter_upper:
                is_correct = True
                match_confidence = pred["confidence"]
                rank_in_predictions = i + 1
                break
        
        # Enhanced result
        result = {
            "target_letter": target_letter_upper,
            "predictions": predictions,
            "is_correct": is_correct,
            "match_confidence": match_confidence,
            "rank_in_predictions": rank_in_predictions,
            "model_used": model_type,
            "processing_time": "< 100ms"
        }
        
        # Save to user history if authenticated
        if current_user:
            prediction_service = PredictionService(db)
            progress_service = ProgressService(db)
            
            # Save prediction history
            prediction_data = PredictionHistoryCreate(
                user_id=current_user.id,
                target_word=target_letter_upper,
                predicted_words=json.dumps(predictions),
                is_correct=is_correct,
                confidence_score=match_confidence,
                model_used=f"psl_{model_type}",
                practice_mode="image_upload"
            )
            prediction_service.save_prediction(prediction_data)
            
            # Update user progress
            progress_update = UserProgressUpdate(
                is_correct=is_correct,
                practice_time=10,  # Estimated time for image upload
                target_word=target_letter_upper
            )
            progress_service.update_progress(current_user.id, progress_update)
        
        return result
        
    except Exception as e:
        print(f"❌ PSL image practice error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def smooth_predictions(recent_predictions: List[List[Dict]], window_size: int = 5) -> List[Dict]:
    """
    Smooth PSL predictions over time using weighted averaging
    Recent predictions get higher weight
    """
    if not recent_predictions:
        return [
            {"letter": "No data", "confidence": 0.0, "rank": 1},
            {"letter": "Processing", "confidence": 0.0, "rank": 2},
            {"letter": "Wait", "confidence": 0.0, "rank": 3},
            {"letter": "Please", "confidence": 0.0, "rank": 4}
        ]
    
    # Use only recent predictions for smoothing
    predictions_to_use = recent_predictions[-window_size:]
    
    # Aggregate letter scores with time-based weighting
    letter_scores = defaultdict(float)
    total_weight = 0
    
    for i, predictions in enumerate(predictions_to_use):
        # More recent predictions get higher weight
        weight = (i + 1) / len(predictions_to_use)
        total_weight += weight
        
        for pred in predictions:
            letter_scores[pred["letter"]] += pred["confidence"] * weight
    
    # Normalize scores
    if total_weight > 0:
        for letter in letter_scores:
            letter_scores[letter] /= total_weight
    
    # Sort and get top 4
    sorted_letters = sorted(letter_scores.items(), key=lambda x: x[1], reverse=True)
    
    smoothed_predictions = []
    for i, (letter, score) in enumerate(sorted_letters[:4]):
        smoothed_predictions.append({
            "letter": letter,
            "confidence": round(score, 4),
            "rank": i + 1,
            "smoothed": True
        })
    
    # Ensure we have 4 predictions
    while len(smoothed_predictions) < 4:
        smoothed_predictions.append({
            "letter": f"Option{len(smoothed_predictions) + 1}",
            "confidence": 0.0,
            "rank": len(smoothed_predictions) + 1,
            "smoothed": True
        })
    
    return smoothed_predictions

async def process_psl_frame_async(
    session_id: str, 
    frame: np.ndarray, 
    model_type: str
) -> Optional[Dict]:
    """
    Process PSL frame asynchronously with proper error handling
    """
    try:
        psl_service = get_psl_service()
        
        if not psl_service.is_model_loaded(model_type):
            return {
                "type": "error",
                "message": f"PSL {model_type} model not loaded"
            }
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def predict():
            return psl_service.predict_frame(frame, model_type)
        
        # Use thread pool for CPU-intensive prediction
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            predictions = await loop.run_in_executor(executor, predict)
        
        return {
            "type": "prediction",
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "frame_processed": True
        }
        
    except Exception as e:
        print(f"❌ PSL frame processing error: {e}")
        return {
            "type": "error",
            "message": f"Frame processing failed: {str(e)}"
        }

@router.websocket("/live-learn")
async def websocket_psl_live_learning(
    websocket: WebSocket,
    model_type: str = "mini",
    target_letter: Optional[str] = None
):
    """
    WebSocket endpoint for live PSL learning with 400ms intervals
    Optimized for real-time letter recognition learning
    """
    session_id = f"psl_{datetime.now().timestamp()}"
    
    try:
        await manager.connect(websocket, session_id)
        
        # Validate model type
        if model_type not in ["mini", "pro"]:
            await manager.send_personal_message({
                "type": "error",
                "message": "Invalid model type. Must be 'mini' or 'pro'"
            }, session_id)
            return
        
        # Get PSL service
        psl_service = get_psl_service()
        
        if not psl_service.is_model_loaded(model_type):
            await manager.send_personal_message({
                "type": "error",
                "message": f"PSL {model_type} model not available"
            }, session_id)
            return
        
        # Send connection success
        await manager.send_personal_message({
            "type": "connected",
            "message": f"PSL Learning connected with {model_type} model",
            "model_type": model_type,
            "frame_interval": PSL_FRAME_INTERVAL,
            "target_letter": target_letter,
            "session_id": session_id
        }, session_id)
        
        # Update session info
        manager.user_sessions[session_id]["model_type"] = model_type
        manager.user_sessions[session_id]["target_letter"] = target_letter
        
        # Frame processing variables
        last_prediction_time = 0
        prediction_buffer = []
        
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                msg_type = data.get("type")
                
                if msg_type == "frame":
                    current_time = asyncio.get_event_loop().time() * 1000  # ms
                    
                    # Check if enough time has passed (400ms interval)
                    if current_time - last_prediction_time < PSL_FRAME_INTERVAL:
                        # Send acknowledgment but skip processing
                        await manager.send_personal_message({
                            "type": "frame_received",
                            "skipped": True,
                            "wait_time": PSL_FRAME_INTERVAL - (current_time - last_prediction_time)
                        }, session_id)
                        continue
                    
                    # Process frame
                    image_data = data.get("frame", "")
                    
                    if not image_data:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "No frame data provided"
                        }, session_id)
                        continue
                    
                    # Handle data URL format
                    if "," in image_data:
                        header, image_data = image_data.split(",", 1)
                    
                    try:
                        # Decode image
                        image_bytes = base64.b64decode(image_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Update frame count
                            manager.user_sessions[session_id]["frame_count"] += 1
                            frame_count = manager.user_sessions[session_id]["frame_count"]
                            
                            # Process frame asynchronously
                            result = await process_psl_frame_async(session_id, frame, model_type)
                            
                            if result and result.get("type") == "prediction":
                                # Add to prediction buffer
                                prediction_buffer.append(result["predictions"])
                                
                                # Keep buffer size manageable
                                if len(prediction_buffer) > PSL_PREDICTION_BUFFER_SIZE:
                                    prediction_buffer.pop(0)
                                
                                # Generate smoothed predictions
                                smoothed_predictions = smooth_predictions(prediction_buffer)
                                
                                # Check if target letter matches
                                target_match = None
                                if target_letter:
                                    for i, pred in enumerate(smoothed_predictions):
                                        if pred["letter"].upper() == target_letter.upper():
                                            target_match = {
                                                "matched": True,
                                                "confidence": pred["confidence"],
                                                "rank": pred["rank"],
                                                "is_high_confidence": pred["confidence"] > PSL_CONFIDENCE_THRESHOLD
                                            }
                                            break
                                    
                                    if not target_match:
                                        target_match = {"matched": False}
                                
                                # Send prediction result
                                await manager.send_personal_message({
                                    "type": "live_prediction",
                                    "predictions": result["predictions"],  # Raw predictions
                                    "smoothed_predictions": smoothed_predictions,  # Smoothed over time
                                    "target_match": target_match,
                                    "frame_count": frame_count,
                                    "buffer_size": len(prediction_buffer),
                                    "model_type": model_type,
                                    "timestamp": result["timestamp"]
                                }, session_id)
                                
                                # High confidence alert
                                if smoothed_predictions[0]["confidence"] > PSL_CONFIDENCE_THRESHOLD:
                                    await manager.send_personal_message({
                                        "type": "high_confidence",
                                        "letter": smoothed_predictions[0]["letter"],
                                        "confidence": smoothed_predictions[0]["confidence"],
                                        "message": f"High confidence: {smoothed_predictions[0]['letter']}"
                                    }, session_id)
                            
                            elif result and result.get("type") == "error":
                                await manager.send_personal_message(result, session_id)
                            
                            # Update last prediction time
                            last_prediction_time = current_time
                            
                        else:
                            await manager.send_personal_message({
                                "type": "error",
                                "message": "Failed to decode frame"
                            }, session_id)
                    
                    except binascii.Error:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Invalid base64 image data"
                        }, session_id)
                    except Exception as e:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Frame processing error: {str(e)}"
                        }, session_id)
                
                elif msg_type == "change_target":
                    # Change target letter during session
                    new_target = data.get("target_letter", "")
                    manager.user_sessions[session_id]["target_letter"] = new_target
                    
                    await manager.send_personal_message({
                        "type": "target_changed",
                        "target_letter": new_target,
                        "message": f"Target changed to: {new_target}"
                    }, session_id)
                
                elif msg_type == "change_model":
                    # Change model during session
                    new_model_type = data.get("model_type", "mini")
                    
                    if new_model_type in ["mini", "pro"] and psl_service.is_model_loaded(new_model_type):
                        manager.user_sessions[session_id]["model_type"] = new_model_type
                        model_type = new_model_type  # Update local variable
                        
                        # Clear prediction buffer when changing models
                        prediction_buffer.clear()
                        
                        await manager.send_personal_message({
                            "type": "model_changed",
                            "model_type": new_model_type,
                            "message": f"Model changed to: {new_model_type}",
                            "buffer_cleared": True
                        }, session_id)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Model {new_model_type} not available"
                        }, session_id)
                
                elif msg_type == "get_stats":
                    # Send session statistics
                    session_data = manager.user_sessions[session_id]
                    session_duration = (datetime.now() - session_data["start_time"]).total_seconds()
                    
                    await manager.send_personal_message({
                        "type": "session_stats",
                        "frame_count": session_data["frame_count"],
                        "session_duration": round(session_duration, 1),
                        "predictions_made": len(prediction_buffer),
                        "avg_fps": round(session_data["frame_count"] / max(session_duration, 1), 2),
                        "target_letter": session_data["target_letter"],
                        "model_type": session_data["model_type"]
                    }, session_id)
                
                elif msg_type == "ping":
                    # Keep alive ping
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, session_id)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    }, session_id)
                    
            except asyncio.TimeoutError:
                # Send ping to check if connection is still alive
                await manager.send_personal_message({
                    "type": "ping_check",
                    "message": "Checking connection..."
                }, session_id)
                
            except WebSocketDisconnect:
                break
                
            except Exception as e:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                }, session_id)
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ PSL WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, session_id)

@router.get("/learning-progress/{user_id}")
async def get_user_psl_progress(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's PSL learning progress"""
    
    # Check if user can access this data
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get PSL-specific prediction history
    psl_predictions = db.query(PredictionHistory).filter(
        PredictionHistory.user_id == user_id,
        PredictionHistory.model_used.like("psl_%")
    ).all()
    
    # Calculate PSL-specific stats
    total_psl_practice = len(psl_predictions)
    correct_psl = sum(1 for p in psl_predictions if p.is_correct)
    psl_accuracy = (correct_psl / total_psl_practice * 100) if total_psl_practice > 0 else 0
    
    # Get letter-wise performance
    letter_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for pred in psl_predictions:
        letter = pred.target_word
        letter_stats[letter]["total"] += 1
        if pred.is_correct:
            letter_stats[letter]["correct"] += 1
    
    # Calculate letter accuracies
    letter_performance = {}
    for letter, stats in letter_stats.items():
        letter_performance[letter] = {
            "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            "total_attempts": stats["total"],
            "correct_attempts": stats["correct"]
        }
    
    return {
        "user_id": user_id,
        "psl_learning_stats": {
            "total_practice_sessions": total_psl_practice,
            "correct_predictions": correct_psl,
            "accuracy_percentage": round(psl_accuracy, 1),
            "letters_practiced": len(letter_stats),
            "letter_performance": letter_performance
        },
        "recent_practice": [
            {
                "target_letter": p.target_word,
                "is_correct": p.is_correct,
                "confidence": p.confidence_score,
                "model_used": p.model_used,
                "timestamp": p.timestamp
            } for p in psl_predictions[-10:]  # Last 10 attempts
        ]
    }
