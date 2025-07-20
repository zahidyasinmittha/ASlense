from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
import tempfile
import os
import asyncio
import json
import base64
from collections import defaultdict
from typing import List, Dict, Optional

from app.core.database import get_db
from app.models import Video, User, UserProgress
from app.real_model_integration import get_pro_model, get_mini_model, predict_video_with_model
from app.auth import get_current_active_user
from app.schemas import EnhancedPredictionResult, PredictionHistoryCreate, UserProgressUpdate
from app.services.user_service import PredictionService, ProgressService

router = APIRouter()

def filter_error_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Filter out error messages from predictions that indicate video processing issues
    """
    error_messages = {
        'video too short', 'need more frames', 'try longer video', 
        'or different sign', 'insufficient frames', 'video error',
        'processing failed', 'no valid frames'
    }
    
    filtered_predictions = []
    for pred in predictions:
        word = pred.get("word", "").lower().strip()
        # Skip predictions that are error messages
        if word not in error_messages and len(word) > 0:
            filtered_predictions.append(pred)
    
    return filtered_predictions

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


@router.websocket("/live-predict")
async def websocket_live_predict(
    websocket: WebSocket,
    model_type: str = "mini"
):
    """
    WebSocket endpoint for real-time ASL prediction from video stream
    Collects frames, processes in batches, returns aggregated predictions
    """
    await websocket.accept()
    
    try:
        # Validate model type
        if model_type not in ["pro", "mini"]:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid model type. Must be 'pro' or 'mini'"
            })
            await websocket.close()
            return
        
        # Get the appropriate model
        if model_type == "pro":
            model = get_pro_model()
        else:
            model = get_mini_model()
        
        if not model or not model.loaded:
            await websocket.send_json({
                "type": "error",
                "message": f"{model_type} model not loaded"
            })
            await websocket.close()
            return
        
        # Send connection success message
        print(f"🔗 WebSocket connected! Model type: {model_type}")
        await websocket.send_json({
            "type": "connected",
            "message": "Ready to receive frames",
            "model_type": model_type
        })
        
        # Frame collection and prediction storage
        frame_buffer = []
        all_predictions = []
        frame_count = 0
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "frame":
                    # Decode base64 image
                    image_data = data.get("frame", "")
                    print(f"🎬 Frame received from frontend, size: {len(image_data)} bytes")
                    if not image_data:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No frame data provided"
                        })
                        continue
                    
                    # Handle data URL format
                    if "," in image_data:
                        header, image_data = image_data.split(",", 1)
                
                    
                    try:
                        image_bytes = base64.b64decode(image_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frame_buffer.append(frame)
                            frame_count += 1
                            print(f"✅ Frame {frame_count} decoded successfully, buffer size: {len(frame_buffer)}")
                            
                            # Send acknowledgment
                            await websocket.send_json({
                                "type": "frame_received",
                                "frame_count": frame_count
                            })
                            
                            # Process frames when we have exactly 20 frames (immediate processing)
                            # Continue collecting more frames after processing
                            should_process = len(frame_buffer) >= 20
                            
                            if should_process:
                                print(f"🎯 Processing batch of {len(frame_buffer)} frames (20-frame batch)...")
                                try:
                                    # Create temporary video from frames (same as video upload)
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                                        height, width = frame_buffer[0].shape[:2]
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                        # Use consistent FPS for 20-frame batches
                                        fps = 15.0
                                        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
                                        
                                        for buffered_frame in frame_buffer:
                                            out.write(buffered_frame)
                                        out.release()
                                        temp_path = temp_file.name
                                    
                                    # Use the same prediction function as video upload
                                    predictions = predict_video_with_model(temp_path, model_type)
                                    print(f"🤖 Model prediction complete, got {len(predictions)} predictions")
                                    
                                    # Filter out error messages before storing predictions
                                    filtered_predictions = filter_error_predictions(predictions)
                                    if len(predictions) != len(filtered_predictions):
                                        print(f"🧹 Filtered {len(predictions) - len(filtered_predictions)} error predictions")
                                    
                                    # Store all valid predictions (not just top 4) for better aggregation
                                    all_predictions.extend(filtered_predictions)
                                    print(f"📊 Latest predictions: {[p['word'] for p in filtered_predictions[:4]]}")
                                    print(f"📈 Total predictions collected: {len(all_predictions)}")
                                    
                                    # Clean up
                                    os.unlink(temp_path)
                                    frame_buffer.clear()
                                    print(f"🧹 Cleanup complete, buffer cleared")
                                    
                                    # Send progress update with current best predictions
                                    await websocket.send_json({
                                        "type": "progress",
                                        "frames_processed": frame_count,
                                        "predictions_count": len(all_predictions),
                                        "latest_predictions": filtered_predictions[:4],
                                        "buffer_processed": True
                                    })
                                    
                                except Exception as e:
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": f"Processing error: {str(e)}"
                                    })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Failed to decode image frame"
                            })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error", 
                            "message": f"Frame decode error: {str(e)}"
                        })
                
                elif msg_type == "analyze":
                    # Aggregate all predictions
                    target_word = data.get("target_word", "")
                    print(f"🔍 Analysis requested for target word: '{target_word}', total predictions: {len(all_predictions)}")
                    
                    if all_predictions:
                        # Simple aggregation: just sum probabilities for each word
                        word_totals = defaultdict(float)
                        
                        for pred in all_predictions:
                            word = pred.get("word", "")
                            confidence = pred.get("confidence", 0.0)
                            word_totals[word] += confidence
                        
                        # Sort by total probability and get top 4
                        final_predictions = []
                        for word, total_probability in word_totals.items():
                            final_predictions.append({
                                "word": word,
                                "confidence": round(total_probability, 4),
                                "rank": 0  # Will be set after sorting
                            })
                        
                        # Sort by total probability (highest first)
                        final_predictions.sort(key=lambda x: x["confidence"], reverse=True)
                        
                        # Set ranks for ALL predictions (keep all words)
                        for i, pred in enumerate(final_predictions):
                            pred["rank"] = i + 1
                        
                        # Use the same enhancement function as video upload
                        enhanced_result = enhance_prediction_result(final_predictions, target_word, model_type)
                        print(f"🎯 Analysis complete! Top prediction: '{enhanced_result.predictions[0]['word']}' (total probability: {enhanced_result.predictions[0]['confidence']:.4f})")
                        print(f"🏆 Match found: {enhanced_result.is_match}, Match confidence: {enhanced_result.match_confidence:.4f}")
                        print(f"📊 Top 4 words by total probability: {[(p['word'], p['confidence']) for p in final_predictions[:4]]}")
                        
                        await websocket.send_json({
                            "type": "final_result",
                            "result": enhanced_result.dict(),
                            "total_frames": frame_count,
                            "total_predictions": len(all_predictions),
                            "processing_method": "simple_probability_sum"
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No predictions available - no frames were processed"
                        })
                    
                    # Clear for next session
                    all_predictions.clear()
                    frame_count = 0
                    frame_buffer.clear()
                    
                elif msg_type == "stop":
                    print(f"🛑 Stop command received, processing remaining {len(frame_buffer)} frames...")
                    # Process any remaining frames in buffer (even if less than 10)
                    if frame_buffer and len(frame_buffer) > 0:
                        try:
                            # Create temporary video from remaining frames (same as video upload)
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                                height, width = frame_buffer[0].shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                # Adjust FPS based on frame count for optimal processing
                                fps = max(10.0, min(15.0, len(frame_buffer)))
                                out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
                                
                                for buffered_frame in frame_buffer:
                                    out.write(buffered_frame)
                                out.release()
                                temp_path = temp_file.name
                            
                            # Use the same prediction function as video upload
                            predictions = predict_video_with_model(temp_path, model_type)
                            print(f"🤖 Final batch prediction complete, got {len(predictions)} predictions")
                            
                            # Filter out error messages before storing predictions
                            filtered_predictions = filter_error_predictions(predictions)
                            if len(predictions) != len(filtered_predictions):
                                print(f"🧹 Final batch: Filtered {len(predictions) - len(filtered_predictions)} error predictions")
                            
                            # Store all valid predictions for better aggregation
                            all_predictions.extend(filtered_predictions)
                            print(f"📊 Final batch predictions: {[p['word'] for p in filtered_predictions[:4]]}")
                            print(f"📈 Total predictions accumulated: {len(all_predictions)}")
                            
                            # Clean up
                            os.unlink(temp_path)
                            frame_buffer.clear()
                            
                            # Send update
                            await websocket.send_json({
                                "type": "progress",
                                "frames_processed": frame_count,
                                "predictions_count": len(all_predictions),
                                "latest_predictions": filtered_predictions[:4],
                                "final_batch": True
                            })
                        except Exception as e:
                            print(f"❌ Final batch processing error: {str(e)}")
                            pass  # Silently handle errors in final processing
                    
                    # Send ready for analysis message
                    await websocket.send_json({
                        "type": "stopped",
                        "total_predictions": len(all_predictions),
                        "total_frames": frame_count,
                        "message": "Recording stopped. Ready to analyze." if all_predictions else "No frames were captured"
                    })
                
                elif msg_type == "ping":
                    # Handle ping to keep connection alive
                    await websocket.send_json({
                        "type": "pong"
                    })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass  # Silently handle connection errors
    finally:
        # Clean up any remaining frames
        frame_buffer.clear()
        all_predictions.clear()
        try:
            await websocket.close()
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
