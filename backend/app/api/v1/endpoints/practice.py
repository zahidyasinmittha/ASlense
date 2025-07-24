from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import cv2
import numpy as np
import tempfile
import os
import json
import base64
import binascii
import torch
import asyncio
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from app.core.database import get_db
from app.models import Video, User, UserProgress
from app.real_model_integration import get_pro_model, get_mini_model, predict_video_with_model
from app.auth import get_current_active_user
from app.schemas import EnhancedPredictionResult, PredictionHistoryCreate, UserProgressUpdate
from app.services.user_service import PredictionService, ProgressService

router = APIRouter()
logger = logging.getLogger(__name__)

# Performance constants
BATCH_SIZE = 15  # Reduced for faster processing
FRAME_SKIP = 1   # Process every 2nd frame for speed (restored to previous)
MAX_BUFFER_SIZE = 400  # Reduced memory usage
HRNET_BATCH_SIZE = 8  # Process HRNet in batches
MEMORY_CLEANUP_INTERVAL = 50  # Clean memory every N frames
ASYNC_PROCESSING_THRESHOLD = 10  # Start async processing after N frames

def process_hrnet_batch(frames, hrnet_model):
    """
    OPTIMIZED: Custom batch processing function for HRNet keypoint extraction with memory management
    """
    try:
        if len(frames) == 0:
            return []
            
        # MEMORY OPTIMIZATION: Process in smaller chunks to avoid memory issues
        batch_size = min(HRNET_BATCH_SIZE, len(frames))
        all_keypoints = []
        
        # Pre-allocate tensors for better memory efficiency
        device = next(hrnet_model.parameters()).device if hasattr(hrnet_model, 'parameters') else 'cpu'
        
        # Convert frames to tensors in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_tensors = []
            
            # OPTIMIZED: Efficient tensor conversion
            for frame in batch_frames:
                # Normalize to [0, 1] and convert to CHW format
                frame_normalized = frame.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).to(device)
                batch_tensors.append(frame_tensor)
            
            if len(batch_tensors) > 1:
                # Stack into batch tensor (N, C, H, W)
                with torch.no_grad():
                    batch_input = torch.stack(batch_tensors)
                    
                    # OPTIMIZED: Process batch through HRNet with memory management
                    if hasattr(hrnet_model, 'forward'):
                        batch_output = hrnet_model.forward(batch_input)
                    else:
                        batch_output = hrnet_model(batch_input)
                    
                    # MEMORY: Move to CPU immediately to free GPU memory
                    batch_output_cpu = batch_output.cpu()
                    
                    # Extract keypoints for each frame
                    for j in range(batch_output_cpu.shape[0]):
                        frame_output = batch_output_cpu[j].numpy()
                        # Convert to expected keypoint format
                        if len(frame_output.shape) == 3:  # (C, H, W)
                            keypoints = extract_keypoints_from_heatmap(frame_output)
                        else:
                            keypoints = frame_output.reshape(-1, 3)
                        all_keypoints.append(keypoints)
                    
                    # MEMORY: Delete tensors to free memory immediately
                    del batch_input, batch_output, batch_output_cpu
                    
            else:
                # Single frame processing
                with torch.no_grad():
                    single_input = batch_tensors[0].unsqueeze(0)
                    if hasattr(hrnet_model, 'forward'):
                        single_output = hrnet_model.forward(single_input)
                    else:
                        single_output = hrnet_model(single_input)
                    
                    frame_output = single_output[0].cpu().numpy()
                    if len(frame_output.shape) == 3:
                        keypoints = extract_keypoints_from_heatmap(frame_output)
                    else:
                        keypoints = frame_output.reshape(-1, 3)
                    all_keypoints.append(keypoints)
                    
                    # MEMORY: Clean up
                    del single_input, single_output
            
            # MEMORY: Clear batch tensors
            del batch_tensors
        
        # OPTIMIZATION: Force garbage collection for large batches
        if len(frames) > 10:
            optimize_memory_usage()
            
        return all_keypoints
        
    except Exception as e:
        print(f"Optimized batch HRNet processing error: {e}")
        return []

def extract_keypoints_from_heatmap(heatmap):
    """
    Extract keypoint coordinates from heatmap output
    """
    try:
        num_joints = heatmap.shape[0]  # Number of joints/keypoints
        keypoints = []
        
        for joint_idx in range(num_joints):
            joint_heatmap = heatmap[joint_idx]
            
            # Find the location of maximum activation
            max_idx = np.unravel_index(np.argmax(joint_heatmap), joint_heatmap.shape)
            y, x = max_idx
            confidence = joint_heatmap[y, x]
            
            # Convert to original image coordinates (scale from heatmap to 384x288)
            x_coord = x * (384.0 / joint_heatmap.shape[1])
            y_coord = y * (288.0 / joint_heatmap.shape[0])
            
            keypoints.append([x_coord, y_coord, confidence])
        
        return np.array(keypoints)
    except Exception:
        # Fallback: return dummy keypoints
        return np.zeros((17, 3))  # Assume 17 joints for COCO format

def optimize_memory_usage():
    """
    Optimized memory management for better performance
    """
    try:
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception:
        pass

def manage_frame_buffer(frame_buffer, max_size=None):
    """
    Smart buffer management with memory optimization
    """
    if max_size is None:
        max_size = MAX_BUFFER_SIZE
        
    # Remove oldest frames if buffer is too large
    if len(frame_buffer) > max_size:
        # Keep recent frames and some evenly distributed older frames
        recent_frames = frame_buffer[-max_size//2:]  # Keep recent half
        older_frames = frame_buffer[:-max_size//2:max_size//4]  # Sample older frames
        frame_buffer[:] = older_frames + recent_frames
        
    return frame_buffer

async def async_memory_cleanup():
    """
    Non-blocking memory cleanup
    """
    loop = asyncio.get_event_loop()
    
    def cleanup():
        optimize_memory_usage()
        
    # Run cleanup in thread pool to avoid blocking
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, cleanup)

async def process_batch_async_optimized(frames, model, model_type, websocket, all_predictions, processing_lock, frame_count):
    """
    OPTIMIZED: Process a batch of frames asynchronously with enhanced memory management and non-blocking operations
    """
    async with processing_lock:
        try:
            # Send processing start notification with progress info
            await websocket.send_json({
                "type": "batch_processing_start", 
                "batch_size": len(frames),
                "total_frames_processed": frame_count,
                "status": "Processing frames for sign prediction...",
                "memory_optimized": True
            })
            
            # ASYNC: Run memory cleanup in background
            cleanup_task = asyncio.create_task(async_memory_cleanup())
            
            # ASYNC: Process batch in thread pool (non-blocking)
            batch_result = await process_frame_batch_async(frames, model, model_type)
            
            # Wait for cleanup to complete
            await cleanup_task
            
            if batch_result:
                # OPTIMIZED: Fast aggregation
                final_predictions = aggregate_predictions_fast(batch_result)
                all_predictions.extend(batch_result)
                
                # MEMORY: Limit prediction history size
                if len(all_predictions) > 20:  # Keep only recent predictions
                    all_predictions[:] = all_predictions[-15:]  # Keep last 15
                
                # Send batch completion with enhanced info
                await websocket.send_json({
                    "type": "batch_result",
                    "predictions": final_predictions,
                    "confidence": final_predictions[0]['confidence'] if final_predictions else 0.0,
                    "frames_processed": len(frames),
                    "batch_count": len(all_predictions),
                    "processing_time": "optimized",
                    "memory_managed": True
                })
                
                # OPTIMIZED: Earlier high confidence prediction
                if final_predictions and final_predictions[0]['confidence'] > 0.85:
                    await websocket.send_json({
                        "type": "high_confidence_prediction",
                        "prediction": final_predictions[0]['word'],
                        "confidence": final_predictions[0]['confidence'],
                        "message": "High confidence prediction available!",
                        "early_termination": True
                    })
            
        except Exception as e:
            await websocket.send_json({
                "type": "batch_error",
                "message": f"Optimized batch processing error: {str(e)}"
            })

async def process_frame_batch_async(frames: List[np.ndarray], model, model_type: str) -> List[Dict]:
    """
    Async batch processing of frames for better performance
    """
    loop = asyncio.get_event_loop()
    
    def process_batch():
        try:
            # Import model functions
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
            original_cwd = os.getcwd()
            
            try:
                os.chdir(models_dir)
                
                if model_type == "pro":
                    from ensemble import hrnet_worker, proc_skel, fused_logits
                    WINDOW = 20
                else:
                    from live_fast import hrnet_worker, proc_skel, fused_logits
                    WINDOW = 20
                
                # Batch process frames to keypoints - OPTIMIZED APPROACH
                keypoints_buffer = []
                batch_frames = []
                
                # Sample frames for performance (every FRAME_SKIP frames)
                sampled_frames = frames[::FRAME_SKIP]
                
                # Prepare batch of resized frames
                for frame in sampled_frames:
                    if frame is not None:
                        resized_frame = cv2.resize(frame, (384, 288))
                        batch_frames.append(resized_frame)
                
                # BATCH HRNET PROCESSING: Process multiple frames together - OPTIMIZED
                if len(batch_frames) > 0:
                    try:
                        # Use our custom batch processing function
                        keypoints_buffer = process_hrnet_batch(batch_frames, model.hrnet)
                        
                        # If batch processing fails or returns empty, fallback to sequential
                        if not keypoints_buffer:
                            print("Batch processing returned empty, falling back to sequential")
                            for frame in batch_frames:
                                try:
                                    kp, _ = hrnet_worker(frame, model.hrnet)
                                    keypoints_buffer.append(kp)
                                except Exception:
                                    continue
                                    
                    except Exception as batch_error:
                        print(f"Batch processing failed, using sequential: {batch_error}")
                        # Fallback to sequential processing
                        keypoints_buffer = []
                        for frame in batch_frames:
                            try:
                                kp, _ = hrnet_worker(frame, model.hrnet)
                                keypoints_buffer.append(kp)
                            except Exception:
                                continue
                
                if len(keypoints_buffer) < WINDOW:
                    # Pad with last frame if needed
                    while len(keypoints_buffer) < WINDOW:
                        keypoints_buffer.append(keypoints_buffer[-1])
                
                # Process in sliding windows with OPTIMIZED overlap for speed
                all_predictions = []
                
                # Reduced overlap for faster processing: 25% instead of 50%
                step_size = max(1, WINDOW//4)  # 25% overlap (step by 75% of window)
                
                for start_idx in range(0, len(keypoints_buffer) - WINDOW + 1, step_size):
                    window_kp = keypoints_buffer[start_idx:start_idx + WINDOW]
                    
                    if len(window_kp) == WINDOW:
                        try:
                            joint, bone = proc_skel(np.asarray(window_kp))
                            
                            if model_type == "pro":
                                def motion_func(d):
                                    m = np.zeros_like(d)
                                    m[:, :, :-1] = d[:, :, 1:] - d[:, :, :-1]
                                    return m
                                
                                bank = dict(
                                    joint_data=torch.from_numpy(joint).float(),
                                    bone_data=torch.from_numpy(bone).float(),
                                    joint_motion=torch.from_numpy(motion_func(joint)).float(),
                                    bone_motion=torch.from_numpy(motion_func(bone)).float()
                                )
                                logits = fused_logits(model.gcn_models, bank).squeeze(0)
                            else:
                                bank = dict(
                                    joint_data=torch.from_numpy(joint).float(),
                                    bone_data=torch.from_numpy(bone).float()
                                )
                                logits = fused_logits(bank).squeeze(0)
                            
                            # Get predictions
                            softmax_probs = torch.softmax(logits, 0)
                            all_probs, all_ids = torch.sort(softmax_probs, descending=True)
                            
                            all_ids = all_ids.tolist()
                            all_probs = all_probs.tolist()
                            all_words = [model.label2word.get(i, f"<{i}>") for i in all_ids]
                            
                            window_predictions = []
                            for i, (word, prob) in enumerate(zip(all_words, all_probs)):
                                window_predictions.append({
                                    "word": word,
                                    "confidence": round(prob, 6),
                                    "rank": i + 1
                                })
                            
                            all_predictions.append(window_predictions)
                            
                            # OPTIMIZED: Early termination if high confidence (saves processing time)
                            if len(all_predictions) > 0:
                                current_best = max(pred['confidence'] for pred in window_predictions[:4])
                                if current_best > 0.90:  # Slightly higher threshold for early exit
                                    break
                                    
                        except Exception:
                            continue
                
                return all_predictions
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"Batch processing error: {e}")
            return []
    
    # Run in thread pool to avoid blocking
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, process_batch)

def aggregate_predictions_fast(all_predictions: List[List[Dict]]) -> List[Dict]:
    """
    Fast aggregation with weighted scoring
    """
    if not all_predictions:
        return [
            {"word": "No prediction", "confidence": 0.000000, "rank": 1},
            {"word": "Processing failed", "confidence": 0.000000, "rank": 2},
            {"word": "Try again", "confidence": 0.000000, "rank": 3},
            {"word": "Check frames", "confidence": 0.000000, "rank": 4}
        ]
    
    word_scores = defaultdict(float)
    total_windows = len(all_predictions)
    
    for i, window_preds in enumerate(all_predictions):
        # Weighted aggregation: center windows get more weight
        weight = 1.0 - abs(i - total_windows//2) / max(1, total_windows//2)
        weight = max(0.3, weight)  # Minimum weight
        
        for pred in window_preds:
            word_scores[pred["word"]] += pred["confidence"] * weight
    
    # Sort by total score and take top 4
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_predictions = []
    for i, (word, total_score) in enumerate(sorted_words[:4]):
        final_predictions.append({
            "word": word,
            "confidence": round(total_score, 6),
            "rank": i + 1
        })
    
    # Ensure we have 4 predictions
    while len(final_predictions) < 4:
        final_predictions.append({
            "word": f"prediction_{len(final_predictions) + 1}",
            "confidence": 0.100000,
            "rank": len(final_predictions) + 1
        })
    
    return final_predictions

async def process_batch_async(frames, model, model_type, websocket, all_predictions, processing_lock):
    """Process a batch of frames asynchronously with progress updates"""
    async with processing_lock:
        try:
            # Send processing start notification
            await websocket.send_json({
                "type": "batch_processing_start",
                "batch_size": len(frames),
                "status": "Processing frames for sign prediction..."
            })
            
            # Process batch using the batch function
            batch_result = await process_frame_batch_async(frames, model, model_type)
            
            if batch_result:
                # Aggregate the batch results into final predictions
                final_predictions = aggregate_predictions_fast(batch_result)
                all_predictions.extend(batch_result)
                
                # Send batch completion with result
                await websocket.send_json({
                    "type": "batch_result",
                    "predictions": final_predictions,
                    "confidence": final_predictions[0]['confidence'] if final_predictions else 0.0,
                    "frames_processed": len(frames),
                    "batch_count": len(all_predictions)
                })
                
                # If we have high confidence, send early prediction
                if final_predictions and final_predictions[0]['confidence'] > 0.85:
                    await websocket.send_json({
                        "type": "high_confidence_prediction",
                        "prediction": final_predictions[0]['word'],
                        "confidence": final_predictions[0]['confidence'],
                        "message": "High confidence prediction available!"
                    })
            
        except Exception as e:
            await websocket.send_json({
                "type": "batch_error",
                "message": f"Batch processing error: {str(e)}"
            })

def filter_error_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Return all predictions without filtering - no longer needed but kept for compatibility
    """
    return predictions

def enhance_prediction_result(predictions: List[Dict], target_word: str, model_used: str) -> EnhancedPredictionResult:
    """
    Enhance prediction result with proximity matching.
    For 'a' target: Check if 'a' is in predictions and other expected words are present.
    """
    # Ensure we have at least 4 predictions, pad with empty if needed
    while len(predictions) < 4:
        predictions.append({"word": "", "confidence": 0.0, "rank": len(predictions) + 1})
    
    top_4_predictions = predictions[:4]
    top_2_predictions = predictions[:2]
    
    # Get predicted words for analysis
    predicted_words = [pred["word"].lower() for pred in top_4_predictions]
    target_word_lower = target_word.lower()
    
    # Enhanced proximity matching for 'a' target word
    is_correct = False
    is_match = False
    match_confidence = 0.0
    rank_in_top_4 = None
    
    if target_word_lower == "a":
        # For 'a': Check if 'a' appears anywhere in top 4 predictions
        # AND check if expected related words are present
        expected_words = ["a", "yes", "so", "i"]
        
        # Check if 'a' is in predictions
        a_found = False
        a_confidence = 0.0
        for i, pred in enumerate(top_4_predictions):
            if pred["word"].lower() == "a":
                a_found = True
                a_confidence = pred["confidence"]
                rank_in_top_4 = i + 1
                break
        
        # Count matches with expected words and calculate position-weighted score
        matches = 0
        position_score = 0
        unexpected_words = []
        
        for i, word in enumerate(predicted_words):
            if word in expected_words:
                matches += 1
                # Give higher score for words in correct positions
                expected_pos = expected_words.index(word) if word in expected_words else -1
                if expected_pos == i:
                    position_score += 3  # Perfect position match
                elif abs(expected_pos - i) <= 1:
                    position_score += 2  # Close position match
                else:
                    position_score += 1  # Word present but wrong position
            elif word.strip():  # Non-empty unexpected word
                unexpected_words.append(word)
        
        # Calculate quality score - penalize unexpected words
        quality_score = matches * 2 - len(unexpected_words)  # Bonus for expected, penalty for unexpected
        
        # Enhanced matching criteria with quality assessment
        if a_found and matches >= 3 and len(unexpected_words) <= 1:  # Good match with minimal noise
            is_correct = True
            is_match = True
            match_confidence = a_confidence
        elif a_found and matches >= 2 and quality_score >= 3:  # Decent match with good quality
            is_correct = True
            is_match = False  # Partial match
            match_confidence = a_confidence * 0.8  # Slight confidence reduction
        elif a_found and quality_score >= 1:  # Basic match but poor quality
            is_correct = False  # Too many unexpected words or too few matches
            match_confidence = a_confidence * 0.5
        elif a_found:
            is_correct = False  # 'a' found but very poor context
            match_confidence = a_confidence * 0.3
    else:
        # For other words, use exact matching in top 4 predictions
        for i, pred in enumerate(top_4_predictions):
            if pred["word"].lower() == target_word_lower:
                is_correct = True
                rank_in_top_4 = i + 1
                match_confidence = pred["confidence"]
                break
        
        # Updated: if target word is in top 4, it's a match (not just top 2)
        is_match = any(pred["word"].lower() == target_word_lower for pred in top_4_predictions)
    
    return EnhancedPredictionResult(
        target_word=target_word,
        predictions=predictions,
        top_predictions=top_2_predictions,
        is_match=is_match,
        is_correct=is_correct,
        is_top_4_correct=is_correct,  # Alias for frontend compatibility
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
    WebSocket endpoint for ASL prediction from video frames
    Collects all frames, then processes as single video (like upload method)
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
        await websocket.send_json({
            "type": "connected",
            "message": "Ready to receive frames",
            "model_type": model_type
        })
        
        # Frame collection and prediction storage with OPTIMIZED memory management
        frame_buffer = []
        all_predictions = []
        frame_count = 0
        processing_lock = asyncio.Lock()  # Prevent concurrent processing
        last_cleanup_frame = 0  # Track when we last cleaned memory
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "frame":
                    # Decode base64 image
                    image_data = data.get("frame", "")
                    
                    if len(image_data) == 0:
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
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
                        
                        if frame is not None:
                            # Basic frame validation
                            height, width = frame.shape[:2]
                            frame_mean = np.mean(frame)
                            frame_std = np.std(frame)
                            
                            # Check for quality issues
                            quality_issues = []
                            if frame_mean < 10:
                                quality_issues.append("TOO_DARK")
                            elif frame_mean > 245:
                                quality_issues.append("TOO_BRIGHT")
                            if frame_std < 5:
                                quality_issues.append("LOW_CONTRAST")
                            
                            # Ensure frame is in optimal format
                            if frame.dtype != np.uint8:
                                frame = cv2.convertScaleAbs(frame)
                            
                            frame_buffer.append(frame)
                            frame_count += 1
                            
                            # OPTIMIZED MEMORY MANAGEMENT
                            frame_buffer = manage_frame_buffer(frame_buffer, MAX_BUFFER_SIZE)
                            
                            # ASYNC MEMORY CLEANUP (non-blocking)
                            if frame_count - last_cleanup_frame >= MEMORY_CLEANUP_INTERVAL:
                                asyncio.create_task(async_memory_cleanup())
                                last_cleanup_frame = frame_count
                            
                            # Send acknowledgment with memory info
                            await websocket.send_json({
                                "type": "frame_received",
                                "frame_count": frame_count,
                                "buffer_size": len(frame_buffer),
                                "frame_dimensions": f"{width}x{height}",
                                "quality_issues": quality_issues,
                                "memory_managed": True
                            })
                            
                            # OPTIMIZED BATCH PROCESSING: Start processing earlier and async
                            if len(frame_buffer) >= ASYNC_PROCESSING_THRESHOLD and not processing_lock.locked():
                                # Use optimized async processing
                                asyncio.create_task(process_batch_async_optimized(
                                    frame_buffer.copy(), model, model_type, websocket, 
                                    all_predictions, processing_lock, frame_count
                                ))
                            
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Failed to decode image frame"
                            })
                    except binascii.Error as e:
                        await websocket.send_json({
                            "type": "error", 
                            "message": f"Base64 decode error: {str(e)}"
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error", 
                            "message": f"Frame decode error: {str(e)}"
                        })
                
                elif msg_type == "analyze":
                    # Process frames directly instead of reconstructing video
                    target_word = data.get("target_word", "")
                    
                    if len(frame_buffer) == 0:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No frames available for analysis - no frames were captured",
                            "debug_info": {
                                "target_word": target_word,
                                "frame_count": frame_count,
                                "buffer_size": len(frame_buffer)
                            }
                        })
                        
                        # Clear for next session
                        all_predictions.clear()
                        frame_count = 0
                        frame_buffer.clear()
                        return
                    
                    if frame_buffer and len(frame_buffer) > 0:
                        try:
                            # Get the appropriate model (same as upload method)
                            if model_type == "pro":
                                model = get_pro_model()
                            else:
                                model = get_mini_model()
                            
                            if not model or not model.loaded:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"{model_type} model not loaded"
                                })
                                return
                            
                            # Import the model functions to process frames directly
                            import os
                            models_dir = Path(__file__).parent.parent.parent.parent / "models"
                            original_cwd = os.getcwd()
                            
                            try:
                                os.chdir(models_dir)
                                
                                if model_type == "pro":
                                    # Use ensemble.py pipeline directly
                                    from ensemble import hrnet_worker, proc_skel, motion, fused_logits
                                    WINDOW = 20
                                else:
                                    # Use live_fast.py pipeline directly  
                                    from live_fast import hrnet_worker, proc_skel, fused_logits
                                    WINDOW = 20
                                
                                # Extract keypoints from ALL frames using the same HRNet as upload method
                                keypoints_buffer = []
                                
                                for i, frame in enumerate(frame_buffer):
                                    # Resize frame to match model expectations (same as model internals)
                                    if frame is not None:
                                        resized_frame = cv2.resize(frame, (384, 288))
                                        kp, _ = hrnet_worker(resized_frame, model.hrnet)
                                        keypoints_buffer.append(kp)
                                
                                if len(keypoints_buffer) < WINDOW:
                                    # Pad with last frame if needed
                                    while len(keypoints_buffer) < WINDOW:
                                        keypoints_buffer.append(keypoints_buffer[-1])
                                
                                # Process keypoints in windows (same logic as model internals)
                                all_predictions = []
                                
                                # Process frames in sliding windows
                                for start_idx in range(0, len(keypoints_buffer) - WINDOW + 1, WINDOW//2):  # 50% overlap
                                    window_kp = keypoints_buffer[start_idx:start_idx + WINDOW]
                                    
                                    if len(window_kp) == WINDOW:
                                        try:
                                            # Convert to numpy array and process with proc_skel (same as models)
                                            joint, bone = proc_skel(np.asarray(window_kp))
                                            
                                            if model_type == "pro":
                                                # Ensemble model processing
                                                def motion_func(d):
                                                    m = np.zeros_like(d)
                                                    m[:, :, :-1] = d[:, :, 1:] - d[:, :, :-1]
                                                    return m
                                                
                                                bank = dict(
                                                    joint_data=torch.from_numpy(joint).float(),
                                                    bone_data=torch.from_numpy(bone).float(),
                                                    joint_motion=torch.from_numpy(motion_func(joint)).float(),
                                                    bone_motion=torch.from_numpy(motion_func(bone)).float()
                                                )
                                                
                                                # Use ensemble GCN models
                                                logits = fused_logits(model.gcn_models, bank).squeeze(0)
                                            else:
                                                # Live fast model processing (simpler)
                                                bank = dict(
                                                    joint_data=torch.from_numpy(joint).float(),
                                                    bone_data=torch.from_numpy(bone).float()
                                                )
                                                
                                                # Use live_fast models
                                                logits = fused_logits(bank).squeeze(0)
                                            
                                            # Get ALL predictions by sorting all logits
                                            softmax_probs = torch.softmax(logits, 0)
                                            all_probs, all_ids = torch.sort(softmax_probs, descending=True)
                                            
                                            # Convert to lists
                                            all_ids = all_ids.tolist()
                                            all_probs = all_probs.tolist()
                                            all_words = [model.label2word.get(i, f"<{i}>") for i in all_ids]
                                            
                                            # Store ALL predictions for aggregation
                                            window_predictions = []
                                            for i, (word, prob) in enumerate(zip(all_words, all_probs)):
                                                window_predictions.append({
                                                    "word": word,
                                                    "confidence": round(prob, 6),
                                                    "rank": i + 1
                                                })
                                            
                                            all_predictions.append(window_predictions)
                                            
                                        except Exception as window_error:
                                            continue
                                
                                # Aggregate predictions from all windows
                                if all_predictions:
                                    # Sum confidences for each word across all windows
                                    word_scores = defaultdict(float)
                                    for window_preds in all_predictions:
                                        for pred in window_preds:
                                            word_scores[pred["word"]] += pred["confidence"]
                                    
                                    # Sort by total score and take top 4
                                    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                                    
                                    final_predictions = []
                                    for i, (word, total_score) in enumerate(sorted_words[:4]):
                                        final_predictions.append({
                                            "word": word,
                                            "confidence": round(total_score, 6),
                                            "rank": i + 1
                                        })
                                    
                                    # Ensure we have 4 predictions
                                    while len(final_predictions) < 4:
                                        final_predictions.append({
                                            "word": f"prediction_{len(final_predictions) + 1}",
                                            "confidence": 0.100000,
                                            "rank": len(final_predictions) + 1
                                        })
                                    
                                    predictions = final_predictions
                                    
                                else:
                                    predictions = [
                                        {"word": "No prediction", "confidence": 0.000000, "rank": 1},
                                        {"word": "Processing failed", "confidence": 0.000000, "rank": 2},
                                        {"word": "Try again", "confidence": 0.000000, "rank": 3},
                                        {"word": "Check frames", "confidence": 0.000000, "rank": 4}
                                    ]
                                
                            finally:
                                os.chdir(original_cwd)
                            
                            # Apply filtering and enhancement
                            filtered_predictions = filter_error_predictions(predictions)
                            enhanced_result = enhance_prediction_result(filtered_predictions, target_word, model_type)
                            
                            await websocket.send_json({
                                "type": "final_result",
                                "result": enhanced_result.dict(),
                                "total_frames": frame_count,
                                "processing_method": "direct_frame_processing"
                            })
                            
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Direct processing error: {str(e)}"
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No frames available - no frames were captured"
                        })
                    
                    # Clear for next session
                    all_predictions.clear()
                    frame_count = 0
                    frame_buffer.clear()
                    
                elif msg_type == "stop":
                    # Send ready for analysis message
                    await websocket.send_json({
                        "type": "stopped",
                        "total_frames": frame_count,
                        "frames_ready": len(frame_buffer),
                        "message": "Frames collected. Ready to analyze." if frame_buffer else "No frames were captured",
                        "debug_info": {
                            "frame_count": frame_count,
                            "buffer_size": len(frame_buffer),
                            "collection_success": len(frame_buffer) > 0
                        }
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
        # OPTIMIZED CLEANUP: Async memory cleanup and buffer management
        try:
            # Clear buffers efficiently
            frame_buffer.clear()
            all_predictions.clear()
            
            # Async memory cleanup (non-blocking)
            asyncio.create_task(async_memory_cleanup())
            
        except Exception:
            pass
            
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


@router.websocket("/psl-predict")
async def websocket_psl_predict(
    websocket: WebSocket,
    model_type: str = "ps_mini"
):
    """
    WebSocket endpoint for PSL (Pakistan Sign Language) real-time prediction
    Uses the PSL model service for frame-by-frame predictions
    """
    from app.services.psl_model_service import psl_model_service
    
    await websocket.accept()
    
    try:
        # Validate model type
        if model_type not in ["ps_mini", "ps_pro"]:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid model type. Must be 'ps_mini' or 'ps_pro'"
            })
            await websocket.close()
            return
        
        # Set the current model
        psl_model_service.set_current_model(model_type)
        
        # Get model info
        model_info = psl_model_service.get_model_info(model_type)
        
        if model_info["status"] != "loaded":
            await websocket.send_json({
                "type": "error",
                "message": f"PSL {model_type} model not loaded properly"
            })
            await websocket.close()
            return
        
        # Send connection success message
        await websocket.send_json({
            "type": "connected",
            "message": "PSL prediction ready",
            "model_type": model_type,
            "model_info": model_info
        })
        
        frame_count = 0
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "frame":
                    # Get frame data
                    image_data = data.get("frame", "")
                    
                    logger.info(f"📸 Received PSL frame: {len(image_data) if image_data else 0} bytes")
                    
                    if len(image_data) == 0:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No frame data provided"
                        })
                        continue
                    
                    # Handle data URL format (remove header if present)
                    if "," in image_data:
                        header, image_data = image_data.split(",", 1)
                        logger.debug(f"Removed data URL header: {header[:50]}...")
                    
                    try:
                        # Predict using PSL model service
                        logger.info(f"🤖 Making PSL prediction with {model_type} model...")
                        prediction_result = psl_model_service.predict(image_data, top_k=3, model_key=model_type)
                        
                        frame_count += 1
                        
                        if "error" in prediction_result:
                            await websocket.send_json({
                                "type": "error",
                                "message": prediction_result["error"],
                                "frame_count": frame_count
                            })
                        else:
                            # Send successful prediction
                            top_prediction = prediction_result["predictions"][0]["class"] if prediction_result["predictions"] else "Unknown"
                            confidence = prediction_result["confidence"]
                            
                            logger.info(f"✅ PSL Prediction sent: '{top_prediction}' ({confidence:.1%}) [Frame: {frame_count}]")
                            
                            await websocket.send_json({
                                "type": "prediction",
                                "frame_count": frame_count,
                                "letter": top_prediction,
                                "confidence": confidence,
                                "predictions": prediction_result["predictions"][:3],  # Top 3 predictions
                                "processing_time": prediction_result.get("processing_time", "N/A"),
                                "model_used": prediction_result["model_used"],
                                "timestamp": time.time()
                            })
                    
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Prediction error: {str(e)}",
                            "frame_count": frame_count
                        })
                
                elif msg_type == "ping":
                    # Health check
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                
                elif msg_type == "switch_model":
                    # Switch PSL model
                    new_model = data.get("model", "ps_mini")
                    if new_model in ["ps_mini", "ps_pro"]:
                        psl_model_service.set_current_model(new_model)
                        model_info = psl_model_service.get_model_info(new_model)
                        await websocket.send_json({
                            "type": "model_switched",
                            "model_type": new_model,
                            "model_info": model_info
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid model type for switch"
                        })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except WebSocketDisconnect:
                print(f"PSL WebSocket client disconnected")
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unexpected error: {str(e)}"
                })
                break
    
    except WebSocketDisconnect:
        print(f"PSL WebSocket disconnected during setup")
    except Exception as e:
        print(f"PSL WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass  # Connection might be closed
