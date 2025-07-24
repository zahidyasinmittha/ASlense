# app/api/v1/endpoints/websocket_psl.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import json
import logging
from typing import Dict, List
import asyncio

from app.core.database import get_db
from app.services.psl_model_service import psl_model_service
from app.auth import get_current_user
from app.models import User

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int = None):
        """Accept a WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: int = None):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def send_to_user(self, message: dict, user_id: int):
        """Send a message to a specific user"""
        if user_id in self.user_connections:
            await self.send_personal_message(message, self.user_connections[user_id])

manager = ConnectionManager()

@router.websocket("/ws/psl-recognition")
async def websocket_psl_recognition(websocket: WebSocket):
    """
    WebSocket endpoint for real-time PSL recognition
    
    Expected message format:
    {
        "type": "predict",
        "image": "base64_image_data",
        "top_k": 3
    }
    
    Response format:
    {
        "type": "prediction",
        "predictions": [
            {"class": "A", "confidence": 0.95},
            {"class": "B", "confidence": 0.03}
        ],
        "confidence": 0.95,
        "processing_time": "150ms"
    }
    """
    await manager.connect(websocket)
    
    try:
        # Send model info on connection
        model_info = psl_model_service.get_model_info()
        await manager.send_personal_message({
            "type": "model_info",
            "data": model_info
        }, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "predict":
                    # Handle prediction request
                    image_data = message.get("image")
                    top_k = message.get("top_k", 3)
                    model_key = message.get("model_key")  # Optional model selection
                    
                    if not image_data:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "No image data provided"
                        }, websocket)
                        continue
                    
                    # Make prediction with specified or current model
                    result = psl_model_service.predict(image_data, top_k, model_key)
                    
                    # Send result back to client
                    await manager.send_personal_message({
                        "type": "prediction",
                        **result
                    }, websocket)
                
                elif message_type == "switch_model":
                    # Handle model switching
                    model_key = message.get("model_key")
                    if model_key:
                        success = psl_model_service.set_model(model_key)
                        if success:
                            model_info = psl_model_service.get_model_info()
                            await manager.send_personal_message({
                                "type": "model_switched",
                                "success": True,
                                "current_model": model_info
                            }, websocket)
                        else:
                            await manager.send_personal_message({
                                "type": "error",
                                "message": f"Failed to switch to model: {model_key}"
                            }, websocket)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "model_key is required"
                        }, websocket)
                
                elif message_type == "get_models":
                    # Send available models
                    available_models = psl_model_service.get_available_models()
                    await manager.send_personal_message({
                        "type": "available_models",
                        "models": available_models
                    }, websocket)
                
                elif message_type == "model_info":
                    # Send model information
                    model_info = psl_model_service.get_model_info()
                    await manager.send_personal_message({
                        "type": "model_info",
                        "data": model_info
                    }, websocket)
                
                elif message_type == "ping":
                    # Handle ping/pong for connection health
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Internal server error"
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.websocket("/ws/psl-recognition/{user_id}")
async def websocket_psl_recognition_authenticated(websocket: WebSocket, user_id: int):
    """
    Authenticated WebSocket endpoint for real-time PSL recognition
    This version can track user sessions and save prediction history
    """
    await manager.connect(websocket, user_id)
    
    try:
        # Send model info on connection
        model_info = psl_model_service.get_model_info()
        await manager.send_personal_message({
            "type": "model_info",
            "data": model_info,
            "user_id": user_id
        }, websocket)
        
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "predict":
                    image_data = message.get("image")
                    top_k = message.get("top_k", 3)
                    session_id = message.get("session_id")
                    
                    if not image_data:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "No image data provided"
                        }, websocket)
                        continue
                    
                    # Make prediction
                    result = psl_model_service.predict(image_data, top_k)
                    
                    # Add session info
                    response = {
                        "type": "prediction",
                        "user_id": user_id,
                        "session_id": session_id,
                        **result
                    }
                    
                    await manager.send_personal_message(response, websocket)
                
                elif message_type == "start_session":
                    # Handle session start
                    await manager.send_personal_message({
                        "type": "session_started",
                        "user_id": user_id,
                        "session_id": message.get("session_id"),
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                elif message_type == "end_session":
                    # Handle session end
                    await manager.send_personal_message({
                        "type": "session_ended",
                        "user_id": user_id,
                        "session_id": message.get("session_id"),
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            
            except Exception as e:
                logger.error(f"Error processing authenticated WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Internal server error"
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"Authenticated WebSocket disconnected for user {user_id}")
    
    except Exception as e:
        logger.error(f"Authenticated WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)
