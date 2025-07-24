# app/api/v1/endpoints/psl_inference.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import base64
import io
from PIL import Image

from app.services.psl_model_service import psl_model_service

router = APIRouter()

@router.get("/models")
async def get_available_models():
    """Get all available PSL models"""
    return psl_model_service.get_available_models()

@router.post("/switch-model")
async def switch_model(model_data: dict):
    """
    Switch to a different PSL model
    
    Expected payload:
    {
        "model_key": "ps_mini" or "ps_pro"
    }
    """
    model_key = model_data.get("model_key")
    if not model_key:
        raise HTTPException(status_code=400, detail="model_key is required")
    
    success = psl_model_service.set_model(model_key)
    if success:
        model_info = psl_model_service.get_model_info()
        return {
            "message": f"Switched to {model_info['model_name']}",
            "current_model": model_info
        }
    else:
        raise HTTPException(status_code=400, detail=f"Failed to switch to model: {model_key}")

@router.get("/model-info")
async def get_model_info():
    """Get PSL model information and status"""
    return psl_model_service.get_model_info()

@router.post("/predict")
async def predict_psl_sign(
    image: UploadFile = File(...),
    top_k: int = 3
):
    """
    Predict PSL sign from uploaded image
    
    Args:
        image: Image file (JPG, PNG, etc.)
        top_k: Number of top predictions to return
        
    Returns:
        Prediction results with confidence scores
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and encode image
        image_bytes = await image.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Make prediction
        result = psl_model_service.predict(image_b64, top_k)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict-base64")
async def predict_psl_sign_base64(
    image_data: dict,
    top_k: int = 3
):
    """
    Predict PSL sign from base64 image data
    
    Expected payload:
    {
        "image": "base64_image_string",
        "top_k": 3
    }
    """
    try:
        image_b64 = image_data.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        top_k = image_data.get("top_k", top_k)
        
        # Make prediction
        result = psl_model_service.predict(image_b64, top_k)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for PSL inference service"""
    model_info = psl_model_service.get_model_info()
    
    return {
        "status": "healthy" if model_info["status"] == "loaded" else "degraded",
        "model_loaded": model_info["status"] == "loaded",
        "service": "PSL Inference API",
        "version": "1.0.0"
    }
