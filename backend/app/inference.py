from fastapi import UploadFile
from typing import Dict, Any

async def run_inference(video: UploadFile, model: str, mode: str = "default") -> Dict[str, Any]:
    """
    Run inference on uploaded video using specified model and mode.
    Returns a dictionary with prediction results.
    """
    # TODO: implement frame extraction + model logic
    return {
        "predicted_text": "Hello",  # Changed from "sign" to "predicted_text" for consistency
        "confidence": 0.95,
        "model_used": model,
        "mode": mode,
        "status": "success"
    }
