from fastapi import APIRouter

# Placeholder router for translation features
router = APIRouter()

@router.get("/")
async def translate_info():
    """Get information about translation features."""
    return {
        "message": "Translation features coming soon",
        "status": "development",
        "features": ["ASL to text", "Text to ASL", "Real-time translation"]
    }

# TODO: Implement translation endpoints
# @router.post("/predict", response_model=Prediction)
# async def predict(
#     video: UploadFile = File(...),
#     model: str = Query("mediapipe"),
#     mode: str = Query("word"),
# ):
#     return await run_inference(video, model, mode)