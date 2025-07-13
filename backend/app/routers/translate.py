from fastapi import APIRouter, UploadFile, File, Query
from app.models import Prediction
from app.inference import run_inference

router = APIRouter()

@router.post("/predict", response_model=Prediction)
async def predict(
    video: UploadFile = File(...),
    model: str = Query("mediapipe"),
    mode: str = Query("word"),
):
    return await run_inference(video, model, mode)