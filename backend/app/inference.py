from fastapi import UploadFile
from app.models import Prediction

async def run_inference(video: UploadFile, model: str, mode: str) -> Prediction:
    # TODO: implement frame extraction + model logic
    return Prediction(sign="Hello", confidence=0.95)
