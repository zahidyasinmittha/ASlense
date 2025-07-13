from pydantic import BaseModel

class Prediction(BaseModel):
    sign: str
    confidence: float
