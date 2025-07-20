import sys
import os
sys.path.append('/d/ASlense/backend')

from app.real_model_integration import predict_video_with_model

# Test upload method with original video
original_video = r"E:\alpha\A.mp4"
print("Testing upload method...")
predictions = predict_video_with_model(original_video, "mini")

print("UPLOAD METHOD RESULTS:")
for i, pred in enumerate(predictions[:4]):
    word = pred.get("word", "")
    confidence = pred.get("confidence", 0.0)
    print(f"  {i+1}. '{word}' (confidence: {confidence})")

print(f"Top prediction confidence: {predictions[0]['confidence']}")
