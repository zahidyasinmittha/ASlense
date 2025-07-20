import cv2
import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.append('d:/ASlense/backend')

from app.real_model_integration import predict_video_with_model

def test_upload_method():
    """
    Test the upload method with the original video to see what predictions it gives
    """
    original_video_path = r"E:\alpha\A.mp4"
    
    print("🎯 TESTING UPLOAD METHOD WITH ORIGINAL VIDEO")
    print("=" * 60)
    
    if not os.path.exists(original_video_path):
        print(f"❌ Video not found: {original_video_path}")
        return
    
    try:
        # Use the exact same prediction function as upload endpoint
        predictions = predict_video_with_model(original_video_path, "mini")
        
        print(f"📊 UPLOAD METHOD RESULTS:")
        print(f"Raw predictions received: {len(predictions)} predictions")
        for i, pred in enumerate(predictions):
            word = pred.get("word", "")
            confidence = pred.get("confidence", 0.0)
            print(f"  {i+1}. '{word}' (confidence: {confidence})")
        
        print(f"\n🎯 Top 4 words: {[p['word'] for p in predictions[:4]]}")
        print(f"🎯 Top prediction: '{predictions[0]['word']}' (confidence: {predictions[0]['confidence']:.6f})")
        
        # Check if these are the high-confidence results we expect
        if predictions[0]['confidence'] > 0.1:
            print(f"✅ HIGH CONFIDENCE DETECTED - This is what WebSocket should match!")
        else:
            print(f"⚠️ Even upload method shows low confidence - may be model issue")
            
    except Exception as e:
        print(f"❌ Error testing upload method: {e}")

if __name__ == "__main__":
    test_upload_method()
