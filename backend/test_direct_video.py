import sys
import os
sys.path.append('/d/ASlense/backend')

from app.real_model_integration import predict_video_with_model

def test_multiple_runs():
    """
    Test if the upload method gives consistent results across multiple runs
    """
    original_video = r"E:\alpha\A.mp4"
    
    print("🔬 TESTING UPLOAD METHOD CONSISTENCY")
    print("=" * 50)
    
    for run in range(3):
        print(f"\n🔄 Run #{run + 1}:")
        predictions = predict_video_with_model(original_video, "mini")
        top_4 = [p['word'] for p in predictions[:4]]
        print(f"  Result: {top_4}")
        print(f"  Confidence: {[p['confidence'] for p in predictions[:4]]}")

if __name__ == "__main__":
    test_multiple_runs()
