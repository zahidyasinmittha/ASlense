import cv2
import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.append('d:/ASlense/backend')

from app.real_model_integration import predict_video_with_model

def test_to4_video():
    """
    Test the upload method with the to_4.mp4 video to see what predictions it gives
    """
    video_path = r"E:\to_4.mp4"
    
    print("🎯 TESTING UPLOAD METHOD WITH TO_4.MP4 VIDEO")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        # First, let's analyze the video properties
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        print(f"📹 VIDEO ANALYSIS:")
        print(f"   File: {video_path}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Dimensions: {width}x{height}")
        print(f"   Frame count: {frame_count}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.2f}s")
        print()
        
        # Use the exact same prediction function as upload endpoint
        print("🤖 Running prediction with mini model...")
        predictions = predict_video_with_model(video_path, "mini")
        
        print(f"📊 UPLOAD METHOD RESULTS FOR TO_4.MP4:")
        print(f"Raw predictions received: {len(predictions)} predictions")
        for i, pred in enumerate(predictions):
            word = pred.get("word", "")
            confidence = pred.get("confidence", 0.0)
            print(f"  {i+1}. '{word}' (confidence: {confidence:.6f})")
        
        print(f"\n🎯 Top 4 words: {[p['word'] for p in predictions[:4]]}")
        print(f"🎯 Top prediction: '{predictions[0]['word']}' (confidence: {predictions[0]['confidence']:.6f})")
        
        # Save these results as reference for WebSocket comparison
        reference_result = [p['word'] for p in predictions[:4]]
        print(f"\n📋 REFERENCE FOR WEBSOCKET COMPARISON:")
        print(f"Expected WebSocket result: {reference_result}")
        
        # Check confidence levels
        if predictions[0]['confidence'] > 0.1:
            print(f"✅ HIGH CONFIDENCE DETECTED - WebSocket should match this!")
        else:
            print(f"⚠️ Low confidence - may be challenging video for model")
            
        print(f"\n💡 Next step: Test this same video with WebSocket method")
        print(f"💡 WebSocket should ideally produce: {reference_result}")
            
    except Exception as e:
        print(f"❌ Error testing upload method: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_to4_video()
