"""
Test script to analyze E:\to_4.mp4 video specifically
This will help us understand what predictions the upload method gives for this video
"""
import sys
import os
sys.path.append('backend')

from app.real_model_integration import predict_video_with_model

def test_e_to4_video():
    """Test the specific E:\to_4.mp4 video"""
    video_path = r"E:\to_4.mp4"
    
    print(f"🎬 Testing video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print(f"   Please check if the file exists at this location")
        return
    
    # Get file info
    file_size = os.path.getsize(video_path)
    print(f"📁 File size: {file_size:,} bytes")
    
    # Analyze with both models
    for model_type in ["mini", "pro"]:
        print(f"\n🤖 Testing with {model_type} model:")
        try:
            predictions = predict_video_with_model(video_path, model_type)
            
            print(f"📋 Raw predictions from {model_type} model:")
            for i, pred in enumerate(predictions):
                word = pred.get("word", "")
                confidence = pred.get("confidence", 0.0)
                print(f"  {i+1}. '{word}' (confidence: {confidence:.4f})")
            
            # Show top 4 specifically
            top_4_words = [p['word'] for p in predictions[:4]]
            print(f"🎯 Top 4 words: {top_4_words}")
            
        except Exception as e:
            print(f"❌ Error with {model_type} model: {str(e)}")
    
    print(f"\n✅ Analysis complete for {video_path}")

if __name__ == "__main__":
    test_e_to4_video()
