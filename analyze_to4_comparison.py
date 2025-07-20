"""
Test E:\to_4.mp4 vs WebSocket method comparison
Shows the exact frame count mismatch issue
"""
import sys
import os
import cv2
sys.path.append('backend')

from app.real_model_integration import predict_video_with_model

def analyze_to4_video():
    """Analyze E:\to_4.mp4 upload method results"""
    video_path = r"E:\to_4.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Analyze video properties
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    file_size = os.path.getsize(video_path)
    
    print(f"🎬 E:\\to_4.mp4 ANALYSIS:")
    print(f"   Frame count: {frame_count} frames (NATURAL)")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Resolution: {width}x{height}")
    print(f"   File size: {file_size:,} bytes")
    print()
    
    # Get upload method predictions
    print("🤖 UPLOAD METHOD PREDICTIONS:")
    try:
        predictions = predict_video_with_model(video_path, "mini")
        upload_words = [p['word'] for p in predictions[:4]]
        print(f"   Result: {upload_words}")
        for i, pred in enumerate(predictions[:4]):
            word = pred.get("word", "")
            confidence = pred.get("confidence", 0.0)
            print(f"   {i+1}. '{word}' (conf: {confidence:.4f})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    print()
    print("🔍 WEBSOCKET METHOD ISSUE:")
    print(f"   ❌ Frontend forces: 49 frames (ARTIFICIAL)")
    print(f"   ✅ Upload method uses: {frame_count} frames (NATURAL)")
    print(f"   📊 Frame count mismatch: {49 - frame_count} extra frames")
    print(f"   ⚠️ This explains different predictions!")
    print()
    
    print("💡 SOLUTION:")
    print(f"   Frontend should capture {frame_count} frames over {duration:.2f}s")
    print(f"   Target FPS: {fps:.1f} (natural)")
    print(f"   Recording duration: {duration:.2f}s (not fixed frame count)")
    print(f"   Expected result: {upload_words}")

if __name__ == "__main__":
    analyze_to4_video()
