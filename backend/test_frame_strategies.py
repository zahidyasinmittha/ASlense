import sys
import os
sys.path.append('/d/ASlense/backend')

import cv2
import numpy as np
import tempfile
from app.real_model_integration import predict_video_with_model

def create_test_video_from_original():
    """
    Extract frames from original video and recreate it to see if we get same results
    """
    original_video = r"E:\alpha\A.mp4"
    
    print("🔬 COMPREHENSIVE FRAME EXTRACTION TEST")
    print("=" * 60)
    
    # Open original video
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("❌ Could not open original video")
        return
    
    # Get original properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Original video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    # Extract ALL frames
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames from original video")
    
    # Test 1: Recreate video with EXACT same properties
    print(f"\n🧪 TEST 1: Exact recreation")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Test prediction
        predictions = predict_video_with_model(temp_file.name, "mini")
        result = [p['word'] for p in predictions[:4]]
        print(f"   Result: {result}")
        print(f"   Expected: ['a', 'yes', 'so', 'i']")
        print(f"   Match: {'✅' if result == ['a', 'yes', 'so', 'i'] else '❌'}")
        
        os.unlink(temp_file.name)
    
    # Test 2: Simulate WebSocket frame extraction (every nth frame)
    print(f"\n🧪 TEST 2: WebSocket-style frame sampling (40 frames)")
    
    # Sample 40 frames evenly distributed
    if len(frames) > 40:
        step = len(frames) / 40
        sampled_frames = []
        for i in range(40):
            idx = int(i * step)
            idx = min(idx, len(frames) - 1)
            sampled_frames.append(frames[idx])
    else:
        sampled_frames = frames
    
    print(f"   Sampled {len(sampled_frames)} frames from {len(frames)} total")
    
    # Create video with same duration (1.63s)
    target_duration = 1.63
    new_fps = len(sampled_frames) / target_duration
    new_fps = max(15.0, min(new_fps, 35.0))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, new_fps, (width, height))
        
        for frame in sampled_frames:
            out.write(frame)
        out.release()
        
        print(f"   Video: {len(sampled_frames)} frames at {new_fps:.1f} FPS")
        
        # Test prediction
        predictions = predict_video_with_model(temp_file.name, "mini")
        result = [p['word'] for p in predictions[:4]]
        print(f"   Result: {result}")
        print(f"   Expected: ['a', 'yes', 'so', 'i']")
        print(f"   Match: {'✅' if result == ['a', 'yes', 'so', 'i'] else '❌'}")
        
        os.unlink(temp_file.name)
    
    # Test 3: Different frame sampling strategies
    print(f"\n🧪 TEST 3: Different sampling strategies")
    
    strategies = [
        ("First 40 frames", frames[:40]),
        ("Last 40 frames", frames[-40:] if len(frames) >= 40 else frames),
        ("Middle 40 frames", frames[len(frames)//2-20:len(frames)//2+20] if len(frames) >= 40 else frames),
        ("Every 2nd frame", frames[::2][:40] if len(frames) >= 80 else frames[::1][:40])
    ]
    
    for strategy_name, strategy_frames in strategies:
        if len(strategy_frames) == 0:
            continue
            
        strategy_fps = len(strategy_frames) / target_duration
        strategy_fps = max(15.0, min(strategy_fps, 35.0))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file.name, fourcc, strategy_fps, (width, height))
            
            for frame in strategy_frames:
                out.write(frame)
            out.release()
            
            predictions = predict_video_with_model(temp_file.name, "mini")
            result = [p['word'] for p in predictions[:4]]
            print(f"   {strategy_name}: {result}")
            
            os.unlink(temp_file.name)
    
    print(f"\n📝 Analysis complete!")

if __name__ == "__main__":
    create_test_video_from_original()
