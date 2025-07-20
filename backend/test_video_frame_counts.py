#!/usr/bin/env python3
"""
Test script to analyze frame counts of uploaded videos
This will help us understand what frame counts the upload method actually receives
"""

import cv2
import os
import glob

def analyze_video_frame_counts():
    """Analyze frame counts of videos in the workspace to understand upload method behavior"""
    
    print("🔍 ANALYZING VIDEO FRAME COUNTS")
    print("=" * 50)
    
    # Look for video files in common locations
    video_patterns = [
        "*.mp4",
        "*.avi", 
        "*.mov",
        "*.mkv",
        "test*.mp4",
        "E:/to_4.mp4"  # The specific video mentioned earlier
    ]
    
    videos_found = []
    for pattern in video_patterns:
        videos_found.extend(glob.glob(pattern))
    
    # Also check in common directories
    common_dirs = [".", "backend", "test_videos", "samples"]
    for directory in common_dirs:
        if os.path.exists(directory):
            for pattern in ["*.mp4", "*.avi", "*.mov"]:
                videos_found.extend(glob.glob(os.path.join(directory, pattern)))
    
    if not videos_found:
        print("❌ No video files found in workspace")
        print("💡 Suggestion: Place some test videos (.mp4, .avi, .mov) in the workspace")
        return
    
    print(f"📹 Found {len(videos_found)} video files:")
    print()
    
    frame_counts = []
    
    for video_path in videos_found:
        try:
            # Analyze video properties
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ Could not open: {video_path}")
                continue
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            print(f"📁 {os.path.basename(video_path)}:")
            print(f"   Frame count: {frame_count}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Resolution: {width}x{height}")
            print(f"   File size: {file_size:,} bytes")
            print(f"   Frames per second captured: {frame_count/duration:.1f}" if duration > 0 else "   Invalid duration")
            print()
            
            frame_counts.append(frame_count)
            
        except Exception as e:
            print(f"❌ Error analyzing {video_path}: {str(e)}")
            print()
    
    if frame_counts:
        print("📊 FRAME COUNT ANALYSIS:")
        print(f"   Minimum frames: {min(frame_counts)}")
        print(f"   Maximum frames: {max(frame_counts)}")
        print(f"   Average frames: {sum(frame_counts)/len(frame_counts):.1f}")
        print(f"   Most common: {max(set(frame_counts), key=frame_counts.count)}")
        print()
        
        # Check if 49 is actually common
        if 49 in frame_counts:
            count_49 = frame_counts.count(49)
            print(f"   Videos with 49 frames: {count_49}/{len(frame_counts)} ({count_49/len(frame_counts)*100:.1f}%)")
        else:
            print(f"   ⚠️ NO videos have 49 frames!")
            
        print()
        print("🎯 RECOMMENDATIONS:")
        if max(frame_counts) - min(frame_counts) > 10:
            print("   ✅ Videos have VARYING frame counts (natural)")
            print("   💡 Frontend should capture variable frame counts, not fixed 49")
            print(f"   💡 Typical range: {min(frame_counts)}-{max(frame_counts)} frames")
        else:
            print(f"   ⚠️ All videos have similar frame counts (~{frame_counts[0]})")
            print("   💡 This might indicate artificial frame count targeting")

def create_natural_frame_frontend_config():
    """Generate config for frontend to use natural frame counts"""
    
    print("\n🔧 FRONTEND CONFIGURATION RECOMMENDATIONS:")
    print("=" * 50)
    print("""
Instead of fixed 49 frames, frontend should:

1. NATURAL RECORDING:
   - Capture frames at natural rate (10-30 FPS)
   - Stop based on time duration (1-3 seconds)
   - Let frame count vary naturally (20-90 frames typical)

2. SUGGESTED FRONTEND LOGIC:
   ```javascript
   // Instead of: captureExactly49Frames()
   // Use: captureForDuration(2000) // 2 seconds
   
   const recordingDuration = 2000; // 2 seconds
   const targetFPS = 20; // Natural capture rate
   const expectedFrames = recordingDuration * targetFPS / 1000;
   // Result: ~40 frames for 2s recording, but could be 35-50
   ```

3. BENEFITS:
   - Matches upload method behavior exactly
   - Natural timing and flow
   - Better ASL recognition
   - No artificial constraints
    """)

if __name__ == "__main__":
    print("🎬 VIDEO FRAME COUNT ANALYZER")
    print("Testing to understand upload method frame count patterns\n")
    
    analyze_video_frame_counts()
    create_natural_frame_frontend_config()
    
    print("\n✅ Analysis complete!")
    print("💡 Use these insights to configure frontend for natural frame counts")
