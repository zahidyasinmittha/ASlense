import cv2
import numpy as np
import os

def analyze_video_details(video_path):
    """
    Analyze detailed characteristics of a video file to understand
    what makes it work well with the ML model
    """
    print(f"🎬 Analyzing video: {video_path}")
    print("=" * 80)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    # Get basic video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
    
    print(f"📊 BASIC VIDEO PROPERTIES:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Codec: {codec}")
    print(f"   File Size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print()
    
    # Analyze frame characteristics
    frame_data = []
    frame_count = 0
    
    print(f"🔍 FRAME-BY-FRAME ANALYSIS:")
    print("Frame | Size (MB) | Brightness | Contrast | Blur Score | Motion")
    print("-" * 70)
    
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame size in memory
        frame_size_mb = frame.nbytes / (1024 * 1024)
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate motion (if we have previous frame)
        motion_score = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff)
        
        # Store frame data
        frame_info = {
            'frame_num': frame_count,
            'size_mb': frame_size_mb,
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score,
            'motion_score': motion_score,
            'dtype': frame.dtype,
            'channels': frame.shape[2] if len(frame.shape) > 2 else 1
        }
        frame_data.append(frame_info)
        
        # Print every 5th frame to avoid spam
        if frame_count % 5 == 0 or frame_count <= 10:
            print(f"{frame_count:5d} | {frame_size_mb:8.3f} | {brightness:10.1f} | {contrast:8.1f} | {blur_score:10.1f} | {motion_score:6.1f}")
        
        prev_gray = gray.copy()
        
        # Limit analysis to first 50 frames for performance
        if frame_count >= 50:
            print(f"... (analyzing first 50 frames of {total_frames})")
            break
    
    cap.release()
    
    # Calculate statistics
    if frame_data:
        avg_brightness = np.mean([f['brightness'] for f in frame_data])
        avg_contrast = np.mean([f['contrast'] for f in frame_data])
        avg_blur = np.mean([f['blur_score'] for f in frame_data])
        avg_motion = np.mean([f['motion_score'] for f in frame_data[1:]])  # Skip first frame (no motion)
        
        print()
        print(f"📈 STATISTICS (first {len(frame_data)} frames):")
        print(f"   Average Brightness: {avg_brightness:.1f}")
        print(f"   Average Contrast: {avg_contrast:.1f}")
        print(f"   Average Blur Score: {avg_blur:.1f} (higher = sharper)")
        print(f"   Average Motion: {avg_motion:.1f}")
        print(f"   Frame Data Type: {frame_data[0]['dtype']}")
        print(f"   Channels: {frame_data[0]['channels']}")
    
    print()
    print(f"🎯 MODEL OPTIMIZATION INSIGHTS:")
    print(f"   Temporal Window: {total_frames} frames over {duration:.2f}s")
    print(f"   Frames per second: {fps:.1f} (model expects ~20-30 FPS)")
    print(f"   Resolution: {width}x{height} (check if model prefers specific size)")
    print(f"   Quality indicators: Blur={avg_blur:.1f}, Contrast={avg_contrast:.1f}")
    
    # Check for ML model compatibility
    print()
    print(f"🤖 ML MODEL COMPATIBILITY CHECK:")
    if 20 <= fps <= 30:
        print(f"   ✅ FPS ({fps:.1f}) in optimal range for ML models")
    else:
        print(f"   ⚠️ FPS ({fps:.1f}) outside optimal range (20-30)")
    
    if 1.0 <= duration <= 3.0:
        print(f"   ✅ Duration ({duration:.1f}s) good for ML processing")
    else:
        print(f"   ⚠️ Duration ({duration:.1f}s) may be suboptimal (prefer 1-3s)")
    
    if avg_blur > 100:
        print(f"   ✅ Good sharpness (blur score: {avg_blur:.1f})")
    else:
        print(f"   ⚠️ Low sharpness (blur score: {avg_blur:.1f})")
    
    if avg_contrast > 30:
        print(f"   ✅ Good contrast ({avg_contrast:.1f})")
    else:
        print(f"   ⚠️ Low contrast ({avg_contrast:.1f})")
    
    print("=" * 80)
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'resolution': (width, height),
        'codec': codec,
        'avg_brightness': avg_brightness,
        'avg_contrast': avg_contrast,
        'avg_blur': avg_blur,
        'avg_motion': avg_motion,
        'frame_data': frame_data[:10]  # Return first 10 frames
    }

def compare_with_reconstructed_video(original_path, reconstructed_path):
    """
    Compare original video with reconstructed video to find differences
    """
    print(f"\n🔍 COMPARING VIDEOS:")
    print(f"Original: {original_path}")
    print(f"Reconstructed: {reconstructed_path}")
    print("=" * 80)
    
    original_data = analyze_video_details(original_path)
    if not original_data:
        return
    
    print(f"\n" + "="*50)
    print(f"RECONSTRUCTED VIDEO ANALYSIS:")
    print(f"="*50)
    
    reconstructed_data = analyze_video_details(reconstructed_path)
    if not reconstructed_data:
        return
    
    # Compare key metrics
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"{'Metric':<20} | {'Original':<15} | {'Reconstructed':<15} | {'Difference'}")
    print("-" * 75)
    print(f"{'FPS':<20} | {original_data['fps']:<15.2f} | {reconstructed_data['fps']:<15.2f} | {reconstructed_data['fps'] - original_data['fps']:+.2f}")
    print(f"{'Duration (s)':<20} | {original_data['duration']:<15.2f} | {reconstructed_data['duration']:<15.2f} | {reconstructed_data['duration'] - original_data['duration']:+.2f}")
    print(f"{'Total Frames':<20} | {original_data['total_frames']:<15d} | {reconstructed_data['total_frames']:<15d} | {reconstructed_data['total_frames'] - original_data['total_frames']:+d}")
    print(f"{'Brightness':<20} | {original_data['avg_brightness']:<15.1f} | {reconstructed_data['avg_brightness']:<15.1f} | {reconstructed_data['avg_brightness'] - original_data['avg_brightness']:+.1f}")
    print(f"{'Contrast':<20} | {original_data['avg_contrast']:<15.1f} | {reconstructed_data['avg_contrast']:<15.1f} | {reconstructed_data['avg_contrast'] - original_data['avg_contrast']:+.1f}")
    print(f"{'Blur Score':<20} | {original_data['avg_blur']:<15.1f} | {reconstructed_data['avg_blur']:<15.1f} | {reconstructed_data['avg_blur'] - original_data['avg_blur']:+.1f}")

if __name__ == "__main__":
    # Analyze the original video
    original_video_path = r"E:\alpha\A.mp4"
    
    print("🎥 VIDEO ANALYSIS TOOL FOR ML MODEL OPTIMIZATION")
    print("=" * 80)
    
    result = analyze_video_details(original_video_path)
    
    # Look for any debug videos in the backend directory to compare
    import glob
    debug_videos = glob.glob(r"d:\ASlense\backend\*debug*.mp4")
    temp_videos = glob.glob(r"C:\Users\*\AppData\Local\Temp\*debug*.mp4")
    
    all_debug_videos = debug_videos + temp_videos
    
    if all_debug_videos:
        print(f"\n🔍 Found {len(all_debug_videos)} debug videos for comparison:")
        for debug_video in all_debug_videos[:3]:  # Compare with first 3
            print(f"Comparing with: {debug_video}")
            compare_with_reconstructed_video(original_video_path, debug_video)
    else:
        print(f"\n📝 No debug videos found for comparison.")
        print(f"Run the WebSocket endpoint to generate debug videos in temp directory.")
    
    print(f"\n✅ Analysis complete!")
