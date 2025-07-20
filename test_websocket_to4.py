import cv2
import numpy as np
import sys
import os
import base64
import asyncio
import websockets
import json

async def test_websocket_with_to4_video():
    """
    Test WebSocket method with to_4.mp4 by reading frames and sending them
    """
    video_path = r"E:\to_4.mp4"
    websocket_url = "ws://127.0.0.1:8000/api/v1/practice/live-predict?model_type=mini"
    
    print("🎯 TESTING WEBSOCKET METHOD WITH TO_4.MP4 VIDEO")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        # First, analyze the video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📹 VIDEO ANALYSIS:")
        print(f"   File: {video_path}")
        print(f"   Dimensions: {width}x{height}")
        print(f"   Frame count: {frame_count}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.2f}s")
        print()
        
        # Read all frames from video
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        print(f"📦 Loaded {len(frames)} frames from video")
        
        # Connect to WebSocket
        print("🔗 Connecting to WebSocket...")
        async with websockets.connect(websocket_url) as websocket:
            
            # Wait for connection message
            response = await websocket.recv()
            conn_data = json.loads(response)
            print(f"✅ Connected: {conn_data.get('message', 'Unknown')}")
            
            # Send frames to WebSocket (simulate real-time capture)
            print(f"📤 Sending {len(frames)} frames to WebSocket...")
            
            for i, frame in enumerate(frames):
                # Encode frame to base64 (like frontend does)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frame_data_url = f"data:image/jpeg;base64,{frame_b64}"
                
                # Send frame
                await websocket.send(json.dumps({
                    "type": "frame",
                    "frame": frame_data_url
                }))
                
                # Wait for acknowledgment
                response = await websocket.recv()
                ack_data = json.loads(response)
                
                if ack_data.get("type") == "frame_received":
                    frame_num = ack_data.get("frame_count", i+1)
                    print(f"✅ Frame {frame_num} acknowledged")
                else:
                    print(f"⚠️ Unexpected response: {ack_data}")
                
                # Small delay to simulate real capture timing
                await asyncio.sleep(0.01)
            
            print(f"📤 All {len(frames)} frames sent successfully")
            
            # Send analyze command with target word 'to' (based on upload results)
            print("🔍 Requesting analysis with target word 'to'...")
            await websocket.send(json.dumps({
                "type": "analyze",
                "target_word": "to"
            }))
            
            # Wait for final result
            print("⏳ Waiting for WebSocket analysis result...")
            response = await websocket.recv()
            result_data = json.loads(response)
            
            if result_data.get("type") == "final_result":
                result = result_data.get("result", {})
                predictions = result.get("predictions", [])
                debug_info = result_data.get("debug_info", {})
                
                print(f"\n📊 WEBSOCKET METHOD RESULTS:")
                print(f"Raw predictions received: {len(predictions)} predictions")
                for i, pred in enumerate(predictions):
                    word = pred.get("word", "")
                    confidence = pred.get("confidence", 0.0)
                    print(f"  {i+1}. '{word}' (confidence: {confidence:.6f})")
                
                ws_words = [p['word'] for p in predictions[:4]]
                upload_words = ['to', 'retrieve', 'hold', 'specific']
                
                print(f"\n🔍 COMPARISON:")
                print(f"WebSocket result: {ws_words}")
                print(f"Upload reference: {upload_words}")
                
                # Analyze differences
                matching_words = set(ws_words).intersection(set(upload_words))
                print(f"\n📊 ANALYSIS:")
                print(f"   Matching words: {list(matching_words)} ({len(matching_words)}/4)")
                print(f"   Accuracy: {len(matching_words)/4*100:.1f}%")
                
                if debug_info:
                    print(f"   Quality assessment: {debug_info.get('quality_assessment', 'unknown')}")
                    print(f"   Video file size: {debug_info.get('video_file_size', 'unknown'):,} bytes")
                
                if ws_words == upload_words:
                    print(f"🎉 PERFECT MATCH! WebSocket exactly matches upload method")
                elif len(matching_words) >= 3:
                    print(f"✅ EXCELLENT MATCH! {len(matching_words)}/4 words correct")
                elif len(matching_words) >= 2:
                    print(f"⚠️ MODERATE MATCH: {len(matching_words)}/4 words correct")
                else:
                    print(f"❌ POOR MATCH: Only {len(matching_words)}/4 words correct")
                    
            else:
                print(f"❌ Unexpected result type: {result_data.get('type')}")
                print(f"Response: {result_data}")
                
    except Exception as e:
        print(f"❌ Error testing WebSocket method: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_with_to4_video())
