import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Play, Square, RotateCcw, Download, AlertCircle, CheckCircle } from 'lucide-react';

const CameraTest: React.FC = () => {
  // Camera states
  const [isRecording, setIsRecording] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<string[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [captureInterval, setCaptureInterval] = useState(150); // ms between frames
  const [notification, setNotification] = useState<{type: 'error' | 'success' | 'info', message: string} | null>(null);
  const [debugUpdateTrigger, setDebugUpdateTrigger] = useState(0);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const debugUpdateRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, [cameraStream]);

  // Auto-hide notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // Update debug info every second when recording
  useEffect(() => {
    if (isRecording) {
      debugUpdateRef.current = window.setInterval(() => {
        setDebugUpdateTrigger(prev => prev + 1);
      }, 1000);
    } else {
      if (debugUpdateRef.current) {
        clearInterval(debugUpdateRef.current);
        debugUpdateRef.current = null;
      }
    }
    
    return () => {
      if (debugUpdateRef.current) {
        clearInterval(debugUpdateRef.current);
      }
    };
  }, [isRecording]);

  const showNotification = useCallback((type: 'error' | 'success' | 'info', message: string) => {
    setNotification({ type, message });
  }, []);

  const startCamera = async () => {
    console.log('🎥 Start camera button clicked');
    console.log('🌐 Current URL:', window.location.href);
    console.log('🔒 Is HTTPS:', window.location.protocol === 'https:');
    
    setIsCameraLoading(true);
    try {
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.log('❌ MediaDevices not supported');
        showNotification('error', 'Camera access is not supported in this browser. Please use a modern browser like Chrome, Firefox, or Edge.');
        return;
      }

      // Try to enumerate devices first
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        console.log('� Available devices:', devices.filter(d => d.kind === 'videoinput'));
      } catch (enumError) {
        console.log('⚠️ Could not enumerate devices:', enumError);
      }

      console.log('�📱 Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1920, min: 1280 },  // Prefer 1080p, minimum 720p
          height: { ideal: 1080, min: 720 },  // Prefer 1080p, minimum 720p
          facingMode: 'user',
          frameRate: { ideal: 30, min: 15 }   // Higher frame rate for better quality
        },
        audio: false 
      });
      
      console.log('✅ Camera stream obtained:', stream);
      console.log('📊 Stream tracks:', stream.getTracks());
      setCameraStream(stream);
      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        console.log('📺 Video element stream set');
        
        // Create promise to wait for video to be ready
        const waitForVideoReady = new Promise<void>((resolve) => {
          const checkReady = () => {
            if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 3) {
              console.log('✅ Video fully ready:', video.videoWidth, 'x', video.videoHeight);
              resolve();
            } else {
              console.log('⏳ Video not ready yet:', {
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                readyState: video.readyState
              });
              setTimeout(checkReady, 100);
            }
          };
          checkReady();
        });
        
        // Add event listeners for video events
        video.addEventListener('loadedmetadata', () => {
          console.log('📹 Video metadata loaded:', {
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            duration: video.duration
          });
        });
        
        video.addEventListener('canplay', () => {
          console.log('🎬 Video can play');
        });
        
        video.addEventListener('loadeddata', () => {
          console.log('📊 Video data loaded');
        });
        
        // Try to play the video
        video.play().then(() => {
          console.log('🎬 Video started playing');
          return waitForVideoReady;
        }).then(() => {
          console.log('🎯 Video is fully ready for capture!');
        }).catch(error => {
          console.log('❌ Error with video:', error);
        });
      }
      setIsRecording(true);
      console.log('🎯 isRecording set to true, video should now be visible');
      
      // Additional check after state update
      setTimeout(() => {
        if (videoRef.current) {
          const video = videoRef.current;
          console.log('🔍 Video state check:', {
            srcObject: !!video.srcObject,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            paused: video.paused,
            readyState: video.readyState,
            currentTime: video.currentTime,
            muted: video.muted,
            autoplay: video.autoplay
          });
        }
      }, 100);
      
      showNotification('success', 'Camera started successfully!');
    } catch (error: any) {
      console.error('❌ Error accessing camera:', error);
      console.log('Error details:', {
        name: error.name,
        message: error.message,
        constraint: error.constraint
      });
      
      let errorMessage = 'Could not access camera. ';
      
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        errorMessage += 'Camera permission was denied. Please allow camera access and try again.';
      } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        errorMessage += 'No camera device found. Please connect a camera and try again.';
      } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        errorMessage += 'Camera is already in use by another application. Please close other apps using the camera and try again.';
      } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
        errorMessage += 'Camera does not support the required settings. Trying with default settings...';
        
        // Try again with less restrictive constraints
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({ 
            video: true,
            audio: false 
          });
          setCameraStream(fallbackStream);
          if (videoRef.current) {
            videoRef.current.srcObject = fallbackStream;
          }
          setIsRecording(true);
          setIsCameraLoading(false);
          showNotification('success', 'Camera started with default settings!');
          return;
        } catch (fallbackError) {
          errorMessage += ' Failed with default settings as well.';
        }
      } else if (error.name === 'NotSupportedError') {
        errorMessage += 'Camera access is not supported in this browser.';
      } else if (error.name === 'SecurityError') {
        errorMessage += 'Camera access blocked due to security restrictions. Please ensure you are using HTTPS or localhost.';
      } else {
        errorMessage += `Unexpected error: ${error.message || 'Unknown error occurred.'}`;
      }
      
      showNotification('error', errorMessage);
    } finally {
      console.log('🔄 Camera loading finished');
      setIsCameraLoading(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setIsRecording(false);
    stopCapturing();
    showNotification('info', 'Camera stopped');
  }, [cameraStream]);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('❌ Missing video or canvas ref');
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      console.log('❌ No canvas context');
      return null;
    }

    // More detailed debugging
    console.log('🔍 Video state for capture:', {
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      readyState: video.readyState,
      paused: video.paused,
      currentTime: video.currentTime,
      networkState: video.networkState,
      srcObject: !!video.srcObject
    });

    // Check if video is actually playing and has dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('❌ Video not ready - dimensions:', video.videoWidth, 'x', video.videoHeight);
      console.log('🔄 Ready state:', video.readyState, '(4 = HAVE_ENOUGH_DATA)');
      
      // If video has a stream but no dimensions, try to wait a bit
      if (video.srcObject && video.readyState < 4) {
        console.log('⏳ Video has stream but not fully loaded, waiting...');
      }
      return null;
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    console.log('📐 Canvas size:', canvas.width, 'x', canvas.height);
    
    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to data URL
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.8);
    console.log('🖼️ Frame captured, data URL length:', frameDataUrl.length);
    
    // Validate frame data
    if (frameDataUrl.length < 1000) {
      console.log('❌ Frame data too small, skipping');
      return null;
    }

    return frameDataUrl;
  }, []);

  const startCapturing = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      showNotification('error', 'Please ensure camera is active');
      return;
    }

    setIsCapturing(true);
    setCapturedFrames([]);
    setFrameCount(0);

    console.log('🎬 Starting frame capture interval');
    
    // Clear any existing interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    let currentFrameCount = 0;
    frameIntervalRef.current = window.setInterval(() => {
      const frameData = captureFrame();
      
      if (frameData) {
        currentFrameCount++;
        setFrameCount(currentFrameCount);
        setCapturedFrames(prev => [...prev, frameData]);
        console.log(`📤 Frame ${currentFrameCount} captured, size: ${Math.round(frameData.length / 1024)}KB`);
      }
    }, captureInterval);

    showNotification('info', 'Frame capture started! Click "Stop Capturing" when done.');
  }, [captureFrame, captureInterval]);

  const stopCapturing = useCallback(() => {
    setIsCapturing(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    showNotification('success', `Frame capture stopped! Captured ${frameCount} frames.`);
  }, [frameCount]);

  const clearFrames = useCallback(() => {
    setCapturedFrames([]);
    setFrameCount(0);
    showNotification('info', 'All captured frames cleared');
  }, []);

  const downloadFrames = useCallback(() => {
    if (capturedFrames.length === 0) {
      showNotification('error', 'No frames to download');
      return;
    }

    // Create a JSON file with all frames data
    const data = {
      timestamp: new Date().toISOString(),
      frameCount: capturedFrames.length,
      captureInterval: captureInterval,
      frames: capturedFrames
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `camera_frames_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('success', `Downloaded ${capturedFrames.length} frames as JSON`);
  }, [capturedFrames, captureInterval]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="relative">
            <Camera className="h-16 w-16 text-blue-600 mx-auto mb-4" />
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              Camera Frame Capture Test
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Test video frame capturing functionality with real-time camera feed
            </p>
          </div>
        </div>

        {/* Notification */}
        {notification && (
          <div className={`fixed top-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg transition-all duration-300 transform ${
            notification.type === 'error' 
              ? 'bg-red-100 border border-red-200 text-red-800'
              : notification.type === 'success'
              ? 'bg-green-100 border border-green-200 text-green-800'
              : 'bg-blue-100 border border-blue-200 text-blue-800'
          }`}>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {notification.type === 'error' && <AlertCircle className="h-5 w-5" />}
                {notification.type === 'success' && <CheckCircle className="h-5 w-5" />}
                {notification.type === 'info' && <AlertCircle className="h-5 w-5" />}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">{notification.message}</p>
              </div>
              <button
                onClick={() => setNotification(null)}
                className="flex-shrink-0 text-gray-400 hover:text-gray-600"
              >
                <span className="sr-only">Close</span>
                ×
              </button>
            </div>
          </div>
        )}

        {/* Settings Panel */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 mb-8 border border-white/20">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Capture Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Capture Interval (ms)
              </label>
              <input
                type="number"
                min="50"
                max="1000"
                step="50"
                value={captureInterval}
                onChange={(e) => setCaptureInterval(Number(e.target.value))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                {(1000 / captureInterval).toFixed(1)} FPS
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Frames Captured
              </label>
              <div className="p-2 bg-gray-100 rounded-lg">
                <span className="text-lg font-bold text-blue-600">{frameCount}</span>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Status
              </label>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  isCapturing 
                    ? 'bg-green-500 animate-pulse' 
                    : isRecording 
                    ? 'bg-yellow-500' 
                    : 'bg-gray-400'
                }`}></div>
                <span className="text-sm font-medium">
                  {isCapturing ? 'Capturing' : isRecording ? 'Camera Active' : 'Inactive'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm overflow-hidden">
              <div className="relative bg-gradient-to-br from-gray-800 via-gray-900 to-black h-[400px] flex items-center justify-center p-2">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  controls={false}
                  style={{ display: isRecording ? 'block' : 'none' }}
                  className="w-full h-full object-cover rounded-lg"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!isRecording && (
                  <div className="text-center text-white z-10">
                    <Camera className="h-24 w-24 mx-auto mb-4 opacity-80 text-blue-400" />
                    <p className="text-lg font-medium">Camera Ready</p>
                    <p className="text-sm opacity-75 mt-2">Click "Start Camera" to begin</p>
                  </div>
                )}

                {isRecording && (
                  <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-500/20 backdrop-blur-sm rounded-lg px-3 py-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium text-white">Live</span>
                  </div>
                )}

                {isCapturing && (
                  <div className="absolute top-4 left-4 flex items-center space-x-2 bg-green-500/20 backdrop-blur-sm rounded-lg px-3 py-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium text-white">Capturing: {frameCount} frames</span>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-6 bg-gradient-to-r from-gray-50 to-blue-50">
                <div className="flex flex-wrap items-center justify-center gap-4">
                  {!isRecording ? (
                    <button
                      onClick={() => {
                        console.log('🖱️ Start Camera button clicked!');
                        startCamera();
                      }}
                      disabled={isCameraLoading}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                    >
                      <Play className="h-5 w-5" />
                      <span>{isCameraLoading ? 'Starting...' : 'Start Camera'}</span>
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={stopCamera}
                        className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                      >
                        <Square className="h-5 w-5" />
                        <span>Stop Camera</span>
                      </button>
                      
                      {!isCapturing ? (
                        <button
                          onClick={startCapturing}
                          className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                        >
                          <Camera className="h-5 w-5" />
                          <span>Start Capturing</span>
                        </button>
                      ) : (
                        <button
                          onClick={stopCapturing}
                          className="flex items-center space-x-2 bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                        >
                          <Square className="h-5 w-5" />
                          <span>Stop Capturing</span>
                        </button>
                      )}
                      
                      {/* Debug button to check video state */}
                      <button
                        onClick={() => {
                          if (videoRef.current) {
                            const video = videoRef.current;
                            console.log('🔍 Manual video check:', {
                              srcObject: !!video.srcObject,
                              videoWidth: video.videoWidth,
                              videoHeight: video.videoHeight,
                              paused: video.paused,
                              readyState: video.readyState,
                              currentTime: video.currentTime,
                              style: video.style.display
                            });
                            
                            // Try to force play
                            if (video.paused) {
                              video.play().then(() => {
                                console.log('✅ Forced video to play');
                              }).catch(err => {
                                console.log('❌ Could not force play:', err);
                              });
                            }
                          }
                        }}
                        className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                      >
                        <AlertCircle className="h-4 w-4" />
                        <span>Debug Video</span>
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Frame Management Panel */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Camera className="h-5 w-5 mr-2 text-blue-600" />
                Captured Frames
              </h3>
              
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">{capturedFrames.length}</div>
                  <div className="text-sm text-gray-500">Total Frames</div>
                </div>

                {capturedFrames.length > 0 && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={downloadFrames}
                        className="flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                      >
                        <Download className="h-4 w-4" />
                        <span>Download</span>
                      </button>
                      
                      <button
                        onClick={clearFrames}
                        className="flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                      >
                        <RotateCcw className="h-4 w-4" />
                        <span>Clear</span>
                      </button>
                    </div>

                    {/* Show last captured frame as preview */}
                    <div className="border rounded-lg overflow-hidden">
                      <div className="text-xs text-gray-500 p-2 bg-gray-50">Latest Frame:</div>
                      <img 
                        src={capturedFrames[capturedFrames.length - 1]} 
                        alt="Latest captured frame"
                        className="w-full h-auto"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Debug Info */}
            <div className="bg-gray-100 rounded-xl p-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                Video Status {debugUpdateTrigger !== undefined && ''}
              </h4>
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Video Dimensions:</span>
                  <span className={`font-mono ${(videoRef.current?.videoWidth || 0) > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {videoRef.current?.videoWidth || 0} × {videoRef.current?.videoHeight || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Ready State:</span>
                  <span className={`font-mono ${(videoRef.current?.readyState || 0) >= 3 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {videoRef.current?.readyState || 0} {(videoRef.current?.readyState || 0) >= 3 ? '(Ready)' : '(Loading)'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Playing:</span>
                  <span className={`font-mono ${videoRef.current?.paused === false ? 'text-green-600' : 'text-red-600'}`}>
                    {videoRef.current?.paused === false ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Current Time:</span>
                  <span className="font-mono">{videoRef.current?.currentTime?.toFixed(2) || '0.00'}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Has Stream:</span>
                  <span className={`font-mono ${videoRef.current?.srcObject ? 'text-green-600' : 'text-red-600'}`}>
                    {videoRef.current?.srcObject ? 'Yes' : 'No'}
                  </span>
                </div>
                
                {/* Capture Readiness Indicator */}
                <div className="mt-2 pt-2 border-t">
                  <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${
                      videoRef.current && 
                      videoRef.current.videoWidth > 0 && 
                      videoRef.current.videoHeight > 0 && 
                      videoRef.current.readyState >= 3 && 
                      !videoRef.current.paused
                        ? 'bg-green-500' 
                        : 'bg-red-500'
                    }`}></div>
                    <span className="text-sm font-medium">
                      {videoRef.current && 
                       videoRef.current.videoWidth > 0 && 
                       videoRef.current.videoHeight > 0 && 
                       videoRef.current.readyState >= 3 && 
                       !videoRef.current.paused
                        ? 'Ready for Capture' 
                        : 'Not Ready'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraTest;
