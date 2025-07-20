import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  Camera, CameraOff, RotateCcw, Target, CheckCircle, AlertCircle, 
  Zap, TrendingUp, Award, Timer, BarChart3, Upload, 
  Video, RefreshCw, Trophy, Star, Wifi, Search, ChevronDown, User
} from 'lucide-react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

interface Prediction {
  word: string;
  confidence: number;
  rank: number;
}

interface PredictionResult {
  target_word: string;
  predictions: Prediction[];
  top_predictions: Prediction[];
  is_match: boolean;
  match_confidence: number;
  model_used: string;
  is_top_4_correct: boolean;
  user_xp_gained?: number;
  user_new_level?: number;
}

interface LivePredictionMessage {
  type: 'connected' | 'frame_received' | 'progress' | 'final_result' | 'error' | 'stopped';
  message?: string;
  frame_count?: number;
  frames_processed?: number;
  predictions_count?: number;
  latest_predictions?: Prediction[];
  result?: PredictionResult;
  total_frames?: number;
  total_predictions?: number;
}

const Practice: React.FC = () => {
  const baseUrl = import.meta.env.VITE_BACKEND_BASEURL || 'http://localhost:8000';
  const wsUrl = baseUrl.replace('http', 'ws');
  const { user, token, makeAuthenticatedRequest } = useAuth();
  
  // Practice mode states
  const [practiceMode, setPracticeMode] = useState<'camera' | 'upload'>('camera');
  const [selectedModel, setSelectedModel] = useState<'mini' | 'pro'>('mini');
  const [targetWord, setTargetWord] = useState('');
  const [availableWords, setAvailableWords] = useState<string[]>([]);
  const [wordSearch, setWordSearch] = useState('');
  const [isWordDropdownOpen, setIsWordDropdownOpen] = useState(false);
  
  // Camera states - RESTORED ORIGINAL
  const [isRecording, setIsRecording] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  
  // NEW: File streaming test states
  const [isFileStreaming, setIsFileStreaming] = useState(false);
  const [fileVideoRef, setFileVideoRef] = useState<HTMLVideoElement | null>(null);
  const [isFileStreamingReady, setIsFileStreamingReady] = useState(false);
  const [isGestureActive, setIsGestureActive] = useState(false);
  const [collectedPredictions, setCollectedPredictions] = useState<Prediction[][]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [capturedFrameCount, setCapturedFrameCount] = useState(0);
  
  // Simple frame capture states (like CameraTest)
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<string[]>([]);
  const [captureInterval, setCaptureInterval] = useState(150); // ms between frames
  
  // WebSocket states for live prediction
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [isLivePredicting, setIsLivePredicting] = useState(false);
  const [livePredictions, setLivePredictions] = useState<Prediction[] | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Video upload states
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  
  // Results states
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [practiceHistory, setPracticeHistory] = useState<any[]>([]);
  const [notification, setNotification] = useState<{type: 'error' | 'success' | 'info', message: string} | null>(null);

  // User progress states
  const [userProgress, setUserProgress] = useState({
    signs_practiced: 0,
    signs_learned: 0,
    total_signs: 136,
    accuracy_rate: 0,
    current_level: 'Beginner',
    current_xp: 0,
    next_level_xp: 100,
    level_progress: 0,
    practice_streak: 0,
    total_practice_time: 0,
    signs_mastered: 0
  });
  const [isLoadingProgress, setIsLoadingProgress] = useState(false);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const wordDropdownRef = useRef<HTMLDivElement>(null);

  // Memoized filtered words computation with debouncing for better performance
  const filteredWords = useMemo(() => {
    if (!wordSearch.trim()) {
      return availableWords;
    }
    
    const searchTerm = wordSearch.toLowerCase();
    
    // Separate exact matches and partial matches for better UX
    const exactMatches: string[] = [];
    const partialMatches: string[] = [];
    
    availableWords.forEach(word => {
      const wordLower = word.toLowerCase();
      if (wordLower === searchTerm) {
        exactMatches.push(word);
      } else if (wordLower.startsWith(searchTerm)) {
        exactMatches.push(word);
      } else if (wordLower.includes(searchTerm)) {
        partialMatches.push(word);
      }
    });
    
    // Combine exact matches first, then partial matches
    return [...exactMatches, ...partialMatches];
  }, [wordSearch, availableWords]);

  // Fetch user progress data
  const fetchUserProgress = useCallback(async () => {
    if (!user || !token) return;
    
    setIsLoadingProgress(true);
    try {
      const response = await makeAuthenticatedRequest(`${baseUrl}/user/progress`);
      
      if (response.ok) {
        const data = await response.json();
        setUserProgress({
          signs_practiced: data.signs_practiced || 0,
          signs_learned: data.signs_learned || 0,
          total_signs: data.total_signs || 136,
          accuracy_rate: Math.round((data.accuracy_rate || 0) * 100) / 100, // Round to 2 decimal places
          current_level: data.current_level || 'Beginner',
          current_xp: data.current_xp || 0,
          next_level_xp: data.next_level_xp || 100,
          level_progress: Math.round(((data.current_xp || 0) / (data.next_level_xp || 100)) * 100),
          practice_streak: data.practice_streak || 0,
          total_practice_time: data.total_practice_time || 0,
          signs_mastered: data.signs_mastered || 0
        });
        
        // User progress data has been updated
      }
    } catch (error) {
      console.error('Error fetching user progress:', error);
    } finally {
      setIsLoadingProgress(false);
    }
  }, [user, token, baseUrl, makeAuthenticatedRequest]);

  // Fetch total signs count
  const fetchTotalSigns = useCallback(async () => {
    try {
      const response = await fetch(`${baseUrl}/videos/count`);
      if (response.ok) {
        const data = await response.json();
        setUserProgress(prev => ({
          ...prev,
          total_signs: data.count || 136
        }));
      }
    } catch (error) {
      console.error('Error fetching total signs:', error);
    }
  }, [baseUrl]);

  // Load available words on component mount
  useEffect(() => {
    loadAvailableWords();
    checkBackendConnection();
    checkBrowserCompatibility();
    fetchTotalSigns();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load user progress when user logs in
  useEffect(() => {
    if (user && token) {
      fetchUserProgress();
    }
  }, [user, token, fetchUserProgress]);

  // Load practice history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('practice_history');
    if (savedHistory) {
      try {
        const history = JSON.parse(savedHistory);
        setPracticeHistory(history);
      } catch (error) {
        console.error('Error loading practice history:', error);
      }
    }
  }, []);

  // Save practice history to localStorage when it changes
  useEffect(() => {
    if (practiceHistory.length > 0) {
      localStorage.setItem('practice_history', JSON.stringify(practiceHistory));
    }
  }, [practiceHistory]);

  // Timer effect for recording
  useEffect(() => {
    let interval: number;
    if (isRecording) {
      interval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  // Auto-hide notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // WebSocket cleanup
  useEffect(() => {
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, [wsConnection]);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wordDropdownRef.current && !wordDropdownRef.current.contains(event.target as Node)) {
        setIsWordDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Memoized notification handler
  const showNotification = useCallback((type: 'error' | 'success' | 'info', message: string) => {
    setNotification({ type, message });
  }, []);

  // Memoized keyboard handler for word input
  const handleWordInputKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!isWordDropdownOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        setIsWordDropdownOpen(true);
        e.preventDefault();
      }
      return;
    }

    if (e.key === 'Escape') {
      setIsWordDropdownOpen(false);
      e.preventDefault();
    } else if (e.key === 'Enter' && filteredWords.length > 0) {
      setTargetWord(filteredWords[0]);
      setWordSearch(filteredWords[0]);
      setIsWordDropdownOpen(false);
      e.preventDefault();
    }
  }, [isWordDropdownOpen, filteredWords]);

  // RESTORED - Original frame capture function for camera
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

    // Check camera vs file streaming mode
    const isFromCamera = !!video.srcObject;
    const isFromFile = !!video.src && !video.srcObject;
    
    console.log('🔍 Video state for capture:', {
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      readyState: video.readyState,
      paused: video.paused,
      currentTime: video.currentTime,
      networkState: video.networkState,
      isFromCamera: isFromCamera,
      isFromFile: isFromFile,
      srcObject: !!video.srcObject,
      src: !!video.src
    });

    // Check if video is actually ready and has dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('❌ Video not ready - dimensions:', video.videoWidth, 'x', video.videoHeight);
      console.log('🔄 Ready state:', video.readyState, '(4 = HAVE_ENOUGH_DATA)');
      
      if (video.srcObject && video.readyState < 4) {
        console.log('⏳ Camera video has stream but not fully loaded, waiting...');
      } else if (video.src && video.readyState < 4) {
        console.log('⏳ File video has src but not fully loaded, waiting...');
      }
      return null;
    }

    // Set canvas size to match video (updated fallback for higher res)
    canvas.width = video.videoWidth || 640;   // Match original to_4.mp4 exactly
    canvas.height = video.videoHeight || 480; // Match original to_4.mp4 exactly
    
    console.log('📐 Canvas size:', canvas.width, 'x', canvas.height);
    
    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to data URL
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.95);  // High quality for better predictions
    const mode = isFromCamera ? 'camera' : 'file';
    console.log(`🖼️ Frame captured from ${mode}, data URL length:`, frameDataUrl.length);
    
    // Validate frame data
    if (frameDataUrl.length < 1000) {
      console.log('❌ Frame data too small, skipping');
      return null;
    }

    return frameDataUrl;
  }, []);

  // RESTORED - Original gesture start function for camera
  const startGesture = useCallback(() => {
    console.log('🚀 Starting gesture - initial checks...');
    console.log('🔍 Initial state:', {
      hasVideo: !!videoRef.current,
      hasCameraStream: !!cameraStream,
      targetWord: targetWord,
      cameraStreamActive: cameraStream?.active,
      isRecording: isRecording
    });
    
    if (!videoRef.current || !cameraStream || !targetWord) {
      console.log('❌ Failed initial checks:', {
        hasVideo: !!videoRef.current,
        hasCameraStream: !!cameraStream,
        hasTargetWord: !!targetWord
      });
      showNotification('error', 'Please ensure camera is active and target word is selected');
      return;
    }
    
    console.log('✅ Initial checks passed, proceeding with gesture start...');
    
    // First, cleanup any existing connections
    if (wsConnection) {
      console.log('🧹 Cleaning up existing WebSocket connection');
      wsConnection.close();
      setWsConnection(null);
    }
    
    if (frameIntervalRef.current) {
      console.log('🧹 Cleaning up existing frame interval');
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Reset states
    setIsGestureActive(true);
    setCollectedPredictions([]);
    setConnectionStatus('disconnected');
    setLivePredictions(null);
    setCapturedFrameCount(0); // Reset frame counter
    
    // Small delay to ensure cleanup is complete
    setTimeout(() => {
      console.log('🔗 Starting new WebSocket connection for camera');
      console.log('🔍 WebSocket URL:', `${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
      setConnectionStatus('connecting');
      const ws = new WebSocket(`${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected for live prediction');
        console.log('🔍 WebSocket state:', {
          readyState: ws.readyState,
          url: ws.url,
          protocol: ws.protocol
        });
        setConnectionStatus('connected');
        setWsConnection(ws);
        
        // Store WebSocket reference for use in interval
        const currentWs = ws;
        
        // Start frame interval AFTER WebSocket is connected
        console.log('🎬 Starting frame interval after WebSocket connection');
        
        // Clear any existing interval (safety check)
        if (frameIntervalRef.current) {
          console.log('🧹 Clearing existing interval before starting new one');
          clearInterval(frameIntervalRef.current);
        }
        
        // Start natural frame capture at 45 FPS (user-controlled duration)
        let frameCount = 0;
        const TARGET_FPS = 45; // Capture at 45 FPS for ultra-high quality ASL recognition
        const FRAME_INTERVAL = 1000 / TARGET_FPS; // 22.22ms between frames for 45 FPS
        console.log(`⏰ Starting 45 FPS frame capture with ${FRAME_INTERVAL}ms interval...`);
        frameIntervalRef.current = window.setInterval(() => {
          console.log('📸 Interval tick - checking conditions');
          
          // Check WebSocket connection
          if (!currentWs || currentWs.readyState !== WebSocket.OPEN) {
            console.log('❌ WebSocket not ready:', currentWs?.readyState);
            return;
          }
          
          // Use the optimized capture function
          const frameDataUrl = captureFrame();
          
          if (frameDataUrl) {
            // Send frame via WebSocket
            currentWs.send(JSON.stringify({
              type: 'frame',
              frame: frameDataUrl
            }));
            frameCount++;
            setCapturedFrameCount(frameCount); // Update UI counter
            console.log(`📤 Frame ${frameCount} sent to WebSocket at 45 FPS, size: ${Math.round(frameDataUrl.length / 1024)}KB`);
            
            // Continue capturing until user stops (natural duration like upload method)
            console.log(`📊 Natural capture at 45 FPS: ${frameCount} frames collected, waiting for user to stop...`);
          }
        }, FRAME_INTERVAL); // Send frame every 22.22ms (45 frames per second)
        
        // Also capture first frame immediately
        setTimeout(() => {
          if (currentWs.readyState === WebSocket.OPEN) {
            const frameDataUrl = captureFrame();
            
            if (frameDataUrl) {
              currentWs.send(JSON.stringify({
                type: 'frame',
                frame: frameDataUrl
              }));
              console.log('📤 First frame sent immediately, size:', Math.round(frameDataUrl.length / 1024) + 'KB');
            } else {
              console.log('❌ First frame capture failed');
            }
          }
        }, 500); // Increased delay to give video more time to initialize
      };

      ws.onmessage = (event) => {
        try {
          const data: LivePredictionMessage = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          switch (data.type) {
            case 'connected':
              console.log('WebSocket connected:', data.message);
              break;
              
            case 'frame_received':
              console.log(`Frame ${data.frame_count} processed`);
              break;
              
            case 'progress':
              console.log(`Progress: ${data.frames_processed} frames processed`);
              if (data.latest_predictions) {
                setLivePredictions(data.latest_predictions);
                
                // If gesture is active, collect predictions for analysis
                if (isGestureActive) {
                  setCollectedPredictions(prev => [...prev, data.latest_predictions!]);
                }
              }
              break;
              
            case 'final_result':
              console.log('Final result received:', data.result);
              if (data.result) {
                handlePredictionResult(data.result);
              }
              break;
              
            case 'stopped':
              console.log('Recording stopped:', data.message);
              // If we get a "no frames" message, provide helpful feedback
              if (data.message && data.message.includes('No frames')) {
                showNotification('error', 'No frames were captured. Please ensure the camera is working and try holding your gesture longer.');
                setIsAnalyzing(false);
              }
              break;
              
            case 'error':
              console.error('WebSocket error:', data.message);
              if (data.message && data.message.includes('No predictions available')) {
                showNotification('error', 'No frames were processed. Please try holding your gesture longer and ensure good lighting.');
              } else {
                showNotification('error', data.message || 'WebSocket error occurred');
              }
              setIsAnalyzing(false);
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        console.log('🔍 Close event details:', {
          readyState: ws.readyState,
          url: ws.url
        });
        setConnectionStatus('disconnected');
        setWsConnection(null);
        setIsLivePredicting(false);
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        console.log('🔍 Error details:', {
          readyState: ws.readyState,
          url: ws.url,
          error: error
        });
        setConnectionStatus('disconnected');
        showNotification('error', 'WebSocket connection failed. Check if backend is running.');
      };
    }, 300); // 300ms delay for cleanup
    
    showNotification('info', 'Gesture recording started! Perform your sign now.');
    console.log('✅ Gesture started successfully');
  }, [videoRef, cameraStream, targetWord, showNotification, wsUrl, selectedModel, wsConnection, connectionStatus, captureFrame]);

  // Optimized gesture end function
  const endGesture = useCallback(async () => {
    if (!isGestureActive) return;
    
    console.log('🛑 Ending gesture...');
    setIsGestureActive(false);
    setIsAnalyzing(true);
    
    // Stop collecting predictions
    if (frameIntervalRef.current) {
      console.log('🧹 Stopping frame interval');
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Send stop message first
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
      console.log('📤 Sending stop message');
      wsConnection.send(JSON.stringify({ type: 'stop' }));
      
      // Wait longer for all frames to be processed
      setTimeout(() => {
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
          console.log('📤 Sending analyze message for target word:', targetWord);
          wsConnection.send(JSON.stringify({ 
            type: 'analyze', 
            target_word: targetWord 
          }));
        } else {
          console.log('❌ WebSocket connection lost before analyze');
          setIsAnalyzing(false);
          showNotification('error', 'Connection lost before analysis could complete');
        }
      }, 1000); // Increased delay to 1 second
    } else {
      console.log('❌ No WebSocket connection available for stopping');
      setIsAnalyzing(false);
      showNotification('error', 'No active connection to stop');
    }
    
    // Don't close WebSocket immediately - wait for final result
    setCollectedPredictions([]);
  }, [isGestureActive, wsConnection, targetWord]);

  // Memoized backend connection check
  const checkBackendConnection = useCallback(async () => {
    try {
      const response = await axios.get(`${baseUrl}/practice/models/status`);
      console.log('Backend models status:', response.data);
    } catch (error) {
      console.error('Backend connection failed:', error);
    }
  }, [baseUrl]);

  // Memoized browser compatibility check
  const checkBrowserCompatibility = useCallback(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showNotification('error', 'Your browser does not support camera access. Please use a modern browser like Chrome, Firefox, or Edge.');
    } else if (!window.WebSocket) {
      showNotification('error', 'Your browser does not support WebSocket connections required for live prediction.');
    }
  }, [showNotification]);

  // Optimized words loading function
  const loadAvailableWords = useCallback(async () => {
    try {
      const response = await axios.get(`${baseUrl}/practice/available-words`);
      setAvailableWords((response.data as any).words || []);
    } catch (error) {
      console.error('Error loading available words:', error);
      // Fallback words
      setAvailableWords([
        'Hello', 'Thank You', 'Please', 'Sorry', 'Yes', 'No',
        'Good', 'Bad', 'Happy', 'Sad', 'Love', 'Family',
        'he', 'study', 'there', 'analyze', 'fine'
      ]);
    }
  }, [baseUrl]);

  // RESTORED - Original camera function
  const startCamera = async () => {
    setIsCameraLoading(true);
    try {
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showNotification('error', 'Camera access is not supported in this browser. Please use a modern browser like Chrome, Firefox, or Edge.');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1920, min: 1280 },  // Prefer 1080p, minimum 720p
          height: { ideal: 1080, min: 720 },  // Prefer 1080p, minimum 720p
          facingMode: 'user',
          frameRate: { ideal: 30, min: 15 }   // Higher frame rate for better quality
        },
        audio: false 
      });
      
      setCameraStream(stream);
      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        console.log('📺 Video element stream set');
        
        // Create promise to wait for video to be ready
        const waitForVideoReady = new Promise<void>((resolve) => {
          const checkReady = () => {
            if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 3) {
              const resolution = `${video.videoWidth}x${video.videoHeight}`;
              const qualityTier = video.videoWidth >= 1920 ? '🎯 1080p' : 
                                 video.videoWidth >= 1280 ? '📺 720p' : 
                                 video.videoWidth >= 640 ? '📹 480p' : '📱 Low-res';
              console.log(`✅ Video fully ready: ${resolution} ${qualityTier}`);
              
              // Log the actual video track settings
              const videoTrack = stream.getVideoTracks()[0];
              if (videoTrack) {
                const settings = videoTrack.getSettings();
                console.log('🔧 Camera settings achieved:', {
                  width: settings.width,
                  height: settings.height,
                  frameRate: settings.frameRate,
                  aspectRatio: settings.aspectRatio
                });
              }
              
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
      showNotification('success', 'Camera started successfully!');
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      
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
      setIsCameraLoading(false);
    }
  };

  // NEW - File streaming test function
  const startFileFrameStreaming = async (video: HTMLVideoElement) => {
    try {
      console.log('🎬 Starting file frame streaming via WebSocket...');
      
      // Initialize WebSocket connection like camera streaming
      let currentWs: WebSocket;
      
      if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        console.log('🔗 Creating new WebSocket connection for file streaming');
        setConnectionStatus('connecting');
        
        // Create WebSocket and wait for connection
        const ws = new WebSocket(`${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
        
        // Wait for WebSocket to connect
        await new Promise<void>((resolve, reject) => {
          ws.onopen = () => {
            console.log('✅ WebSocket connected for file streaming');
            setConnectionStatus('connected');
            setWsConnection(ws);
            resolve();
          };
          
          ws.onerror = (error) => {
            console.error('❌ WebSocket connection failed:', error);
            reject(new Error('WebSocket connection failed'));
          };
          
          // Set up message handlers
          ws.onmessage = (event) => {
            try {
              const data: LivePredictionMessage = JSON.parse(event.data);
              console.log('File streaming WebSocket message:', data);
              
              switch (data.type) {
                case 'connected':
                  console.log('File streaming WebSocket connected:', data.message);
                  break;
                  
                case 'frame_received':
                  console.log(`File frame ${data.frame_count} processed`);
                  break;
                  
                case 'progress':
                  console.log(`File streaming progress: ${data.frames_processed} frames processed`);
                  if (data.latest_predictions) {
                    setLivePredictions(data.latest_predictions);
                  }
                  break;
                  
                case 'final_result':
                  console.log('File streaming final result:', data.result);
                  setIsAnalyzing(false);
                  setIsFileStreaming(false);
                  if (data.result) {
                    handlePredictionResult(data.result);
                  }
                  break;
                  
                case 'stopped':
                  console.log('File streaming stopped:', data.message);
                  setIsFileStreamingReady(true);
                  showNotification('success', 'Video frames processed! Ready to analyze.');
                  break;
                  
                case 'error':
                  console.error('File streaming WebSocket error:', data.message);
                  setIsAnalyzing(false);
                  setIsFileStreaming(false);
                  setIsFileStreamingReady(false);
                  showNotification('error', data.message || 'File streaming error');
                  break;
              }
            } catch (error) {
              console.error('Error parsing WebSocket message:', error);
            }
          };
          
          ws.onclose = () => {
            console.log('File streaming WebSocket closed');
            setConnectionStatus('disconnected');
            setWsConnection(null);
          };
        });
        
        currentWs = ws;
      } else {
        currentWs = wsConnection;
      }
      
      // Now WebSocket is connected, start streaming frames
      console.log('🎬 WebSocket ready, starting frame streaming...');
      showNotification('info', 'WebSocket connected! Starting frame capture...');
      
      // Create canvas for frame extraction
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Could not get canvas context');
      }
      
      // Ensure video has valid dimensions
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        throw new Error(`Invalid video dimensions: ${video.videoWidth}x${video.videoHeight}`);
      }
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log(`📐 Canvas setup: ${canvas.width}x${canvas.height} (video: ${video.videoWidth}x${video.videoHeight})`);
      
      // Test canvas drawing capability
      ctx.fillStyle = 'red';
      ctx.fillRect(0, 0, 10, 10);
      const testData = canvas.toDataURL('image/jpeg', 0.95);  // High quality for better predictions
      console.log(`🧪 Canvas test successful, test image size: ${testData.length} bytes`);
      
      // Stream frames with natural frame extraction at 25 FPS (like upload method)
      // No artificial frame count - let the video determine its natural duration
      let frameCount = 0;
      let isStreaming = true;
      
      // Use consistent 45 FPS for ultra-high quality ASL recognition
      const targetFPS = 45; // Ultra-high FPS for ASL gesture videos
      const frameInterval = Math.max(22.22, 1000 / targetFPS); // At least 22.22ms between frames, target 22.22ms for 45 FPS
      
      // Pre-load video for better frame extraction
      video.preload = 'auto';
      video.load(); // Force reload to ensure all metadata is available
      
      // Wait a bit for video to be fully loaded before starting
      await new Promise(resolve => {
        if (video.readyState >= 2) {
          resolve(undefined);
        } else {
          video.addEventListener('loadeddata', () => resolve(undefined), { once: true });
        }
      });
      
      console.log(`🎬 Video fully loaded - readyState: ${video.readyState}, duration: ${video.duration}s, dimensions: ${video.videoWidth}x${video.videoHeight}`);
      console.log(`🎬 Starting natural frame streaming: video duration=${video.duration.toFixed(3)}s, interval=${frameInterval.toFixed(1)}ms`);
      
      // Safety timeout to prevent infinite loops (max 10 seconds regardless of video duration)
      const maxStreamingTime = Math.max(video.duration * 1000 + 2000, 10000); // Video duration + 2s buffer, min 10s
      const streamingStartTime = Date.now();
      
      console.log(`⏰ Safety timeout set for ${maxStreamingTime}ms to prevent infinite loops`);
      
      const streamFrames = () => {
        console.log(`🔍 streamFrames called - frameCount: ${frameCount}, isStreaming: ${isStreaming}, currentWs state: ${currentWs?.readyState}, video: paused=${video.paused}, ended=${video.ended}, currentTime=${video.currentTime.toFixed(3)}s, duration=${video.duration.toFixed(3)}s`);
        
        // Safety timeout check
        const elapsed = Date.now() - streamingStartTime;
        if (elapsed > maxStreamingTime) {
          console.log(`⏰ SAFETY TIMEOUT: Ending streaming after ${elapsed}ms to prevent infinite loop`);
          isStreaming = false;
        }
        
        if (!isStreaming || !currentWs || currentWs.readyState !== WebSocket.OPEN) {
          console.log('🛑 Stopping file frame streaming - conditions not met');
          return;
        }
        
        try {
          // Natural frame capture - let video play at its natural pace
          // Enhanced condition: check for sufficient video data AND proper timing
          const hasVideoData = video.readyState >= 2;  // HAVE_CURRENT_DATA or better
          const videoNotAtEnd = !video.ended && video.currentTime < video.duration * 0.99;
          const canCaptureFrame = hasVideoData && videoNotAtEnd;
          
          if (canCaptureFrame) {
            // Capture frame from current video position (natural timing)
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const frameData = canvas.toDataURL('image/jpeg', 0.95);  // High quality for better predictions
            frameCount++;
            
            console.log(`📸 Sending file frame ${frameCount} (${frameData.length} bytes) at time ${video.currentTime.toFixed(3)}s`);
            
            // Send frame through WebSocket
            currentWs.send(JSON.stringify({
              type: 'frame',
              frame: frameData
            }));
          } else {
            console.log(`⏭️ Skipping frame capture - readyState: ${video.readyState}, ended: ${video.ended}, currentTime: ${video.currentTime.toFixed(3)}s/${video.duration.toFixed(3)}s`);
            
            // If we're consistently unable to capture frames, end the streaming
            if (video.currentTime >= video.duration * 0.98) {
              console.log(`🏁 Ending due to video being 98%+ complete`);
              isStreaming = false;
            }
          }
          
          // Continue streaming until video ends (natural duration)
          // Enhanced end-of-video detection: check multiple conditions
          const isVideoAtEnd = video.ended || 
                              (video.currentTime >= video.duration - 0.05) ||  // Within 50ms of end
                              (video.currentTime >= video.duration * 0.99);     // 99% complete
          
          const shouldContinue = isStreaming && !isVideoAtEnd;
          
          if (shouldContinue) {
            // Advance video time at 45 FPS
            video.currentTime = Math.min(video.currentTime + (1/targetFPS), video.duration - 0.01);
            setTimeout(streamFrames, frameInterval);
          } else {
            console.log(`🏁 Natural frame streaming completed - frameCount: ${frameCount}`);
            console.log(`📊 Video end detected: ended=${video.ended}, currentTime=${video.currentTime.toFixed(3)}s, duration=${video.duration.toFixed(3)}s`);
            console.log(`📊 Natural capture complete: ${frameCount} frames from ${video.duration.toFixed(2)}s video`);
            
            // Natural timing achieved - no artificial constraints
            isStreaming = false;
            setIsRecording(false);
            setIsFileStreamingReady(true);
            console.log(`✅ Natural frame count achieved: ${frameCount} frames (matches upload method)`);
            
            // Send stop signal to backend
            if (currentWs && currentWs.readyState === WebSocket.OPEN) {
              currentWs.send(JSON.stringify({ type: 'stop' }));
            }
            showNotification('success', 'Video streaming complete! Click Analyze to get results.');
          }
          
        } catch (error) {
          console.error('❌ Error streaming frame:', error);
        }
      };
      
      // Start video playback and frame streaming
      video.currentTime = 0;
      console.log('🎬 Starting video playback...');
      
      try {
        await video.play();
        console.log('✅ Video playback started successfully');
        console.log(`📹 Video initial state: paused=${video.paused}, ended=${video.ended}, currentTime=${video.currentTime}, duration=${video.duration}, readyState=${video.readyState}`);
        
        // Start frame streaming immediately
        console.log('🎯 Starting frame streaming loop...');
        streamFrames();
        
        // Also set up a comprehensive monitoring system
        let monitoringInterval: number;
        
        const startMonitoring = () => {
          monitoringInterval = setInterval(() => {
            const progress = (video.currentTime / video.duration * 100).toFixed(1);
            console.log(`🔍 Monitoring: frameCount=${frameCount}, time=${video.currentTime.toFixed(3)}s/${video.duration.toFixed(3)}s (${progress}%), paused=${video.paused}, ended=${video.ended}, isStreaming=${isStreaming}`);
            
            // If video has ended, complete the capture
            if ((video.ended || video.currentTime >= video.duration) && isStreaming) {
              console.log(`✅ Video ended naturally with ${frameCount} frames captured`);
              isStreaming = false;
              if (currentWs && currentWs.readyState === WebSocket.OPEN) {
                currentWs.send(JSON.stringify({ type: 'stop' }));
              }
              clearInterval(monitoringInterval);
            }
            
            // Stop monitoring when complete
            if (!isStreaming) {
              console.log('🛑 Clearing monitoring interval');
              clearInterval(monitoringInterval);
            }
          }, 200); // Check every 200ms for responsive monitoring
        };
        
        startMonitoring();
        
        // No backup frame capture needed - 100% efficiency achieved with perfect timing
        
      } catch (playError) {
        console.error('❌ Error starting video playback:', playError);
        showNotification('error', 'Failed to start video playback');
        throw playError;
      }
      
    } catch (error: any) {
      console.error('❌ File frame streaming error:', error);
      showNotification('error', `Failed to stream video frames: ${error.message}`);
      setIsRecording(false);
      setIsFileStreaming(false);
    }
  };

  // Analyze file streaming results
  const analyzeFileStreaming = async () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      showNotification('error', 'WebSocket connection not available');
      return;
    }
    
    if (!targetWord.trim()) {
      showNotification('error', 'Please enter a target word');
      return;
    }
    
    console.log('🔍 Requesting analysis for target word:', targetWord);
    setIsFileStreamingReady(false);
    setIsAnalyzing(true);
    
    // Send analyze command
    wsConnection.send(JSON.stringify({
      type: 'analyze',
      target_word: targetWord
    }));
    
    showNotification('info', 'Analyzing video predictions...');
  };

  const startFileStreaming = async () => {
    console.log('🔧 Starting file streaming test mode...');
    
    setIsCameraLoading(true);
    setIsFileStreamingReady(false);
    setIsAnalyzing(false);
    try {
      // Test mode - no video file required
      console.log('� File streaming test mode activated');
      setIsFileStreaming(true);
      showNotification('success', 'File streaming test mode activated! You can now test the streaming functionality.');
      
      // Add file selection capability
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'video/*';
      
      input.onchange = async (event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file && videoRef.current) {
          console.log('📹 Video file selected:', file.name);
          try {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.muted = true;
            video.playsInline = true;
            
            // Wait for video to load
            await new Promise<void>((resolve, reject) => {
              video.addEventListener('loadedmetadata', () => {
                console.log('📹 File video metadata loaded:', {
                  videoWidth: video.videoWidth,
                  videoHeight: video.videoHeight,
                  duration: video.duration,
                  readyState: video.readyState,
                  networkState: video.networkState
                });
                resolve();
              });
              video.addEventListener('error', (e) => {
                console.error('❌ Video loading error:', e);
                reject(e);
              });
              video.load();
            });

            // Set the video for display and streaming
            videoRef.current.src = video.src;
            videoRef.current.load();
            
            setIsFileStreaming(true);
            setIsRecording(true); // Start recording mode for file streaming
            showNotification('success', `Video ${file.name} loaded! Initializing WebSocket...`);
            
            // Start streaming frames to backend via WebSocket
            await startFileFrameStreaming(video);
            
          } catch (error: any) {
            console.error('Error loading video file:', error);
            showNotification('error', 'Failed to load video file');
          }
        } else {
          setIsFileStreaming(true);
          showNotification('success', 'File streaming test mode activated!');
        }
        setIsCameraLoading(false);
      };
      
      // Auto-trigger file selection
      setTimeout(() => input.click(), 100);
    } catch (error: any) {
      console.error('Error in file streaming test:', error);
      showNotification('error', 'Failed to activate file streaming test mode');
    } finally {
      setIsCameraLoading(false);
    }
  };

  // Optimized camera stop function
  const stopCamera = useCallback(() => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setIsRecording(false);
    setSessionTime(0);
    stopLivePrediction();
  }, [cameraStream]);

  // Optimized live prediction stop function  
  const stopLivePrediction = useCallback(() => {
    setIsLivePredicting(false);
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (wsConnection) {
      wsConnection.send(JSON.stringify({ type: 'stop' }));
      wsConnection.close();
      setWsConnection(null);
    }
    setConnectionStatus('disconnected');
    setLivePredictions(null);
  }, [wsConnection]);

  // Debug function to check video state
  const debugVideoState = useCallback(() => {
    if (videoRef.current) {
      const video = videoRef.current;
      console.log('🔍 Manual video debug:', {
        srcObject: !!video.srcObject,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        paused: video.paused,
        readyState: video.readyState,
        currentTime: video.currentTime,
        style: video.style.display,
        isRecording: isRecording,
        streamActive: cameraStream?.active
      });
      
      // Try to force play if paused
      if (video.paused && video.srcObject) {
        video.play().then(() => {
          console.log('✅ Forced video to play');
        }).catch(err => {
          console.log('❌ Could not force play:', err);
        });
      }
    }
  }, [isRecording, cameraStream]);

  // Simple frame capture functions (like CameraTest)
  const startCapturing = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      showNotification('error', 'Please ensure camera is active');
      return;
    }

    setIsCapturing(true);
    setCapturedFrames([]);
    setCapturedFrameCount(0);

    console.log('🎬 Starting simple frame capture interval');
    
    // Clear any existing interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    let currentFrameCount = 0;
    frameIntervalRef.current = window.setInterval(() => {
      const frameData = captureFrame();
      
      if (frameData) {
        currentFrameCount++;
        setCapturedFrameCount(currentFrameCount);
        setCapturedFrames(prev => [...prev, frameData]);
        console.log(`📤 Frame ${currentFrameCount} captured, size: ${Math.round(frameData.length / 1024)}KB`);
        
        // Step 3: Also send to WebSocket if connected
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
          try {
            wsConnection.send(JSON.stringify({
              type: 'frame',
              frame: frameData
            }));
            console.log(`🌐 Frame ${currentFrameCount} sent to WebSocket`);
          } catch (error) {
            console.error('❌ Error sending frame to WebSocket:', error);
          }
        }
      }
    }, captureInterval);

    const message = wsConnection && wsConnection.readyState === WebSocket.OPEN 
      ? 'Frame capture + WebSocket sending started!' 
      : 'Frame capture started! Connect WebSocket to also send frames.';
    showNotification('info', message);
  }, [captureFrame, captureInterval, showNotification, wsConnection]);

  const stopCapturing = useCallback(() => {
    setIsCapturing(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    console.log(`✅ Frame capture stopped! Captured ${capturedFrames.length} frames.`);
    showNotification('success', `Frame capture stopped! Captured ${capturedFrames.length} frames.`);
  }, [capturedFrames.length, showNotification]);

  const clearFrames = useCallback(() => {
    setCapturedFrames([]);
    setCapturedFrameCount(0);
    showNotification('info', 'All captured frames cleared');
  }, [showNotification]);

  // Step 1: Basic WebSocket connection function
  const connectWebSocket = useCallback(() => {
    console.log('🔗 Step 1: Connecting to WebSocket...');
    
    if (wsConnection) {
      console.log('🧹 Closing existing WebSocket connection');
      wsConnection.close();
      setWsConnection(null);
    }
    
    setConnectionStatus('connecting');
    const ws = new WebSocket(`${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
    
    ws.onopen = () => {
      console.log('✅ Step 1: WebSocket connected successfully!');
      setConnectionStatus('connected');
      setWsConnection(ws);
      showNotification('success', 'WebSocket connected! Ready for frame sending.');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📨 Step 4: WebSocket message received:', data);
        
        // Handle different message types from backend
        switch (data.type) {
          case 'connected':
            console.log('🔗 Backend confirmed connection:', data.message);
            break;
            
          case 'frame_received':
            console.log(`✅ Backend received frame ${data.frame_count || 'unknown'}`);
            break;
            
          case 'progress':
            console.log(`📊 Backend progress: ${data.frames_processed || 0} frames processed`);
            if (data.latest_predictions) {
              console.log('🔮 Latest predictions:', data.latest_predictions);
            }
            break;
            
          case 'final_result':
            console.log('🎯 Final result received:', data.result);
            if (data.result) {
              showNotification('success', `Prediction complete! Top result: ${data.result.predictions[0]?.word || 'Unknown'}`);
            }
            break;
            
          case 'error':
            console.error('❌ Backend error:', data.message);
            showNotification('error', data.message || 'Backend error occurred');
            break;
            
          default:
            console.log('📝 Other message type:', data.type, data);
        }
      } catch (error) {
        console.error('❌ Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      setConnectionStatus('disconnected');
      setWsConnection(null);
    };
    
    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      setConnectionStatus('disconnected');
      showNotification('error', 'WebSocket connection failed. Check if backend is running.');
    };
  }, [wsUrl, selectedModel, wsConnection, showNotification]);

  // Step 2: Send single frame to WebSocket
  const sendFrameToWebSocket = useCallback(() => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      showNotification('error', 'WebSocket not connected. Please connect first.');
      return;
    }
    
    console.log('📤 Step 2: Sending single frame to WebSocket...');
    
    const frameData = captureFrame();
    if (!frameData) {
      showNotification('error', 'Failed to capture frame. Check camera state.');
      return;
    }
    
    try {
      wsConnection.send(JSON.stringify({
        type: 'frame',
        frame: frameData
      }));
      
      console.log('✅ Step 2: Frame sent successfully!', {
        frameSize: Math.round(frameData.length / 1024) + 'KB'
      });
      showNotification('success', `Frame sent! Size: ${Math.round(frameData.length / 1024)}KB`);
    } catch (error) {
      console.error('❌ Error sending frame:', error);
      showNotification('error', 'Failed to send frame to WebSocket');
    }
  }, [wsConnection, captureFrame, showNotification]);

  // Step 5: Send analyze command to WebSocket
  const analyzeFrames = useCallback(() => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      showNotification('error', 'WebSocket not connected.');
      return;
    }
    
    if (!targetWord) {
      showNotification('error', 'Please select a target word for analysis.');
      return;
    }
    
    console.log('🔍 Step 5: Sending analyze command...');
    
    try {
      // First send stop command
      wsConnection.send(JSON.stringify({ type: 'stop' }));
      
      // Then send analyze command
      setTimeout(() => {
        wsConnection.send(JSON.stringify({ 
          type: 'analyze', 
          target_word: targetWord 
        }));
        
        console.log('✅ Step 5: Analyze command sent for word:', targetWord);
        showNotification('info', `Analyzing frames for word: ${targetWord}`);
      }, 500);
      
    } catch (error) {
      console.error('❌ Error sending analyze command:', error);
      showNotification('error', 'Failed to send analyze command');
    }
  }, [wsConnection, targetWord, showNotification]);

  // Test frame capture function (like CameraTest)
  const testFrameCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('❌ Missing video or canvas ref for test');
      showNotification('error', 'Camera not ready for test capture');
      return;
    }

    const video = videoRef.current;
    console.log('🧪 Testing frame capture...');
    console.log('🔍 Video state before test:', {
      isRecording: isRecording,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      readyState: video.readyState,
      paused: video.paused,
      currentTime: video.currentTime,
      srcObject: !!video.srcObject,
      display: video.style.display,
      visibility: getComputedStyle(video).visibility,
      cameraStreamActive: cameraStream?.active
    });
    
    const frameDataUrl = captureFrame();
    
    if (frameDataUrl) {
      console.log('✅ Test capture successful!', {
        frameSize: Math.round(frameDataUrl.length / 1024) + 'KB',
        dataUrlLength: frameDataUrl.length
      });
      showNotification('success', `Test capture successful! Frame size: ${Math.round(frameDataUrl.length / 1024)}KB`);
    } else {
      console.log('❌ Test capture failed');
      showNotification('error', 'Test capture failed - check video state');
    }
  }, [captureFrame, showNotification, isRecording, cameraStream]);

  // Optimized video upload handler
  const handleVideoUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedVideo(file);
      const url = URL.createObjectURL(file);
      setVideoPreview(url);
    } else {
      showNotification('error', 'Please select a valid video file');
    }
  }, [showNotification]);

  // Optimized video prediction function
  const predictUploadedVideo = useCallback(async () => {
    if (!uploadedVideo || !targetWord) {
      showNotification('error', 'Please select a video file and target word');
      return;
    }

    // Check if user is authenticated
    if (!user || !token) {
      showNotification('error', 'Please log in to use video prediction features');
      return;
    }

    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('video_file', uploadedVideo);
      formData.append('target_word', targetWord);
      formData.append('model_type', selectedModel);
      
      const response = await axios.post(`${baseUrl}/practice/predict-video`, formData, {
        headers: { 
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        }
      });
      
      handlePredictionResult(response.data as PredictionResult);
    } catch (error: any) {
      console.error('Error in video prediction:', error);
      
      if (error.response?.status === 401) {
        showNotification('error', 'Authentication failed. Please log in again.');
      } else if (error.response?.status === 403) {
        showNotification('error', 'Access denied. Please check your permissions.');
      } else {
        showNotification('error', 'Video prediction failed. Please try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  }, [uploadedVideo, targetWord, selectedModel, baseUrl, showNotification, user, token]);

  // Optimized prediction result handler with user tracking
  const handlePredictionResult = useCallback(async (result: PredictionResult) => {
    setPredictionResult(result);
    setIsAnalyzing(false);
    
    // Close WebSocket connection after getting result
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
      setConnectionStatus('disconnected');
    }
    
    // Submit to backend for user tracking if authenticated
    if (user && token) {
      try {
        const response = await makeAuthenticatedRequest(`${baseUrl}/practice/predict-with-user`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            target_word: result.target_word,
            predicted_word: result.predictions[0]?.word || '',
            confidence: result.predictions[0]?.confidence || 0,
            is_correct: result.is_top_4_correct, // Use top 4 logic for correctness
            model_used: result.model_used,
            practice_mode: practiceMode, // Include practice mode
            top_4_predictions: result.predictions.slice(0, 4).map(p => p.word)
          })
        });

        if (response.ok) {
          const userData = await response.json();
          if (userData.xp_gained) {
            result.user_xp_gained = userData.xp_gained;
          }
          if (userData.new_level) {
            result.user_new_level = userData.new_level;
          }
          
          // Refresh user progress after successful prediction tracking
          fetchUserProgress();
        }
      } catch (error) {
        console.error('Error tracking user progress:', error);
      }
    }
    
    // Add to practice history
    const historyEntry = {
      word: result.target_word,
      predictions: result.predictions,
      isMatch: result.is_match,
      isTop4Correct: result.is_top_4_correct,
      confidence: result.match_confidence,
      timestamp: new Date().toISOString(),
      model: result.model_used
    };
    
    setPracticeHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10 entries
    
    // User progress is now updated via fetchUserProgress() after backend tracking
  }, [user, token, baseUrl, fetchUserProgress, wsConnection, practiceMode, makeAuthenticatedRequest]);

  // Optimized results clearing function
  const clearResults = useCallback(() => {
    setPredictionResult(null);
    setUploadedVideo(null);
    setVideoPreview(null);
    setLivePredictions(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Optimized mode and model change handlers
  const handlePracticeModeChange = useCallback((mode: 'camera' | 'upload') => {
    setPracticeMode(mode);
    // Clear results when switching modes
    clearResults();
  }, [clearResults]);

  const handleModelChange = useCallback((model: 'mini' | 'pro') => {
    setSelectedModel(model);
  }, []);

  const handleWordSelection = useCallback((word: string) => {
    setTargetWord(word);
    setWordSearch(word);
    setIsWordDropdownOpen(false);
  }, []);

  const handleWordSearchChange = useCallback((value: string) => {
    setWordSearch(value);
    setIsWordDropdownOpen(true);
  }, []);

  const handleClearWordSearch = useCallback(() => {
    setWordSearch('');
    setTargetWord('');
  }, []);

  // Memoized utility functions for better performance
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const getConfidenceColor = useCallback((confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }, []);

  const getConfidenceIcon = useCallback((confidence: number) => {
    if (confidence >= 0.8) return CheckCircle;
    if (confidence >= 0.6) return AlertCircle;
    return AlertCircle;
  }, []);

  // Memoized static data
  const models = useMemo(() => [
    { id: 'mini', name: 'AS Mini', accuracy: '85%+', speed: 'Fast', description: 'Optimized for real-time prediction' },
    { id: 'pro', name: 'AS Pro', accuracy: '90%+', speed: 'Medium', description: 'High accuracy ensemble model' },
  ], []);

  const practiceGoals = useMemo(() => [
    { 
      title: 'Daily Practice', 
      current: userProgress.signs_practiced, 
      target: 5, 
      unit: 'signs', 
      color: 'blue' 
    },
    { 
      title: 'Weekly Streak', 
      current: userProgress.practice_streak, 
      target: 7, 
      unit: 'days', 
      color: 'green' 
    },
    { 
      title: 'Accuracy Goal', 
      current: Math.round(userProgress.accuracy_rate), 
      target: 95, 
      unit: '%', 
      color: 'purple' 
    },
  ], [userProgress]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header with enhanced styling */}
        <div className="text-center mb-12 animate-fade-in-up">
          <div className="relative group mb-8">
            {/* <div className="absolute -inset-8 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 blur-xl opacity-20 group-hover:opacity-30 transition duration-1000 animate-pulse rounded-full"></div> */}
            <div className="relative">
              <Target className="h-20 w-20 text-blue-600 mx-auto mb-4 animate-bounce-gentle" />
              <h1 className="text-5xl pb-2.5 md:text-6xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
                Perfect Your Signs
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                AI-powered ASL recognition with real-time feedback and gesture analysis
              </p>
            </div>
          </div>
          
          {/* Status Indicators */}
          <div className="flex items-center justify-center mt-4 space-x-6">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {connectionStatus === 'connected' ? (
                <>
                  <Wifi className="h-5 w-5 text-green-600" />
                  <span className="text-sm text-green-600 font-medium">Live prediction active</span>
                </>
              ) : connectionStatus === 'connecting' ? (
                <>
                  <RefreshCw className="h-5 w-5 text-yellow-600 animate-spin" />
                  <span className="text-sm text-yellow-600 font-medium">Connecting...</span>
                </>
              ) : (
                <>
                  <Wifi className="h-5 w-5 text-green-600" />
                  <span className="text-sm text-green-600">Ready for practice</span>
                </>
              )}
            </div>

            {/* Authentication Status */}
            <div className="flex items-center space-x-2">
              {user && token ? (
                <>
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <span className="text-sm text-green-600 font-medium">
                    Logged in as {user.fullName || user.email}
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 text-orange-500" />
                  <span className="text-sm text-orange-500 font-medium">
                    Please log in for full features
                  </span>
                </>
              )}
            </div>
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

        {/* Practice Mode Selector with enhanced styling */}
        <div className="p-8 mb-8 animate-fade-in-up animation-delay-200 border border-white/20">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Choose Your Practice Mode</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <button
              onClick={() => handlePracticeModeChange('camera')}
              className={`group p-8 rounded-2xl border-3 transition-all duration-500 transform hover:scale-105 ${
                practiceMode === 'camera'
                  ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-purple-50 text-blue-700 shadow-2xl shadow-blue-500/25'
                  : 'border-gray-200 bg-green-100 text-gray-700 hover:border-blue-300 hover:shadow-xl'
              }`}
            >
              <div className="relative">
                <Camera className="h-12 w-12 mx-auto mb-4 group-hover:animate-bounce" />
                <div className="font-bold text-lg mb-2">Live Camera</div>
                <div className="text-sm opacity-75">Real-time gesture recognition with instant feedback</div>
                {practiceMode === 'camera' && (
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            </button>
            
            <button
              onClick={() => handlePracticeModeChange('upload')}
              className={`group p-8 rounded-2xl border-3 transition-all duration-500 transform hover:scale-105 ${
                practiceMode === 'upload'
                  ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 text-purple-700 shadow-2xl shadow-purple-500/25'
                  : 'border-gray-200 bg-green-100 text-gray-700 hover:border-purple-300 hover:shadow-xl'
              }`}
            >
              <div className="relative">
                <Upload className="h-12 w-12 mx-auto mb-4 group-hover:animate-bounce" />
                <div className="font-bold text-lg mb-2">Video Upload</div>
                <div className="text-sm opacity-75">Upload and analyze pre-recorded sign videos</div>
                {practiceMode === 'upload' && (
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            </button>
          </div>
        </div>

        {/* Practice Settings with enhanced styling */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 animate-fade-in-up animation-delay-400 border border-white/20 relative" style={{ zIndex: 1000 }}>
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Practice Configuration</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 relative">
            {/* Target Word Selection */}
            <div className="relative" ref={wordDropdownRef}>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Word to Practice
              </label>
              
              {/* Search Input */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  placeholder="Search for a word to practice..."
                  value={wordSearch}
                  onChange={(e) => handleWordSearchChange(e.target.value)}
                  onFocus={() => setIsWordDropdownOpen(true)}
                  onKeyDown={handleWordInputKeyDown}
                  className="w-full pl-10 pr-10 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={() => setIsWordDropdownOpen(!isWordDropdownOpen)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                >
                  <ChevronDown className={`h-5 w-5 text-gray-400 transform transition-transform ${isWordDropdownOpen ? 'rotate-180' : ''}`} />
                </button>
                
                {/* Clear search button */}
                {wordSearch && (
                  <button
                    type="button"
                    onClick={handleClearWordSearch}
                    className="absolute inset-y-0 right-8 pr-2 flex items-center text-gray-400 hover:text-gray-600"
                  >
                    ×
                  </button>
                )}
              </div>

              {/* Selected Word Display */}
              {targetWord && !isWordDropdownOpen && (
                <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-blue-800">Selected: {targetWord}</span>
                    <button
                      onClick={handleClearWordSearch}
                      className="text-blue-600 hover:text-blue-800"
                    >
                      ×
                    </button>
                  </div>
                </div>
              )}

              {/* Dropdown */}
              {isWordDropdownOpen && (
                <div className="absolute w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-2xl max-h-60 overflow-y-auto"
                     style={{ zIndex: 999999 }}>
                  {/* Quick Access Section */}
                  {!wordSearch.trim() && (
                    <div className="border-b bg-gray-50">
                      <div className="p-2 text-xs font-semibold text-gray-600 uppercase">Quick Access</div>
                      <div className="flex flex-wrap gap-1 p-2">
                        {['Hello', 'Thank You', 'Please', 'Yes', 'No', 'Good'].map((word) => (
                          <button
                            key={word}
                            onClick={() => handleWordSelection(word)}
                            className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                          >
                            {word}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {filteredWords.length > 0 ? (
                    <>
                      <div className="p-2 text-xs text-gray-500 border-b">
                        {filteredWords.length} word{filteredWords.length !== 1 ? 's' : ''} found
                      </div>
                      {filteredWords.map((word) => {
                        const searchTerm = wordSearch.toLowerCase();
                        const wordLower = word.toLowerCase();
                        const isExactMatch = wordLower === searchTerm;
                        const isStartsWithMatch = wordLower.startsWith(searchTerm);
                        
                        // Highlight matching text
                        const highlightText = () => {
                          if (!searchTerm) return word;
                          
                          const index = wordLower.indexOf(searchTerm);
                          if (index === -1) return word;
                          
                          const before = word.slice(0, index);
                          const match = word.slice(index, index + searchTerm.length);
                          const after = word.slice(index + searchTerm.length);
                          
                          return (
                            <>
                              {before}
                              <span className="bg-yellow-200 font-semibold">{match}</span>
                              {after}
                            </>
                          );
                        };
                        
                        return (
                          <button
                            key={word}
                            onClick={() => handleWordSelection(word)}
                            className={`w-full text-left px-4 py-2 hover:bg-blue-50 focus:bg-blue-50 focus:outline-none flex items-center justify-between ${
                              targetWord === word ? 'bg-blue-100 text-blue-800 font-medium' : 'text-gray-900'
                            }`}
                          >
                            <span>{highlightText()}</span>
                            {wordSearch && (
                              <span className="text-xs text-gray-400 ml-2 flex-shrink-0">
                                {isExactMatch ? '✓ exact' : isStartsWithMatch ? '→ starts' : '⊃ contains'}
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </>
                  ) : (
                    <div className="p-4 text-center text-gray-500">
                      {wordSearch ? `No words found matching "${wordSearch}"` : 'No words available'}
                    </div>
                  )}
                  
                  {/* Close dropdown button */}
                  <div className="border-t bg-gray-50">
                    <button
                      onClick={() => setIsWordDropdownOpen(false)}
                      className="w-full p-2 text-sm text-gray-600 hover:text-gray-800"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}

              {!targetWord && (
                <p className="mt-2 text-sm text-amber-600 flex items-center">
                  <AlertCircle className="h-4 w-4 mr-1" />
                  Please select a target word to enable camera and prediction features
                </p>
              )}
              
              {/* Search Tips */}
              <div className="mt-2 text-xs text-gray-500">
                <span>💡 Tips: Use ↓ to open dropdown, Enter to select first match, Esc to close</span>
              </div>
            </div>

            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Detection Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value as 'mini' | 'pro')}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.accuracy} ({model.speed})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* Main Practice Area */}
          <div className="xl:col-span-3">
            {/* Gesture Instructions */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 mb-6 border border-blue-200">
              <h3 className="text-lg font-bold text-blue-900 mb-3 flex items-center">
                <Target className="h-5 w-5 mr-2" />
                How to Practice with Gesture Detection
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
                  <div>
                    <div className="font-semibold text-blue-800">Select Word</div>
                    <div className="text-blue-700">Choose a target word to practice</div>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
                  <div>
                    <div className="font-semibold text-green-800">Start Gesture (Natural Duration)</div>
                    <div className="text-green-700">Perform your sign naturally - duration will vary by gesture</div>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
                  <div>
                    <div className="font-semibold text-purple-800">End & Analyze</div>
                    <div className="text-purple-700">Click "End Gesture" to get results</div>
                  </div>
                </div>
              </div>
            </div>
            {practiceMode === 'camera' ? (
              /* Camera Mode */
              <div className="bg-white rounded-xl shadow-sm overflow-hidden animate-fade-in-up animation-delay-600">
                <div className="relative bg-gradient-to-br from-gray-800 via-gray-900 to-black h-[500px] flex items-center justify-center p-2">
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
                      <Video className="h-24 w-24 mx-auto mb-4 opacity-80 text-blue-400" />
                      <p className="text-lg font-medium">Dual Mode Practice</p>
                      <p className="text-sm opacity-75 mt-2">Choose Camera Streaming or File Upload Test</p>
                      <div className="mt-4 space-y-2">
                        <p className="text-sm text-green-300">✅ Camera Streaming: Real-time gesture recognition</p>
                        <p className="text-sm text-blue-300">🔧 File Streaming: Upload video for testing</p>
                        {!uploadedVideo && (
                          <p className="text-sm text-yellow-300">⚠️ Upload a video file to enable file streaming test</p>
                        )}
                        {uploadedVideo && !targetWord && (
                          <p className="text-sm text-orange-300">💡 Select a target word for better analysis (optional for testing)</p>
                        )}
                      </div>
                    </div>
                  )}

                  {isRecording && (
                    <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-500/20 backdrop-blur-sm rounded-lg px-3 py-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium text-white">
                        {cameraStream ? 'Camera Live' : 'File Test'} - {formatTime(sessionTime)}
                      </span>
                    </div>
                  )}

                  {isCapturing && (
                    <div className="absolute top-4 left-4 flex items-center space-x-2 bg-green-500/20 backdrop-blur-sm rounded-lg px-3 py-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium text-white">
                        Capturing: {capturedFrameCount} frames
                      </span>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                      <div className="text-center text-white">
                        <RefreshCw className="h-12 w-12 mx-auto mb-4 animate-spin" />
                        <p className="text-lg font-medium">Processing your sign...</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-6 bg-gradient-to-r from-gray-50 to-blue-50">
                  <div className="flex flex-col items-center space-y-4">
                    {/* Dual Mode Buttons */}
                    {!isRecording ? (
                      <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4 w-full">
                        {/* Camera Streaming Button */}
                        <button
                          onClick={startCamera}
                          disabled={!targetWord || isCameraLoading}
                          className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none w-full sm:w-auto"
                        >
                          {isCameraLoading ? (
                            <RefreshCw className="h-5 w-5 animate-spin" />
                          ) : (
                            <Video className="h-5 w-5" />
                          )}
                          <span>{isCameraLoading ? 'Loading Camera...' : 'Start Camera Streaming'}</span>
                        </button>
                        
                        {/* File Streaming Test Button */}
                        <button
                          onClick={startFileStreaming}
                          disabled={false}
                          className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none w-full sm:w-auto"
                          title="Test file streaming functionality"
                        >
                          <Upload className="h-5 w-5" />
                          <span>Test File Streaming</span>
                        </button>
                        
                        {/* Analyze Video Button - appears when file streaming is ready */}
                        {isFileStreamingReady && (
                          <button
                            onClick={analyzeFileStreaming}
                            disabled={isAnalyzing}
                            className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none w-full sm:w-auto"
                            title="Analyze the uploaded video frames"
                          >
                            {isAnalyzing ? (
                              <RefreshCw className="h-5 w-5 animate-spin" />
                            ) : (
                              <Target className="h-5 w-5" />
                            )}
                            <span>{isAnalyzing ? 'Analyzing...' : 'Analyze Video'}</span>
                          </button>
                        )}
                        
                        {/* Debug Info */}
                        <div className="text-xs text-gray-500 mt-2">
                          Debug: uploadedVideo={uploadedVideo ? 'YES' : 'NO'}; isCameraLoading={isCameraLoading ? 'YES' : 'NO'}; disabled={(!uploadedVideo || isCameraLoading) ? 'YES' : 'NO'}
                        </div>
                      </div>
                    ) : (
                      <>
                        {!isCapturing ? (
                          <>
                            <button
                              onClick={startCapturing}
                              className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                            >
                              <Target className="h-5 w-5" />
                              <span>Start Capturing</span>
                            </button>
                            
                            <button
                              onClick={testFrameCapture}
                              className="flex items-center space-x-3 px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                            >
                              <Camera className="h-5 w-5" />
                              <span>Test Single Frame</span>
                            </button>
                            
                            <button
                              onClick={connectWebSocket}
                              disabled={connectionStatus === 'connecting'}
                              className={`flex items-center space-x-3 px-6 py-4 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 ${
                                connectionStatus === 'connected' 
                                  ? 'bg-gradient-to-r from-green-500 to-green-600' 
                                  : 'bg-gradient-to-r from-purple-600 to-purple-700'
                              }`}
                            >
                              {connectionStatus === 'connecting' ? (
                                <RefreshCw className="h-5 w-5 animate-spin" />
                              ) : connectionStatus === 'connected' ? (
                                <Wifi className="h-5 w-5" />
                              ) : (
                                <Wifi className="h-5 w-5" />
                              )}
                              <span>
                                {connectionStatus === 'connecting' ? 'Connecting...' : 
                                 connectionStatus === 'connected' ? 'Connected' : 'Connect WebSocket'}
                              </span>
                            </button>
                            
                            {connectionStatus === 'connected' && (
                              <>
                                <button
                                  onClick={sendFrameToWebSocket}
                                  className="flex items-center space-x-3 px-6 py-4 bg-gradient-to-r from-orange-600 to-orange-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                                >
                                  <Upload className="h-5 w-5" />
                                  <span>Send Frame</span>
                                </button>
                                
                                <button
                                  onClick={analyzeFrames}
                                  disabled={!targetWord || capturedFrameCount === 0}
                                  className="flex items-center space-x-3 px-6 py-4 bg-gradient-to-r from-yellow-600 to-yellow-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50"
                                >
                                  <Target className="h-5 w-5" />
                                  <span>Analyze Frames</span>
                                </button>
                              </>
                            )}
                          </>
                        ) : (
                          <button
                            onClick={stopCapturing}
                            className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl animate-pulse"
                          >
                            <CheckCircle className="h-5 w-5" />
                            <span>Capturing ({capturedFrameCount} frames)</span>
                          </button>
                        )}
                        
                        <button
                          onClick={stopCamera}
                          className="flex items-center space-x-3 px-6 py-4 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                        >
                          <CameraOff className="h-5 w-5" />
                          <span>Stop Camera</span>
                        </button>
                        
                        {capturedFrames.length > 0 && (
                          <button
                            onClick={clearFrames}
                            className="flex items-center space-x-3 px-4 py-2 bg-orange-500 text-white rounded-lg font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl text-sm"
                          >
                            <RotateCcw className="h-4 w-4" />
                            <span>Clear {capturedFrames.length} Frames</span>
                          </button>
                        )}
                        
                        <button
                          onClick={debugVideoState}
                          className="flex items-center space-x-3 px-4 py-2 bg-yellow-500 text-black rounded-lg font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl text-sm"
                        >
                          <span>Debug Video</span>
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Upload Mode */
              <div className="bg-white rounded-xl shadow-sm overflow-hidden animate-fade-in-up animation-delay-600">
                <div className="relative bg-gradient-to-br from-purple-800 via-purple-900 to-black h-[500px] flex items-center justify-center p-2">
                  {videoPreview ? (
                    <video
                      src={videoPreview}
                      controls
                      className="w-full h-full object-cover rounded-lg shadow-lg"
                    />
                  ) : (
                    <div className="text-center text-white z-10">
                      <Upload className="h-24 w-24 mx-auto mb-4 opacity-80 text-purple-400" />
                      <p className="text-lg font-medium">Upload Video</p>
                      <p className="text-sm opacity-75 mt-2">Select a video file to analyze</p>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                      <div className="text-center text-white">
                        <RefreshCw className="h-12 w-12 mx-auto mb-4 animate-spin" />
                        <p className="text-lg font-medium">Analyzing video...</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-6 bg-gradient-to-r from-gray-50 to-purple-50">
                  <div className="flex items-center justify-center space-x-4">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      className="hidden"
                    />
                    
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="flex items-center space-x-3 px-6 py-4 bg-white text-purple-700 border-2 border-purple-200 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-sm hover:shadow-md"
                    >
                      <Video className="h-5 w-5" />
                      <span>Select Video</span>
                    </button>
                    
                    {uploadedVideo && (
                      <button
                        onClick={predictUploadedVideo}
                        disabled={isProcessing || !targetWord || !user || !token}
                        className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50"
                        title={!user || !token ? 'Please log in to analyze videos' : !targetWord ? 'Please select a target word' : ''}
                      >
                        <Target className="h-5 w-5" />
                        <span>
                          {!user || !token ? 'Log In Required' : 'Analyze Video'}
                        </span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Sidebar */}
          <div className="space-y-6">
            {/* Frame Capture Display */}
            {practiceMode === 'camera' && isRecording && (
              <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Camera className="h-5 w-5 mr-2 text-blue-600" />
                  Captured Frames
                </h3>
                
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600">{capturedFrames.length}</div>
                    <div className="text-sm text-gray-500">Total Frames</div>
                  </div>

                  {isCapturing && (
                    <div className="flex items-center justify-center space-x-2 p-3 bg-green-50 rounded-lg">
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium text-green-700">
                        Capturing at {(1000 / captureInterval).toFixed(1)} FPS
                      </span>
                    </div>
                  )}

                  {capturedFrames.length > 0 && (
                    <>
                      {/* Show last captured frame as preview */}
                      <div className="border rounded-lg overflow-hidden">
                        <div className="text-xs text-gray-500 p-2 bg-gray-50">Latest Frame:</div>
                        <img 
                          src={capturedFrames[capturedFrames.length - 1]} 
                          alt="Latest captured frame"
                          className="w-full h-auto"
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={clearFrames}
                          className="flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                          <RotateCcw className="h-4 w-4" />
                          <span>Clear</span>
                        </button>
                        
                        <div className="text-center text-xs text-gray-500 p-2">
                          {Math.round(capturedFrames[capturedFrames.length - 1]?.length / 1024) || 0}KB each
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Results Display */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-800">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="h-5 w-5 mr-2 text-blue-600" />
                Prediction Results
              </h3>
              
              {predictionResult ? (
                <div className="space-y-4">
                  {/* Match Result */}
                  <div className={`p-4 rounded-lg ${
                    predictionResult.is_match 
                      ? 'bg-green-50 border border-green-200' 
                      : predictionResult.is_top_4_correct
                      ? 'bg-yellow-50 border border-yellow-200'
                      : 'bg-red-50 border border-red-200'
                  }`}>
                    <div className="flex items-center space-x-3 mb-2">
                      {predictionResult.is_match ? (
                        <>
                          <Trophy className="h-6 w-6 text-green-600" />
                          <span className="font-bold text-green-800">Perfect Match!</span>
                        </>
                      ) : predictionResult.is_top_4_correct ? (
                        <>
                          <CheckCircle className="h-6 w-6 text-yellow-600" />
                          <span className="font-bold text-yellow-800">Close! In Top 4</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-6 w-6 text-red-600" />
                          <span className="font-bold text-red-800">Not Quite Right</span>
                        </>
                      )}
                    </div>
                    <div className="text-sm text-gray-600">
                      Target: <span className="font-medium">{predictionResult.target_word}</span>
                    </div>
                    {predictionResult.is_match && (
                      <div className="text-sm text-green-600 mt-1">
                        Confidence: {(predictionResult.match_confidence * 100).toFixed(1)}%
                      </div>
                    )}
                    {predictionResult.is_top_4_correct && !predictionResult.is_match && (
                      <div className="text-sm text-yellow-600 mt-1">
                        ✨ Good job! Your sign was recognized in the top 4 predictions
                      </div>
                    )}
                    {predictionResult.user_xp_gained && (
                      <div className="text-sm text-blue-600 mt-1">
                        🎉 +{predictionResult.user_xp_gained} XP earned!
                        {predictionResult.user_new_level && ` Level up to ${predictionResult.user_new_level}!`}
                      </div>
                    )}
                  </div>

                  {/* Top 4 Predictions */}
                  <div>
                    <h4 className="font-medium text-gray-700 mb-3 flex items-center">
                      Top 4 Predictions
                      <span className="ml-2 text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                        Any of these count as correct!
                      </span>
                    </h4>
                    <div className="space-y-2">
                      {predictionResult.predictions.slice(0, 4).map((prediction, index) => {
                        const isTargetWord = prediction.word.toLowerCase() === predictionResult.target_word.toLowerCase();
                        return (
                          <div key={index} className={`flex items-center justify-between p-3 rounded-lg ${
                            isTargetWord ? 'bg-green-50 border border-green-200' : 'bg-gray-50'
                          }`}>
                            <div className="flex items-center space-x-3">
                              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                index === 0 ? 'bg-yellow-100 text-yellow-800' :
                                index === 1 ? 'bg-gray-100 text-gray-800' :
                                index === 2 ? 'bg-orange-100 text-orange-800' :
                                'bg-blue-100 text-blue-800'
                              }`}>
                                {index + 1}
                              </div>
                              <span className={`font-medium ${isTargetWord ? 'text-green-800' : ''}`}>
                                {prediction.word}
                                {isTargetWord && <span className="ml-2 text-green-600">✓ Target</span>}
                              </span>
                            </div>
                            <div className="flex items-center space-x-2">
                              {React.createElement(getConfidenceIcon(prediction.confidence * 100), {
                                className: `h-4 w-4 ${getConfidenceColor(prediction.confidence * 100)}`
                              })}
                              <span className={`text-sm font-medium ${getConfidenceColor(prediction.confidence * 100)}`}>
                                {(prediction.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="font-medium">Practice a sign to see results</p>
                  <p className="text-sm mt-1">Select a target word and start practicing</p>
                </div>
              )}
            </div>

            {/* Your Progress */}
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-900 border border-blue-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Trophy className="h-5 w-5 mr-2 text-blue-600" />
                Your Progress
                {isLoadingProgress && (
                  <RefreshCw className="h-4 w-4 ml-2 text-blue-600 animate-spin" />
                )}
              </h3>
              
              {user && token ? (
                <>
                  {/* Signs Learned Progress */}
                  <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">Signs Learned</span>
                      <span className="text-sm font-bold text-blue-600">
                        {userProgress.signs_learned}/{userProgress.total_signs}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div 
                        className="h-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-1000"
                        style={{ width: `${userProgress.total_signs > 0 ? (userProgress.signs_learned / userProgress.total_signs) * 100 : 0}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {userProgress.total_signs > 0 ? Math.round((userProgress.signs_learned / userProgress.total_signs) * 100 * 10) / 10 : 0}% Complete
                    </div>
                  </div>

                  {/* Level Progress */}
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">Level</span>
                      <span className="text-sm font-bold text-purple-600">{userProgress.current_level}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-xs text-gray-600 mb-1">
                      <span>{userProgress.current_xp} XP</span>
                      <span>•</span>
                      <span>Next level: {userProgress.next_level_xp} XP</span>
                    </div>
                    <div className="flex items-center space-x-2 text-xs text-gray-600">
                      <span>Beginner</span>
                      <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-1 bg-gradient-to-r from-green-400 to-blue-500 rounded-full transition-all duration-1000" 
                          style={{ width: `${Math.max(0, Math.min(100, userProgress.level_progress || 0))}%` }}
                        ></div>
                      </div>
                      <span>Expert</span>
                    </div>
                  </div>

                  {/* Quick Stats Grid */}
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="text-center p-3 bg-white/60 rounded-lg">
                      <div className="text-lg font-bold text-blue-600">{userProgress.signs_mastered}</div>
                      <div className="text-xs text-gray-600">Signs Mastered</div>
                    </div>
                    <div className="text-center p-3 bg-white/60 rounded-lg">
                      <div className="text-lg font-bold text-green-600">
                        {Math.max(0, userProgress.total_signs - userProgress.signs_learned)}
                      </div>
                      <div className="text-xs text-gray-600">Signs Remaining</div>
                    </div>
                  </div>

                  {/* Additional Stats */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="text-center p-3 bg-white/40 rounded-lg">
                      <div className="text-lg font-bold text-purple-600">{userProgress.accuracy_rate.toFixed(1)}%</div>
                      <div className="text-xs text-gray-600">Accuracy</div>
                    </div>
                    <div className="text-center p-3 bg-white/40 rounded-lg">
                      <div className="text-lg font-bold text-orange-600">{userProgress.practice_streak}</div>
                      <div className="text-xs text-gray-600">Day Streak</div>
                    </div>
                  </div>

                  {/* Level Badges */}
                  <div className="mt-4 pt-4 border-t border-blue-200">
                    <div className="text-xs text-gray-600 mb-2">Achievement Levels</div>
                    <div className="flex justify-between">
                      <div className="flex flex-col items-center">
                        <div className={`w-6 h-6 rounded-full text-white text-xs flex items-center justify-center font-bold ${
                          ['Beginner', 'Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level)
                            ? 'bg-green-500' : 'bg-gray-200 text-gray-400'
                        }`}>
                          {['Beginner', 'Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level) ? '✓' : '1'}
                        </div>
                        <span className={`text-xs mt-1 ${
                          ['Beginner', 'Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level) 
                            ? 'text-green-600' : 'text-gray-500'
                        }`}>Beginner</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <div className={`w-6 h-6 rounded-full text-xs flex items-center justify-center ${
                          ['Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level)
                            ? 'bg-green-500 text-white font-bold' : 'bg-gray-200 text-gray-400'
                        }`}>
                          {['Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level) ? '✓' : '2'}
                        </div>
                        <span className={`text-xs mt-1 ${
                          ['Intermediate', 'Advanced', 'Expert'].includes(userProgress.current_level) 
                            ? 'text-green-600' : 'text-gray-500'
                        }`}>Intermediate</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <div className={`w-6 h-6 rounded-full text-xs flex items-center justify-center ${
                          ['Advanced', 'Expert'].includes(userProgress.current_level)
                            ? 'bg-green-500 text-white font-bold' : 'bg-gray-200 text-gray-400'
                        }`}>
                          {['Advanced', 'Expert'].includes(userProgress.current_level) ? '✓' : '3'}
                        </div>
                        <span className={`text-xs mt-1 ${
                          ['Advanced', 'Expert'].includes(userProgress.current_level) 
                            ? 'text-green-600' : 'text-gray-400'
                        }`}>Advanced</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <div className={`w-6 h-6 rounded-full text-xs flex items-center justify-center ${
                          userProgress.current_level === 'Expert'
                            ? 'bg-green-500 text-white font-bold' : 'bg-gray-200 text-gray-400'
                        }`}>
                          {userProgress.current_level === 'Expert' ? '✓' : '4'}
                        </div>
                        <span className={`text-xs mt-1 ${
                          userProgress.current_level === 'Expert' 
                            ? 'text-green-600' : 'text-gray-400'
                        }`}>Expert</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <div className="mb-3">
                    <User className="h-12 w-12 mx-auto text-gray-300" />
                  </div>
                  <p className="text-sm font-medium text-gray-600 mb-2">Login Required</p>
                  <p className="text-xs text-gray-500">
                    Please log in to track your learning progress and achievements
                  </p>
                </div>
              )}
            </div>

            {/* Practice Stats */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1000">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 mr-2 text-purple-600" />
                Practice Stats
              </h3>
              
              {user && token ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <Timer className="h-5 w-5 text-blue-600 mx-auto mb-1" />
                    <div className="text-lg font-bold text-blue-600">{formatTime(sessionTime)}</div>
                    <div className="text-xs text-blue-600">Session Time</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <Award className="h-5 w-5 text-green-600 mx-auto mb-1" />
                    <div className="text-lg font-bold text-green-600">{userProgress.practice_streak}</div>
                    <div className="text-xs text-green-600">Day Streak</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <Star className="h-5 w-5 text-purple-600 mx-auto mb-1" />
                    <div className="text-lg font-bold text-purple-600">{userProgress.accuracy_rate}%</div>
                    <div className="text-xs text-purple-600">Accuracy Rate</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded-lg">
                    <TrendingUp className="h-5 w-5 text-orange-600 mx-auto mb-1" />
                    <div className="text-lg font-bold text-orange-600">{userProgress.signs_practiced}</div>
                    <div className="text-xs text-orange-600">Signs Practiced</div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  <p className="text-sm">Login to view your practice statistics</p>
                </div>
              )}
            </div>

            {/* Practice Goals */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Award className="h-5 w-5 mr-2 text-yellow-600" />
                Practice Goals
              </h3>
              <div className="space-y-4">
                {practiceGoals.map((goal, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700">{goal.title}</span>
                      <span className={`text-sm font-bold text-${goal.color}-600`}>
                        {goal.current}/{goal.target} {goal.unit}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div 
                        className={`h-2 rounded-full bg-gradient-to-r from-${goal.color}-500 to-${goal.color}-400 transition-all duration-1000`}
                        style={{ width: `${Math.min(100, (goal.current / goal.target) * 100)}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Practice History */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1400">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
                Recent Practice
              </h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {practiceHistory.length > 0 ? (
                  practiceHistory.map((practice, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium text-gray-900">{practice.word}</div>
                        <div className="flex items-center space-x-2 text-xs text-gray-500">
                          <span>{new Date(practice.timestamp).toLocaleTimeString()}</span>
                          <span>•</span>
                          <span className="capitalize">{practice.model}</span>
                        </div>
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                        practice.isMatch 
                          ? 'bg-green-100 text-green-700' 
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {practice.isMatch ? '✓ Match' : '✗ Miss'}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-500">
                    <p className="text-sm">No practice history yet</p>
                    <p className="text-xs mt-1">Start practicing to see your progress</p>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1600">
              <div className="space-y-3">
                <button
                  onClick={clearResults}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors duration-200"
                >
                  <RotateCcw className="h-4 w-4" />
                  <span>Clear Results</span>
                </button>
              </div>
            </div>

            {/* Practice Tips */}
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 animate-fade-in-up animation-delay-1800">
              <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-yellow-500" />
                Practice Tips
              </h3>
              <ul className="space-y-2 text-sm text-blue-800">
                {[
                  'Ensure good lighting for better detection',
                  'Keep your hands clearly visible in frame',
                  'Practice slowly and deliberately',
                  'Use live prediction for instant feedback',
                  'Try both camera and video upload modes',
                  'Compare results between Pro and Mini models'
                ].map((tip, index) => (
                  <li key={index} className="flex items-start">
                    <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 mr-3 flex-shrink-0"></div>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Practice;
