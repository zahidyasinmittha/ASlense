import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  Camera, CameraOff, Target, CheckCircle, AlertCircle, 
  Timer, BarChart3, Video, RefreshCw, Trophy, User, 
  Wifi, Search, Upload
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
  
  // Practice mode states - Camera and Upload modes
  const [practiceMode, setPracticeMode] = useState<'camera' | 'upload'>('camera');
  const [selectedModel, setSelectedModel] = useState<'mini' | 'pro'>('mini');
  const [targetWord, setTargetWord] = useState('');
  const [availableWords, setAvailableWords] = useState<string[]>([]);
  const [wordSearch, setWordSearch] = useState('');
  const [isWordDropdownOpen, setIsWordDropdownOpen] = useState(false);
  
  // Camera states - Clean and simple
  const [isRecording, setIsRecording] = useState(false);
  // Simple session timer - starts from 0 each time
  const [sessionTime, setSessionTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [capturedFrameCount, setCapturedFrameCount] = useState(0);
  
  // WebSocket states for live prediction
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [isLivePredicting, setIsLivePredicting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Video upload states (for upload mode only)
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  
  // Results states
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [practiceHistory, setPracticeHistory] = useState<any[]>(() => {
    // Load practice history from localStorage on component mount
    try {
      const savedHistory = localStorage.getItem('asl-practice-history');
      return savedHistory ? JSON.parse(savedHistory) : [];
    } catch (error) {
      console.error('Error loading practice history from localStorage:', error);
      return [];
    }
  });
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
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const wordDropdownRef = useRef<HTMLDivElement>(null);

  // Memoized filtered words computation
  const filteredWords = useMemo(() => {
    if (!wordSearch.trim()) {
      return availableWords;
    }
    
    const searchTerm = wordSearch.toLowerCase();
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
    
    return [...exactMatches, ...partialMatches];
  }, [wordSearch, availableWords]);

  // Fetch user progress data
  const fetchUserProgress = useCallback(async () => {
    if (!user || !token) return;
    
    try {
      const response = await makeAuthenticatedRequest(`${baseUrl}/user/progress`);
      
      if (response.ok) {
        const data = await response.json();
        setUserProgress({
          signs_practiced: data.signs_practiced || 0,
          signs_learned: data.signs_learned || 0,
          total_signs: data.total_signs || 136,
          accuracy_rate: Math.round((data.accuracy_rate || 0) * 100) / 100,
          current_level: data.current_level || 'Beginner',
          current_xp: data.current_xp || 0,
          next_level_xp: data.next_level_xp || 100,
          level_progress: Math.round(((data.current_xp || 0) / (data.next_level_xp || 100)) * 100),
          practice_streak: data.practice_streak || 0,
          total_practice_time: data.total_practice_time || 0,
          signs_mastered: data.signs_mastered || 0
        });
      }
    } catch (error) {
      console.error('Error fetching user progress:', error);
    }
  }, [user, token, baseUrl, makeAuthenticatedRequest]);

  // Load available words
  const loadAvailableWords = useCallback(async () => {
    try {
      const response = await axios.get(`${baseUrl}/practice/available-words`);
      setAvailableWords((response.data as any).words || []);
    } catch (error) {
      console.error('Error loading available words:', error);
      setAvailableWords([
        'Hello', 'Thank You', 'Please', 'Sorry', 'Yes', 'No',
        'Good', 'Bad', 'Happy', 'Sad', 'Love', 'Family',
        'he', 'study', 'there', 'analyze', 'fine'
      ]);
    }
  }, [baseUrl]);

  // Load recent practice history from backend
  const fetchRecentPracticeHistory = useCallback(async () => {
    if (!user || !token) return;
    
    try {
      const response = await makeAuthenticatedRequest(`${baseUrl}/practice/recent-history`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.recent_practice && Array.isArray(data.recent_practice)) {
          // Merge with localStorage data, prioritizing backend data
          const backendHistory = data.recent_practice.map((item: any) => ({
            word: item.target_word,
            predictions: item.predictions || [],
            isMatch: item.is_correct,
            isTop4Correct: item.is_top_4_correct,
            confidence: item.confidence,
            timestamp: item.created_at || new Date().toISOString(),
            model: item.model_used || 'unknown'
          }));
          
          // Combine with localStorage data, remove duplicates
          setPracticeHistory(prev => {
            const combined = [...backendHistory, ...prev];
            const unique = combined.filter((item, index, self) => 
              index === self.findIndex(t => 
                t.word === item.word && 
                Math.abs(new Date(t.timestamp).getTime() - new Date(item.timestamp).getTime()) < 60000
              )
            );
            return unique.slice(0, 10); // Keep only 10 most recent
          });
        }
      }
    } catch (error) {
      console.error('Error fetching recent practice history:', error);
    }
  }, [user, token, baseUrl, makeAuthenticatedRequest]);

  // Effects
  useEffect(() => {
    loadAvailableWords();
  }, [loadAvailableWords]);

  useEffect(() => {
    if (user && token) {
      fetchUserProgress();
      fetchRecentPracticeHistory();
    }
  }, [user, token, fetchUserProgress, fetchRecentPracticeHistory]);

  // Simple timer effect - just counts seconds from component mount
  useEffect(() => {
    if (user) {
      console.log('⏰ Practice Component - Starting simple timer');
      
      // Update timer every second
      const timerInterval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 1000);

      return () => {
        console.log('⏹️ Practice Component - Clearing timer');
        clearInterval(timerInterval);
      };
    }
  }, [user]);

  // Save practice history to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('asl-practice-history', JSON.stringify(practiceHistory));
    } catch (error) {
      console.error('Error saving practice history to localStorage:', error);
    }
  }, [practiceHistory]);

  // Timer effect for recording - Remove this as we now use session manager
  // useEffect(() => {
  //   let interval: number;
  //   if (isRecording) {
  //     interval = setInterval(() => {
  //       setSessionTime(prev => prev + 1);
  //     }, 1000);
  //   }
  //   return () => clearInterval(interval);
  // }, [isRecording]);

  // Auto-hide notification
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
      // Note: Don't end session here as we want it to persist across page reloads
      // Session will only end on browser close, tab close, or logout via session manager
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

  // Notification handler
  const showNotification = useCallback((type: 'error' | 'success' | 'info', message: string) => {
    setNotification({ type, message });
  }, []);

  // Frame capture function - Clean and simple
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return null;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.95);
    
    if (frameDataUrl.length < 1000) {
      return null;
    }

    return frameDataUrl;
  }, []);

  // Camera functions - Clean and simple
  const startCamera = async () => {
    setIsCameraLoading(true);
    try {
      // Check for basic camera support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showNotification('error', 'Camera access is not supported in this browser. Please use Chrome, Firefox, or Safari.');
        return;
      }

      // Check if we're in a secure context (HTTPS or localhost)
      if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        showNotification('error', 'Camera access requires HTTPS or localhost. Please use a secure connection.');
        return;
      }

      // Check camera permissions first
      try {
        const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
        console.log('Camera permission status:', permissions.state);
        if (permissions.state === 'denied') {
          showNotification('error', 'Camera permission is denied. Please enable camera access in your browser settings.');
          return;
        }
      } catch (permError) {
        console.log('Could not check camera permissions:', permError);
        // Continue anyway as some browsers don't support permissions API
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1920, min: 1280 },
          height: { ideal: 1080, min: 720 },
          facingMode: 'user',
          frameRate: { ideal: 30, min: 15 }
        },
        audio: false 
      });
      
      setCameraStream(stream);
      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        
        const waitForVideoReady = new Promise<void>((resolve) => {
          const checkReady = () => {
            if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 3) {
              resolve();
            } else {
              setTimeout(checkReady, 100);
            }
          };
          checkReady();
        });
        
        video.play().then(() => {
          return waitForVideoReady;
        }).then(() => {
          console.log('Camera is ready!');
        }).catch(error => {
          console.log('Error with video:', error);
        });
      }
      setIsRecording(true);
      showNotification('success', 'Camera started successfully!');
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        code: error.code,
        constraint: error.constraint
      });
      
      let errorMessage = 'Could not access camera. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Camera permission was denied. Please allow camera access and try again.';
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera device found. Please check if a camera is connected.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application. Please close other apps using the camera.';
      } else if (error.name === 'OverconstrainedError') {
        errorMessage += 'Camera does not support the required settings. Trying with basic settings...';
        // Try again with simpler constraints
        try {
          const simpleStream = await navigator.mediaDevices.getUserMedia({ 
            video: true,
            audio: false 
          });
          setCameraStream(simpleStream);
          if (videoRef.current) {
            videoRef.current.srcObject = simpleStream;
            await videoRef.current.play();
          }
          setIsRecording(true);
          showNotification('success', 'Camera started with basic settings!');
          return;
        } catch (retryError) {
          console.error('Retry with simple settings failed:', retryError);
          errorMessage += ' Basic camera access also failed.';
        }
      } else if (error.name === 'NotSupportedError') {
        errorMessage += 'Camera access is not supported in this browser or requires HTTPS.';
      } else {
        errorMessage += `Please check your camera settings. Error: ${error.message || 'Unknown error'}`;
      }
      
      showNotification('error', errorMessage);
    } finally {
      setIsCameraLoading(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setIsRecording(false);
    // Don't reset session time - let it persist via session manager
    
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
    }
    setConnectionStatus('disconnected');
    setIsLivePredicting(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }, [cameraStream, wsConnection]);

  // WebSocket connection - Clean and simple
  const connectWebSocket = useCallback(() => {
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
    }
    
    setConnectionStatus('connecting');
    console.log('🏃 PRACTICE: Attempting WebSocket connection...');
    console.log(`🏃 PRACTICE: URL: ${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
    console.log(`🏃 PRACTICE: Base URL: ${baseUrl}`);
    console.log(`🏃 PRACTICE: WS URL: ${wsUrl}`);
    console.log(`🏃 PRACTICE: Model: ${selectedModel}`);
    
    const ws = new WebSocket(`${wsUrl}/practice/live-predict?model_type=${selectedModel}`);
    
    ws.onopen = (event) => {
      console.log('✅ PRACTICE: WebSocket connected successfully!', event);
      setConnectionStatus('connected');
      setWsConnection(ws);
      showNotification('success', 'Connected to prediction service!');
    };
    
    ws.onmessage = (event) => {
      try {
        const data: LivePredictionMessage = JSON.parse(event.data);
        console.log('📥 PRACTICE: Received message:', data);
        
        switch (data.type) {
          case 'connected':
            console.log('🔗 PRACTICE: Connection confirmed by server');
            break;
            
          case 'frame_received':
            setCapturedFrameCount(data.frame_count || 0);
            break;
            
          case 'final_result':
            if (data.result) {
              handlePredictionResult(data.result);
            }
            break;
            
          case 'error':
            console.log('❌ PRACTICE: Server error:', data.message);
            showNotification('error', data.message || 'Prediction error occurred');
            break;
        }
      } catch (error) {
        console.error('❌ PRACTICE: Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = (event) => {
      console.log('🔌 PRACTICE: WebSocket closed:', event.code, event.reason);
      setConnectionStatus('disconnected');
      setWsConnection(null);
    };
    
    ws.onerror = (error) => {
      console.error('❌ PRACTICE: WebSocket error:', error);
      console.log('❌ PRACTICE: WebSocket state:', ws.readyState);
      setConnectionStatus('disconnected');
      showNotification('error', 'Connection failed. Check if backend is running.');
    };
  }, [wsUrl, selectedModel, wsConnection, showNotification, baseUrl]);

  // Auto prediction - Start sending frames immediately when connected
  const startPrediction = useCallback(async () => {
    if (!targetWord) {
      showNotification('error', 'Please select a target word first');
      return;
    }

    if (!videoRef.current || !cameraStream) {
      showNotification('error', 'Camera is not active');
      return;
    }

    // Connect WebSocket if not connected
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      
      // Wait for connection with timeout
      let attempts = 0;
      const checkConnection = async (): Promise<boolean> => {
        return new Promise((resolve) => {
          const check = () => {
            if ((wsConnection && wsConnection.readyState === WebSocket.OPEN) || attempts >= 10) {
              resolve(wsConnection?.readyState === WebSocket.OPEN || false);
            } else {
              attempts++;
              setTimeout(check, 500);
            }
          };
          check();
        });
      };
      
      const connected = await checkConnection();
      if (!connected) {
        showNotification('error', 'Failed to connect to prediction service');
        return;
      }
    }

    setIsLivePredicting(true);
    setCapturedFrameCount(0);
    
    // Auto-start sending frames to backend
    let frameCount = 0;
    frameIntervalRef.current = window.setInterval(() => {
      if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        return;
      }
      
      const frameDataUrl = captureFrame();
      
      if (frameDataUrl) {
        wsConnection.send(JSON.stringify({
          type: 'frame',
          frame: frameDataUrl
        }));
        frameCount++;
        setCapturedFrameCount(frameCount);
      }
    }, 150); // Auto-capture frame every 150ms
    
    showNotification('info', `Auto-prediction started for: ${targetWord}`);
  }, [targetWord, videoRef, cameraStream, connectWebSocket, wsConnection, captureFrame]);

  // Stop prediction and auto-analyze
  const stopPredictionAndAnalyze = useCallback(() => {
    setIsLivePredicting(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN && targetWord) {
      wsConnection.send(JSON.stringify({ type: 'stop' }));
      
      // Auto-analyze after stopping
      setTimeout(() => {
        wsConnection.send(JSON.stringify({ 
          type: 'analyze', 
          target_word: targetWord 
        }));
        showNotification('info', `Auto-analyzing ${capturedFrameCount} frames for: ${targetWord}`);
        setIsProcessing(true);
      }, 500);
    }
  }, [wsConnection, targetWord, capturedFrameCount]);

  // Handle prediction result
  const handlePredictionResult = useCallback(async (result: PredictionResult) => {
    setPredictionResult(result);
    setIsProcessing(false);
    
    // Close WebSocket connection after getting result
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
      setConnectionStatus('disconnected');
    }
    
    // No need to track predictions in session - just submit to backend
    
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
            is_correct: result.is_top_4_correct,
            model_used: result.model_used,
            practice_mode: practiceMode,
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
    
    setPracticeHistory(prev => [historyEntry, ...prev.slice(0, 9)]);
  }, [user, token, baseUrl, fetchUserProgress, wsConnection, practiceMode, makeAuthenticatedRequest]);

  // Video upload functions (for upload mode only)
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

  const predictUploadedVideo = useCallback(async () => {
    if (!uploadedVideo || !targetWord) {
      showNotification('error', 'Please select a video file and target word');
      return;
    }

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

  // Word selection handlers
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

  // Clear results
  const clearResults = useCallback(() => {
    setPredictionResult(null);
    setUploadedVideo(null);
    setVideoPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Clear practice history
  const clearPracticeHistory = useCallback(() => {
    setPracticeHistory([]);
    try {
      localStorage.removeItem('asl-practice-history');
    } catch (error) {
      console.error('Error clearing practice history from localStorage:', error);
    }
  }, []);

  // Mode change handler
  const handlePracticeModeChange = useCallback((mode: 'camera' | 'upload') => {
    setPracticeMode(mode);
    clearResults();
    // No session tracking needed
  }, [clearResults]);

  // Utility functions
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const getConfidenceColor = useCallback((confidence: number) => {
    if (confidence >= 0.001) return 'text-green-600';
    if (confidence >= 0.0006) return 'text-yellow-600';
    return 'text-red-600';
  }, []);

  // Memoized data
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
        {/* Header */}
        <div className="text-center mb-12">
          <div className="relative group mb-8">
            <div className="relative">
              <Target className="h-20 w-20 text-blue-600 mx-auto mb-4" />
              <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 bg-clip-text text-transparent mb-4">
                ASL Practice
              </h1>
              <p className="text-xl text-gray-600 font-medium">
                Practice American Sign Language with real-time AI recognition
              </p>
            </div>
          </div>
          
          {/* Status Indicators */}
          <div className="flex items-center justify-center mt-4 space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
              <span className="text-sm font-medium text-gray-600">
                Camera {isRecording ? 'Active' : 'Inactive'}
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' :
                connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 'bg-gray-300'
              }`}></div>
              <span className="text-sm font-medium text-gray-600">
                {connectionStatus === 'connected' ? 'Connected' :
                 connectionStatus === 'connecting' ? 'Connecting' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* Notification */}
        {notification && (
          <div className={`fixed top-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg transition-all duration-300 ${
            notification.type === 'error' 
              ? 'bg-red-100 border border-red-200 text-red-800'
              : notification.type === 'success'
              ? 'bg-green-100 border border-green-200 text-green-800'
              : 'bg-blue-100 border border-blue-200 text-blue-800'
          }`}>
            <div className="flex items-start space-x-3">
              {notification.type === 'error' && <AlertCircle className="h-5 w-5 mt-0.5" />}
              {notification.type === 'success' && <CheckCircle className="h-5 w-5 mt-0.5" />}
              {notification.type === 'info' && <AlertCircle className="h-5 w-5 mt-0.5" />}
              <p className="font-medium">{notification.message}</p>
            </div>
          </div>
        )}

        {/* Practice Mode Selector */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 border border-white/20">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Choose Your Practice Mode</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <button
              onClick={() => handlePracticeModeChange('camera')}
              className={`group p-8 rounded-2xl border-3 transition-all duration-500 transform hover:scale-105 ${
                practiceMode === 'camera'
                  ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-purple-50 text-blue-700 shadow-2xl shadow-blue-500/25'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-blue-300 hover:shadow-xl'
              }`}
            >
              <div className="flex flex-col items-center text-center">
                <Camera className="h-16 w-16 mb-4 group-hover:scale-110 transition-transform duration-300" />
                <h3 className="text-2xl font-bold mb-2">Live Camera</h3>
                <p className="text-sm opacity-75">Real-time ASL recognition using your camera</p>
              </div>
            </button>
            
            <button
              onClick={() => handlePracticeModeChange('upload')}
              className={`group p-8 rounded-2xl border-3 transition-all duration-500 transform hover:scale-105 ${
                practiceMode === 'upload'
                  ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 text-purple-700 shadow-2xl shadow-purple-500/25'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-purple-300 hover:shadow-xl'
              }`}
            >
              <div className="flex flex-col items-center text-center">
                <Upload className="h-16 w-16 mb-4 group-hover:scale-110 transition-transform duration-300" />
                <h3 className="text-2xl font-bold mb-2">Video Upload</h3>
                <p className="text-sm opacity-75">Upload and analyze pre-recorded ASL videos</p>
              </div>
            </button>
          </div>
        </div>

        {/* Practice Configuration */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 border border-white/20">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Practice Configuration</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Target Word Selection */}
            <div className="space-y-4" ref={wordDropdownRef}>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Target Word</label>
              <div className="relative">
                <div className="flex items-center space-x-2">
                  <div className="relative flex-1">
                    <input
                      type="text"
                      value={wordSearch}
                      onChange={(e) => handleWordSearchChange(e.target.value)}
                      placeholder="Search for a word to practice..."
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-white shadow-sm"
                      onFocus={() => setIsWordDropdownOpen(true)}
                    />
                    <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                  </div>
                  
                  {wordSearch && (
                    <button
                      onClick={handleClearWordSearch}
                      className="p-2 text-gray-500 hover:text-red-500 transition-colors"
                      title="Clear selection"
                    >
                      Ã—
                    </button>
                  )}
                </div>

                {isWordDropdownOpen && filteredWords.length > 0 && (
                  <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-xl shadow-lg max-h-60 overflow-y-auto">
                    {filteredWords.slice(0, 10).map((word, index) => (
                      <button
                        key={`${word}-${index}`}
                        onClick={() => handleWordSelection(word)}
                        className="w-full px-4 py-3 text-left hover:bg-blue-50 transition-colors border-b border-gray-100 last:border-b-0"
                      >
                        {word}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              
              {targetWord && (
                <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <span className="text-blue-800 font-medium">Selected: {targetWord}</span>
                </div>
              )}
            </div>

            {/* Model Selection */}
            <div className="space-y-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">AI Model</label>
              <div className="grid grid-cols-1 gap-3">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id as 'mini' | 'pro')}
                    className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                      selectedModel === model.id
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 bg-white text-gray-700 hover:border-blue-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="text-left">
                        <div className="font-bold">{model.name}</div>
                        <div className="text-sm opacity-75">{model.description}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-green-600">{model.accuracy}</div>
                        <div className="text-xs text-gray-500">{model.speed}</div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          
          {/* Main Practice Area */}
          <div className="xl:col-span-3">
            
            {practiceMode === 'camera' ? (
              /* Camera Mode - Clean 2-Button Interface */
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl overflow-hidden border border-white/30 mb-8">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Camera className="h-12 w-10 text-white" />
                      <div>
                        <h3 className="text-xl font-bold text-white">Live Camera</h3>
                        <p className="text-blue-100">Real-time ASL recognition</p>
                      </div>
                    </div>
                    
                    {isRecording && (
                      <div className="flex items-center space-x-4 text-white">
                        <div className="flex items-center space-x-2">
                          <Timer className="h-5 w-5" />
                          <span className="font-mono text-lg">{formatTime(sessionTime)}</span>
                        </div>
                        {isLivePredicting && (
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                            <span className="text-sm">Predicting</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div className="relative bg-black">
                  <video
                    ref={videoRef}
                    className="w-full h-120 object-cover"
                    autoPlay
                    muted
                    playsInline
                  />
                  <canvas
                    ref={canvasRef}
                    className="hidden"
                  />
                  
                  {!isRecording && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
                      <div className="text-center text-white">
                        <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p className="text-lg font-medium">Camera not active</p>
                        <p className="text-gray-300">Click "Start Camera" to begin</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-6 bg-gradient-to-r from-gray-50 to-blue-50">
                  <div className="flex flex-col items-center space-y-4">
                    
                    {/* Simple 2-Button Interface */}
                    <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 w-full">
                      
                      {/* Button 1: Camera Start/Stop */}
                      {!isRecording ? (
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
                          <span>{isCameraLoading ? 'Loading Camera...' : 'Start Camera'}</span>
                        </button>
                      ) : (
                        <button
                          onClick={stopCamera}
                          className="flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl w-full sm:w-auto"
                        >
                          <CameraOff className="h-5 w-5" />
                          <span>Stop Camera</span>
                        </button>
                      )}

                      {/* Button 2: Prediction Start/Stop */}
                      {isRecording && (
                        <button
                          onClick={() => {
                            if (!isLivePredicting) {
                              startPrediction();
                            } else {
                              stopPredictionAndAnalyze();
                            }
                          }}
                          disabled={!targetWord || connectionStatus === 'connecting' || isProcessing}
                          className={`flex items-center space-x-3 px-8 py-4 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none w-full sm:w-auto ${
                            isLivePredicting 
                              ? 'bg-gradient-to-r from-red-600 to-red-700 animate-pulse' 
                              : 'bg-gradient-to-r from-blue-600 to-purple-600'
                          }`}
                        >
                          {connectionStatus === 'connecting' ? (
                            <RefreshCw className="h-5 w-5 animate-spin" />
                          ) : isProcessing ? (
                            <RefreshCw className="h-5 w-5 animate-spin" />
                          ) : isLivePredicting ? (
                            <CheckCircle className="h-5 w-5" />
                          ) : (
                            <Target className="h-5 w-5" />
                          )}
                          <span>
                            {connectionStatus === 'connecting' 
                              ? 'Connecting...' 
                              : isProcessing
                              ? 'Analyzing...'
                              : isLivePredicting 
                                ? `Stop & Analyze (${capturedFrameCount} frames)`
                                : 'Start Prediction'
                            }
                          </span>
                        </button>
                      )}
                    </div>

                    {/* Status indicators */}
                    {isRecording && (
                      <div className="flex items-center justify-center space-x-4 text-sm">
                        {isLivePredicting && (
                          <div className="flex items-center space-x-2 bg-green-100 rounded-lg px-3 py-2">
                            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                            <span className="text-green-700 font-medium">
                              Auto-Capturing: {targetWord} ({capturedFrameCount} frames)
                            </span>
                          </div>
                        )}
                        
                        {connectionStatus === 'connected' && (
                          <div className="flex items-center space-x-2 bg-blue-100 rounded-lg px-3 py-2">
                            <Wifi className="h-4 w-4 text-blue-600" />
                            <span className="text-blue-700 font-medium">Connected</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Instructions */}
                    {!targetWord && (
                      <p className="text-sm text-amber-600 flex items-center justify-center">
                        <AlertCircle className="h-4 w-4 mr-1" />
                        Please select a target word first
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Upload Mode - Improved Layout */
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl overflow-hidden border border-white/30 mb-8">
                <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6">
                  <div className="flex items-center space-x-3">
                    <Upload className="h-8 w-8 text-white" />
                    <div>
                      <h3 className="text-xl font-bold text-white">Video Upload</h3>
                      <p className="text-purple-100">Upload and analyze ASL videos</p>
                    </div>
                  </div>
                </div>

                {/* Video Display Section - Moved down with better spacing */}
                <div className="relative bg-black">
                  {videoPreview ? (
                    <div className="relative">
                      <video
                        src={videoPreview}
                        controls
                        className="w-full h-120 object-contain bg-black"
                        controlsList="nodownload"
                      />
                      <div className="absolute top-3 left-3">
                        <div className="bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm font-medium">
                          Preview: {uploadedVideo?.name}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-80 flex items-center justify-center bg-gray-900">
                      <div className="text-center text-white">
                        <Upload className="h-20 w-20 mx-auto mb-4 opacity-50" />
                        <p className="text-xl font-medium mb-2">No Video Selected</p>
                        <p className="text-gray-300">Click "Choose Video File" to upload an ASL video</p>
                        <p className="text-gray-400 text-sm mt-2">Supported formats: MP4, AVI, MOV, WebM</p>
                      </div>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center">
                      <div className="text-center text-white">
                        <RefreshCw className="h-20 w-20 mx-auto mb-6 animate-spin" />
                        <p className="text-xl font-medium mb-2">Processing Video...</p>
                        <p className="text-gray-300">Analyzing ASL signs using AI models</p>
                        <div className="mt-4 bg-white bg-opacity-20 rounded-full h-2 w-64 mx-auto">
                          <div className="bg-purple-400 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

              {/* Enhanced upload information */}
              <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 border-t border-purple-100">
                <div className="text-center">
                  <p className="text-sm text-purple-700 font-medium">
                    Upload your ASL video and our AI will analyze it for the selected target word
                  </p>
                  <p className="text-xs text-purple-600 mt-1">
                    Best results with clear hand movements and good lighting
                  </p>
                </div>
              </div>

              {/* Upload Controls Section - Moved to top */}
              <div className="p-6 bg-gradient-to-r from-gray-50 to-purple-50 border-b border-purple-100">
                  <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      className="hidden"
                    />
                    
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl w-full sm:w-auto"
                    >
                      <Upload className="h-5 w-5" />
                      <span>Choose Video File</span>
                    </button>
                    
                    <button
                      onClick={predictUploadedVideo}
                      disabled={!uploadedVideo || !targetWord || isProcessing}
                      className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none w-full sm:w-auto"
                    >
                      {isProcessing ? (
                        <RefreshCw className="h-5 w-5 animate-spin" />
                      ) : (
                        <BarChart3 className="h-5 w-5" />
                      )}
                      <span>{isProcessing ? 'Processing...' : 'Analyze Video'}</span>
                    </button>
                  </div>

                  {/* Status indicators */}
                  {uploadedVideo && (
                    <div className="mt-4 flex items-center justify-center">
                      <div className="flex items-center space-x-2 bg-purple-100 rounded-lg px-4 py-2">
                        <CheckCircle className="h-4 w-4 text-purple-600" />
                        <span className="text-purple-700 font-medium text-sm">
                          Video ready: {uploadedVideo.name}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Instructions */}
                  {!targetWord && (
                    <div className="mt-4 flex items-center justify-center">
                      <p className="text-sm text-amber-600 flex items-center">
                        <AlertCircle className="h-4 w-4 mr-1" />
                        Please select a target word first
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Results Section */}
            {predictionResult && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/30">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-bold text-gray-900">Prediction Results</h3>
                  <button
                    onClick={clearResults}
                    className="text-gray-500 hover:text-red-500 transition-colors"
                  >
                    Clear Results
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Target vs Result */}
                  <div className="space-y-4">
                    <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl border border-blue-200">
                      <h4 className="text-lg font-semibold text-gray-700 mb-2">Target Word</h4>
                      <p className="text-3xl font-bold text-blue-600">{predictionResult.target_word}</p>
                    </div>

                    <div className={`text-center p-6 rounded-xl border ${
                      predictionResult.is_top_4_correct || predictionResult.target_word.toLowerCase() === predictionResult.predictions[0]?.word?.toLowerCase()
                        ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
                        : 'bg-gradient-to-br from-red-50 to-pink-50 border-red-200'
                    }`}>
                      <h4 className="text-lg font-semibold text-gray-700 mb-2">Top Prediction</h4>
                      <p className={`text-3xl font-bold ${
                        predictionResult.is_top_4_correct || predictionResult.target_word.toLowerCase() === predictionResult.predictions[0]?.word?.toLowerCase() ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {predictionResult.predictions[0]?.word || 'No prediction'}
                      </p>
                      <p className={`text-sm font-medium mt-1 ${getConfidenceColor(predictionResult.predictions[0]?.confidence || 0)}`}>
                        {((predictionResult.predictions[0]?.confidence || 0) * 100).toFixed(4)}% confidence
                      </p>
                    </div>
                  </div>

                  {/* All Predictions */}
                  <div className="space-y-3">
                    <h4 className="text-lg font-semibold text-gray-700">All Predictions</h4>
                    {predictionResult.predictions.slice(0, 4).map((pred, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border"
                      >
                        <div className="flex items-center space-x-3">
                          <span className="flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-600 rounded-full text-sm font-bold">
                            {index + 1}
                          </span>
                          <span className="font-medium text-gray-900">{pred.word}</span>
                        </div>
                        <div className="text-right">
                          <span className={`font-semibold ${getConfidenceColor(pred.confidence)}`}>
                            {(pred.confidence * 100).toFixed(4)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Performance Feedback */}
                <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {predictionResult.is_top_4_correct || predictionResult.target_word.toLowerCase() === predictionResult.predictions[0]?.word?.toLowerCase() ? (
                        <CheckCircle className="h-6 w-6 text-green-600" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-600" />
                      )}
                      <span className="font-medium text-gray-700">
                        {predictionResult.is_top_4_correct || predictionResult.target_word.toLowerCase() === predictionResult.predictions[0]?.word?.toLowerCase() ? 'âœ… Your prediction is correct!' : 'âŒ Prediction not matched - Keep practicing!'}
                      </span>
                    </div>
                    
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Model: {predictionResult.model_used}</p>
                      {predictionResult.user_xp_gained && (
                        <p className="text-sm text-green-600">+{predictionResult.user_xp_gained} XP earned!</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Clear Success Message */}
                {(predictionResult.is_top_4_correct || predictionResult.target_word.toLowerCase() === predictionResult.predictions[0]?.word?.toLowerCase()) && (
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl text-center">
                    <div className="flex items-center justify-center space-x-2">
                      <CheckCircle className="h-8 w-8 text-green-600" />
                      <h4 className="text-xl font-bold text-green-700">
                        ðŸŽ‰ Your prediction is correct!
                      </h4>
                    </div>
                    <p className="text-green-600 mt-2">
                      You successfully performed the ASL sign for "{predictionResult.target_word}"
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            
            {/* User Progress */}
            {user && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-white/30">
                <div className="flex items-center space-x-3 mb-4">
                  <User className="h-6 w-6 text-blue-600" />
                  <h3 className="text-lg font-bold text-gray-900">Your Progress</h3>
                </div>

                <div className="space-y-4">
                  {/* Level & XP */}
                  <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl">
                    <div className="flex items-center justify-center space-x-2 mb-2">
                      <Trophy className="h-5 w-5 text-purple-600" />
                      <span className="font-semibold text-purple-600">{userProgress.current_level}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${userProgress.level_progress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600">
                      {userProgress.current_xp} / {userProgress.next_level_xp} XP
                    </p>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <p className="text-2xl font-bold text-blue-600">{userProgress.signs_practiced}</p>
                      <p className="text-xs text-gray-600">Signs Practiced</p>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <p className="text-2xl font-bold text-green-600">{Math.round(userProgress.accuracy_rate)}%</p>
                      <p className="text-xs text-gray-600">Accuracy</p>
                    </div>
                    <div className="text-center p-3 bg-yellow-50 rounded-lg">
                      <p className="text-2xl font-bold text-yellow-600">{userProgress.practice_streak}</p>
                      <p className="text-xs text-gray-600">Day Streak</p>
                    </div>
                    <div className="text-center p-3 bg-purple-50 rounded-lg">
                      <p className="text-2xl font-bold text-purple-600">{userProgress.signs_mastered}</p>
                      <p className="text-xs text-gray-600">Mastered</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Practice Goals */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-white/30">
              <div className="flex items-center space-x-3 mb-4">
                <Target className="h-6 w-6 text-green-600" />
                <h3 className="text-lg font-bold text-gray-900">Practice Goals</h3>
              </div>

              <div className="space-y-4">
                {practiceGoals.map((goal, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700">{goal.title}</span>
                      <span className="text-sm font-bold text-gray-900">
                        {goal.current}/{goal.target} {goal.unit}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-500 ${
                          goal.color === 'blue' ? 'bg-blue-500' :
                          goal.color === 'green' ? 'bg-green-500' :
                          'bg-purple-500'
                        }`}
                        style={{ width: `${Math.min((goal.current / goal.target) * 100, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Practice History */}
            {practiceHistory.length > 0 && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-white/30">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="h-6 w-6 text-orange-600" />
                    <h3 className="text-lg font-bold text-gray-900">Recent Practice</h3>
                  </div>
                  <button
                    onClick={clearPracticeHistory}
                    className="text-xs text-gray-500 hover:text-red-500 transition-colors px-2 py-1 rounded"
                    title="Clear practice history"
                  >
                    Clear
                  </button>
                </div>

                <div className="space-y-3">
                  {practiceHistory.slice(0, 3).map((entry, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border">
                      <div className="flex items-center space-x-3">
                        {entry.isMatch || entry.isTop4Correct ? (
                          <CheckCircle className="h-5 w-5 text-green-600" />
                        ) : (
                          <AlertCircle className="h-5 w-5 text-red-600" />
                        )}
                        <div>
                          <p className="font-medium text-gray-900">{entry.word}</p>
                          <p className="text-xs text-gray-500">
                            {new Date(entry.timestamp).toLocaleDateString()} â€¢ {entry.model}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-700">
                          {entry.predictions[0]?.word || 'No prediction'}
                        </p>
                        <p className="text-xs text-gray-500">
                          {((entry.confidence || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Practice;
