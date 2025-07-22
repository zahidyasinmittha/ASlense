import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Languages, 
  Camera, 
  Type, 
  Target, 
  Clock, 
  Zap, 
  CheckCircle, 
  MessageSquare, 
  Hand, 
  History, 
  X,
  Video,
  Square,
  Settings,
  Monitor
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { translationSessionManager } from '../services/translationSessionManager';

interface TranslationResult {
  result: string;
  confidence: number;
  processingTime: number;
  timestamp: Date;
  mode: 'sign-to-text' | 'text-to-sign';
  predictionType?: 'sentence' | 'word';
  modelUsed?: string;
}

interface Prediction {
  word: string;
  confidence: number;
  rank: number;
  timestamp?: string;
  allPredictions?: Prediction[]; // For storing all 4 predictions in a batch
}

interface LivePredictionMessage {
  type: 'connected' | 'frame_received' | 'progress' | 'live_prediction' | 'final_result' | 'error' | 'stopped' | 'batch_result' | 'high_confidence_prediction';
  message?: string;
  frame_count?: number;
  latest_predictions?: Prediction[];
  predictions?: Prediction[];
  confidence?: number;
  model_used?: string;
  prediction?: string; // For single prediction format
  result?: {
    predicted_text: string;
    confidence: number;
    processing_time?: number;
    model_used?: string;
    prediction_mode?: string;
    frames_processed?: number;
  };
}

interface SessionStats {
  totalTranslations: number;
  averageAccuracy: number;
  sessionTime: number;
}

interface RecentTranslation {
  result: string;
  confidence: number;
  timestamp: Date;
  mode: 'sign-to-text' | 'text-to-sign';
  predictionType?: 'sentence' | 'word';
}

interface ModelOption {
  id: string;
  name: string;
  accuracy: number;
  speed: string;
  description: string;
}

const Translate: React.FC = () => {
  const { user, token, makeAuthenticatedRequest } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<number | null>(null);
  
  // Base URL for API calls
  const baseUrl = 'http://localhost:8000';
  const wsUrl = baseUrl.replace('http', 'ws');
  
  // State management
  const [translationMode, setTranslationMode] = useState<'sign-to-text' | 'text-to-sign'>('sign-to-text');
  const [predictionMode, setPredictionMode] = useState<'sentence' | 'word'>('sentence');
  const [selectedModel, setSelectedModel] = useState('mini-fastsmooth'); // Default to first sentence model
  const [detectedText, setDetectedText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [lastResult, setLastResult] = useState<TranslationResult | null>(null);
  
  // Real-time prediction states
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [isLivePredicting, setIsLivePredicting] = useState(false);
  const [liveResult, setLiveResult] = useState<string>('');
  const [capturedFrameCount, setCapturedFrameCount] = useState(0);
  const [currentPredictions, setCurrentPredictions] = useState<Prediction[]>([]);
  const [accumulatedPredictions, setAccumulatedPredictions] = useState<Prediction[]>([]);
  
  const [sessionStats, setSessionStats] = useState<SessionStats>(() => {
    // Initialize with current session stats if session exists
    const stats = translationSessionManager.getSessionStats();
    return stats ? {
      totalTranslations: stats.translations,
      averageAccuracy: stats.accuracy,
      sessionTime: stats.session_time
    } : {
      totalTranslations: 0,
      averageAccuracy: 0,
      sessionTime: 0
    };
  });
  const [recentTranslations, setRecentTranslations] = useState<RecentTranslation[]>([]);
  
  // User progress state (same as Practice module)
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

  // Reset selected model when prediction mode changes
  useEffect(() => {
    if (predictionMode === 'sentence') {
      setSelectedModel('mini-fastsmooth'); // Default to first sentence model
    } else {
      setSelectedModel('mini'); // Default to first word model
    }
  }, [predictionMode]);

  // Available models - different models based on prediction mode
  const getSentenceModels = (): ModelOption[] => [
    {
      id: 'mini-fastsmooth',
      name: 'AS Mini + FastSmooth-ASL LLM',
      accuracy: 87,
      speed: 'Fast',
      description: 'Fast sentence prediction with lightweight models'
    },
    {
      id: 'pro-refined',
      name: 'AS Pro + RefinedAgent-ASL',
      accuracy: 95,
      speed: 'Medium',
      description: 'High accuracy sentence prediction with advanced agents'
    }
  ];

  const getWordModels = (): ModelOption[] => [
    {
      id: 'mini',
      name: 'AS Mini',
      accuracy: 85,
      speed: 'Fast',
      description: 'Fast and efficient word prediction'
    },
    {
      id: 'pro',
      name: 'AS Pro',
      accuracy: 92,
      speed: 'Medium',
      description: 'High accuracy word prediction'
    }
  ];

  // Get current models based on prediction mode
  const models: ModelOption[] = predictionMode === 'sentence' ? getSentenceModels() : getWordModels();

  // Fetch user progress data (same as Practice module)
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

  // Initialize session on component mount
  useEffect(() => {
    if (user) {
      translationSessionManager.startSession(user.id);
      
      // Fetch user progress from backend initially
      fetchUserProgress();
      
      // Immediately update stats from existing session
      const stats = translationSessionManager.getSessionStats();
      if (stats) {
        setSessionStats({
          totalTranslations: stats.translations,
          averageAccuracy: Math.round(userProgress.accuracy_rate), // Use backend accuracy
          sessionTime: stats.session_time
        });
      }
      
      // Update stats every second
      const statsInterval = setInterval(() => {
        const stats = translationSessionManager.getSessionStats();
        if (stats) {
          setSessionStats({
            totalTranslations: stats.translations,
            averageAccuracy: Math.round(userProgress.accuracy_rate), // Use backend accuracy
            sessionTime: stats.session_time
          });
        }
      }, 1000);

      // Refresh user progress every 30 seconds to keep accuracy updated
      const progressInterval = setInterval(() => {
        fetchUserProgress();
      }, 30000);

      return () => {
        clearInterval(statsInterval);
        clearInterval(progressInterval);
        // Note: Don't end session here as we want it to persist across page reloads
        // Session will only end on browser close, tab close, or logout via session manager
      };
    }
  }, [user, fetchUserProgress, userProgress.accuracy_rate]);

  // Cleanup camera on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Load recent translations
  useEffect(() => {
    const history = translationSessionManager.getRecentTranslations();
    setRecentTranslations(history);
  }, []);

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // OPTIMIZED Frame capture function for better performance
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

    // OPTIMIZATION 1: Reduce resolution for faster processing (maintain aspect ratio)
    const targetWidth = 640; // Reduced from 640 for 4x faster processing
    const targetHeight = 480; // Reduced from 480
    const aspectRatio = video.videoWidth / video.videoHeight;
    
    let finalWidth = targetWidth;
    let finalHeight = Math.round(targetWidth / aspectRatio);
    
    if (finalHeight > targetHeight) {
      finalHeight = targetHeight;
      finalWidth = Math.round(targetHeight * aspectRatio);
    }

    canvas.width = finalWidth;
    canvas.height = finalHeight;
    
    // OPTIMIZATION 2: Use faster image smoothing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'low'; // Faster than 'high'
    
    ctx.drawImage(video, 0, 0, finalWidth, finalHeight);
    
    // OPTIMIZATION 3: Lower JPEG quality for faster encoding and smaller transfer
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.7); // Reduced from 0.95
    
    // OPTIMIZATION 4: Minimum size check for valid frames
    if (frameDataUrl.length < 500) { // Reduced threshold
      return null;
    }

    return frameDataUrl;
  }, []);

  // WebSocket connection for real-time prediction
  const connectWebSocket = useCallback(() => {
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
    }
    
    setConnectionStatus('connecting');
    
    // Get model type based on selected model
    const modelType = predictionMode === 'word' ? 
      (selectedModel === 'pro' ? 'pro' : 'mini') : 
      (selectedModel === 'pro-refined' ? 'pro' : 'mini');
    
    // Direct connection to translate endpoint - match Practice URL pattern
    const translateWsUrl = `${wsUrl}/translate/live-translate?model_type=${modelType}&prediction_mode=${predictionMode}`;
    
    console.log('🏃 TRANSLATE: Attempting WebSocket connection...');
    console.log(`🏃 TRANSLATE: URL: ${translateWsUrl}`);
    console.log(`🏃 TRANSLATE: Base URL: ${baseUrl}`);
    console.log(`🏃 TRANSLATE: WS URL: ${wsUrl}`);
    console.log(`🏃 TRANSLATE: Model: ${modelType}`);
    console.log(`🏃 TRANSLATE: Mode: ${predictionMode}`);
    
    const ws = new WebSocket(translateWsUrl);
    
    ws.onopen = () => {
      console.log('✅ TRANSLATE: WebSocket connected successfully!');
      setConnectionStatus('connected');
      setWsConnection(ws);
    };
    
    ws.onmessage = (event) => {
      try {
        const data: LivePredictionMessage = JSON.parse(event.data);
        console.log('📥 TRANSLATE: Received message:', data);
          
          switch (data.type) {
            case 'connected':
              console.log('🔗 TRANSLATE: Connection confirmed by server');
              break;
              
            case 'live_prediction':
              // Real-time prediction result every 20 frames (word mode only)
              if (data.predictions && data.predictions.length > 0) {
                console.log('🔍 FRONTEND: Received live prediction confidence values:', data.predictions.map(p => ({word: p.word, confidence: p.confidence})));
                
                setCurrentPredictions(data.predictions);
                const topPrediction = data.predictions[0];
                
                // Update live result for video overlay (still show top prediction prominently)
                setLiveResult(topPrediction.word);
                
                // Add ALL TOP 4 predictions to accumulated list - BATCH ENTRY
                setAccumulatedPredictions(prev => {
                  const batchEntry: Prediction = {
                    word: `Batch ${Math.floor(Date.now() / 1000)}`, // Unique batch identifier
                    confidence: 0, // Will be calculated from top prediction
                    rank: 0,
                    timestamp: new Date().toLocaleTimeString(),
                    // Add all 4 predictions as metadata
                    allPredictions: data.predictions // Store all 4 predictions
                  };
                  
                  // Add latest batch at the BEGINNING of array (top of list)
                  return [batchEntry, ...prev];
                });
                
                // Update Translation Result section with LATEST prediction immediately
                const result: TranslationResult = {
                  result: topPrediction.word,
                  confidence: topPrediction.confidence, // Keep as 0-1 format
                  processingTime: 0, // Real-time, no processing time
                  timestamp: new Date(),
                  mode: 'sign-to-text',
                  predictionType: predictionMode,
                  modelUsed: selectedModel
                };
                
                console.log('🔍 FRONTEND: Setting lastResult with confidence:', result.confidence);
                
                // Update ONLY detectedText to show latest word without affecting accumulated list
                setDetectedText(result.result);
                setLastResult(result);
              }
              break;
              
            case 'final_result':
              // Final result for both sentence and word modes
              if (data.result) {
                setIsProcessing(false); // Stop processing indicator
                
                const result: TranslationResult = {
                  result: data.result.predicted_text,
                  confidence: data.result.confidence,
                  processingTime: data.result.processing_time || 0,
                  timestamp: new Date(),
                  mode: 'sign-to-text',
                  predictionType: predictionMode,
                  modelUsed: data.result.model_used
                };

                setDetectedText(result.result);
                setLastResult(result);
                
                // Add to session
                translationSessionManager.addTranslation({
                  result: result.result,
                  confidence: result.confidence * 100, // Convert to 0-100 for session tracking
                  mode: 'sign-to-text',
                  isCorrect: result.confidence > 0.85 // Use 0-1 format for comparison
                });
                
                // Update recent translations
                const updatedHistory = translationSessionManager.getRecentTranslations();
                setRecentTranslations(updatedHistory);
              }
              break;
              
            case 'error':
              console.log('❌ TRANSLATE: Server error:', data.message);
              break;
          }
        } catch (error) {
          console.error('❌ TRANSLATE: Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log('🔌 TRANSLATE: WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        setWsConnection(null);
      };
      
      ws.onerror = (error) => {
        console.error('❌ TRANSLATE: WebSocket error:', error);
        console.log('❌ TRANSLATE: WebSocket state:', ws.readyState);
        setConnectionStatus('disconnected');
      };
    
  }, [wsUrl, selectedModel, predictionMode]);

  const startCamera = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Camera access is not supported in this browser.');
        return;
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
          setIsCameraActive(true);
        }).catch(() => {
          // Camera setup error handled in catch block below
        });
      }
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      
      let errorMessage = 'Could not access camera. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Camera permission was denied.';
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera device found.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else {
        errorMessage += 'Please check your camera settings.';
      }
      
      alert(errorMessage);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
      setIsRecording(false);
      setIsLivePredicting(false);
    }
    
    // Close WebSocket connection
    if (wsConnection) {
      wsConnection.close();
      setWsConnection(null);
    }
    setConnectionStatus('disconnected');
    
    // Clear frame interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Clear live predictions
    setLiveResult('');
    setCurrentPredictions([]);
    setCapturedFrameCount(0);
  };

  const startPrediction = async () => {
    if (!isCameraActive || isRecording) return;
    
    setIsRecording(true);
    setIsLivePredicting(true);
    setLiveResult('');
    setCurrentPredictions([]);
    setAccumulatedPredictions([]); // Clear accumulated predictions on new session
    
    // Connect WebSocket if not connected
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      
      // Wait for connection
      let attempts = 0;
      while ((!wsConnection || wsConnection.readyState !== WebSocket.OPEN) && attempts < 10) {
        await new Promise(resolve => setTimeout(resolve, 500));
        attempts++;
      }
      
      if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        setIsRecording(false);
        setIsLivePredicting(false);
        return;
      }
    }
    
    // Start sending frames - match Practice section timing exactly (150ms)
    let frameCount = 0;
    frameIntervalRef.current = window.setInterval(() => {
      if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        return;
      }
      
      const frameDataUrl = captureFrame();
      
      if (frameDataUrl) {
        frameCount++;
        setCapturedFrameCount(frameCount); // Update frame count immediately like Practice
        
        // Send EVERY frame to backend like Practice section
        wsConnection.send(JSON.stringify({
          type: 'frame',
          frame: frameDataUrl,
          prediction_mode: predictionMode
        }));
      }
    }, 150); // Match Practice section: capture frame every 150ms
  };

  const stopPrediction = () => {
    setIsRecording(false);
    setIsLivePredicting(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // For sentence mode, start processing
    if (predictionMode === 'sentence') {
      setIsProcessing(true);
    }
    
    // Send stop message to WebSocket for final processing (especially for sentence mode)
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
      wsConnection.send(JSON.stringify({
        type: 'stop'
      }));
    }
    
    // For word mode, create final result from live prediction if available
    if (predictionMode === 'word' && liveResult && currentPredictions.length > 0) {
      const topPrediction = currentPredictions[0];
      const result: TranslationResult = {
        result: liveResult,
        confidence: topPrediction.confidence,
        processingTime: 0, // Real-time, no processing time
        timestamp: new Date(),
        mode: 'sign-to-text',
        predictionType: predictionMode,
        modelUsed: selectedModel
      };

      setDetectedText(result.result);
      setLastResult(result);
      
      // Add to session
      translationSessionManager.addTranslation({
        result: result.result,
        confidence: topPrediction.confidence * 100, // Convert to 0-100 for session tracking
        mode: 'sign-to-text',
        isCorrect: topPrediction.confidence > 0.85 // Use 0-1 format for comparison
      });
      
      // Update recent translations
      const updatedHistory = translationSessionManager.getRecentTranslations();
      setRecentTranslations(updatedHistory);
    }
    
    // For sentence mode, the final result will come from WebSocket final_result message
    // and setIsProcessing(false) will be called in the WebSocket message handler
  };

  const translateTextToSign = async () => {
    if (!targetText.trim()) return;

    setIsProcessing(true);
    
    try {
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 2500));
      
      // Simulate sign description result
      const confidence = Math.floor(80 + Math.random() * 20);
      const processingTime = Math.floor(1200 + Math.random() * 1800);
      
      const signDescriptions = [
        "Start with both hands at chest level, move right hand forward in a greeting gesture, then point to yourself and make questioning expression with raised eyebrows.",
        "Begin with a wave motion using your dominant hand, then touch your chest with your index finger, followed by pointing forward with open palm.",
        "Initiate contact with a friendly hand gesture, point to self, then extend open palm toward the other person with questioning facial expression."
      ];
      
      const result: TranslationResult = {
        result: signDescriptions[Math.floor(Math.random() * signDescriptions.length)],
        confidence,
        processingTime,
        timestamp: new Date(),
        mode: 'text-to-sign'
      };

      setLastResult(result);
      
      // Add to session
      translationSessionManager.addTranslation({
        result: result.result,
        confidence: confidence,
        mode: 'text-to-sign',
        isCorrect: confidence > 80 // Consider high confidence as correct
      });
      
      // Update recent translations
      const updatedHistory = translationSessionManager.getRecentTranslations();
      setRecentTranslations(updatedHistory);
      
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearResults = () => {
    setDetectedText('');
    setLastResult(null);
    setTargetText('');
    setAccumulatedPredictions([]); // Clear accumulated predictions
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in-up">
          <div className="relative group mb-6">
            <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000 animate-pulse"></div>
            <h1 className="relative text-4xl font-bold text-gray-800 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              ASL Translator
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Use your camera to translate sign language into text in real-time, or translate text into sign descriptions
          </p>
        </div>

        {/* Translation Mode Selector */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-xl shadow-lg p-2 border border-gray-200">
            <button
              onClick={() => setTranslationMode('sign-to-text')}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                translationMode === 'sign-to-text'
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              <Camera className="inline-block w-5 h-5 mr-2" />
              Sign to Text
            </button>
            <button
              onClick={() => setTranslationMode('text-to-sign')}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                translationMode === 'text-to-sign'
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              <Type className="inline-block w-5 h-5 mr-2" />
              Text to Sign
            </button>
          </div>
        </div>

        {/* Session Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center">
              <div className="p-3 bg-gradient-to-r from-blue-100 to-blue-200 rounded-lg">
                <Languages className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Translations</p>
                <p className="text-2xl font-bold text-gray-800">{sessionStats.totalTranslations}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center">
              <div className="p-3 bg-gradient-to-r from-green-100 to-green-200 rounded-lg">
                <Target className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Avg. Accuracy</p>
                <p className="text-2xl font-bold text-gray-800">{sessionStats.averageAccuracy}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center">
              <div className="p-3 bg-gradient-to-r from-purple-100 to-purple-200 rounded-lg">
                <Clock className="h-8 w-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Session Time</p>
                <p className="text-2xl font-bold text-gray-800">{formatTime(sessionStats.sessionTime)}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              {translationMode === 'sign-to-text' ? (
                <><Camera className="w-5 h-5 mr-2" />Camera Detection</>
              ) : (
                <><Type className="w-5 h-5 mr-2" />Enter Text</>
              )}
            </h2>
            
            {translationMode === 'sign-to-text' ? (
              <div className="space-y-6">
                {/* Model Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    <Settings className="inline w-4 h-4 mr-1" />
                    Select Model
                  </label>
                  <div className="grid grid-cols-1 gap-3">
                    {models.map((model) => (
                      <label key={model.id} className="cursor-pointer">
                        <input
                          type="radio"
                          name="model"
                          value={model.id}
                          checked={selectedModel === model.id}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          className="sr-only"
                        />
                        <div className={`p-3 rounded-lg border transition-all duration-200 ${
                          selectedModel === model.id
                            ? 'border-purple-500 bg-purple-50 ring-2 ring-purple-200'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}>
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="font-medium text-gray-900">{model.name}</h4>
                              <p className="text-sm text-gray-600">{model.description}</p>
                            </div>
                            <div className="text-right text-sm">
                              <div className="text-green-600 font-medium">{model.accuracy}% accuracy</div>
                              <div className="text-gray-500">{model.speed}</div>
                            </div>
                          </div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Prediction Mode Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    <Monitor className="inline w-4 h-4 mr-1" />
                    Prediction Mode
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setPredictionMode('sentence')}
                      className={`p-3 rounded-lg border transition-all duration-200 ${
                        predictionMode === 'sentence'
                          ? 'border-blue-500 bg-blue-50 text-blue-700'
                          : 'border-gray-200 hover:border-gray-300 text-gray-700'
                      }`}
                    >
                      <MessageSquare className="w-5 h-5 mx-auto mb-1" />
                      <div className="text-sm font-medium">Sentence</div>
                      <div className="text-xs">Full phrases</div>
                    </button>
                    <button
                      onClick={() => setPredictionMode('word')}
                      className={`p-3 rounded-lg border transition-all duration-200 ${
                        predictionMode === 'word'
                          ? 'border-blue-500 bg-blue-50 text-blue-700'
                          : 'border-gray-200 hover:border-gray-300 text-gray-700'
                      }`}
                    >
                      <Type className="w-5 h-5 mx-auto mb-1" />
                      <div className="text-sm font-medium">Word</div>
                      <div className="text-xs">Individual words</div>
                    </button>
                  </div>
                </div>

                {/* Camera Feed */}
                <div className="relative">
                  <div className="bg-gray-900 rounded-lg overflow-hidden aspect-video">
                    <video
                      ref={videoRef}
                      autoPlay
                      muted
                      className="w-full h-full object-cover"
                    />
                    {!isCameraActive && (
                      <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                        <div className="text-center text-white">
                          <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p className="text-sm">Camera not active</p>
                        </div>
                      </div>
                    )}
                    
                    {/* Live Prediction Overlay - SHOW ALL TOP 4 PREDICTIONS */}
                    {isLivePredicting && currentPredictions.length > 0 && (
                      <div className="absolute bottom-4 left-4 right-4">
                        <div className="bg-black bg-opacity-80 text-white p-4 rounded-lg">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-xs text-gray-300">Live Predictions (Every 20 Frames)</span>
                            <span className="text-xs text-gray-300">Frames: {capturedFrameCount}</span>
                          </div>
                          
                          {/* Display ALL TOP 4 Predictions */}
                          <div className="space-y-2">
                            {currentPredictions.slice(0, 4).map((pred, index) => (
                              <div 
                                key={`live-${pred.word}-${index}`}
                                className={`flex justify-between items-center p-2 rounded ${
                                  index === 0 
                                    ? 'bg-green-600 bg-opacity-80' 
                                    : 'bg-white bg-opacity-20'
                                }`}
                              >
                                <div className="flex items-center space-x-2">
                                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    index === 0 
                                      ? 'bg-white text-green-600' 
                                      : 'bg-gray-600 text-white'
                                  }`}>
                                    {index + 1}
                                  </span>
                                  <span className={`font-medium ${
                                    index === 0 ? 'text-white text-lg' : 'text-gray-200 text-sm'
                                  }`}>
                                    {pred.word}
                                  </span>
                                  {index === 0 && (
                                    <span className="text-xs bg-white text-green-600 px-2 py-1 rounded-full font-bold">
                                      BEST
                                    </span>
                                  )}
                                </div>
                                <span className={`font-semibold text-xs px-2 py-1 rounded ${
                                  index === 0 
                                    ? 'bg-white text-green-600' 
                                    : 'bg-gray-600 text-white'
                                }`}>
                                  {(pred.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                          
                          <div className="mt-2 pt-2 border-t border-gray-500 text-xs text-gray-300 text-center">
                            Real-time GCN Model • Next update in {20 - (capturedFrameCount % 20)} frames
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Connection Status */}
                    {predictionMode === 'word' && connectionStatus !== 'disconnected' && (
                      <div className="absolute top-4 right-4">
                        <div className={`px-2 py-1 rounded-full text-xs ${
                          connectionStatus === 'connected' 
                            ? 'bg-green-500 text-white' 
                            : 'bg-yellow-500 text-black'
                        }`}>
                          {connectionStatus === 'connected' ? 'Live' : 'Connecting...'}
                        </div>
                      </div>
                    )}
                  </div>
                  <canvas ref={canvasRef} className="hidden" />
                </div>

                {/* Camera Controls */}
                <div className="flex gap-3">
                  {!isCameraActive ? (
                    <>
                      <button
                        onClick={startCamera}
                        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors duration-200 flex items-center justify-center"
                      >
                        <Video className="w-4 h-4 mr-2" />
                        Start Camera
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={stopCamera}
                        className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors duration-200 flex items-center justify-center"
                      >
                        <Square className="w-4 h-4 mr-2" />
                        Stop Camera
                      </button>
                      {!isRecording ? (
                        <button
                          onClick={startPrediction}
                          disabled={isProcessing}
                          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                        >
                          <Zap className="w-4 h-4 mr-2" />
                          Start {predictionMode === 'word' ? 'Live Detection' : 'Detection'}
                        </button>
                      ) : (
                        <button
                          onClick={stopPrediction}
                          className="flex-1 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors duration-200 flex items-center justify-center"
                        >
                          <Square className="w-4 h-4 mr-2" />
                          Stop Detection ({capturedFrameCount} frames)
                        </button>
                      )}
                    </>
                  )}
                </div>

                {isLivePredicting && predictionMode === 'word' && (
                  <div className="text-center">
                    <div className="inline-flex items-center px-4 py-2 bg-blue-50 rounded-lg">
                      <div className="animate-pulse w-3 h-3 bg-blue-600 rounded-full mr-3"></div>
                      <span className="text-blue-700">
                        Live word detection active...
                      </span>
                    </div>
                  </div>
                )}
                
                {isProcessing && predictionMode === 'sentence' && (
                  <div className="text-center">
                    <div className="inline-flex items-center px-4 py-2 bg-blue-50 rounded-lg">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                      <span className="text-blue-700">
                        Processing sentence...
                      </span>
                    </div>
                  </div>
                )}
                
                {!isRecording && !isProcessing && predictionMode === 'sentence' && connectionStatus === 'connected' && (
                  <div className="text-center">
                    <div className="inline-flex items-center px-4 py-2 bg-green-50 rounded-lg">
                      <div className="w-3 h-3 bg-green-600 rounded-full mr-3"></div>
                      <span className="text-green-700">
                        Ready for sentence detection
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div>
                <textarea
                  value={targetText}
                  onChange={(e) => setTargetText(e.target.value)}
                  placeholder="Enter text to translate to sign language..."
                  className="w-full h-32 p-4 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
                />
                <div className="mt-4 flex justify-between">
                  <span className="text-sm text-gray-500">
                    {targetText.length}/500 characters
                  </span>
                  <button
                    onClick={translateTextToSign}
                    disabled={!targetText.trim() || isProcessing}
                    className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline-block mr-2"></div>
                        Translating...
                      </>
                    ) : (
                      <>
                        <Languages className="inline-block w-4 h-4 mr-2" />
                        Translate
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                <Zap className="w-5 h-5 mr-2" />
                Translation Result
              </h2>
              {(detectedText || lastResult) && (
                <button
                  onClick={clearResults}
                  className="text-gray-500 hover:text-red-500 transition-colors duration-300"
                  title="Clear results"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
            
            <div className="space-y-4">
              {translationMode === 'sign-to-text' ? (
                (detectedText || accumulatedPredictions.length > 0) ? (
                  <div className="space-y-3">
                    {/* Real-time live prediction display */}
                    {detectedText && (
                      <div className={`p-4 rounded-lg border transition-all duration-300 ${
                        isLivePredicting && predictionMode === 'word'
                          ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-300 animate-pulse'
                          : 'bg-gradient-to-r from-green-50 to-blue-50 border-green-200'
                      }`}>
                        <div className="flex items-center mb-2">
                          {isLivePredicting && predictionMode === 'word' ? (
                            <>
                              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 animate-ping"></div>
                              <span className="font-medium text-blue-800">Live Prediction (Real-time):</span>
                            </>
                          ) : (
                            <>
                              <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                              <span className="font-medium text-green-800">Current Prediction:</span>
                            </>
                          )}
                          {isLivePredicting && predictionMode === 'word' && (
                            <span className="ml-auto text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                              Frame: {capturedFrameCount}
                            </span>
                          )}
                        </div>
                        <p className={`text-lg font-semibold ${
                          isLivePredicting && predictionMode === 'word' ? 'text-blue-900' : 'text-gray-800'
                        }`}>
                          {detectedText}
                        </p>
                        {lastResult && (
                          <div className={`mt-3 pt-3 border-t ${
                            isLivePredicting && predictionMode === 'word' ? 'border-blue-200' : 'border-green-200'
                          }`}>
                            <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
                              <div>
                                Model: <span className="font-medium">{selectedModel || 'AS Mini'}</span>
                              </div>
                              <div>
                                Confidence: <span className="font-medium">{(lastResult.confidence * 100).toFixed(1)}%</span>
                              </div>
                              {isLivePredicting && predictionMode === 'word' && (
                                <div>
                                  Updated: <span className="font-medium text-blue-600">Live</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Accumulated predictions - Show ALL TOP 4 for each 20-frame batch */}
                    {accumulatedPredictions.length > 0 && (
                      <div className="p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200 transition-all duration-300">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center">
                            <History className="w-5 h-5 text-gray-600 mr-2" />
                            <span className="font-medium text-gray-800">Real-time Prediction Batches:</span>
                          </div>
                          <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded-full">
                            {accumulatedPredictions.length} batches
                          </span>
                        </div>
                        <div className="space-y-3 max-h-[36rem] overflow-y-auto">
                          {accumulatedPredictions.map((batchEntry, index) => {
                            const batchNumber = index + 1; // Simple sequential numbering
                            const isLatest = index === accumulatedPredictions.length - 1;
                            return (
                              <div 
                                key={`batch-${index}-${batchEntry.timestamp}`} 
                                className={`p-3 rounded-lg border transition-all duration-200 ${
                                  isLatest
                                    ? 'bg-blue-100 border-blue-300 ring-2 ring-blue-200' 
                                    : 'bg-white border-gray-200'
                                }`}
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className={`text-sm font-medium ${
                                    isLatest ? 'text-blue-800' : 'text-gray-700'
                                  }`}>
                                    Batch #{batchNumber}
                                    {isLatest && (
                                      <span className="ml-2 text-xs bg-blue-200 text-blue-700 px-2 py-1 rounded-full">
                                        Latest
                                      </span>
                                    )}
                                  </span>
                                {batchEntry.timestamp && (
                                  <span className="text-xs text-gray-500">
                                    {batchEntry.timestamp}
                                  </span>
                                )}
                              </div>
                              
                              {/* Display all 4 predictions for this batch */}
                              {batchEntry.allPredictions && batchEntry.allPredictions.length > 0 ? (
                                <div className="space-y-1">
                                  {batchEntry.allPredictions.slice(0, 4).map((pred, predIndex) => (
                                    <div 
                                      key={`pred-${predIndex}`}
                                      className={`flex justify-between items-center p-2 rounded text-sm ${
                                        predIndex === 0 
                                          ? 'bg-green-100 border border-green-300' 
                                          : 'bg-gray-50 border border-gray-200'
                                      }`}
                                    >
                                      <div className="flex items-center space-x-2">
                                        <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                                          predIndex === 0 
                                            ? 'bg-green-600 text-white' 
                                            : 'bg-gray-400 text-white'
                                        }`}>
                                          {predIndex + 1}
                                        </span>
                                        <span className={`font-medium ${
                                          predIndex === 0 ? 'text-green-800' : 'text-gray-700'
                                        }`}>
                                          {pred.word}
                                        </span>
                                      </div>
                                      <span className={`text-xs px-2 py-1 rounded font-semibold ${
                                        predIndex === 0 
                                          ? 'bg-green-200 text-green-800' 
                                          : 'bg-gray-200 text-gray-600'
                                      }`}>
                                        {(pred.confidence * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-gray-500 italic">No predictions available</div>
                              )}
                            </div>
                            );
                          })}
                        </div>
                        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-600 flex items-center justify-between">
                          <div className="flex items-center">
                            <div className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></div>
                            Every 20 frames • TOP 4 per batch
                          </div>
                          <span className="text-gray-500">Showing all {accumulatedPredictions.length} batches</span>
                        </div>
                      </div>
                    )}

                    {/* Current Live Predictions - Show TOP 4 like Practice section */}
                    {isLivePredicting && predictionMode === 'word' && currentPredictions.length > 0 && (
                      <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200 transition-all duration-300">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center">
                            <Target className="w-5 h-5 text-green-600 mr-2" />
                            <span className="font-medium text-green-800">Current Live Predictions (Top 4):</span>
                          </div>
                          <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full animate-pulse">
                            Frame: {capturedFrameCount}
                          </span>
                        </div>
                        <div className="space-y-2">
                          {currentPredictions.slice(0, 4).map((pred, index) => (
                            <div 
                              key={`${pred.word}-${index}-current`} 
                              className={`flex justify-between items-center p-3 rounded-lg transition-all duration-200 ${
                                index === 0
                                  ? 'bg-gradient-to-r from-green-100 to-emerald-100 border border-green-300 ring-2 ring-green-200' 
                                  : 'bg-white border border-gray-200 hover:bg-gray-50'
                              }`}
                            >
                              <div className="flex items-center space-x-3">
                                <span className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${
                                  index === 0 
                                    ? 'bg-green-200 text-green-800' 
                                    : 'bg-blue-100 text-blue-600'
                                }`}>
                                  {index + 1}
                                </span>
                                <span className={`font-medium ${
                                  index === 0 ? 'text-green-800' : 'text-gray-700'
                                }`}>
                                  {pred.word}
                                  {index === 0 && (
                                    <span className="ml-2 text-xs bg-green-200 text-green-700 px-2 py-1 rounded-full">
                                      Best
                                    </span>
                                  )}
                                </span>
                              </div>
                              <div className="text-right">
                                <span className={`font-semibold text-sm px-2 py-1 rounded ${
                                  index === 0
                                    ? 'bg-green-200 text-green-800' 
                                    : 'bg-gray-100 text-gray-600'
                                }`}>
                                  {(pred.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-3 pt-3 border-t border-green-200 text-xs text-green-600 flex items-center justify-between">
                          <div className="flex items-center">
                            <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-ping"></div>
                            Live predictions • Updated every 20 frames
                          </div>
                          <span className="text-green-500">Real-time</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                    <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Translation results will appear here</p>
                    <p className="text-sm mt-2">
                      {predictionMode === 'word' ? 'Real-time word predictions every 20 frames' : 'Start translation to see results'}
                    </p>
                  </div>
                )
              ) : (
                lastResult ? (
                  <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                    <div className="flex items-center mb-2">
                      <CheckCircle className="w-5 h-5 text-purple-600 mr-2" />
                      <span className="font-medium text-purple-800">Sign Description:</span>
                    </div>
                    <p className="text-gray-800 text-lg mb-3">{lastResult.result}</p>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Model:</span>
                        <span className="font-medium">{selectedModel || 'AS Mini'}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Confidence:</span>
                        <span className="font-medium">{lastResult.confidence}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Processing Time:</span>
                        <span className="font-medium">{lastResult.processingTime}ms</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                    <Hand className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Sign descriptions will appear here</p>
                  </div>
                )
              )}
            </div>
          </div>
        </div>

        {/* Recent Translations */}
        {recentTranslations.length > 0 && (
          <div className="mt-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <History className="w-5 h-5 mr-2" />
                Recent Translations
              </h3>
              <div className="space-y-3">
                {recentTranslations.slice(0, 3).map((translation, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300">
                    <div className="flex-1">
                      <p className="text-sm text-gray-600">
                        {translation.mode === 'sign-to-text' ? (
                          `Sign → Text ${translation.predictionType ? `(${translation.predictionType})` : ''}`
                        ) : (
                          'Text → Sign'
                        )}
                      </p>
                      <p className="text-gray-800 truncate">{translation.result}</p>
                    </div>
                    <div className="text-right text-sm text-gray-500">
                      <p>{translation.confidence}% confidence</p>
                      <p>{new Date(translation.timestamp).toLocaleTimeString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Translate;
