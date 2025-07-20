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
  
  // Base URL for API calls
  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  
  // State management
  const [translationMode, setTranslationMode] = useState<'sign-to-text' | 'text-to-sign'>('sign-to-text');
  const [predictionMode, setPredictionMode] = useState<'sentence' | 'word'>('sentence');
  const [selectedModel, setSelectedModel] = useState('hrnet-fast');
  const [detectedText, setDetectedText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [lastResult, setLastResult] = useState<TranslationResult | null>(null);
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

  // Available models
  const models: ModelOption[] = [
    {
      id: 'hrnet-fast',
      name: 'HRNet Fast',
      accuracy: 89,
      speed: 'Fast',
      description: 'Optimized for real-time performance'
    },
    {
      id: 'hrnet-accurate',
      name: 'HRNet Accurate',
      accuracy: 94,
      speed: 'Medium',
      description: 'Higher accuracy, moderate speed'
    },
    {
      id: 'ensemble-model',
      name: 'Ensemble Model',
      accuracy: 96,
      speed: 'Slow',
      description: 'Best accuracy, slower processing'
    }
  ];

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

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Camera access denied or unavailable');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
      setIsRecording(false);
    }
  };

  const startPrediction = async () => {
    if (!isCameraActive || isRecording) return;
    
    setIsRecording(true);
    setIsProcessing(true);
    
    try {
      // Simulate processing time based on prediction mode
      const processingTime = predictionMode === 'sentence' ? 3000 + Math.random() * 2000 : 1500 + Math.random() * 1000;
      await new Promise(resolve => setTimeout(resolve, processingTime));
      
      // Simulate prediction results
      const confidence = Math.floor(85 + Math.random() * 15);
      const actualProcessingTime = Math.floor(processingTime);
      
      const sentenceResults = [
        "Hello, how are you today?",
        "Thank you very much",
        "Nice to meet you",
        "What is your name?",
        "Have a good day"
      ];
      
      const wordResults = [
        "Hello",
        "Thank",
        "You", 
        "Please",
        "Good",
        "Day",
        "Name"
      ];
      
      const resultText = predictionMode === 'sentence' 
        ? sentenceResults[Math.floor(Math.random() * sentenceResults.length)]
        : wordResults[Math.floor(Math.random() * wordResults.length)];
      
      const result: TranslationResult = {
        result: resultText,
        confidence,
        processingTime: actualProcessingTime,
        timestamp: new Date(),
        mode: 'sign-to-text',
        predictionType: predictionMode
      };

      setDetectedText(result.result);
      setLastResult(result);
      
      // Add to session
      translationSessionManager.addTranslation({
        result: result.result,
        confidence: confidence,
        mode: 'sign-to-text',
        isCorrect: confidence > 85
      });
      
      // Update recent translations
      const updatedHistory = translationSessionManager.getRecentTranslations();
      setRecentTranslations(updatedHistory);
      
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsProcessing(false);
      setIsRecording(false);
    }
  };

  const stopPrediction = () => {
    setIsRecording(false);
    setIsProcessing(false);
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
                  </div>
                  <canvas ref={canvasRef} className="hidden" />
                </div>

                {/* Camera Controls */}
                <div className="flex gap-3">
                  {!isCameraActive ? (
                    <button
                      onClick={startCamera}
                      className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors duration-200 flex items-center justify-center"
                    >
                      <Video className="w-4 h-4 mr-2" />
                      Start Camera
                    </button>
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
                          Start Detection
                        </button>
                      ) : (
                        <button
                          onClick={stopPrediction}
                          className="flex-1 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors duration-200 flex items-center justify-center"
                        >
                          <Square className="w-4 h-4 mr-2" />
                          Stop Detection
                        </button>
                      )}
                    </>
                  )}
                </div>

                {isProcessing && (
                  <div className="text-center">
                    <div className="inline-flex items-center px-4 py-2 bg-blue-50 rounded-lg">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                      <span className="text-blue-700">
                        {predictionMode === 'sentence' ? 'Analyzing sentence...' : 'Detecting word...'}
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
                detectedText ? (
                  <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                    <div className="flex items-center mb-2">
                      <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                      <span className="font-medium text-green-800">Detected Text:</span>
                    </div>
                    <p className="text-gray-800 text-lg">{detectedText}</p>
                    {lastResult && (
                      <div className="mt-3 pt-3 border-t border-green-200">
                        <p className="text-sm text-gray-600">
                          Confidence: <span className="font-medium">{lastResult.confidence}%</span>
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                    <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Translation results will appear here</p>
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
