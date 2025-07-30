import React, { useState, useRef, useCallback } from 'react';
import { Camera, StopCircle, Play, Zap, Award, Target, RefreshCw, CheckCircle, XCircle, Timer, Upload } from 'lucide-react';
import { usePSLWebSocket } from '../hooks/usePSLWebSocket';

interface PredictionResult {
  letter: string;
  confidence: number;
  timestamp: number;
}

interface ModelConfig {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  speed: 'fast' | 'medium' | 'slow';
  color: string;
  icon: React.ReactNode;
  processingTime: number; // in milliseconds
}

const PSLPractice: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('ps-mini');
  const [isRecording, setIsRecording] = useState(false);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [sessionStats, setSessionStats] = useState({
    correct: 0,
    total: 0,
    streak: 0,
    bestStreak: 0
  });
  const [targetLetter, setTargetLetter] = useState<string>('');
  const [practiceMode, setPracticeMode] = useState<'free' | 'guided'>('free');
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [testImage, setTestImage] = useState<string | null>(null);
  const [testPrediction, setTestPrediction] = useState<PredictionResult | null>(null);
  const [frameCount, setFrameCount] = useState<number>(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  // WebSocket connection for real-time PSL prediction
  const pslWebSocket = usePSLWebSocket({
    modelType: selectedModel === 'ps-mini' ? 'ps_mini' : 'ps_pro',
    onPrediction: (prediction) => {
      const result: PredictionResult = {
        letter: prediction.letter,
        confidence: prediction.confidence,
        timestamp: prediction.timestamp
      };

      setCurrentPrediction(result);
      setPredictions(prev => [result, ...prev.slice(0, 7)]);
      
      // If this is from a test image, also set test prediction
      if (testImage) {
        setTestPrediction(result);
      }

      // Update stats if in guided mode
      if (practiceMode === 'guided' && targetLetter) {
        const isCorrect = prediction.letter === targetLetter;
        setSessionStats(prev => ({
          correct: prev.correct + (isCorrect ? 1 : 0),
          total: prev.total + 1,
          streak: isCorrect ? prev.streak + 1 : 0,
          bestStreak: Math.max(prev.bestStreak, isCorrect ? prev.streak + 1 : prev.streak)
        }));

        if (isCorrect) {
          // Auto-select next target letter after 2 seconds
          setTimeout(() => {
            const nextLetter = commonPSLLetters[Math.floor(Math.random() * commonPSLLetters.length)];
            setTargetLetter(nextLetter);
          }, 2000);
        }
      }
    },
    onError: (error) => {
      console.error('PSL WebSocket error:', error);
      setConnectionStatus('disconnected');
    },
    onConnected: () => {
      setConnectionStatus('connected');
    }
  });

  // Sync connection status with WebSocket hook state
  React.useEffect(() => {
    if (pslWebSocket.isConnected) {
      setConnectionStatus('connected');
    } else if (pslWebSocket.isConnecting) {
      setConnectionStatus('connecting');
    } else {
      setConnectionStatus('disconnected');
    }
  }, [pslWebSocket.isConnected, pslWebSocket.isConnecting]);

  const models: ModelConfig[] = [
    {
      id: 'ps-mini',
      name: 'PS Mini',
      description: 'Fast inference with high accuracy',
      accuracy: 98,
      speed: 'fast',
      color: 'bg-green-500',
      icon: <Zap className="h-5 w-5" />,
      processingTime: 400
    },
    {
      id: 'ps-pro',
      name: 'PS Pro',
      description: 'Premium accuracy with optimal speed',
      accuracy: 99,
      speed: 'medium',
      color: 'bg-blue-500',
      icon: <Award className="h-5 w-5" />,
      processingTime: 400
    }
  ];

  const commonPSLLetters = [
    'Alif', 'Bay', 'Pay', 'Tay', 'Taay', 'Say', 'Chay', 'Khay', 'Dal', '1-Hay',
    'Daal', 'Ray', 'Zay', 'Seen', 'Sheen', 'Suad', 'Tuey', 'Ain', 'Ghain',
    'Fay', 'Kaf', 'Lam', 'Meem', 'Nuun', 'Wao'
  ];

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        try {
          await videoRef.current.play();
          setCameraStarted(true);
        } catch (playError) {
          console.error('Video play error:', playError);
        }
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Camera access failed. Please check permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    setCameraStarted(false);
    setIsRecording(false);
    
    // Stop WebSocket prediction if running
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    pslWebSocket.disconnect();
    setConnectionStatus('disconnected');
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [pslWebSocket]);

  const startPrediction = () => {
    if (!cameraStarted || !videoRef.current) {
      console.error('Camera not started');
      return;
    }

    if (!pslWebSocket.isConnected) {
      console.error('WebSocket not connected. Please connect first.');
      return;
    }
    
    setIsRecording(true);
    setFrameCount(0); // Reset frame count when starting
    
    // Start capturing and sending frames immediately since WebSocket is already connected
    const captureFrame = () => {
      if (videoRef.current && pslWebSocket.isConnected) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (ctx) {
          const video = videoRef.current;
          const videoWidth = video.videoWidth;
          const videoHeight = video.videoHeight;
          
          // Calculate crop area for hand focus (center 60% of the frame)
          const cropRatio = 0.6;
          const cropWidth = videoWidth * cropRatio;
          const cropHeight = videoHeight * cropRatio;
          const cropX = (videoWidth - cropWidth) / 2;
          const cropY = (videoHeight - cropHeight) / 2;
          
          // Set canvas size to the cropped area
          canvas.width = cropWidth;
          canvas.height = cropHeight;
          
          if (videoWidth > 0 && videoHeight > 0) {
            // Draw the cropped area onto canvas
            ctx.drawImage(
              video,
              cropX, cropY, cropWidth, cropHeight,  // Source crop area
              0, 0, cropWidth, cropHeight           // Destination area (full canvas)
            );
            
            // Increment frame count
            setFrameCount(prev => prev + 1);
            
            // Convert to base64 and send to WebSocket
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            pslWebSocket.sendFrame(frameData);
          }
        }
      }
    };

    // Send frames every 400ms for balanced speed and stability
    frameIntervalRef.current = window.setInterval(captureFrame, 400);
  };

  const stopPrediction = () => {
    setIsRecording(false);
    setFrameCount(0); // Reset frame count when stopping
    
    // Stop frame capture
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Disconnect WebSocket
    pslWebSocket.disconnect();
    setConnectionStatus('disconnected');
  };

  const resetSession = () => {
    setSessionStats({ correct: 0, total: 0, streak: 0, bestStreak: 0 });
    setPredictions([]);
    setCurrentPrediction(null);
    setTargetLetter('');
    setPracticeMode('free');
  };

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    
    // If WebSocket is connected, switch the model
    if (pslWebSocket.isConnected) {
      const wsModelType = modelId === 'ps-mini' ? 'ps_mini' : 'ps_pro';
      pslWebSocket.switchModel(wsModelType);
    }
  };

  const startGuidedPractice = () => {
    setPracticeMode('guided');
    const randomLetter = commonPSLLetters[Math.floor(Math.random() * commonPSLLetters.length)];
    setTargetLetter(randomLetter);
    setSessionStats({ correct: 0, total: 0, streak: 0, bestStreak: 0 });
    setPredictions([]);
    setCurrentPrediction(null);
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageData = e.target?.result as string;
        setTestImage(imageData);
        
        // Test prediction on uploaded image
        if (pslWebSocket.isConnected) {
          testImagePrediction(imageData);
        } else {
          alert('Please connect to WebSocket first to test the image');
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const testImagePrediction = (imageData: string) => {
    if (!pslWebSocket.isConnected) {
      console.error('WebSocket not connected');
      return;
    }

    // Send the image for prediction
    pslWebSocket.sendFrame(imageData);
    
    // The result will come through the normal onPrediction callback
  };

  const selectedModelConfig = models.find(m => m.id === selectedModel);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="relative group mb-8">
              <div className="absolute -inset-4 bg-white/20 blur-xl rounded-full opacity-70 group-hover:opacity-100 transition duration-1000"></div>
              <Target className="relative h-16 w-16 mx-auto animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">PSL Practice Studio</h1>
            <p className="text-xl text-blue-100 max-w-3xl mx-auto">
              Practice your Pakistan Sign Language skills with AI-powered recognition models. 
              Get real-time feedback and track your progress.
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8">
        {/* Model Selection */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Choose Your AI Model</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {models.map((model) => (
              <div
                key={model.id}
                className={`relative cursor-pointer rounded-xl p-6 border-2 transition-all duration-300 ${
                  selectedModel === model.id
                    ? 'border-indigo-500 bg-indigo-50 shadow-lg scale-105'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                onClick={() => handleModelChange(model.id)}
              >
                <div className="flex items-start space-x-4">
                  <div className={`p-3 rounded-full text-white ${model.color}`}>
                    {model.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 mb-2">{model.name}</h3>
                    <p className="text-gray-600 mb-4">{model.description}</p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Accuracy:</span>
                        <div className="font-semibold text-green-600">{model.accuracy}%</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Speed:</span>
                        <div className="font-semibold text-blue-600 capitalize">{model.speed}</div>
                      </div>
                    </div>
                  </div>
                </div>
                {selectedModel === model.id && (
                  <div className="absolute top-4 right-4">
                    <CheckCircle className="h-6 w-6 text-indigo-500" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Practice Mode Selection */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Practice Mode</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <button
              onClick={() => setPracticeMode('free')}
              className={`p-6 rounded-xl border-2 text-left transition-all duration-300 ${
                practiceMode === 'free'
                  ? 'border-indigo-500 bg-indigo-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center space-x-3 mb-3">
                <Play className="h-6 w-6 text-indigo-600" />
                <h3 className="text-xl font-bold text-gray-900">Free Practice</h3>
              </div>
              <p className="text-gray-600">Practice any sign and get instant recognition feedback</p>
            </button>
            
            <button
              onClick={startGuidedPractice}
              className={`p-6 rounded-xl border-2 text-left transition-all duration-300 ${
                practiceMode === 'guided'
                  ? 'border-indigo-500 bg-indigo-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center space-x-3 mb-3">
                <Target className="h-6 w-6 text-indigo-600" />
                <h3 className="text-xl font-bold text-gray-900">Guided Practice</h3>
              </div>
              <p className="text-gray-600">Follow prompts and track your accuracy progress</p>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Camera Feed</h2>
                <div className="flex items-center space-x-2">
                  {selectedModelConfig && (
                    <span className={`px-3 py-1 rounded-full text-white text-sm ${selectedModelConfig.color}`}>
                      {selectedModelConfig.name}
                    </span>
                  )}
                  {processingTime > 0 && (
                    <span className="flex items-center space-x-1 text-sm text-gray-600">
                      <Timer className="h-4 w-4" />
                      <span>{processingTime}ms</span>
                    </span>
                  )}
                  {/* WebSocket Connection Status */}
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    connectionStatus === 'connected' ? 'bg-green-100 text-green-800' :
                    connectionStatus === 'connecting' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {connectionStatus === 'connected' ? '🟢 Connected' :
                     connectionStatus === 'connecting' ? '🟡 Connecting' :
                     '🔴 Disconnected'}
                  </span>
                </div>
              </div>
              
              <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ width: '100%' }}>
                <video
                 className="w-full h-[60vh] sm:h-[60vh] md:h-[50vh] lg:h-[70vh] object-cover"
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{ 
                    objectFit: 'cover',
                    backgroundColor: '#000000'
                  }}
                />
                {!cameraStarted && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white">
                      <Camera className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">Click Start Camera to begin</p>
                    </div>
                  </div>
                )}
                
                {/* Target Letter Overlay for Guided Mode */}
                {practiceMode === 'guided' && targetLetter && (
                  <div className="absolute top-4 left-4 bg-black/70 text-white px-4 py-2 rounded-lg">
                    <div className="text-sm">Show this sign:</div>
                    <div className="text-2xl font-bold">{targetLetter}</div>
                  </div>
                )}

                {/* Frame Count Display */}
                {isRecording && (
                  <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-2 rounded-lg">
                    <div className="text-sm">Frame:</div>
                    <div className="text-lg font-mono">{frameCount}</div>
                  </div>
                )}

                {/* Current Prediction Overlay */}
                {currentPrediction && (
                  <div className="absolute bottom-4 right-4 bg-black/70 text-white px-4 py-2 rounded-lg">
                    <div className="text-sm">Detected:</div>
                    <div className="text-xl font-bold">{currentPrediction.letter}</div>
                    <div className="text-sm">{(currentPrediction.confidence * 100).toFixed(1)}%</div>
                  </div>
                )}

                {/* Hand Focus Area Overlay */}
                {cameraStarted && (
                  <div className="absolute inset-0 pointer-events-none">
                    {/* Crop area indicator - center 60% */}
                    <div 
                      className="absolute border-2 border-green-400 border-dashed bg-green-400/10"
                      style={{
                        left: '20%',
                        top: '20%',
                        width: '60%',
                        height: '60%'
                      }}
                    >
                      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-green-400 text-sm font-medium bg-black/50 px-2 py-1 rounded">
                        Hand Focus Area
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex justify-center mt-4 space-x-4">
                {!cameraStarted ? (
                  <button
                    onClick={startCamera}
                    className="flex items-center space-x-2 px-3 md:px-6 py-3 rounded-lg font-semibold bg-indigo-500 hover:bg-indigo-600 text-white transition-colors duration-300"
                  >
                    <Camera className="h-5 w-5" />
                    <span>Start Camera</span>
                  </button>
                ) : (
                  <>
                    {/* Connection Button */}
                    {connectionStatus === 'disconnected' && (
                      <button
                        onClick={() => {
                          setConnectionStatus('connecting');
                          pslWebSocket.connect();
                        }}
                        className="flex items-center space-x-2 px-3 md:px-6 py-3 rounded-lg font-semibold bg-blue-500 hover:bg-blue-600 text-white transition-colors duration-300"
                      >
                        <Zap className="h-5 w-5" />
                        <span>Connect</span>
                      </button>
                    )}

                    {/* Prediction Controls - only show when connected */}
                    {connectionStatus === 'connected' && (
                      <>
                        {!isRecording ? (
                          <button
                            onClick={startPrediction}
                            className="flex items-center space-x-2 px-3 md:px-6 py-3 rounded-lg font-semibold bg-green-500 hover:bg-green-600 text-white transition-colors duration-300"
                          >
                            <Play className="h-5 w-5" />
                            <span>Start Prediction</span>
                          </button>
                        ) : (
                          <button
                            onClick={stopPrediction}
                            className="flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold bg-red-500 hover:bg-red-600 text-white transition-colors duration-300"
                          >
                            <StopCircle className="h-5 w-5" />
                            <span>Stop Prediction</span>
                          </button>
                        )}
                      </>
                    )}

                    {/* Disconnect Button - show when connected or connecting */}
                    {(connectionStatus === 'connected' || connectionStatus === 'connecting') && (
                      <button
                        onClick={() => {
                          pslWebSocket.disconnect();
                          setConnectionStatus('disconnected');
                          if (isRecording) {
                            stopPrediction();
                          }
                        }}
                        className="flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold bg-orange-500 hover:bg-orange-600 text-white transition-colors duration-300"
                      >
                        <XCircle className="h-5 w-5" />
                        <span>Disconnect</span>
                      </button>
                    )}
                    
                    {/* Image Upload Test Button */}
                    <label
                      className="flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold bg-purple-500 hover:bg-purple-600 text-white transition-colors duration-300 cursor-pointer"
                    >
                      <Upload className="h-5 w-5" />
                      <span>Test Image</span>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                      />
                    </label>
                    
                    <button
                      onClick={stopCamera}
                      className="flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700 text-white transition-colors duration-300"
                    >
                      <Camera className="h-5 w-5" />
                      <span>Stop Camera</span>
                    </button>
                  </>
                )}

                {(predictions.length > 0 || sessionStats.total > 0) && (
                  <button
                    onClick={resetSession}
                    className="flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold bg-gray-500 hover:bg-gray-600 text-white transition-colors duration-300"
                  >
                    <RefreshCw className="h-5 w-5" />
                    <span>Reset</span>
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Test Image Results */}
            {testImage && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Test Image Prediction</h3>
                <div className="space-y-4">
                  <div className="relative">
                    <img 
                      src={testImage} 
                      alt="Test upload" 
                      className="w-full max-w-xs rounded-lg shadow-md mx-auto"
                    />
                  </div>
                  {testPrediction && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-gray-900 mb-2">
                          {testPrediction.letter}
                        </div>
                        <div className="text-lg text-gray-600">
                          {(testPrediction.confidence * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                    </div>
                  )}
                  <button
                    onClick={() => {
                      setTestImage(null);
                      setTestPrediction(null);
                    }}
                    className="w-full px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
                  >
                    Clear Test
                  </button>
                </div>
              </div>
            )}

            {/* Session Stats */}
            {practiceMode === 'guided' && sessionStats.total > 0 && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Session Stats</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {sessionStats.total > 0 ? Math.round((sessionStats.correct / sessionStats.total) * 100) : 0}%
                    </div>
                    <div className="text-sm text-gray-600">Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{sessionStats.streak}</div>
                    <div className="text-sm text-gray-600">Current Streak</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{sessionStats.bestStreak}</div>
                    <div className="text-sm text-gray-600">Best Streak</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-600">{sessionStats.total}</div>
                    <div className="text-sm text-gray-600">Total Signs</div>
                  </div>
                </div>
              </div>
            )}

            {/* Recent Predictions */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Recent Predictions</h3>
              {predictions.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No predictions yet. Start practicing to see results!</p>
              ) : (
                <div className="space-y-3">
                  {predictions.map((prediction, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center space-x-3">
                        {practiceMode === 'guided' && targetLetter && (
                          prediction.letter === targetLetter ? (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          ) : (
                            <XCircle className="h-5 w-5 text-red-500" />
                          )
                        )}
                        <span className="font-semibold text-gray-900">{prediction.letter}</span>
                      </div>
                      <span className="text-sm text-gray-600">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Model Performance */}
            {selectedModelConfig && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Model Performance</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Accuracy</span>
                      <span className="font-semibold">{selectedModelConfig.accuracy}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${selectedModelConfig.accuracy}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Speed</span>
                      <span className="font-semibold capitalize">{selectedModelConfig.speed}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          selectedModelConfig.speed === 'fast' ? 'bg-green-500 w-full' :
                          selectedModelConfig.speed === 'medium' ? 'bg-yellow-500 w-3/4' :
                          'bg-red-500 w-1/2'
                        }`}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PSLPractice;
