import React, { useState, useRef, useCallback } from 'react';
import { Camera, StopCircle, Play, Zap, Award, Target, RefreshCw, CheckCircle, XCircle, Timer } from 'lucide-react';

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
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const models: ModelConfig[] = [
    {
      id: 'ps-mini',
      name: 'PS Mini',
      description: 'Fast inference with high accuracy',
      accuracy: 98,
      speed: 'fast',
      color: 'bg-green-500',
      icon: <Zap className="h-5 w-5" />,
      processingTime: 150
    },
    {
      id: 'ps-pro',
      name: 'PS Pro',
      description: 'Premium accuracy with optimal speed',
      accuracy: 99,
      speed: 'medium',
      color: 'bg-blue-500',
      icon: <Award className="h-5 w-5" />,
      processingTime: 300
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
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Error accessing camera. Please ensure camera permissions are granted.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const simulatePrediction = useCallback(() => {
    const model = models.find(m => m.id === selectedModel);
    if (!model) return;

    const startTime = Date.now();
    
    setTimeout(() => {
      // Simulate model prediction
      const randomLetter = commonPSLLetters[Math.floor(Math.random() * commonPSLLetters.length)];
      const baseConfidence = model.accuracy / 100;
      const confidence = Math.min(0.99, baseConfidence + (Math.random() * 0.1 - 0.05));
      
      const prediction: PredictionResult = {
        letter: randomLetter,
        confidence: confidence,
        timestamp: Date.now()
      };

      setCurrentPrediction(prediction);
      setPredictions(prev => [prediction, ...prev.slice(0, 9)]);
      setProcessingTime(Date.now() - startTime);

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
    }, model.processingTime);
  }, [selectedModel, practiceMode, targetLetter, models, commonPSLLetters]);

  const toggleRecording = async () => {
    if (!isRecording) {
      await startCamera();
      setIsRecording(true);
      // Start continuous prediction
      const interval = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState === 4) {
          simulatePrediction();
        }
      }, 1000);
      
      // Store interval in a ref or state to clear it later
      (window as any).predictionInterval = interval;
    } else {
      setIsRecording(false);
      stopCamera();
      clearInterval((window as any).predictionInterval);
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

  const resetSession = () => {
    setSessionStats({ correct: 0, total: 0, streak: 0, bestStreak: 0 });
    setPredictions([]);
    setCurrentPrediction(null);
    setTargetLetter('');
    setPracticeMode('free');
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
                onClick={() => setSelectedModel(model.id)}
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
                </div>
              </div>
              
              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video mb-6">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                {!isRecording && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white">
                      <Camera className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">Click Start to begin practice</p>
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

                {/* Current Prediction Overlay */}
                {currentPrediction && (
                  <div className="absolute bottom-4 right-4 bg-black/70 text-white px-4 py-2 rounded-lg">
                    <div className="text-sm">Detected:</div>
                    <div className="text-xl font-bold">{currentPrediction.letter}</div>
                    <div className="text-sm">{(currentPrediction.confidence * 100).toFixed(1)}%</div>
                  </div>
                )}
              </div>

              <div className="flex justify-center space-x-4">
                <button
                  onClick={toggleRecording}
                  className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-colors duration-300 ${
                    isRecording
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                  }`}
                >
                  {isRecording ? (
                    <>
                      <StopCircle className="h-5 w-5" />
                      <span>Stop Practice</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5" />
                      <span>Start Practice</span>
                    </>
                  )}
                </button>

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
