import React, { useState, useEffect } from 'react';
import { Camera, CameraOff, RotateCcw, Target, CheckCircle, AlertCircle, Settings, Zap, TrendingUp, Award, Timer, BarChart3 } from 'lucide-react';

const Practice: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [selectedModel, setSelectedModel] = useState('mediapipe');
  const [detectedWord, setDetectedWord] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [practiceScore, setPracticeScore] = useState(0);
  const [sessionTime, setSessionTime] = useState(0);
  const [practiceStreak] = useState(7);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const models = [
    { id: 'mediapipe', name: 'MediaPipe v2.1', accuracy: '95%', speed: 'Fast', description: 'Best for real-time detection' },
    { id: 'openpose', name: 'OpenPose Enhanced', accuracy: '87%', speed: 'Medium', description: 'High accuracy for complex signs' },
    { id: 'custom', name: 'ASLense Custom', accuracy: '92%', speed: 'Fast', description: 'Optimized for ASL vocabulary' },
  ];

  const recentPractice = [
    { word: 'Hello', score: 95, time: '2 min ago', difficulty: 'Beginner' },
    { word: 'Thank You', score: 88, time: '5 min ago', difficulty: 'Beginner' },
    { word: 'Please', score: 92, time: '8 min ago', difficulty: 'Intermediate' },
    { word: 'Love', score: 97, time: '12 min ago', difficulty: 'Beginner' },
    { word: 'Family', score: 85, time: '15 min ago', difficulty: 'Intermediate' },
  ];

  const practiceGoals = [
    { title: 'Daily Practice', current: 3, target: 5, unit: 'signs', color: 'blue' },
    { title: 'Weekly Streak', current: practiceStreak, target: 7, unit: 'days', color: 'green' },
    { title: 'Accuracy Goal', current: 92, target: 95, unit: '%', color: 'purple' },
  ];

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      setIsProcessing(true);
      // Simulate detection
      const words = ['Hello', 'Thank You', 'Please', 'Sorry', 'Love', 'Happy', 'Family', 'Friend'];
      const randomWord = words[Math.floor(Math.random() * words.length)];
      const randomConfidence = Math.floor(Math.random() * 30) + 70;
      const randomScore = Math.floor(Math.random() * 20) + 80;
      
      setTimeout(() => {
        setDetectedWord(randomWord);
        setConfidence(randomConfidence);
        setPracticeScore(randomScore);
        setIsProcessing(false);
      }, 1500);
    } else {
      setDetectedWord('');
      setConfidence(0);
      setPracticeScore(0);
      setIsProcessing(false);
    }
  };

  const clearResults = () => {
    setDetectedWord('');
    setConfidence(0);
    setPracticeScore(0);
    setIsRecording(false);
    setIsProcessing(false);
    setSessionTime(0);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 80) return CheckCircle;
    if (confidence >= 60) return AlertCircle;
    return AlertCircle;
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'from-green-500 to-emerald-500';
    if (score >= 70) return 'from-yellow-500 to-orange-500';
    return 'from-red-500 to-pink-500';
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
    const floatingElements = [
    { emoji: '🤟', delay: 0, duration: 4 },
    { emoji: '👋', delay: 1, duration: 5 },
    { emoji: '✋', delay: 2, duration: 3 },
    { emoji: '👌', delay: 3, duration: 6 },
    { emoji: '🤲', delay: 4, duration: 4 },
    { emoji: '👐', delay: 5, duration: 5 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in-up">
          <div className="relative group mb-6">
            <div className="absolute -inset-4 bg-gradient-to-r blur opacity-25 group-hover:opacity-40 transition duration-1000 animate-pulse"></div>
            <Target className="relative h-16 w-16 text-blue-600 mx-auto animate-bounce-gentle" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">Perfect Your Signs</h1>
          <p className="text-xl text-gray-600">
            AI-powered gesture recognition for skill improvement
          </p>
          {floatingElements.map((element, i) => (
            <div
              key={i}
              className="absolute text-4xl opacity-20 animate-float pointer-events-none"
              style={{
                left: `${10 + (i * 15)}%`,
                top: `${20 + (i * 10)}%`,
                animationDelay: `${element.delay}s`,
                animationDuration: `${element.duration}s`
              }}
            >
              {element.emoji}
            </div>
          ))}
           {/* Rotating rings */}
                      <div className="absolute inset-0 border-2 border-blue-400/30 rounded-full animate-spin" style={{ animationDuration: '20s' }}></div>
                      <div className="absolute inset-4 border-2 border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}></div>
                      <div className="absolute inset-8 border-2 border-pink-400/30 rounded-full animate-spin" style={{ animationDuration: '10s' }}></div>
        </div>

        {/* Practice Stats Bar */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8 animate-fade-in-up animation-delay-200">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Timer className="h-5 w-5 text-blue-600 mr-2" />
                <span className="text-sm font-medium text-gray-600">Session Time</span>
              </div>
              <div className="text-2xl font-bold text-blue-600">{formatTime(sessionTime)}</div>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Award className="h-5 w-5 text-green-600 mr-2" />
                <span className="text-sm font-medium text-gray-600">Practice Streak</span>
              </div>
              <div className="text-2xl font-bold text-green-600">{practiceStreak} days</div>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <BarChart3 className="h-5 w-5 text-purple-600 mr-2" />
                <span className="text-sm font-medium text-gray-600">Avg. Accuracy</span>
              </div>
              <div className="text-2xl font-bold text-purple-600">92%</div>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <TrendingUp className="h-5 w-5 text-orange-600 mr-2" />
                <span className="text-sm font-medium text-gray-600">Signs Practiced</span>
              </div>
              <div className="text-2xl font-bold text-orange-600">47</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm overflow-hidden animate-fade-in-up animation-delay-400">
              <div className="relative bg-gradient-to-br from-gray-800 via-gray-900 to-black h-96 flex items-center justify-center overflow-hidden">
                {/* Animated Background Grid */}
                <div className="absolute inset-0 opacity-10">
                  <div className="grid grid-cols-8 grid-rows-6 h-full w-full">
                    {[...Array(48)].map((_, i) => (
                      <div
                        key={i}
                        className="border border-white/20 animate-pulse"
                        style={{ animationDelay: `${i * 50}ms` }}
                      ></div>
                    ))}
                  </div>
                </div>

                {/* Floating particles */}
                {[...Array(15)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute w-1 h-1 bg-blue-400/30 rounded-full animate-float"
                    style={{
                      left: `${Math.random() * 100}%`,
                      top: `${Math.random() * 100}%`,
                      animationDelay: `${Math.random() * 3}s`,
                      animationDuration: `${2 + Math.random() * 3}s`
                    }}
                  ></div>
                ))}

                <div className="relative text-center text-white z-10">
                  <div className="relative group">
                    <div className={`absolute -inset-8 rounded-full blur transition-all duration-300 ${
                      isRecording ? 'bg-red-500/30 animate-pulse-glow' : 'bg-blue-500/20'
                    }`}></div>
                    <Camera className={`relative h-24 w-24 mx-auto mb-4 opacity-80 transition-all duration-300 ${
                      isRecording ? 'text-red-400 animate-pulse' : 'text-blue-400'
                    } ${isProcessing ? 'animate-spin' : ''}`} />
                  </div>
                  <p className="text-lg font-medium">
                    {isProcessing ? 'Processing your gesture...' : 
                     isRecording ? 'Recording your gestures...' : 'Camera feed ready'}
                  </p>
                  <p className="text-sm opacity-75 mt-2">
                    {isProcessing ? 'AI is analyzing your sign' :
                     isRecording ? 'Make your sign clearly' : 'Click start to begin practice'}
                  </p>
                  
                  {isRecording && (
                    <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-500/20 backdrop-blur-sm rounded-lg px-3 py-2 animate-slide-down">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium">Recording</span>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="absolute top-4 left-4 flex items-center space-x-2 bg-blue-500/20 backdrop-blur-sm rounded-lg px-3 py-2 animate-slide-down">
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium">Processing</span>
                    </div>
                  )}
                </div>
                
                {/* Overlay for detected gestures */}
                {detectedWord && !isProcessing && (
                  <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg p-4 animate-slide-up">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white mb-2 animate-fade-in-up">{detectedWord}</div>
                      <div className="flex items-center justify-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {React.createElement(getConfidenceIcon(confidence), {
                            className: `h-5 w-5 ${getConfidenceColor(confidence)} animate-bounce`
                          })}
                          <span className={`font-medium ${getConfidenceColor(confidence)}`}>
                            {confidence}% confidence
                          </span>
                        </div>
                        {practiceScore > 0 && (
                          <div className={`px-3 py-1 rounded-full text-white text-sm font-medium bg-gradient-to-r ${getScoreColor(practiceScore)} animate-scale-in`}>
                            Score: {practiceScore}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-6 bg-gradient-to-r from-gray-50 to-blue-50">
                <div className="flex items-center justify-center space-x-4">
                  <button
                    onClick={toggleRecording}
                    disabled={isProcessing}
                    className={`group flex items-center space-x-3 px-8 py-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed ${
                      isRecording
                        ? 'bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800'
                        : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700'
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <CameraOff className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                        <span>Stop Practice</span>
                      </>
                    ) : (
                      <>
                        <Camera className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                        <span>Start Practice</span>
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={clearResults}
                    className="group flex items-center space-x-2 px-6 py-4 bg-white text-gray-700 rounded-xl border-2 border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-all duration-300 transform hover:scale-105 shadow-sm hover:shadow-md"
                  >
                    <RotateCcw className="h-5 w-5 group-hover:rotate-180 transition-transform duration-500" />
                    <span>Clear</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Settings & Results */}
          <div className="space-y-6">
            {/* Model Selection */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-600">
              <div className="flex items-center space-x-2 mb-4">
                <Settings className="h-5 w-5 text-gray-600" />
                <h3 className="text-lg font-semibold text-gray-900">Detection Model</h3>
              </div>
              <div className="space-y-3">
                {models.map((model) => (
                  <label key={model.id} className="group flex items-center space-x-3 cursor-pointer p-3 rounded-lg hover:bg-gray-50 transition-colors duration-200">
                    <input
                      type="radio"
                      name="model"
                      value={model.id}
                      checked={selectedModel === model.id}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500 transition-colors duration-200"
                    />
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <div className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-200">
                          {model.name}
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                            {model.accuracy}
                          </span>
                          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                            {model.speed}
                          </span>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">{model.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Detection Results */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-800">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="h-5 w-5 mr-2 text-blue-600" />
                Detection Results
              </h3>
              
              {detectedWord ? (
                <div className="space-y-4 animate-fade-in-up">
                  <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900 mb-2 animate-bounce">{detectedWord}</div>
                    <div className="flex items-center justify-center space-x-4 mb-3">
                      {React.createElement(getConfidenceIcon(confidence), {
                        className: `h-5 w-5 ${getConfidenceColor(confidence)}`
                      })}
                      <span className={`font-medium ${getConfidenceColor(confidence)}`}>
                        {confidence}% confidence
                      </span>
                    </div>
                    {practiceScore > 0 && (
                      <div className={`inline-block px-4 py-2 rounded-full text-white font-medium bg-gradient-to-r ${getScoreColor(practiceScore)} animate-pulse`}>
                        Practice Score: {practiceScore}/100
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Confidence</span>
                      <span className="text-gray-600">{confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ${
                          confidence >= 80 ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 
                          confidence >= 60 ? 'bg-gradient-to-r from-yellow-500 to-orange-500' : 
                          'bg-gradient-to-r from-red-500 to-pink-500'
                        } animate-shimmer`}
                        style={{ width: `${confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Target className="h-12 w-12 mx-auto mb-4 opacity-50 animate-pulse" />
                  <p className="font-medium">Start practicing to see results</p>
                  <p className="text-sm mt-1">Your detection results will appear here</p>
                </div>
              )}
            </div>

            {/* Practice Goals */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1000">
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
                        className={`h-2 rounded-full bg-gradient-to-r from-${goal.color}-500 to-${goal.color}-400 transition-all duration-1000 animate-shimmer`}
                        style={{ width: `${(goal.current / goal.target) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Practice */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
                Recent Practice
              </h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {recentPractice.map((practice, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200 animate-fade-in-left" style={{ animationDelay: `${index * 100}ms` }}>
                    <div>
                      <div className="font-medium text-gray-900">{practice.word}</div>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>{practice.time}</span>
                        <span>•</span>
                        <span className={`px-2 py-1 rounded-full ${
                          practice.difficulty === 'Beginner' ? 'bg-green-100 text-green-700' :
                          practice.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {practice.difficulty}
                        </span>
                      </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-white text-sm font-medium bg-gradient-to-r ${getScoreColor(practice.score)}`}>
                      {practice.score}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Practice Tips */}
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 animate-fade-in-up animation-delay-1400">
              <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-yellow-500 animate-bounce" />
                Practice Tips
              </h3>
              <ul className="space-y-3 text-sm text-blue-800">
                {[
                  'Ensure good lighting for better detection',
                  'Keep your hands clearly visible in frame',
                  'Practice slowly and deliberately',
                  'Maintain consistent hand positioning',
                  'Use appropriate facial expressions'
                ].map((tip, index) => (
                  <li key={index} className="flex items-start animate-fade-in-left" style={{ animationDelay: `${1500 + index * 100}ms` }}>
                    <div className={`w-2 h-2 rounded-full mt-2 mr-3 flex-shrink-0 ${
                      index % 5 === 0 ? 'bg-blue-500' :
                      index % 5 === 1 ? 'bg-purple-500' :
                      index % 5 === 2 ? 'bg-pink-500' :
                      index % 5 === 3 ? 'bg-green-500' :
                      'bg-yellow-500'
                    }`}></div>
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