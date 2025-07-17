import React, { useState, useEffect } from 'react';
import { Camera, Square, Play, RefreshCw, Settings, Mic, Zap, Activity, Globe, Download, Share2, Copy, Volume2 } from 'lucide-react';

const Translate: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('mediapipe');
  const [inputMode, setInputMode] = useState('word');
  const [isRecording, setIsRecording] = useState(false);
  const [detectedText, setDetectedText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionStats, setSessionStats] = useState({ translations: 0, accuracy: 0, time: 0 });
  const [translationHistory, setTranslationHistory] = useState([
    { text: 'Hello, how are you?', timestamp: '2 min ago', confidence: 95, mode: 'sentence' },
    { text: 'Thank you very much', timestamp: '5 min ago', confidence: 88, mode: 'word' },
    { text: 'Nice to meet you', timestamp: '8 min ago', confidence: 92, mode: 'sentence' },
    { text: 'Please', timestamp: '12 min ago', confidence: 94, mode: 'word' },
    { text: 'Good morning', timestamp: '15 min ago', confidence: 89, mode: 'sentence' },
  ]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => {
        setSessionStats(prev => ({ ...prev, time: prev.time + 1 }));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const models = [
    { id: 'mediapipe', name: 'MediaPipe v2.1', accuracy: '95%', speed: 'Fast', description: 'Best for real-time detection', color: 'blue' },
    { id: 'openpose', name: 'OpenPose Enhanced', accuracy: '87%', speed: 'Medium', description: 'High accuracy for complex signs', color: 'purple' },
    { id: 'custom', name: 'ASLense Custom', accuracy: '92%', speed: 'Fast', description: 'Optimized for ASL vocabulary', color: 'green' },
  ];

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    
    if (!isRecording) {
      setIsProcessing(true);
      // Simulate detection
      const words = ['Hello', 'Thank You', 'Please', 'How are you?', 'Nice to meet you'];
      const sentences = [
        'Hello, how are you today?',
        'Thank you for your help.',
        'Please wait a moment.',
        'I am learning sign language.',
        'Nice to meet you, my friend.',
        'Good morning, have a great day!',
        'See you later, take care!'
      ];
      
      const results = inputMode === 'word' ? words : sentences;
      const randomResult = results[Math.floor(Math.random() * results.length)];
      const confidence = Math.floor(Math.random() * 20) + 80;
      
      setTimeout(() => {
        setDetectedText(randomResult);
        setIsProcessing(false);
        // Add to history
        setTranslationHistory(prev => [
          { text: randomResult, timestamp: 'Just now', confidence, mode: inputMode },
          ...prev.slice(0, 4)
        ]);
        // Update stats
        setSessionStats(prev => ({
          ...prev,
          translations: prev.translations + 1,
          accuracy: Math.round((prev.accuracy * prev.translations + confidence) / (prev.translations + 1))
        }));
      }, inputMode === 'word' ? 1000 : 2500);
    } else {
      setDetectedText('');
      setIsProcessing(false);
    }
  };

  const clearResults = () => {
    setDetectedText('');
    setIsRecording(false);
    setIsProcessing(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      speechSynthesis.speak(utterance);
    }
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
      <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in-up">
          <div className="relative group mb-6">
            <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000 animate-pulse"></div>
            <Globe className="relative h-16 w-16 text-green-600 mx-auto animate-spin" style={{ animationDuration: '3s' }} />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">Real-Time Translation</h1>
          <p className="text-xl text-gray-600">
            Convert sign language gestures to text instantly with AI
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

        {/* Session Stats */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8 animate-fade-in-up animation-delay-200">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 animate-counter">{sessionStats.translations}</div>
              <div className="text-sm text-gray-600">Translations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600 animate-counter">{sessionStats.accuracy}%</div>
              <div className="text-sm text-gray-600">Avg. Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{formatTime(sessionStats.time)}</div>
              <div className="text-sm text-gray-600">Session Time</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600 capitalize">{inputMode}</div>
              <div className="text-sm text-gray-600">Current Mode</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Settings Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Model Selection */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-left">
              <div className="flex items-center space-x-2 mb-4">
                <Settings className="h-5 w-5 text-gray-600" />
                <h3 className="text-lg font-semibold text-gray-900">AI Model</h3>
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
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-200">
                        {model.name}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">{model.description}</div>
                      <div className="flex items-center space-x-2 mt-2">
                        <span className={`text-xs px-2 py-1 bg-${model.color}-100 text-${model.color}-700 rounded-full`}>
                          {model.accuracy}
                        </span>
                        <span className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded-full">
                          {model.speed}
                        </span>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Input Mode */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-left animation-delay-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Activity className="h-5 w-5 mr-2 text-purple-600" />
                Input Mode
              </h3>
              <div className="space-y-3">
                <label className="group flex items-center space-x-3 cursor-pointer p-4 rounded-lg hover:bg-gray-50 transition-colors duration-200 border-2 border-transparent hover:border-blue-200">
                  <input
                    type="radio"
                    name="inputMode"
                    value="word"
                    checked={inputMode === 'word'}
                    onChange={(e) => setInputMode(e.target.value)}
                    className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-200 flex items-center">
                      <Zap className="h-4 w-4 mr-2 text-yellow-500" />
                      Word Detection
                    </div>
                    <div className="text-sm text-gray-500">Real-time word recognition</div>
                  </div>
                </label>
                <label className="group flex items-center space-x-3 cursor-pointer p-4 rounded-lg hover:bg-gray-50 transition-colors duration-200 border-2 border-transparent hover:border-purple-200">
                  <input
                    type="radio"
                    name="inputMode"
                    value="sentence"
                    checked={inputMode === 'sentence'}
                    onChange={(e) => setInputMode(e.target.value)}
                    className="w-4 h-4 text-purple-600 focus:ring-purple-500"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-purple-600 transition-colors duration-200 flex items-center">
                      <Mic className="h-4 w-4 mr-2 text-purple-500" />
                      Sentence Mode
                    </div>
                    <div className="text-sm text-gray-500">Record complete sentences</div>
                  </div>
                </label>
              </div>
            </div>

            {/* Translation History */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-left animation-delay-400">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Globe className="h-5 w-5 mr-2 text-green-600" />
                Recent Translations
              </h3>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {translationHistory.map((item, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200 group animate-fade-in-up" style={{ animationDelay: `${index * 100}ms` }}>
                    <div className="text-sm font-medium text-gray-900 mb-1 line-clamp-2">{item.text}</div>
                    <div className="flex justify-between items-center text-xs text-gray-500">
                      <div className="flex items-center space-x-2">
                        <span>{item.timestamp}</span>
                        <span className={`px-2 py-1 rounded-full ${
                          item.mode === 'word' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'
                        }`}>
                          {item.mode}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full">
                          {item.confidence}%
                        </span>
                        <button
                          onClick={() => copyToClipboard(item.text)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 rounded transition-all duration-200"
                        >
                          <Copy className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-left animation-delay-600">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={clearResults}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-300 transform hover:scale-105"
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>Clear Results</span>
                </button>
                <button className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-300 transform hover:scale-105">
                  <Download className="h-4 w-4" />
                  <span>Export History</span>
                </button>
                <button className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-all duration-300 transform hover:scale-105">
                  <Share2 className="h-4 w-4" />
                  <span>Share Session</span>
                </button>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Camera Feed */}
            <div className="bg-white rounded-xl shadow-sm overflow-hidden animate-fade-in-up animation-delay-800">
              <div className="relative bg-gradient-to-br from-gray-800 via-blue-900 to-purple-900 h-96 flex items-center justify-center overflow-hidden">
                {/* Animated Background */}
                <div className="absolute inset-0">
                  <div className="absolute top-0 left-0 w-full h-full">
                    {[...Array(30)].map((_, i) => (
                      <div
                        key={i}
                        className="absolute w-2 h-2 bg-white/10 rounded-full animate-float"
                        style={{
                          left: `${Math.random() * 100}%`,
                          top: `${Math.random() * 100}%`,
                          animationDelay: `${Math.random() * 3}s`,
                          animationDuration: `${3 + Math.random() * 2}s`
                        }}
                      ></div>
                    ))}
                  </div>
                </div>

                {/* Grid overlay */}
                <div className="absolute inset-0 opacity-5">
                  <div className="grid grid-cols-12 grid-rows-8 h-full w-full">
                    {[...Array(96)].map((_, i) => (
                      <div
                        key={i}
                        className="border border-white/30 animate-pulse"
                        style={{ animationDelay: `${i * 20}ms` }}
                      ></div>
                    ))}
                  </div>
                </div>

                <div className="relative text-center text-white z-10">
                  <div className="relative group">
                    <div className={`absolute -inset-8 rounded-full blur transition-all duration-300 ${
                      isRecording 
                        ? inputMode === 'word' 
                          ? 'bg-blue-500/30 animate-pulse-glow' 
                          : 'bg-purple-500/30 animate-pulse-glow'
                        : 'bg-gray-500/20'
                    }`}></div>
                    <Camera className={`relative h-24 w-24 mx-auto mb-4 opacity-90 transition-all duration-300 ${
                      isRecording 
                        ? inputMode === 'word'
                          ? 'text-blue-400 animate-pulse'
                          : 'text-purple-400 animate-pulse'
                        : 'text-gray-400'
                    } ${isProcessing ? 'animate-spin' : ''}`} />
                  </div>
                  <p className="text-lg font-medium">
                    {isProcessing 
                      ? 'Processing your signs...'
                      : isRecording 
                        ? inputMode === 'word' 
                          ? 'Detecting signs in real-time...' 
                          : 'Recording sentence...'
                        : 'Camera ready for translation'
                    }
                  </p>
                  <p className="text-sm opacity-75 mt-2">
                    {isProcessing
                      ? 'AI is analyzing your gestures'
                      : inputMode === 'word' 
                        ? 'Live translation will appear below'
                        : 'Complete sentences will be processed after recording'
                    }
                  </p>
                  
                  {isRecording && (
                    <div className={`absolute top-4 right-4 flex items-center space-x-2 backdrop-blur-sm rounded-lg px-3 py-2 animate-slide-down ${
                      inputMode === 'word' ? 'bg-blue-500/20' : 'bg-purple-500/20'
                    }`}>
                      <div className={`w-3 h-3 rounded-full animate-pulse ${
                        inputMode === 'word' ? 'bg-blue-500' : 'bg-purple-500'
                      }`}></div>
                      <span className="text-sm font-medium">
                        {inputMode === 'word' ? 'Detecting...' : 'Recording...'}
                      </span>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="absolute top-4 left-4 flex items-center space-x-2 bg-yellow-500/20 backdrop-blur-sm rounded-lg px-3 py-2 animate-slide-down">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium">Processing...</span>
                    </div>
                  )}
                </div>
                
                {/* Recording overlay for sentence mode */}
                {inputMode === 'sentence' && isRecording && (
                  <div className="absolute inset-0 bg-purple-500/10 border-4 border-purple-500/50 rounded-lg animate-pulse"></div>
                )}
              </div>

              {/* Controls */}
              <div className="p-6 bg-gradient-to-r from-gray-50 to-blue-50">
                <div className="flex items-center justify-center space-x-4">
                  {inputMode === 'word' ? (
                    <button
                      onClick={toggleRecording}
                      disabled={isProcessing}
                      className={`group flex items-center space-x-3 px-8 py-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed ${
                        isRecording
                          ? 'bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800'
                          : 'bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800'
                      }`}
                    >
                      <Camera className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                      <span>{isRecording ? 'Stop Detection' : 'Start Detection'}</span>
                    </button>
                  ) : (
                    <button
                      onClick={toggleRecording}
                      disabled={isProcessing}
                      className={`group flex items-center space-x-3 px-8 py-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed ${
                        isRecording
                          ? 'bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800'
                          : 'bg-gradient-to-r from-purple-600 to-purple-700 text-white hover:from-purple-700 hover:to-purple-800'
                      }`}
                    >
                      {isRecording ? (
                        <>
                          <Square className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                          <span>Stop Recording</span>
                        </>
                      ) : (
                        <>
                          <Play className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                          <span>Start Recording</span>
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Results */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-up animation-delay-1000">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Globe className="h-5 w-5 mr-2 text-green-600" />
                {inputMode === 'word' ? 'Live Translation' : 'Sentence Translation'}
              </h3>
              
              {isProcessing ? (
                <div className="flex items-center justify-center py-12">
                  <div className="relative">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                    <div className="absolute inset-0 animate-ping rounded-full h-12 w-12 border border-blue-400 opacity-25"></div>
                  </div>
                  <span className="ml-4 text-gray-600 font-medium">Processing your signs...</span>
                </div>
              ) : detectedText ? (
                <div className="space-y-6 animate-fade-in-up">
                  <div className="bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 rounded-xl p-6 border border-blue-100">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="text-2xl font-bold text-gray-900 mb-3 animate-typewriter">{detectedText}</div>
                        <div className="text-sm text-blue-700 font-medium">
                          {inputMode === 'word' ? 'Live detection result' : 'Sentence translation complete'}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => speakText(detectedText)}
                          className="p-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition-colors duration-200"
                          title="Speak text"
                        >
                          <Volume2 className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => copyToClipboard(detectedText)}
                          className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors duration-200"
                          title="Copy to clipboard"
                        >
                          <Copy className="h-4 w-4" />
                        </button>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                          <span className="text-sm text-green-600 font-medium">Active</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-gray-600">Model</span>
                      <span className="font-medium text-gray-900">
                        {models.find(m => m.id === selectedModel)?.name.split(' ')[0]}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-gray-600">Mode</span>
                      <span className="font-medium text-gray-900 capitalize">
                        {inputMode === 'word' ? 'Real-time' : 'Batch'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-gray-600">Confidence</span>
                      <span className="font-medium text-green-600">
                        {translationHistory[0]?.confidence || 0}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-gray-600">Status</span>
                      <span className="font-medium text-green-600">Complete</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="relative mb-4">
                    <Globe className="h-16 w-16 mx-auto opacity-20 animate-pulse" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-8 h-8 border-2 border-gray-300 border-dashed rounded-full animate-spin opacity-30"></div>
                    </div>
                  </div>
                  <div className="text-lg font-medium mb-2">
                    {inputMode === 'word' ? 'Start detection to see live results' : 'Record a sentence to translate'}
                  </div>
                  <div className="text-sm">
                    {inputMode === 'word' 
                      ? 'Real-time word detection will appear here instantly'
                      : 'Complete sentences will be translated after recording'
                    }
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Translate;