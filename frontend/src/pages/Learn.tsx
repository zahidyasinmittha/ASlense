import React, { useState } from 'react';
import { Play, X, BookOpen, Hash, Heart, MessageSquare, Users, Star, Trophy, Clock, Search, Filter, Volume2 } from 'lucide-react';

const Learn: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState('alphabet');
  const [selectedWord, setSelectedWord] = useState<any>(null);
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState('all');

  const categories = [
    { id: 'alphabet', name: 'Alphabet', icon: BookOpen, color: 'blue', count: 26 },
    { id: 'numbers', name: 'Numbers', icon: Hash, color: 'green', count: 10 },
    { id: 'daily', name: 'Daily Vocabulary', icon: MessageSquare, color: 'purple', count: 50 },
    { id: 'emotions', name: 'Emotions', icon: Heart, color: 'red', count: 20 },
    { id: 'phrases', name: 'Phrases', icon: Users, color: 'orange', count: 30 },
  ];

  const wordsData = {
    alphabet: [
      { id: 1, title: 'Letter A', description: 'Make a fist with your thumb resting on the side of your index finger.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-a.mp4', thumbnail: '🅰️' },
      { id: 2, title: 'Letter B', description: 'Hold your hand up with all four fingers extended and thumb folded across your palm.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-b.mp4', thumbnail: '🅱️' },
      { id: 3, title: 'Letter C', description: 'Form a C shape with your thumb and fingers.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-c.mp4', thumbnail: '🅲' },
      { id: 4, title: 'Letter D', description: 'Point your index finger up while other fingers touch your thumb.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-d.mp4', thumbnail: '🅳' },
      { id: 5, title: 'Letter E', description: 'Curl all fingers down to touch your thumb.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-e.mp4', thumbnail: '🅴' },
      { id: 6, title: 'Letter F', description: 'Touch your thumb to your index finger, other fingers up.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-f.mp4', thumbnail: '🅵' },
    ],
    numbers: [
      { id: 7, title: 'Number 1', description: 'Point your index finger up while keeping other fingers closed.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-1.mp4', thumbnail: '1️⃣' },
      { id: 8, title: 'Number 2', description: 'Extend your index and middle fingers in a V shape.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-2.mp4', thumbnail: '2️⃣' },
      { id: 9, title: 'Number 3', description: 'Extend your index, middle, and ring fingers.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-3.mp4', thumbnail: '3️⃣' },
      { id: 10, title: 'Number 4', description: 'Hold up four fingers with thumb folded.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-4.mp4', thumbnail: '4️⃣' },
      { id: 11, title: 'Number 5', description: 'Spread all five fingers wide.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-5.mp4', thumbnail: '5️⃣' },
    ],
    daily: [
      { id: 12, title: 'Hello', description: 'Wave your hand in a friendly greeting motion.', difficulty: 'Beginner', duration: '45s', videoUrl: 'https://example.com/video-hello.mp4', thumbnail: '👋' },
      { id: 13, title: 'Thank You', description: 'Touch your chin with your fingertips and move your hand forward.', difficulty: 'Beginner', duration: '45s', videoUrl: 'https://example.com/video-thanks.mp4', thumbnail: '🙏' },
      { id: 14, title: 'Please', description: 'Place your hand on your chest and move it in a circular motion.', difficulty: 'Intermediate', duration: '60s', videoUrl: 'https://example.com/video-please.mp4', thumbnail: '🤲' },
      { id: 15, title: 'Sorry', description: 'Make a fist and rub it in a circular motion on your chest.', difficulty: 'Beginner', duration: '45s', videoUrl: 'https://example.com/video-sorry.mp4', thumbnail: '😔' },
      { id: 16, title: 'Water', description: 'Tap your chin with the W handshape.', difficulty: 'Intermediate', duration: '50s', videoUrl: 'https://example.com/video-water.mp4', thumbnail: '💧' },
      { id: 17, title: 'Food', description: 'Bring your fingertips to your mouth repeatedly.', difficulty: 'Beginner', duration: '40s', videoUrl: 'https://example.com/video-food.mp4', thumbnail: '🍽️' },
    ],
    emotions: [
      { id: 18, title: 'Happy', description: 'Brush your hands up your chest with a joyful expression.', difficulty: 'Beginner', duration: '45s', videoUrl: 'https://example.com/video-happy.mp4', thumbnail: '😊' },
      { id: 19, title: 'Sad', description: 'Run your fingers down your face like tears.', difficulty: 'Beginner', duration: '45s', videoUrl: 'https://example.com/video-sad.mp4', thumbnail: '😢' },
      { id: 20, title: 'Excited', description: 'Alternate touching your chest with both hands in an upward motion.', difficulty: 'Intermediate', duration: '60s', videoUrl: 'https://example.com/video-excited.mp4', thumbnail: '🤩' },
      { id: 21, title: 'Angry', description: 'Claw your fingers and pull them away from your face.', difficulty: 'Intermediate', duration: '55s', videoUrl: 'https://example.com/video-angry.mp4', thumbnail: '😠' },
      { id: 22, title: 'Love', description: 'Cross your arms over your chest and hug yourself.', difficulty: 'Beginner', duration: '40s', videoUrl: 'https://example.com/video-love.mp4', thumbnail: '❤️' },
    ],
    phrases: [
      { id: 23, title: 'How are you?', description: 'Point to the person, then sign "how" and point back to them.', difficulty: 'Intermediate', duration: '90s', videoUrl: 'https://example.com/video-how-are-you.mp4', thumbnail: '❓' },
      { id: 24, title: 'Nice to meet you', description: 'Sign "nice", "to", "meet", and "you" in sequence.', difficulty: 'Advanced', duration: '120s', videoUrl: 'https://example.com/video-nice-meet.mp4', thumbnail: '🤝' },
      { id: 25, title: 'I love you', description: 'Extend your thumb, index, and pinky fingers.', difficulty: 'Beginner', duration: '30s', videoUrl: 'https://example.com/video-i-love-you.mp4', thumbnail: '🤟' },
      { id: 26, title: 'Good morning', description: 'Sign "good" then "morning" with sunrise motion.', difficulty: 'Intermediate', duration: '75s', videoUrl: 'https://example.com/video-good-morning.mp4', thumbnail: '🌅' },
      { id: 27, title: 'See you later', description: 'Point to your eyes, then to the person, then wave goodbye.', difficulty: 'Advanced', duration: '100s', videoUrl: 'https://example.com/video-see-later.mp4', thumbnail: '👋' },
    ],
  };

  const currentWords = wordsData[selectedCategory as keyof typeof wordsData];

  const filteredWords = currentWords.filter(word => {
    const matchesSearch = word.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         word.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDifficulty = difficultyFilter === 'all' || word.difficulty.toLowerCase() === difficultyFilter;
    return matchesSearch && matchesDifficulty;
  });

  const openVideoModal = (word: any) => {
    setSelectedWord(word);
    setIsVideoModalOpen(true);
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
  };

  const closeVideoModal = () => {
    setIsVideoModalOpen(false);
    setSelectedWord(null);
    document.body.style.overflow = 'unset'; // Restore scrolling
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'text-green-600 bg-green-100 border-green-200';
      case 'Intermediate': return 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'Advanced': return 'text-red-600 bg-red-100 border-red-200';
      default: return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getDifficultyIcon = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return '🟢';
      case 'Intermediate': return '🟡';
      case 'Advanced': return '🔴';
      default: return '⚪';
    }
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
      <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-8 ">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in-up ">
          <div className="relative group mb-6">
            <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000 animate-pulse"></div>
            <BookOpen className="relative h-16 w-16 text-blue-600 mx-auto animate-bounce-gentle" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">Master ASL Signs</h1>
          <p className="text-xl text-gray-600">Interactive video lessons for every skill level</p>
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

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar */}
          <div className="lg:w-1/4">
            <div className="bg-white rounded-xl shadow-sm p-6 sticky top-24 animate-fade-in-left">
              <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
                <Filter className="h-5 w-5 mr-2 text-blue-600" />
                Categories
              </h2>
              
              {/* Search */}
              <div className="mb-6">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search signs..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:border-gray-400"
                  />
                </div>
              </div>

              {/* Difficulty Filter */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Difficulty Level</label>
                <select
                  value={difficultyFilter}
                  onChange={(e) => setDifficultyFilter(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="all">All Levels</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </div>

              {/* Categories */}
              <nav className="space-y-2 stagger-animation">
                {categories.map((category) => {
                  const Icon = category.icon;
                  return (
                    <button
                      key={category.id}
                      onClick={() => {
                        setSelectedCategory(category.id);
                        setSearchTerm('');
                        setDifficultyFilter('all');
                      }}
                      className={`w-full flex items-center justify-between px-4 py-3 rounded-lg text-left transition-all duration-300 transform hover:scale-105 group card-hover ${
                        selectedCategory === category.id
                          ? `bg-gradient-to-r from-${category.color}-50 to-${category.color}-100 text-${category.color}-700 border-${category.color}-200 border shadow-md`
                          : 'text-gray-700 hover:bg-gray-50 hover:shadow-sm'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <Icon className={`h-5 w-5 ${selectedCategory === category.id ? `text-${category.color}-600` : 'text-gray-500'} group-hover:scale-110 transition-transform duration-300 animate-wiggle`} />
                        <div>
                          <span className="font-medium">{category.name}</span>
                          <div className="text-xs text-gray-500">{category.count} signs</div>
                        </div>
                      </div>
                      {selectedCategory === category.id && (
                        <div className={`w-3 h-3 bg-${category.color}-600 rounded-full animate-pulse`}></div>
                      )}
                    </button>
                  );
                })}
              </nav>

              {/* Progress Summary */}
              <div className="mt-8 p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg animate-scale-in animation-delay-500">
                <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                  <Trophy className="h-4 w-4 text-yellow-500 mr-2 animate-bounce" />
                  Your Progress
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Signs Learned</span>
                    <span className="font-medium text-blue-600">47/136</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000 animate-shimmer" style={{ width: '35%' }}></div>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Beginner</span>
                    <span>Expert</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:w-3/4">
            {/* Category Header */}
            <div className="bg-white rounded-xl shadow-sm p-6 mb-6 animate-fade-in-right">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {React.createElement(categories.find(c => c.id === selectedCategory)?.icon || BookOpen, {
                    className: `h-8 w-8 text-${categories.find(c => c.id === selectedCategory)?.color}-600 animate-bounce-gentle`
                  })}
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">
                      {categories.find(c => c.id === selectedCategory)?.name}
                    </h2>
                    <p className="text-gray-600">
                      {filteredWords.length} of {currentWords.length} signs
                      {searchTerm && ` matching "${searchTerm}"`}
                      {difficultyFilter !== 'all' && ` (${difficultyFilter})`}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-500">Total Duration</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {Math.round(filteredWords.reduce((acc, word) => acc + parseInt(word.duration), 0) / 60)} min
                  </div>
                </div>
              </div>
            </div>

            {/* Words Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 stagger-animation">
              {filteredWords.map((word, index) => (
                <div
                  key={word.id}
                  onClick={() => openVideoModal(word)}
                  className="group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 cursor-pointer word-card animate-scale-in"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  {/* Thumbnail */}
                  <div className="relative bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 h-48 rounded-t-xl flex items-center justify-center overflow-hidden">
                    <div className="absolute inset-0 bg-black/20"></div>
                    <div className="relative text-6xl group-hover:scale-110 transition-transform duration-300 animate-float" style={{ animationDelay: `${index * 200}ms` }}>
                      {word.thumbnail}
                    </div>
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
                      <Play className="h-12 w-12 text-white opacity-0 group-hover:opacity-100 transition-all duration-300 transform scale-75 group-hover:scale-100 animate-pulse" />
                    </div>
                    
                    {/* Difficulty Badge */}
                    <div className={`absolute top-3 right-3 px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(word.difficulty)} backdrop-blur-sm`}>
                      <span className="mr-1">{getDifficultyIcon(word.difficulty)}</span>
                      {word.difficulty}
                    </div>

                    {/* Duration Badge */}
                    <div className="absolute bottom-3 left-3 flex items-center space-x-1 bg-black/50 backdrop-blur-sm rounded-full px-2 py-1 text-white text-xs">
                      <Clock className="h-3 w-3" />
                      <span>{word.duration}</span>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors duration-300">
                      {word.title}
                    </h3>
                    <p className="text-gray-600 text-sm leading-relaxed mb-4 line-clamp-2">
                      {word.description}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Volume2 className="h-4 w-4 text-gray-400" />
                        <span className="text-xs text-gray-500">Audio included</span>
                      </div>
                      <button className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 transition-colors duration-300 group-hover:translate-x-1 transform">
                        <span className="text-sm font-medium">Watch</span>
                        <Play className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* No Results */}
            {filteredWords.length === 0 && (
              <div className="text-center py-12 animate-fade-in-up">
                <div className="text-6xl mb-4 animate-bounce">🔍</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">No signs found</h3>
                <p className="text-gray-600 mb-4">
                  Try adjusting your search terms or difficulty filter
                </p>
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setDifficultyFilter('all');
                  }}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-300 transform hover:scale-105"
                >
                  Clear Filters
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Video Modal */}
      {isVideoModalOpen && selectedWord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in-up">
          <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto animate-scale-in">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200 sticky top-0 bg-white z-10">
              <div className="flex items-center space-x-4">
                <div className="text-3xl">{selectedWord.thumbnail}</div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{selectedWord.title}</h2>
                  <div className="flex items-center space-x-3 mt-1">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(selectedWord.difficulty)}`}>
                      {getDifficultyIcon(selectedWord.difficulty)} {selectedWord.difficulty}
                    </span>
                    <div className="flex items-center space-x-1 text-gray-500 text-sm">
                      <Clock className="h-4 w-4" />
                      <span>{selectedWord.duration}</span>
                    </div>
                  </div>
                </div>
              </div>
              <button
                onClick={closeVideoModal}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-300 hover:scale-110"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {/* Video Player */}
            <div className="relative bg-gray-900 h-96 flex items-center justify-center">
              <div className="text-center text-white">
                <Play className="h-24 w-24 mx-auto mb-4 opacity-80 animate-pulse" />
                <p className="text-lg font-medium">Video Player</p>
                <p className="text-sm opacity-75 mt-2">
                  Interactive ASL demonstration for "{selectedWord.title}"
                </p>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
                  <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
                    <BookOpen className="h-5 w-5 mr-2" />
                    How to Sign
                  </h3>
                  <p className="text-blue-800 leading-relaxed">
                    {selectedWord.description}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
                  <h3 className="font-semibold text-purple-900 mb-3 flex items-center">
                    <MessageSquare className="h-5 w-5 mr-2" />
                    Usage Tips
                  </h3>
                  <ul className="text-purple-800 space-y-2 text-sm">
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Practice slowly at first to build muscle memory
                    </li>
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Maintain clear hand positioning
                    </li>
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Use appropriate facial expressions
                    </li>
                  </ul>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-between mt-6">
                <button className="flex items-center space-x-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-300 hover:scale-105">
                  <Star className="h-4 w-4" />
                  <span>Add to Favorites</span>
                </button>
                
                <div className="flex space-x-3">
                  <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 hover:scale-105 shadow-lg">
                    Practice This Sign
                  </button>
                  <button
                    onClick={closeVideoModal}
                    className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-300 hover:scale-105"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Learn;