import React, { useState, useEffect } from 'react';
import { BookOpen, Eye, Star, ChevronRight, ImageIcon, Target, X, Play, Lightbulb, ArrowLeft, ArrowRight } from 'lucide-react';
import { pslAPI } from '../services/api';

interface PSLAlphabetEntry {
  id: number;
  letter: string;
  file_path: string;
  label: string;
  difficulty: string;
  description: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

const PSLLearn: React.FC = () => {
  const [selectedLetter, setSelectedLetter] = useState<PSLAlphabetEntry | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [pslEntries, setPslEntries] = useState<PSLAlphabetEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create image arrays for each letter based on database entries
  const getImagesForLetter = (letter: string): string[] => {
    // Find all entries for this letter
    const letterEntries = pslEntries.filter(entry => {
      // Extract the base letter name (before the underscore)
      const baseLetter = entry.letter.split('_')[0];
      return baseLetter === letter;
    });
    
    // Return the file paths with /img/ prefix, filtering out video files
    return letterEntries
      .filter(entry => entry.file_path.match(/\.(jpg|jpeg|png|JPG|JPEG|PNG)$/i))
      .map(entry => `/img/${entry.file_path}`)
      .slice(0, 3); // Limit to 3 images
  };

  const fetchPslEntries = async () => {
    try {
      setLoading(true);

      // Fetch all PSL entries for complete alphabet display
      const response = await pslAPI.getAll({ skip: 0, limit: 200 });
      
      setPslEntries(response.data as PSLAlphabetEntry[]);
      setError(null);
    } catch (error) {
      console.error('Error fetching PSL entries:', error);
      setError('Error loading PSL alphabet entries');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPslEntries();
  }, []);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'hard': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const handleLetterClick = (entry: PSLAlphabetEntry) => {
    setSelectedLetter(entry);
    setCurrentImageIndex(0);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedLetter(null);
    setCurrentImageIndex(0);
  };

  const nextImage = () => {
    if (selectedLetter) {
      const baseLetter = selectedLetter.letter.split('_')[0];
      const images = getImagesForLetter(baseLetter);
      if (currentImageIndex < images.length - 1) {
        setCurrentImageIndex(currentImageIndex + 1);
      }
    }
  };

  const prevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1);
    }
  };

  // Handle keyboard events for modal
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isModalOpen) {
        closeModal();
      } else if (event.key === 'ArrowLeft' && isModalOpen) {
        prevImage();
      } else if (event.key === 'ArrowRight' && isModalOpen) {
        nextImage();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isModalOpen, currentImageIndex]);

  // Filter entries by difficulty and get unique letters (images only)
  const getUniqueLetters = (difficulty: string) => {
    const seenLetters = new Set<string>();
    const result: PSLAlphabetEntry[] = [];
    
    // Filter entries by difficulty and only include image files
    pslEntries
      .filter(entry => 
        entry.difficulty.toLowerCase() === difficulty.toLowerCase() &&
        entry.file_path.match(/\.(jpg|jpeg|png|JPG|JPEG|PNG)$/i)
      )
      .forEach(entry => {
        const baseLetter = entry.letter.split('_')[0];
        
        // If we haven't seen this letter yet, add it
        if (!seenLetters.has(baseLetter)) {
          seenLetters.add(baseLetter);
          result.push(entry);
        }
      });
    
    return result;
  };

  const easyLetters = getUniqueLetters('easy');
  const mediumLetters = getUniqueLetters('medium');
  const hardLetters = getUniqueLetters('hard');

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading PSL alphabet...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-4">⚠️</div>
          <p className="text-red-600">{error}</p>
          <button 
            onClick={() => fetchPslEntries()}
            className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-br from-green-600 via-teal-600 to-blue-600 text-white">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="relative group mb-8">
              <div className="absolute -inset-4 bg-white/20 blur-xl rounded-full opacity-70 group-hover:opacity-100 transition duration-1000"></div>
              <BookOpen className="relative h-16 w-16 mx-auto animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">Learn PSL Alphabet</h1>
            <p className="text-xl text-green-100 max-w-3xl mx-auto">
              Master the complete Pakistan Sign Language alphabet with visual demonstrations. 
              Learn all 26 letters through clear, easy-to-follow images and step-by-step guidance.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4 text-sm">
              <div className="flex items-center space-x-2 bg-white/20 rounded-full px-4 py-2">
                <ImageIcon className="h-4 w-4" />
                <span>26 Alphabet Signs</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/20 rounded-full px-4 py-2">
                <Eye className="h-4 w-4" />
                <span>Visual Learning</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/20 rounded-full px-4 py-2">
                <Target className="h-4 w-4" />
                <span>Progressive Difficulty</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 -mt-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-all duration-300">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 rounded-full">
                <Target className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Easy Letters</h3>
                <p className="text-2xl font-bold text-green-600">{easyLetters.length}</p>
                <p className="text-sm text-gray-600">Perfect for beginners</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-all duration-300">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-yellow-100 rounded-full">
                <Star className="h-6 w-6 text-yellow-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Medium Letters</h3>
                <p className="text-2xl font-bold text-yellow-600">{mediumLetters.length}</p>
                <p className="text-sm text-gray-600">Intermediate practice</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-all duration-300">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-red-100 rounded-full">
                <BookOpen className="h-6 w-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Hard Letters</h3>
                <p className="text-2xl font-bold text-red-600">{hardLetters.length}</p>
                <p className="text-sm text-gray-600">Advanced challenges</p>
              </div>
            </div>
          </div>
        </div>

        {/* Learning Sections */}
        <div className="space-y-8">
          {/* Easy Letters */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <div className="w-4 h-4 bg-green-500 rounded-full mr-3"></div>
              Easy Letters ({easyLetters.length})
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">
              {easyLetters.map((entry) => (
                <div
                  key={entry.id}
                  className="group cursor-pointer bg-gray-50 rounded-xl p-6 hover:bg-green-50 hover:shadow-lg transition-all duration-300"
                  onClick={() => handleLetterClick(entry)}
                >
                  <img
                    src={`/img/${entry.file_path}`}
                    alt={entry.description}
                    className="w-full h-40 object-contain rounded-lg mb-4"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = '/img/placeholder.jpg';
                    }}
                  />
                  <h3 className="text-2xl font-bold text-center text-gray-900 mb-3">{entry.letter.split('_')[0]}</h3>
                  <p className="text-sm text-gray-600 text-center mb-2">{entry.label}</p>
                  <div className={`px-3 py-2 rounded-full text-sm font-medium text-center ${getDifficultyColor(entry.difficulty)}`}>
                    {entry.difficulty}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Medium Letters */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <div className="w-4 h-4 bg-yellow-500 rounded-full mr-3"></div>
              Medium Letters ({mediumLetters.length})
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">
              {mediumLetters.map((entry) => (
                <div
                  key={entry.id}
                  className="group cursor-pointer bg-gray-50 rounded-xl p-6 hover:bg-yellow-50 hover:shadow-lg transition-all duration-300"
                  onClick={() => handleLetterClick(entry)}
                >
                  <img
                    src={`/img/${entry.file_path}`}
                    alt={entry.description}
                    className="w-full h-40 object-contain rounded-lg mb-4"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = '/img/placeholder.jpg';
                    }}
                  />
                  <h3 className="text-2xl font-bold text-center text-gray-900 mb-3">{entry.letter.split('_')[0]}</h3>
                  <p className="text-sm text-gray-600 text-center mb-2">{entry.label}</p>
                  <div className={`px-3 py-2 rounded-full text-sm font-medium text-center ${getDifficultyColor(entry.difficulty)}`}>
                    {entry.difficulty}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hard Letters */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <div className="w-4 h-4 bg-red-500 rounded-full mr-3"></div>
              Hard Letters ({hardLetters.length})
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">
              {hardLetters.map((entry) => (
                <div
                  key={entry.id}
                  className="group cursor-pointer bg-gray-50 rounded-xl p-6 hover:bg-red-50 hover:shadow-lg transition-all duration-300"
                  onClick={() => handleLetterClick(entry)}
                >
                  <img
                    src={`/img/${entry.file_path}`}
                    alt={entry.description}
                    className="w-full h-48 object-contain rounded-lg mb-4"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = '/img/placeholder.jpg';
                    }}
                  />
                  <h3 className="text-2xl font-bold text-center text-gray-900 mb-3">{entry.letter.split('_')[0]}</h3>
                  <p className="text-sm text-gray-600 text-center mb-2">{entry.label}</p>
                  <div className={`px-3 py-2 rounded-full text-sm font-medium text-center ${getDifficultyColor(entry.difficulty)}`}>
                    {entry.difficulty}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-8 bg-gradient-to-br from-green-600 to-blue-600 rounded-2xl p-8 text-white text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Practice?</h2>
          <p className="text-xl text-green-100 mb-6">
            Now that you've learned the alphabet, test your skills with our practice mode!
          </p>
          <button className="bg-white text-green-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors duration-300 flex items-center mx-auto">
            Start Practicing
            <ChevronRight className="h-5 w-5 ml-2" />
          </button>
        </div>

        {/* Modal */}
        {isModalOpen && selectedLetter && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={closeModal}
          >
            <div 
              className="bg-white rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <h2 className="text-3xl font-bold text-gray-900">
                  Letter {selectedLetter.letter.split('_')[0]}
                </h2>
                <button
                  onClick={closeModal}
                  className="p-2 hover:bg-gray-100 rounded-full transition-colors duration-200"
                >
                  <X className="h-6 w-6 text-gray-500" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6">
                <div className="flex flex-col lg:flex-col gap-8">
                  {/* Images Section */}
                  <div className="flex-1 space-y-4">
                    {/* Main Image Display */}
                    <div className="relative">
                      <img
                        src={getImagesForLetter(selectedLetter.letter.split('_')[0])[currentImageIndex]}
                        alt={`${selectedLetter.letter} sign view ${currentImageIndex + 1}`}
                        className="w-full h-80 object-contain rounded-xl border-4 border-gray-200"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = '/img/placeholder.jpg';
                        }}
                      />
                      <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-sm font-semibold ${getDifficultyColor(selectedLetter.difficulty)}`}>
                        {selectedLetter.difficulty}
                      </div>
                      
                      {/* Image Navigation */}
                      {getImagesForLetter(selectedLetter.letter.split('_')[0]).length > 1 && (
                        <>
                          <button
                            onClick={prevImage}
                            disabled={currentImageIndex === 0}
                            className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-white bg-opacity-80 hover:bg-opacity-100 p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                          >
                            <ArrowLeft className="h-5 w-5 text-gray-700" />
                          </button>
                          <button
                            onClick={nextImage}
                            disabled={currentImageIndex === getImagesForLetter(selectedLetter.letter.split('_')[0]).length - 1}
                            className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-white bg-opacity-80 hover:bg-opacity-100 p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                          >
                            <ArrowRight className="h-5 w-5 text-gray-700" />
                          </button>
                        </>
                      )}
                    </div>

                    {/* Image Thumbnails */}
                    <div className="flex space-x-2 justify-center">
                      {getImagesForLetter(selectedLetter.letter.split('_')[0]).map((image, index) => (
                        <button
                          key={index}
                          onClick={() => setCurrentImageIndex(index)}
                          className={`w-16 h-16 rounded-lg overflow-hidden border-2 transition-all duration-200 ${
                            index === currentImageIndex ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-300 hover:border-gray-400'
                          }`}
                        >
                          <img
                            src={image}
                            alt={`View ${index + 1}`}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.src = '/img/placeholder.jpg';
                            }}
                          />
                        </button>
                      ))}
                    </div>

                    {/* Image Counter */}
                    <div className="text-center text-sm text-gray-600">
                      {currentImageIndex + 1} of {getImagesForLetter(selectedLetter.letter.split('_')[0]).length} views
                    </div>
                    
                    {/* Practice Button */}
                    <button className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 px-4 rounded-xl font-semibold hover:from-green-700 hover:to-teal-700 transition-all duration-300 flex items-center justify-center space-x-2">
                      <Play className="h-5 w-5" />
                      <span>Practice This Letter</span>
                    </button>
                  </div>

                  {/* Letter Details */}
                  <div className="flex-1 space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-3 flex items-center">
                        <Lightbulb className="h-5 w-5 mr-2 text-yellow-500" />
                        How to Sign
                      </h3>
                      <p className="text-gray-700 leading-relaxed text-lg">
                        {selectedLetter.description}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-3">
                        Multiple Views
                      </h3>
                      <div className="space-y-3">
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            <strong>View 1:</strong> Front angle - Shows the main hand position
                          </p>
                        </div>
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            <strong>View 2:</strong> Side angle - Shows hand depth and positioning
                          </p>
                        </div>
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            <strong>View 3:</strong> Detail view - Close-up of finger positions
                          </p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-3">
                        Tips for Practice
                      </h3>
                      <div className="space-y-3">
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            Use all three image views to understand the complete hand position
                          </p>
                        </div>
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            Practice in front of a mirror to check your hand positioning
                          </p>
                        </div>
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            Hold the sign for 2-3 seconds to build muscle memory
                          </p>
                        </div>
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                          <p className="text-gray-700">
                            Repeat 10-15 times for effective learning
                          </p>
                        </div>
                        {selectedLetter.difficulty === 'Hard' && (
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                            <p className="text-gray-700">
                              This is a challenging letter - take your time and practice slowly
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-xl p-4">
                      <h4 className="font-semibold text-gray-900 mb-2">Difficulty Level</h4>
                      <div className="flex items-center space-x-2">
                        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(selectedLetter.difficulty)}`}>
                          {selectedLetter.difficulty}
                        </div>
                        <span className="text-gray-600">
                          {selectedLetter.difficulty === 'Easy' && '• Great for beginners'}
                          {selectedLetter.difficulty === 'Medium' && '• Intermediate level'}
                          {selectedLetter.difficulty === 'Hard' && '• Advanced challenge'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50 rounded-b-2xl">
                <button
                  onClick={closeModal}
                  className="px-6 py-2 text-gray-600 hover:text-gray-800 transition-colors duration-200"
                >
                  Close
                </button>
                <div className="text-sm text-gray-500">
                  Use ← → arrow keys to navigate images • ESC to close
                </div>
                <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 flex items-center space-x-2">
                  <span>Next Letter</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PSLLearn;
