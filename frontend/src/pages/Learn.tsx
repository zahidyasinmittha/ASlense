import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Play,
  X,
  BookOpen,
  Hash,
  Heart,
  MessageSquare,
  Users,
  Star,
  Trophy,
  Clock,
  Search,
  Filter,
  Volume2,
} from "lucide-react";
import VideoPlayer from "../components/VideoPlayer";
import { useAuth } from "../contexts/AuthContext";

const Learn: React.FC = () => {
  const baseUrl = import.meta.env.VITE_BACKEND_BASEURL;
  const { user, token, makeAuthenticatedRequest } = useAuth();
  const [selectedCategory, setSelectedCategory] = useState("Alphabet");
  const [selectedWord, setSelectedWord] = useState<any>(null);
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("all");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  // User progress state
  const [userProgress, setUserProgress] = useState({
    signs_practiced: 0,
    signs_learned: 0,
    total_signs: 136,
    accuracy_rate: 0.0,
    current_level: 'Beginner',
    current_xp: 0,
    next_level_xp: 100,
    level_progress: 0,
    practice_streak: 0,
    total_practice_time: 0,
    signs_mastered: 0,
    experience_points: 0,
    level: 'Beginner'
  });
  // Note: totalSigns removed as we now use userProgress.total_signs throughout

  const categories = [
    {
      id: "Alphabet",
      name: "Alphabet",
      icon: BookOpen,
      color: "blue",
      count: 56,
    },
    { id: "Numbers", name: "Numbers", icon: Hash, color: "green", count: 24 },
    {
      id: "Daily Vocabulary",
      name: "Daily Vocabulary",
      icon: MessageSquare,
      color: "purple",
      count: 3758,
    },
    { id: "Emotions", name: "Emotions", icon: Heart, color: "red", count: 536 },
    { id: "Phrases", name: "Phrases", icon: Users, color: "orange", count: 40 },
  ];

  interface VideoFromApi {
    id: number;
    title: string;
    description: string;
    difficulty: "Beginner" | "Intermediate" | "Advanced";
    duration: string;
    video_file: string;
    thumbnail: string;
    category: string;
    word: string;
  }

  interface Video {
    id: number;
    title: string;
    description: string;
    difficulty: "Beginner" | "Intermediate" | "Advanced";
    duration: string;
    thumbnail: string;
    thumbnailUrl?: string;
    videoUrl: string;
    category: string;
    word: string;
  }
   
  interface VideosResponse {
  videos: VideoFromApi[];
  total: number;
  page: number;
}

  const [currentWords, setCurrentWords] = useState<Video[]>([]);
  const [filteredWords, setFilteredWords] = useState<Video[]>([]);

  // Fetch user progress data
  const fetchUserProgress = async () => {
    if (!user || !token) return;
    
    try {
      const response = await makeAuthenticatedRequest(`${baseUrl}/user/progress`);
      if (response.ok) {
        const data = await response.json();
        console.log('Learn page received progress data:', data);
        setUserProgress(data);
      }
    } catch (error) {
      console.error('Error fetching user progress:', error);
    }
  };

  // Load user progress on component mount
  useEffect(() => {
    fetchUserProgress();
  }, [user, token]);

  // Reset all data when category changes
  useEffect(() => {
    setCurrentWords([]);
    setFilteredWords([]);
    setPage(1);
    setHasMore(true);
    setSearchTerm("");
    setDifficultyFilter("all");
  }, [selectedCategory]);

  // Fetch videos when category or page changes
  useEffect(() => {
    const fetchVideos = async () => {
      if (loading || (!hasMore && page > 1)) return;

      setLoading(true);
      try {
        const res = await axios.get<VideosResponse>(
          `${baseUrl}/learn/videos?category=${selectedCategory}&page=${page}&limit=20`
        );
        const apiVideos = res.data?.videos || [];

        const newVideos: Video[] = apiVideos.map((v: VideoFromApi) => ({
          id: v.id,
          title: v.title || v.word || "Untitled",
          description: v.description || "No description available",
          difficulty: v.difficulty || "Beginner",
          duration: v.duration || "30s",
          videoUrl: `${baseUrl}/learn/stream/${v.video_file}`,
          thumbnailUrl: v.thumbnail ? `/thumbnails/${encodeURIComponent(v.thumbnail)}` : undefined,
          thumbnail: v.thumbnail || "📹",
          category: v.category || "",
          word: v.word || "",
        }));

        console.log("Fetched videos:", newVideos);


        if (page === 1) {
          // First page - replace all data
          setCurrentWords(newVideos);
          setFilteredWords(newVideos);
        } else {
          // Subsequent pages - append new videos without duplicates
          setCurrentWords((prev) => {
            const existingIds = new Set(prev.map((v) => v.id));
            const uniqueNew = newVideos.filter((v) => !existingIds.has(v.id));
            return [...prev, ...uniqueNew];
          });
          setFilteredWords((prev) => {
            const existingIds = new Set(prev.map((v) => v.id));
            const uniqueNew = newVideos.filter((v) => !existingIds.has(v.id));
            return [...prev, ...uniqueNew];
          });
        }

        // Check if there are more pages
        setHasMore(newVideos.length === 20);
      } catch (error) {
        console.error("Error fetching videos:", error);
        setHasMore(false);
      } finally {
        setLoading(false);
      }
    };
    fetchVideos();
  }, [selectedCategory, page]);

  // Handle infinite scroll for the video container
  useEffect(() => {
    const handleScroll = (event: Event) => {
      const container = event.target as HTMLElement;
      if (
        container.scrollTop + container.clientHeight >=
          container.scrollHeight - 100 &&
        !loading &&
        hasMore &&
        searchTerm === "" &&
        difficultyFilter === "all"
      ) {
        setPage((prev) => prev + 1);
      }
    };

    // Find the video container and add scroll listener
    const videoContainer = document.querySelector('.video-scroll-container');
    if (videoContainer) {
      videoContainer.addEventListener("scroll", handleScroll);
      return () => videoContainer.removeEventListener("scroll", handleScroll);
    }
  }, [loading, hasMore, searchTerm, difficultyFilter]);

  // Handle search and filtering
  useEffect(() => {
    const performSearch = async () => {
      if (searchTerm || difficultyFilter !== "all") {
        setLoading(true);
        try {
          const res = await axios.get<VideosResponse>(`${baseUrl}/learn/search`, {
            params: {
              query: searchTerm || "",
              difficulty: difficultyFilter,
              page: 1,
              limit: 100,
            },
          });

          const searchResults: Video[] = (res.data?.videos || []).map(
            (v: VideoFromApi) => ({
              id: v.id,
              title: v.title || v.word || "Untitled",
              description: v.description || "No description available",
              difficulty: v.difficulty || "Beginner",
              duration: v.duration || "30s",
              videoUrl: `${baseUrl}/learn/stream/${v.video_file}`,
              thumbnailUrl: v.thumbnail ? `/thumbnails/${encodeURIComponent(v.thumbnail)}` : undefined,
              thumbnail: v.thumbnail || "📹",
              category: v.category || "",  
              word: v.word || "",
            })
          );

          setFilteredWords(searchResults);
        } catch (error) {
          console.error("Error searching videos:", error);
          setFilteredWords([]);
        } finally {
          setLoading(false);
        }
      } else {
        // No search/filter active - show current words
        setFilteredWords(currentWords);
      }
    };

    const debounceTimer = setTimeout(performSearch, 300);
    return () => clearTimeout(debounceTimer);
  }, [searchTerm, difficultyFilter, currentWords]);

  const handleCategoryChange = (categoryId: string) => {
    setSelectedCategory(categoryId);
  };

  const openVideoModal = (word: Video) => {
    setSelectedWord(word);
    setIsVideoModalOpen(true);
    document.body.style.overflow = "hidden";
  };

  const closeVideoModal = () => {
    setIsVideoModalOpen(false);
    setSelectedWord(null);
    document.body.style.overflow = "unset";
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "text-green-600 bg-green-100 border-green-200";
      case "Intermediate":
        return "text-yellow-600 bg-yellow-100 border-yellow-200";
      case "Advanced":
        return "text-red-600 bg-red-100 border-red-200";
      default:
        return "text-gray-600 bg-gray-100 border-gray-200";
    }
  };

  const getDifficultyIcon = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "🟢";
      case "Intermediate":
        return "🟡";
      case "Advanced":
        return "🔴";
      default:
        return "⚪";
    }
  };

  const floatingElements = [
    { emoji: "🤟", delay: 0, duration: 4 },
    { emoji: "👋", delay: 1, duration: 5 },
    { emoji: "✋", delay: 2, duration: 3 },
    { emoji: "👌", delay: 3, duration: 6 },
    { emoji: "🤲", delay: 4, duration: 4 },
    { emoji: "👐", delay: 5, duration: 5 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in-up">
          <div className="relative group mb-6">
            <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000 animate-pulse"></div>
            <BookOpen className="relative h-16 w-16 text-blue-600 mx-auto animate-bounce-gentle" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Master ASL Signs
          </h1>
          <p className="text-xl text-gray-600">
            Interactive video lessons for every skill level
          </p>
          {floatingElements.map((element, i) => (
            <div
              key={i}
              className="absolute text-4xl opacity-20 animate-float pointer-events-none"
              style={{
                left: `${10 + i * 15}%`,
                top: `${20 + i * 10}%`,
                animationDelay: `${element.delay}s`,
                animationDuration: `${element.duration}s`,
              }}
            >
              {element.emoji}
            </div>
          ))}
          {/* Rotating rings */}
          <div
            className="absolute inset-0 border-2 border-blue-400/30 rounded-full animate-spin"
            style={{ animationDuration: "20s" }}
          ></div>
          <div
            className="absolute inset-4 border-2 border-purple-400/30 rounded-full animate-spin"
            style={{ animationDuration: "15s", animationDirection: "reverse" }}
          ></div>
          <div
            className="absolute inset-8 border-2 border-pink-400/30 rounded-full animate-spin"
            style={{ animationDuration: "10s" }}
          ></div>
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
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Difficulty Level
                </label>
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
              <nav className="space-y-2 stagger-animation mb-6">
                {categories.map((category) => {
                  const Icon = category.icon;
                  return (
                    <button
                      key={category.id}
                      onClick={() => handleCategoryChange(category.id)}
                      className={`w-full flex items-center justify-between px-4 py-3 rounded-lg text-left transition-all duration-300 transform hover:scale-105 group card-hover ${
                        selectedCategory === category.id
                          ? `bg-gradient-to-r from-${category.color}-50 to-${category.color}-100 text-${category.color}-700 border-${category.color}-200 border shadow-md`
                          : "text-gray-700 hover:bg-gray-50 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <Icon
                          className={`h-5 w-5 ${
                            selectedCategory === category.id
                              ? `text-${category.color}-600`
                              : "text-gray-500"
                          } group-hover:scale-110 transition-transform duration-300 animate-wiggle`}
                        />
                        <div>
                          <span className="font-medium">{category.name}</span>
                          <div className="text-xs text-gray-500">
                            {category.count} signs
                          </div>
                        </div>
                      </div>
                      {selectedCategory === category.id && (
                        <div
                          className={`w-3 h-3 bg-${category.color}-600 rounded-full animate-pulse`}
                        ></div>
                      )}
                    </button>
                  );
                })}
              </nav>

              {/* Enhanced Progress Summary */}
              <div className="mt-8 p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-sm animate-scale-in animation-delay-500 border border-blue-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Trophy className="h-5 w-5 text-blue-600 mr-2" />
                  Your Progress
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
                  </>
                ) : (
                  <div className="text-center py-6 text-gray-500">
                    <div className="mb-3">
                      <BookOpen className="h-12 w-12 mx-auto text-gray-300" />
                    </div>
                    <p className="text-sm font-medium text-gray-600 mb-2">Login Required</p>
                    <p className="text-xs text-gray-500">
                      Please log in to track your learning progress
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:w-3/4">
            {/* Category Header */}
            <div className="bg-white rounded-xl shadow-sm p-6 mb-6 animate-fade-in-right">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {React.createElement(
                    categories.find((c) => c.id === selectedCategory)?.icon ||
                      BookOpen,
                    {
                      className: `h-8 w-8 text-${
                        categories.find((c) => c.id === selectedCategory)?.color
                      }-600 animate-bounce-gentle`,
                    }
                  )}
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">
                      {categories.find((c) => c.id === selectedCategory)?.name}
                    </h2>
                    <p className="text-gray-600">
                      {filteredWords.length} of {currentWords.length} signs
                      {searchTerm && ` matching "${searchTerm}"`}
                      {difficultyFilter !== "all" && ` (${difficultyFilter})`}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-500">Total Duration</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {Math.round(
                      filteredWords.reduce((acc, word) => {
                        const duration = word.duration.replace(/\D/g, "");
                        return acc + (parseInt(duration) || 0);
                      }, 0) / 60
                    )}{" "}
                    min
                  </div>
                </div>
              </div>
            </div>

            {/* Videos Container with Independent Scrolling */}
            <div className="bg-white rounded-xl shadow-sm p-6 animate-fade-in-right relative">
              <div className="h-[98vh] overflow-y-auto scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-gray-100 scroll-smooth video-scroll-container">
                {/* Videos Grid - This scrolls independently */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 stagger-animation pb-6">
                  {filteredWords.map((word, index) => (
                    <div
                      key={`video-${word.id}-${selectedCategory}-${index}`}
                      onClick={() => openVideoModal(word)}
                      className="group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 cursor-pointer word-card animate-scale-in"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      {/* Thumbnail */}
                      <div
                        className="relative bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 h-48 rounded-t-xl flex items-center justify-center overflow-hidden"
                        style={{ 
                          backgroundImage: word.thumbnailUrl ? `url(${word.thumbnailUrl})` : undefined, 
                          backgroundSize: 'cover', 
                          backgroundPosition: 'center' 
                        }}
                      >
                        <div className="absolute inset-0 bg-black/20"></div>
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
                          <Play className="h-12 w-12 text-white opacity-0 group-hover:opacity-100 transition-all duration-300 transform scale-75 group-hover:scale-100 animate-pulse" />
                        </div>

                        {/* Difficulty Badge */}
                        <div
                          className={`absolute top-3 right-3 px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(
                            word.difficulty
                          )} backdrop-blur-sm`}
                        >
                          <span className="mr-1">
                            {getDifficultyIcon(word.difficulty)}
                          </span>
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
                            <span className="text-xs text-gray-500">
                              Audio included
                            </span>
                          </div>
                          <button className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 transition-colors duration-300 group-hover:translate-x-1 transform">
                            <span className="text-sm font-medium">Watch</span>
                            <Play className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Loading State */}
                  {loading && (
                    <div className="col-span-full text-center py-8">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                      <p className="mt-4 text-gray-600">Loading more signs...</p>
                    </div>
                  )}

                  {/* No Results */}
                  {filteredWords.length === 0 && !loading && (
                    <div className="col-span-full text-center py-12 animate-fade-in-up">
                      <div className="text-6xl mb-4 animate-bounce">🔍</div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">
                        No signs found
                      </h3>
                      <p className="text-gray-600 mb-4">
                        Try adjusting your search terms or difficulty filter
                      </p>
                      <button
                        onClick={() => {
                          setSearchTerm("");
                          setDifficultyFilter("all");
                        }}
                        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-300 transform hover:scale-105"
                      >
                        Clear Filters
                      </button>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Scroll to Top Button for Video Area */}
              <button
                onClick={() => {
                  const container = document.querySelector('.video-scroll-container');
                  container?.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-110"
                title="Scroll to top"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              </button>
            </div>
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
                  <h2 className="text-2xl font-bold text-gray-900">
                    {selectedWord.title}
                  </h2>
                  <div className="flex items-center space-x-3 mt-1">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(
                        selectedWord.difficulty
                      )}`}
                    >
                      {getDifficultyIcon(selectedWord.difficulty)}{" "}
                      {selectedWord.difficulty}
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

            {/* Enhanced Video Player */}
            <div className="relative bg-gray-900 rounded-lg overflow-hidden">
              <VideoPlayer
                src={selectedWord.videoUrl}
                thumbnail={selectedWord.thumbnailUrl}
                title={selectedWord.title}
                className="w-full"
              />
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-6">
              {/* How to Sign */}
              <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
                  <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
                    <BookOpen className="h-5 w-5 mr-2" />
                    How to Sign
                  </h3>
                  <p className="text-blue-800 leading-relaxed">
                    {selectedWord.description}
                  </p>
                </div>

                {/* Usage Tips */}
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
                  <h3 className="font-semibold text-purple-900 mb-3 flex items-center">
                    <MessageSquare className="h-5 w-5 mr-2" />
                    Usage Tips
                  </h3>
                  <ul className="text-purple-800 space-y-2 text-sm">
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Practice slowly at first to build muscle memory.
                    </li>
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Maintain clear hand positioning.
                    </li>
                    <li className="flex items-start">
                      <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      Use appropriate facial expressions.
                    </li>
                  </ul>
                </div>
              </section>

              {/* Action Buttons */}
              <div className="flex flex-col md:flex-row md:justify-between gap-4">
                <button className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-300 hover:scale-105">
                  <Star className="h-4 w-4" />
                  Add to Favorites
                </button>

                <div className="flex gap-3">
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
