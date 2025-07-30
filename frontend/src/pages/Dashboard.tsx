import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link } from 'react-router-dom';
import { userAPI } from '../services/api';

// ==================== INTERFACES ====================

interface UserProgress {
  id?: number;
  user_id?: number;
  total_signs_practiced: number;
  correct_predictions: number;
  total_predictions: number;
  accuracy_rate: number;
  practice_streak: number;
  longest_streak: number;
  total_practice_time: number;
  level: string;
  experience_points: number;
  last_practice_date?: string;
}

interface Achievement {
  id: number;
  title: string;
  description: string;
  iconType: string;
  unlockedAt: string;
}

interface RecentPrediction {
  id: number;
  user_id: number;
  target_word: string;
  predicted_words: string; // JSON string
  is_correct: boolean;
  confidence_score?: number;
  model_used: string;
  practice_mode: string;
  session_id?: string;
  timestamp: string;
}

const Dashboard: React.FC = () => {
  const { user, token, logout } = useAuth();
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([])
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchDashboardData();
    }
  }, [token]);

  const fetchDashboardData = async () => {
    try {
      // Fetch dashboard data (includes progress)
      const dashboardResponse = await userAPI.getDashboard();
      const dashboardData = dashboardResponse.data as any;
      setProgress(dashboardData.progress);
      setRecentPredictions(dashboardData.recent_predictions || []);

      // Fetch achievements (if endpoint exists)
      try {
        const achievementsResponse = await userAPI.getAchievements();
        const achievementsData = achievementsResponse.data;
        setAchievements(achievementsData as any);
      } catch (error) {
        // Achievements endpoint might not exist, ignore error
        console.log('Achievements endpoint not available');
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getProgressPercentage = () => {
    if (!progress) return 0;
    // Calculate XP progress using experience_points
    const currentLevel = parseInt(progress.level.replace(/\D/g, '')) || 1;
    const xpForNextLevel = currentLevel * 100; // Simple calculation
    return progress.experience_points > 0 ? (progress.experience_points / xpForNextLevel) * 100 : 0;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-gray-200 mx-auto"></div>
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-600 border-t-transparent absolute top-0 left-1/2 transform -translate-x-1/2"></div>
          </div>
          <div className="mt-8">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
              Loading Dashboard
            </h2>
            <p className="text-gray-600 animate-pulse">Preparing your learning insights...</p>
          </div>
          {/* Floating elements during loading */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-blue-400 rounded-full animate-ping delay-300"></div>
            <div className="absolute top-1/3 right-1/4 w-3 h-3 bg-purple-400 rounded-full animate-ping delay-700"></div>
            <div className="absolute bottom-1/4 left-1/3 w-2 h-2 bg-pink-400 rounded-full animate-ping delay-1000"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-r from-pink-400/20 to-indigo-400/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 w-60 h-60 bg-gradient-to-r from-purple-400/10 to-blue-400/10 rounded-full blur-2xl animate-bounce delay-500"></div>
      </div>

      {/* Header */}
      <header className="relative bg-white/80 backdrop-blur-lg shadow-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <div className="relative group">
                <h1 className="relative text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent px-4 py-2">
                  ASLense
                </h1>
              </div>
              {user?.role === 'admin' && (
                <span className="ml-4 px-3 py-1 bg-gradient-to-r from-red-500 to-pink-500 text-white text-sm font-medium rounded-full shadow-lg animate-pulse">
                  ⚡ Admin
                </span>
              )}
            </div>
            <div className="flex items-center space-x-6">
              <div className="text-gray-700 font-medium">
                <span className="text-gray-500">Welcome back,</span>
                <br />
                <span className="text-lg text-gray-800">{user?.fullName || user?.username}!</span>
              </div>
              <button
                onClick={logout}
                className="group hidden md:block relative bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <span className="relative z-10">Sign Out</span>
                <div className="absolute inset-0 bg-white/20 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Welcome Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Your ASL Learning Journey
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Continue your sign language learning adventure with personalized practice and insights
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          <Link
            to="/practice"
            className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-500 transform hover:scale-105 border border-white/50 hover:border-blue-200/50 overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 group-hover:from-blue-500/10 group-hover:to-purple-500/10 transition-all duration-500"></div>
            <div className="relative flex items-center">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl p-4 shadow-lg group-hover:shadow-xl transition-all duration-300 group-hover:scale-110">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors duration-300">Practice</h3>
                <p className="text-gray-600 mt-1">Test your ASL skills with AI</p>
                <div className="mt-3 text-sm text-blue-600 font-medium group-hover:text-blue-700">
                  Start practicing →
                </div>
              </div>
            </div>
            <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-xl transform translate-x-8 -translate-y-8 group-hover:scale-150 transition-transform duration-500"></div>
          </Link>

          <Link
            to="/learn"
            className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-500 transform hover:scale-105 border border-white/50 hover:border-green-200/50 overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-emerald-500/5 group-hover:from-green-500/10 group-hover:to-emerald-500/10 transition-all duration-500"></div>
            <div className="relative flex items-center">
              <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-4 shadow-lg group-hover:shadow-xl transition-all duration-300 group-hover:scale-110">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C20.832 18.477 19.246 18 17.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-bold text-gray-900 group-hover:text-green-600 transition-colors duration-300">Learn</h3>
                <p className="text-gray-600 mt-1">Browse video lessons</p>
                <div className="mt-3 text-sm text-green-600 font-medium group-hover:text-green-700">
                  Explore lessons →
                </div>
              </div>
            </div>
            <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-green-400/20 to-emerald-400/20 rounded-full blur-xl transform translate-x-8 -translate-y-8 group-hover:scale-150 transition-transform duration-500"></div>
          </Link>

          {user?.role === 'admin' && (
            <Link
              to="/admin"
              className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-500 transform hover:scale-105 border border-white/50 hover:border-red-200/50 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-pink-500/5 group-hover:from-red-500/10 group-hover:to-pink-500/10 transition-all duration-500"></div>
              <div className="relative flex items-center">
                <div className="bg-gradient-to-br from-red-500 to-pink-600 rounded-2xl p-4 shadow-lg group-hover:shadow-xl transition-all duration-300 group-hover:scale-110">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div className="ml-6">
                  <h3 className="text-xl font-bold text-gray-900 group-hover:text-red-600 transition-colors duration-300">Admin Panel</h3>
                  <p className="text-gray-600 mt-1">Manage users & system</p>
                  <div className="mt-3 text-sm text-red-600 font-medium group-hover:text-red-700">
                    Open admin →
                  </div>
                </div>
              </div>
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-red-400/20 to-pink-400/20 rounded-full blur-xl transform translate-x-8 -translate-y-8 group-hover:scale-150 transition-transform duration-500"></div>
            </Link>
          )}
        </div>

        {/* Progress Overview */}
        {progress && (
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-gray-900 mb-8 text-center">Your Learning Progress</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Level Progress */}
              <div className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/50 hover:shadow-2xl transition-all duration-500 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 group-hover:from-blue-500/10 group-hover:to-purple-500/10 transition-all duration-500"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-6">
                    <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl p-3 shadow-lg">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <span className="text-sm font-medium text-gray-500 bg-gray-100 px-3 py-1 rounded-full">{progress.experience_points} XP</span>
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">Level Progress</h3>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">{progress.level}</span>
                  </div>
                  <div className="relative">
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full transition-all duration-1000 shadow-inner"
                        style={{ width: `${getProgressPercentage()}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Current Level</span>
                      <span>{getProgressPercentage().toFixed(1)}%</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-4 font-medium">Keep practicing to level up! 🚀</p>
                </div>
                <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-xl group-hover:scale-125 transition-transform duration-500"></div>
              </div>

              {/* Streak Info */}
              <div className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/50 hover:shadow-2xl transition-all duration-500 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-red-500/5 group-hover:from-orange-500/10 group-hover:to-red-500/10 transition-all duration-500"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-6">
                    <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-2xl p-3 shadow-lg">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                      </svg>
                    </div>
                    <span className="text-2xl">🔥</span>
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Streaks</h3>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center p-4 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl border border-orange-100">
                      <span className="text-gray-700 font-medium">Current Streak</span>
                      <span className="text-2xl font-bold text-orange-600">{progress.practice_streak}</span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                      <span className="text-gray-700 font-medium">Longest Streak</span>
                      <span className="text-2xl font-bold text-green-600">{progress.longest_streak}</span>
                    </div>
                  </div>
                </div>
                <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-orange-400/20 to-red-400/20 rounded-full blur-xl group-hover:scale-125 transition-transform duration-500"></div>
              </div>

              {/* Accuracy Stats */}
              <div className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/50 hover:shadow-2xl transition-all duration-500 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-emerald-500/5 group-hover:from-green-500/10 group-hover:to-emerald-500/10 transition-all duration-500"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-6">
                    <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-3 shadow-lg">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div className="text-3xl font-bold text-green-600">{progress?.accuracy_rate?.toFixed(1) || '0.0'}%</div>
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Performance</h3>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
                      <span className="text-gray-700 font-medium">Total Predictions</span>
                      <span className="text-xl font-bold text-blue-600">{progress?.total_predictions || 0}</span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                      <span className="text-gray-700 font-medium">Correct</span>
                      <span className="text-xl font-bold text-green-600">{progress?.correct_predictions || 0}</span>
                    </div>
                  </div>
                </div>
                <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-green-400/20 to-emerald-400/20 rounded-full blur-xl group-hover:scale-125 transition-transform duration-500"></div>
              </div>
            </div>
          </div>
        )}

        {/* Recent Activity & Achievements */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Predictions */}
          <div className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/50 hover:shadow-2xl transition-all duration-500 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 group-hover:from-indigo-500/8 group-hover:to-purple-500/8 transition-all duration-500"></div>
            <div className="relative">
              <div className="flex items-center mb-6">
                <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl p-3 shadow-lg mr-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900">Recent Practice</h3>
              </div>
              <div className="space-y-4 max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
                {recentPredictions.length > 0 ? (
                  recentPredictions.slice(0, 5).map((prediction) => {
                    // Parse predicted_words JSON string to get the main prediction
                    let predictedWord = 'Unknown';
                    let confidence = 0;
                    try {
                      const predictions = JSON.parse(prediction.predicted_words);
                      if (predictions.length > 0) {
                        predictedWord = predictions[0].word;
                        confidence = predictions[0].confidence;
                      }
                    } catch (e) {
                      predictedWord = prediction.predicted_words;
                      confidence = prediction.confidence_score || 0;
                    }

                    return (
                      <div key={prediction.id} className="group/item relative bg-gradient-to-r from-gray-50 to-white rounded-xl p-4 border border-gray-200/50 hover:border-gray-300/50 hover:shadow-md transition-all duration-300">
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className="font-bold text-gray-900 text-lg">{prediction.target_word}</span>
                              <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                                prediction.is_correct 
                                  ? 'bg-green-100 text-green-800 border border-green-200' 
                                  : 'bg-red-100 text-red-800 border border-red-200'
                              }`}>
                                {prediction.is_correct ? '✓ Correct' : '✗ Incorrect'}
                              </div>
                            </div>
                            <p className="text-sm text-gray-600 mb-1">
                              <span className="font-medium">Predicted:</span> {predictedWord} 
                              <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                                {(confidence * 100).toFixed(1)}%
                              </span>
                            </p>
                            <div className="flex items-center space-x-3 text-xs text-gray-500">
                              <span className="bg-gray-100 px-2 py-1 rounded">{prediction.model_used}</span>
                              <span>{new Date(prediction.timestamp).toLocaleDateString()}</span>
                            </div>
                          </div>
                          <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg ${
                            prediction.is_correct 
                              ? 'bg-gradient-to-br from-green-400 to-green-600' 
                              : 'bg-gradient-to-br from-red-400 to-red-600'
                          }`}>
                            {prediction.is_correct ? '✓' : '✗'}
                          </div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center py-12">
                    <div className="bg-gradient-to-br from-gray-100 to-gray-200 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <p className="text-gray-500 font-medium">No recent practice sessions</p>
                    <p className="text-gray-400 text-sm mt-1">Start practicing to see your progress here!</p>
                  </div>
                )}
              </div>
            </div>
            <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-indigo-400/20 to-purple-400/20 rounded-full blur-xl group-hover:scale-125 transition-transform duration-500"></div>
          </div>

          {/* Achievements */}
          <div className="group relative bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/50 hover:shadow-2xl transition-all duration-500 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-yellow-500/5 to-orange-500/5 group-hover:from-yellow-500/8 group-hover:to-orange-500/8 transition-all duration-500"></div>
            <div className="relative">
              <div className="flex items-center mb-6">
                <div className="bg-gradient-to-br from-yellow-500 to-orange-600 rounded-2xl p-3 shadow-lg mr-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900">Recent Achievements</h3>
              </div>
              <div className="space-y-4 max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
                {achievements.length > 0 ? (
                  achievements.slice(0, 5).map((achievement) => (
                    <div key={achievement.id} className="flex items-center p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl border border-yellow-200/50 hover:border-yellow-300/50 hover:shadow-md transition-all duration-300">
                      <div className="bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full p-3 mr-4 shadow-lg">
                        <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="flex-1">
                        <p className="font-bold text-gray-900 mb-1">{achievement.title}</p>
                        <p className="text-sm text-gray-600">{achievement.description}</p>
                        <p className="text-xs text-yellow-600 font-medium mt-1">
                          🏆 Unlocked {new Date(achievement.unlockedAt).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12">
                    <div className="bg-gradient-to-br from-yellow-100 to-orange-200 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                      </svg>
                    </div>
                    <p className="text-gray-500 font-medium">No achievements yet</p>
                    <p className="text-gray-400 text-sm mt-1">Keep practicing to unlock achievements!</p>
                  </div>
                )}
              </div>
            </div>
            <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-yellow-400/20 to-orange-400/20 rounded-full blur-xl group-hover:scale-125 transition-transform duration-500"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
