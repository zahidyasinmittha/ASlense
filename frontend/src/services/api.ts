import axios from 'axios';

// Get base URL from environment variables
const BASE_URL = import.meta.env.VITE_BACKEND_BASEURL || 'http://localhost:8000';
const WS_URL = BASE_URL.replace('http', 'ws');

// Create axios instance with default configuration
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000, // 60 seconds timeout (increased from 30s for video processing)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Export API configuration for other files to use
export const API_CONFIG = {
  BASE_URL,
  WS_URL,
  ENDPOINTS: {
    // Authentication endpoints
    AUTH: {
      LOGIN: '/auth/login',
      REGISTER: '/auth/register',
      REFRESH: '/auth/refresh',
      LOGOUT: '/auth/logout',
      ME: '/auth/me',
    },
    
    // User endpoints
    USER: {
      DASHBOARD: '/user/dashboard',
      PROFILE: '/user/profile',
      PROGRESS: '/user/progress',
      ACHIEVEMENTS: '/user/achievements',
      SETTINGS: '/user/settings',
    },
    
    // Translation endpoints
    TRANSLATE: {
      START_SESSION: '/translate/start-session',
      END_SESSION: (sessionId: string) => `/translate/end-session/${sessionId}`,
      TRANSLATE_TEXT: '/translate/translate',
      VIDEO_PREDICT: '/translate/video-predict',
      LLM_TEST: '/translate/llm-test',
      LIVE_TRANSLATE_WS: (modelType: string, predictionMode: string) => 
        `${WS_URL}/translate/live-translate?model_type=${modelType}&prediction_mode=${predictionMode}`,
    },
    
    // Practice endpoints
    PRACTICE: {
      START_SESSION: '/practice/start-session',
      END_SESSION: (sessionId: string) => `/practice/end-session/${sessionId}`,
      MODELS_STATUS: '/practice/models/status',
      AVAILABLE_WORDS: '/practice/available-words',
      PREDICT_VIDEO: '/practice/predict-video',
      PREDICT_FRAMES: '/practice/predict-frames',
      SAVE_USER_PREDICTION: '/practice/predict-with-user',
      PING: '/practice/ping',
      LIVE_PRACTICE_WS: (modelType: string, category: string) => 
        `${WS_URL}/practice/live-practice?model_type=${modelType}&category=${category}`,
    },
    
    // Learn endpoints
    LEARN: {
      VIDEOS: '/learn/videos',
      SEARCH: '/learn/search',
      CATEGORIES: '/learn/categories',
    },
    
    // Videos endpoints
    VIDEOS: {
      LIST: '/videos',
      BY_ID: (id: number) => `/videos/${id}`,
      COUNT: '/videos/count',
      SEARCH: '/videos/search',
    },
    
    // Admin endpoints
    ADMIN: {
      STATS: '/admin-api/stats',
      USERS: '/admin-api/users',
      VIDEOS: '/admin-api/videos',
      METRICS: '/admin-api/metrics',
      USER_BY_ID: (userId: number) => `/admin-api/users/${userId}`,
      TOGGLE_USER_STATUS: (userId: number) => `/admin-api/users/${userId}/toggle-status`,
    },
  }
};

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token'); // Changed from 'access_token' to 'auth_token'
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle common errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    // If the error is 401 and we haven't already tried to refresh
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          const response = await axios.post(`${BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token } = response.data as { access_token: string };
          localStorage.setItem('auth_token', access_token); // Changed from 'access_token' to 'auth_token'

          // Retry the original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, redirect to login
        localStorage.removeItem('auth_token'); // Changed from 'access_token' to 'auth_token'
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// API endpoints
export const authAPI = {
  login: (credentials: { username: string; password: string }) =>
    api.post(API_CONFIG.ENDPOINTS.AUTH.LOGIN, credentials),
  
  register: (userData: { username: string; email: string; password: string; full_name?: string }) =>
    api.post(API_CONFIG.ENDPOINTS.AUTH.REGISTER, userData),
  
  refresh: (refreshToken: string) =>
    api.post(API_CONFIG.ENDPOINTS.AUTH.REFRESH, { refresh_token: refreshToken }),
  
  me: () => api.get(API_CONFIG.ENDPOINTS.AUTH.ME),
};

export const adminAPI = {
  getStats: () => api.get(API_CONFIG.ENDPOINTS.ADMIN.STATS),
  getUsers: (skip = 0, limit = 100) => api.get(`${API_CONFIG.ENDPOINTS.ADMIN.USERS}?skip=${skip}&limit=${limit}`),
  getVideos: () => api.get(API_CONFIG.ENDPOINTS.ADMIN.VIDEOS),
  getMetrics: () => api.get(API_CONFIG.ENDPOINTS.ADMIN.METRICS),
  
  createUser: (userData: any) => api.post(API_CONFIG.ENDPOINTS.ADMIN.USERS, userData),
  updateUser: (userId: number, userData: any) => api.put(API_CONFIG.ENDPOINTS.ADMIN.USER_BY_ID(userId), userData),
  deleteUser: (userId: number) => api.delete(API_CONFIG.ENDPOINTS.ADMIN.USER_BY_ID(userId)),
  toggleUserStatus: (userId: number) => api.patch(API_CONFIG.ENDPOINTS.ADMIN.TOGGLE_USER_STATUS(userId)),
};

export const userAPI = {
  getDashboard: () => api.get(API_CONFIG.ENDPOINTS.USER.DASHBOARD),
  getProgress: () => api.get(API_CONFIG.ENDPOINTS.USER.PROGRESS),
  getAchievements: () => api.get(API_CONFIG.ENDPOINTS.USER.ACHIEVEMENTS),
  updateProfile: (userData: any) => api.put(API_CONFIG.ENDPOINTS.USER.PROFILE, userData),
  updateSettings: (settings: any) => api.put(API_CONFIG.ENDPOINTS.USER.SETTINGS, settings),
};

export const videosAPI = {
  getVideos: (params: any) => api.get(API_CONFIG.ENDPOINTS.VIDEOS.LIST, { params }),
  getVideoById: (id: number) => api.get(API_CONFIG.ENDPOINTS.VIDEOS.BY_ID(id)),
  getVideoCount: () => api.get(API_CONFIG.ENDPOINTS.VIDEOS.COUNT),
  searchVideos: (params: any) => api.get(API_CONFIG.ENDPOINTS.VIDEOS.SEARCH, { params }),
};

export const practiceAPI = {
  getModelsStatus: () => api.get(API_CONFIG.ENDPOINTS.PRACTICE.MODELS_STATUS),
  getAvailableWords: () => api.get(API_CONFIG.ENDPOINTS.PRACTICE.AVAILABLE_WORDS),
  
  predictVideo: (formData: FormData) => 
    api.post(API_CONFIG.ENDPOINTS.PRACTICE.PREDICT_VIDEO, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000 // 5 minutes for video processing with LLM
    }),
  
  predictFrames: (data: any) => api.post(API_CONFIG.ENDPOINTS.PRACTICE.PREDICT_FRAMES, data),
  
  saveUserPrediction: (data: any) => api.post(API_CONFIG.ENDPOINTS.PRACTICE.SAVE_USER_PREDICTION, data),
  
  ping: () => api.get(API_CONFIG.ENDPOINTS.PRACTICE.PING),
  
  startSession: (sessionData: any) => api.post(API_CONFIG.ENDPOINTS.PRACTICE.START_SESSION, sessionData),
  
  endSession: (sessionId: string, sessionData: any) => 
    api.post(API_CONFIG.ENDPOINTS.PRACTICE.END_SESSION(sessionId), sessionData),
};

export const translateAPI = {
  startSession: (sessionData: any) => api.post(API_CONFIG.ENDPOINTS.TRANSLATE.START_SESSION, sessionData),
  
  endSession: (sessionId: string, sessionData: any) => 
    api.post(API_CONFIG.ENDPOINTS.TRANSLATE.END_SESSION(sessionId), sessionData),
  
  translateText: (translationData: any) => api.post(API_CONFIG.ENDPOINTS.TRANSLATE.TRANSLATE_TEXT, translationData),
  
  predictVideo: (formData: FormData) => 
    api.post(API_CONFIG.ENDPOINTS.TRANSLATE.VIDEO_PREDICT, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000 // 5 minutes for video processing with LLM
    }),
  
  testLLM: () => api.post(API_CONFIG.ENDPOINTS.TRANSLATE.LLM_TEST),
};

export const learnAPI = {
  getVideos: (params: any) => api.get(API_CONFIG.ENDPOINTS.LEARN.VIDEOS, { params }),
  searchVideos: (params: any) => api.get(API_CONFIG.ENDPOINTS.LEARN.SEARCH, { params }),
  getCategories: () => api.get(API_CONFIG.ENDPOINTS.LEARN.CATEGORIES),
};

// WebSocket utilities
export const WebSocketAPI = {
  getLiveTranslateUrl: (modelType: string, predictionMode: string) =>
    API_CONFIG.ENDPOINTS.TRANSLATE.LIVE_TRANSLATE_WS(modelType, predictionMode),
  
  getLivePracticeUrl: (modelType: string, category: string) =>
    API_CONFIG.ENDPOINTS.PRACTICE.LIVE_PRACTICE_WS(modelType, category),
  
  createWebSocket: (url: string): WebSocket => new WebSocket(url),
};

// Utility functions
export const ApiUtils = {
  getFullUrl: (endpoint: string): string => `${BASE_URL}${endpoint}`,
  getWebSocketUrl: (endpoint: string): string => `${WS_URL}${endpoint}`,
  handleApiError: (error: any): string => {
    if (error.response?.data?.detail) {
      return error.response.data.detail;
    }
    if (error.message) {
      return error.message;
    }
    return 'An unexpected error occurred';
  },
};

export default api;
