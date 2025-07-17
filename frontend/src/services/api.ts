import axios from 'axios';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

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
          const response = await axios.post('http://localhost:8000/auth/refresh', {
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
    api.post('/auth/login', credentials),
  
  register: (userData: { username: string; email: string; password: string; full_name?: string }) =>
    api.post('/auth/register', userData),
  
  refresh: (refreshToken: string) =>
    api.post('/auth/refresh', { refresh_token: refreshToken }),
  
  me: () => api.get('/auth/me'),
};

export const adminAPI = {
  getStats: () => api.get('/admin-api/stats'),
  getUsers: (skip = 0, limit = 100) => api.get(`/admin-api/users?skip=${skip}&limit=${limit}`),
  getVideos: () => api.get('/admin-api/videos'),
  getMetrics: () => api.get('/admin-api/metrics'),
  
  createUser: (userData: any) => api.post('/admin-api/users', userData),
  updateUser: (userId: number, userData: any) => api.put(`/admin-api/users/${userId}`, userData),
  deleteUser: (userId: number) => api.delete(`/admin-api/users/${userId}`),
  toggleUserStatus: (userId: number) => api.patch(`/admin-api/users/${userId}/toggle-status`),
};

export const userAPI = {
  getDashboard: () => api.get('/user/dashboard'),
  getProgress: () => api.get('/user/progress'),
  getAchievements: () => api.get('/user/achievements'),
  updateProfile: (userData: any) => api.put('/user/profile', userData),
};

export const videosAPI = {
  getVideos: (params: any) => api.get('/videos', { params }),
  getVideoById: (id: number) => api.get(`/videos/${id}`),
  getVideoCount: () => api.get('/videos/count'),
  searchVideos: (params: any) => api.get('/videos/search', { params }),
};

export const practiceAPI = {
  getModelsStatus: () => api.get('/practice/models/status'),
  getAvailableWords: () => api.get('/practice/available-words'),
  
  predictVideo: (formData: FormData) => 
    api.post('/practice/predict-video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }),
  
  predictFrames: (data: any) => api.post('/practice/predict-frames', data),
  
  saveUserPrediction: (data: any) => api.post('/practice/predict-with-user', data),
  
  ping: () => api.get('/practice/ping'),
};

export const learnAPI = {
  getVideos: (params: any) => api.get('/learn/videos', { params }),
  searchVideos: (params: any) => api.get('/learn/search', { params }),
  getCategories: () => api.get('/learn/categories'),
};

export default api;
