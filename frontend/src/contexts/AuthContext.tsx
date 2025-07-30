import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { translationSessionManager } from '../services/translationSessionManager';
import { authAPI } from '../services/api';
import {BASE_URL} from '../services/api';

export interface User {
  id: number;
  username: string;
  email: string;
  role: 'admin' | 'user';
  fullName?: string;
  createdAt: string;
}

export interface AuthContextType {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  login: (email: string, password: string) => Promise<boolean>;
  register: (userData: RegisterData) => Promise<boolean>;
  logout: () => void;
  refreshAccessToken: () => Promise<boolean>;
  makeAuthenticatedRequest: (url: string, options?: RequestInit) => Promise<Response>;
  loading: boolean;
  isAdmin: boolean;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  fullName?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load auth data from localStorage on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('auth_token');
    const savedRefreshToken = localStorage.getItem('refresh_token');
    const savedUser = localStorage.getItem('auth_user');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setRefreshToken(savedRefreshToken);
      setUser(JSON.parse(savedUser));
    }
    setLoading(false);
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setLoading(true);
      const response = await authAPI.login({ username: email, password });
      const data = response.data as {
        access_token: string;
        refresh_token: string;
        user: User;
      };
      setToken(data.access_token);
      setRefreshToken(data.refresh_token);
      setUser(data.user);
      // Save to localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token || '');
      localStorage.setItem('auth_user', JSON.stringify(data.user));
      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const register = async (userData: RegisterData): Promise<boolean> => {
    try {
      setLoading(true);
      // Map frontend field names to backend field names
      const backendData = {
        username: userData.username,
        email: userData.email,
        password: userData.password,
        full_name: userData.fullName || '', // Map fullName to full_name
        role: 'user'
      };
      const response = await authAPI.register(backendData);
      const data = response.data as {
        access_token: string;
        refresh_token: string;
        user: User;
      };
      setToken(data.access_token);
      setRefreshToken(data.refresh_token);
      setUser(data.user);
      // Save to localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token || '');
      localStorage.setItem('auth_user', JSON.stringify(data.user));
      return true;
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Refresh access token using refresh token
  const refreshAccessToken = async (): Promise<boolean> => {
    if (!refreshToken) {
      console.error('No refresh token available');
      return false;
    }

    try {
      const response = await fetch(`${BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: refreshToken,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setToken(data.access_token);
        
        // Update refresh token if provided
        if (data.refresh_token) {
          setRefreshToken(data.refresh_token);
          localStorage.setItem('refresh_token', data.refresh_token);
        }
        
        // Save new access token
        localStorage.setItem('auth_token', data.access_token);
        
        return true;
      } else {
        // Refresh token is invalid, logout user
        logout();
        return false;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      logout();
      return false;
    }
  };

  // Make authenticated request with automatic token refresh
  const makeAuthenticatedRequest = async (url: string, options: RequestInit = {}): Promise<Response> => {
    // Add authorization header
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
    };

    let response = await fetch(url, {
      ...options,
      headers,
    });

    // If unauthorized, try to refresh token and retry
    if (response.status === 401 && refreshToken) {
      const refreshSuccess = await refreshAccessToken();
      
      if (refreshSuccess) {
        // Retry request with new token
        const newHeaders = {
          ...options.headers,
          'Authorization': `Bearer ${token}`,
        };
        
        response = await fetch(url, {
          ...options,
          headers: newHeaders,
        });
      }
    }

    return response;
  };

  const logout = () => {
    // End active sessions before clearing user data
    if (translationSessionManager.isSessionActive()) {
      translationSessionManager.endSession('logout');
    }
    // No practice session cleanup needed - simple timer will reset on page reload
    
    setUser(null);
    setToken(null);
    setRefreshToken(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('auth_user');
  };

  const isAdmin = user?.role === 'admin';

  const value: AuthContextType = {
    user,
    token,
    refreshToken,
    login,
    register,
    logout,
    refreshAccessToken,
    makeAuthenticatedRequest,
    loading,
    isAdmin,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
