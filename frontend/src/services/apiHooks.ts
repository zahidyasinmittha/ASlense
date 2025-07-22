/**
 * Custom hooks for API calls
 * Provides convenient hooks for common API operations
 */

import { useState, useCallback } from 'react';
import { API_CONFIG, translateAPI, practiceAPI, userAPI, ApiUtils } from './api';

/**
 * Hook for translation API calls
 */
export const useTranslationAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictVideo = useCallback(async (formData: FormData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await translateAPI.predictVideo(formData);
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const startSession = useCallback(async (sessionData: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await translateAPI.startSession(sessionData);
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const endSession = useCallback(async (sessionId: string, sessionData: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await translateAPI.endSession(sessionId, sessionData);
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    predictVideo,
    startSession,
    endSession,
    loading,
    error,
    clearError: () => setError(null),
  };
};

/**
 * Hook for practice API calls
 */
export const usePracticeAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictVideo = useCallback(async (formData: FormData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await practiceAPI.predictVideo(formData);
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const getModelsStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await practiceAPI.getModelsStatus();
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    predictVideo,
    getModelsStatus,
    loading,
    error,
    clearError: () => setError(null),
  };
};

/**
 * Hook for user API calls
 */
export const useUserAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getProgress = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await userAPI.getProgress();
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const updateProfile = useCallback(async (userData: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await userAPI.updateProfile(userData);
      return response.data;
    } catch (err: any) {
      const errorMessage = ApiUtils.handleApiError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    getProgress,
    updateProfile,
    loading,
    error,
    clearError: () => setError(null),
  };
};

/**
 * Hook for WebSocket connections
 */
export const useWebSocket = () => {
  const [connection, setConnection] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');

  const connect = useCallback((url: string, onMessage?: (event: MessageEvent) => void) => {
    if (connection) {
      connection.close();
    }

    setStatus('connecting');
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus('connected');
      setConnection(ws);
    };

    ws.onclose = () => {
      setStatus('disconnected');
      setConnection(null);
    };

    ws.onerror = () => {
      setStatus('disconnected');
      setConnection(null);
    };

    if (onMessage) {
      ws.onmessage = onMessage;
    }

    return ws;
  }, [connection]);

  const disconnect = useCallback(() => {
    if (connection) {
      connection.close();
      setConnection(null);
      setStatus('disconnected');
    }
  }, [connection]);

  const send = useCallback((data: any) => {
    if (connection && connection.readyState === WebSocket.OPEN) {
      connection.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, [connection]);

  return {
    connection,
    status,
    connect,
    disconnect,
    send,
  };
};

/**
 * Hook for getting API configuration
 */
export const useAPIConfig = () => {
  return {
    baseUrl: API_CONFIG.BASE_URL,
    wsUrl: API_CONFIG.WS_URL,
    endpoints: API_CONFIG.ENDPOINTS,
    getTranslateWebSocketUrl: (modelType: string, predictionMode: string) =>
      API_CONFIG.ENDPOINTS.TRANSLATE.LIVE_TRANSLATE_WS(modelType, predictionMode),
    getPracticeWebSocketUrl: (modelType: string, category: string) =>
      API_CONFIG.ENDPOINTS.PRACTICE.LIVE_PRACTICE_WS(modelType, category),
  };
};
