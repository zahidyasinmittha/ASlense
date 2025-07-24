import { useRef, useEffect, useCallback, useState } from 'react';

interface PSLPrediction {
  letter: string;
  confidence: number;
  timestamp: number;
}

interface PSLWebSocketMessage {
  type: 'connected' | 'prediction' | 'error' | 'model_switched' | 'pong';
  message?: string;
  model_type?: string;
  model_info?: any;
  frame_count?: number;
  letter?: string;
  confidence?: number;
  predictions?: Array<{ class: string; confidence: number }>;
  processing_time?: string;
  model_used?: string;
  timestamp?: number;
}

interface UsePSLWebSocketProps {
  onPrediction: (prediction: PSLPrediction) => void;
  onError: (error: string) => void;
  onConnected?: (modelInfo: any) => void;
  modelType?: 'ps_mini' | 'ps_pro';
}

export const usePSLWebSocket = ({
  onPrediction,
  onError,
  onConnected,
  modelType = 'ps_mini'
}: UsePSLWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || isConnecting) {
      return;
    }

    setIsConnecting(true);
    
    try {
      // Connect to the PSL WebSocket endpoint
      const wsUrl = `ws://localhost:8000/api/v1/practice/psl-predict?model_type=${modelType}`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        
        // Start ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };

      ws.onmessage = (event) => {
        try {
          const data: PSLWebSocketMessage = JSON.parse(event.data);
          
          switch (data.type) {
            case 'connected':
              if (onConnected && data.model_info) {
                onConnected(data.model_info);
              }
              break;
              
            case 'prediction':
              if (data.letter && data.confidence !== undefined && data.timestamp) {
                onPrediction({
                  letter: data.letter,
                  confidence: data.confidence,
                  timestamp: data.timestamp
                });
              }
              break;
              
            case 'error':
              console.error('PSL WebSocket error:', data.message);
              onError(data.message || 'Unknown WebSocket error');
              break;
              
            case 'model_switched':
              if (onConnected && data.model_info) {
                onConnected(data.model_info);
              }
              break;
              
            case 'pong':
              // Keep-alive response
              break;
              
            default:
              console.warn('Unknown WebSocket message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          onError('Failed to parse server response');
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setIsConnecting(false);
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        // Attempt to reconnect after a delay (unless manually closed)
        if (event.code !== 1000) { // 1000 is normal closure
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('PSL WebSocket error:', error);
        setIsConnecting(false);
        onError('WebSocket connection failed');
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setIsConnecting(false);
      onError('Failed to establish WebSocket connection');
    }
  }, [modelType, onPrediction, onError, onConnected, isConnecting]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const sendFrame = useCallback((frameData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        frame: frameData
      }));
      return true;
    } else {
      return false;
    }
  }, []);

  const switchModel = useCallback((newModelType: 'ps_mini' | 'ps_pro') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'switch_model',
        model: newModelType
      }));
      return true;
    }
    return false;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    sendFrame,
    switchModel,
    isConnected,
    isConnecting
  };
};
