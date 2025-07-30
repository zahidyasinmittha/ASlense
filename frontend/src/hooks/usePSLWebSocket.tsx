// Example integration for PSLPractice.tsx
// Add this to your existing PSLPractice component

import { useEffect, useRef, useState } from 'react';
import {BASE_URL} from '../services/api';

// Add this interface for the WebSocket service
interface PSLPrediction {
  class: string;
  confidence: number;
}

interface PSLPredictionResult {
  predictions: PSLPrediction[];
  confidence: number;
  processing_time: string;
  error?: string;
}

// Add this custom hook for PSL WebSocket
const usePSLWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [predictions, setPredictions] = useState<PSLPrediction[]>([]);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = () => {
    try {
      wsRef.current = new WebSocket(`${BASE_URL}/api/v1/ws/psl-recognition`);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('PSL WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'prediction':
            setPredictions(data.predictions || []);
            break;
          case 'model_info':
            setModelInfo(data.data);
            break;
          case 'error':
            console.error('PSL WebSocket error:', data.message);
            break;
        }
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('PSL WebSocket disconnected');
      };

      wsRef.current.onerror = (error) => {
        console.error('PSL WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect to PSL WebSocket:', error);
    }
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  };

  const predict = (imageDataUrl: string) => {
    if (!isConnected || !wsRef.current) {
      console.error('WebSocket not connected');
      return;
    }

    try {
      const imageData = imageDataUrl.includes(',') 
        ? imageDataUrl.split(',')[1] 
        : imageDataUrl;

      const message = {
        type: 'predict',
        image: imageData,
        top_k: 3
      };

      wsRef.current.send(JSON.stringify(message));
    } catch (error) {
      console.error('Error sending prediction request:', error);
    }
  };

  useEffect(() => {
    connect();
    return () => disconnect();
  }, []);

  return {
    isConnected,
    predictions,
    modelInfo,
    predict,
    connect,
    disconnect
  };
};

// Example usage in your PSLPractice component:
/*

// Add this to your PSLPractice component
const { isConnected, predictions, modelInfo, predict } = usePSLWebSocket();

// Modify your existing camera capture function
const captureAndPredict = () => {
  if (cameraRef.current && canvasRef.current) {
    const canvas = canvasRef.current;
    const video = cameraRef.current;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get image data as base64
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      
      // Send to PSL model for prediction
      predict(imageDataUrl);
    }
  }
};

// Add this to your JSX to show real-time predictions
<div className="mt-4 bg-gray-50 rounded-lg p-4">
  <h3 className="font-semibold text-gray-900 mb-2">Real-time Predictions</h3>
  {!isConnected ? (
    <p className="text-red-600">Connecting to PSL model...</p>
  ) : (
    <div className="space-y-2">
      {predictions.map((pred, index) => (
        <div key={index} className="flex justify-between items-center">
          <span className="font-medium">{pred.class}</span>
          <span className="text-sm text-gray-600">
            {(pred.confidence * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  )}
</div>

// Add this to show model status
{modelInfo && (
  <div className="mt-2 text-xs text-gray-500">
    Model: {modelInfo.model_name} | 
    Status: {modelInfo.status} | 
    Accuracy: {modelInfo.accuracy}
  </div>
)}

*/

export { usePSLWebSocket };
