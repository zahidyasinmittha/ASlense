// Frontend WebSocket PSL Recognition Service
// Add this to your frontend services folder

class PSLWebSocketService {
  private ws: WebSocket | null = null;
  private isConnected: boolean = false;
  private messageHandlers: Map<string, Array<(data: any) => void>> = new Map();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;

  constructor() {
    // Initialize in constructor
  }

  // Connect to WebSocket
  connect(baseUrl = 'ws://localhost:8000') {
    try {
      this.ws = new WebSocket(`${baseUrl}/api/v1/ws/psl-recognition`);
      
      this.ws.onopen = () => {
        console.log('PSL WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.handleMessage('connected', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data.type, data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('PSL WebSocket disconnected');
        this.isConnected = false;
        this.handleMessage('disconnected', { status: 'disconnected' });
        this.attemptReconnect();
      };

      this.ws.onerror = (error: Event) => {
        console.error('PSL WebSocket error:', error);
        this.handleMessage('error', { error: 'WebSocket connection error' });
      };

    } catch (error) {
      console.error('Failed to connect to PSL WebSocket:', error);
    }
  }

  // Disconnect from WebSocket
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.isConnected = false;
    }
  }

  // Attempt to reconnect
  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
    }
  }

  // Send prediction request
  predict(imageDataUrl: string, topK = 3, modelKey?: string): boolean {
    if (!this.isConnected || !this.ws) {
      console.error('WebSocket not connected');
      return false;
    }

    try {
      // Remove data URL prefix if present
      const imageData = imageDataUrl.includes(',') 
        ? imageDataUrl.split(',')[1] 
        : imageDataUrl;

      const message: any = {
        type: 'predict',
        image: imageData,
        top_k: topK
      };

      // Add model key if specified
      if (modelKey) {
        message.model_key = modelKey;
      }

      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending prediction request:', error);
      return false;
    }
  }

  // Switch model
  switchModel(modelKey: string): boolean {
    if (!this.isConnected || !this.ws) {
      console.error('WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'switch_model',
        model_key: modelKey
      };
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error switching model:', error);
      return false;
    }
  }

  // Get available models
  getAvailableModels(): boolean {
    if (!this.isConnected || !this.ws) {
      console.error('WebSocket not connected');
      return false;
    }

    try {
      const message = { type: 'get_models' };
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error requesting available models:', error);
      return false;
    }
  }

  // Get model info
  getModelInfo() {
    if (!this.isConnected || !this.ws) {
      console.error('WebSocket not connected');
      return false;
    }

    try {
      const message = { type: 'model_info' };
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error requesting model info:', error);
      return false;
    }
  }

  // Send ping to keep connection alive
  ping() {
    if (!this.isConnected || !this.ws) {
      return false;
    }

    try {
      const message = {
        type: 'ping',
        timestamp: Date.now()
      };
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending ping:', error);
      return false;
    }
  }

  // Register message handler
  onMessage(type: string, handler: (data: any) => void): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);
  }

  // Remove message handler
  offMessage(type: string, handler: (data: any) => void): void {
    if (this.messageHandlers.has(type)) {
      const handlers = this.messageHandlers.get(type)!;
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  // Handle incoming messages
  private handleMessage(type: string, data: any): void {
    if (this.messageHandlers.has(type)) {
      this.messageHandlers.get(type)!.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in message handler for ${type}:`, error);
        }
      });
    }
  }

  // Get connection status
  getStatus() {
    return {
      connected: this.isConnected,
      readyState: this.ws ? this.ws.readyState : WebSocket.CLOSED,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// Usage example:
// const pslService = new PSLWebSocketService();

// // Set up event handlers
// pslService.onMessage('prediction', (data) => {
//   console.log('Prediction result:', data.predictions);
//   console.log('Confidence:', data.confidence);
//   // Update UI with prediction results
// });

// pslService.onMessage('model_info', (data) => {
//   console.log('Model info:', data.data);
//   // Update UI with model status
// });

// pslService.onMessage('error', (data) => {
//   console.error('PSL service error:', data.message);
//   // Show error to user
// });

// // Connect to service
// pslService.connect();

// // Make prediction from canvas
// const canvas = document.getElementById('cameraCanvas');
// const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
// pslService.predict(imageDataUrl, 3);

export default PSLWebSocketService;
