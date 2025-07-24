# PSL Model Integration Guide

This guide explains how to integrate your trained PS Mini (MobileNet V2) model with the ASLense backend for real-time PSL recognition.

## 🚀 Quick Setup

### 1. Prepare Model Files

First, ensure you have your trained model files:

```bash
# Your training script should generate these files:
psl_mobilenet_v2_model.pth    # Trained model weights
label_encoder.pkl             # Label encoder for class mapping
```

### 2. Setup Backend

```bash
# Navigate to backend directory
cd backend

# Run the setup script
python setup_psl_model.py

# Copy your model files to the psl models directory
# (Create this directory if it doesn't exist)
mkdir "psl models"
copy /path/to/your/psl_mobilenet_v2_model.pth "psl models/"
copy /path/to/your/label_encoder.pkl "psl models/"

# Test the model loading
python test_psl_model.py

# Install additional dependencies if needed
pip install -r requirements.txt
```

### 3. Start Backend with PSL Support

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 📡 API Endpoints

### REST API Endpoints

```bash
# Get model information
GET /api/v1/psl-inference/model-info

# Predict from uploaded image
POST /api/v1/psl-inference/predict
# Body: multipart/form-data with image file

# Predict from base64 image
POST /api/v1/psl-inference/predict-base64
# Body: {"image": "base64_string", "top_k": 3}

# Health check
GET /api/v1/psl-inference/health
```

### WebSocket Endpoints

```bash
# Real-time PSL recognition
ws://localhost:8000/api/v1/ws/psl-recognition

# Authenticated real-time recognition
ws://localhost:8000/api/v1/ws/psl-recognition/{user_id}
```

## 🔌 Frontend Integration

### Option 1: WebSocket (Recommended for real-time)

```typescript
import { usePSLWebSocket } from '@/hooks/usePSLWebSocket';

const MyComponent = () => {
  const { isConnected, predictions, predict } = usePSLWebSocket();

  const handleCameraFrame = (canvas: HTMLCanvasElement) => {
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
    predict(imageDataUrl);
  };

  return (
    <div>
      {isConnected ? (
        <div>
          <h3>Predictions:</h3>
          {predictions.map((pred, idx) => (
            <div key={idx}>
              {pred.class}: {(pred.confidence * 100).toFixed(1)}%
            </div>
          ))}
        </div>
      ) : (
        <p>Connecting to PSL model...</p>
      )}
    </div>
  );
};
```

### Option 2: REST API (For single predictions)

```typescript
const predictPSLSign = async (imageFile: File) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('top_k', '3');

  const response = await fetch('/api/v1/psl-inference/predict', {
    method: 'POST',
    body: formData,
  });

  return await response.json();
};
```

## 📝 WebSocket Message Formats

### Client → Server Messages

```javascript
// Prediction request
{
  "type": "predict",
  "image": "base64_image_data",
  "top_k": 3
}

// Get model info
{
  "type": "model_info"
}

// Ping (keep alive)
{
  "type": "ping",
  "timestamp": 1642781234567
}
```

### Server → Client Messages

```javascript
// Prediction result
{
  "type": "prediction",
  "predictions": [
    {"class": "A", "confidence": 0.95},
    {"class": "B", "confidence": 0.03},
    {"class": "C", "confidence": 0.02}
  ],
  "confidence": 0.95,
  "processing_time": "150ms"
}

// Model information
{
  "type": "model_info",
  "data": {
    "model_name": "PS Mini",
    "status": "loaded",
    "accuracy": "98%",
    "speed": "Fast (150ms)",
    "num_classes": 26,
    "device": "cuda"
  }
}

// Error message
{
  "type": "error",
  "message": "Error description"
}
```

## 🛠️ Integration with Existing PSLPractice

### Update your PSLPractice.tsx:

1. **Import the WebSocket hook:**
```typescript
import { usePSLWebSocket } from '@/hooks/usePSLWebSocket';
```

2. **Add to your component:**
```typescript
const { isConnected, predictions, modelInfo, predict } = usePSLWebSocket();
```

3. **Modify your camera capture:**
```typescript
const captureAndPredict = () => {
  if (cameraRef.current && canvasRef.current) {
    const canvas = canvasRef.current;
    const video = cameraRef.current;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      predict(imageDataUrl); // Send to real PSL model
    }
  }
};
```

4. **Update your model selection to use real models:**
```typescript
const models = [
  {
    id: 'ps-mini',
    name: 'PS Mini',
    accuracy: '98%',
    speed: 'Fast',
    description: 'Real MobileNet V2 model',
    isReal: true, // Flag to indicate this is a real model
  },
  // ... your existing mock models
];
```

## 🔧 Troubleshooting

### Model Not Loading
```bash
# Check if model files exist
ls "psl models/"

# Check file sizes
du -h "psl models/"*

# Test model loading
python test_psl_model.py
```

### WebSocket Connection Issues
- Ensure backend is running on port 8000
- Check if CORS is properly configured
- Verify WebSocket URL: `ws://localhost:8000/api/v1/ws/psl-recognition`

### Performance Issues
- Model runs on GPU if available (CUDA)
- Reduce image quality for faster processing
- Use WebSocket for real-time vs REST for single predictions

## 📊 Model Performance

- **PS Mini Model**: MobileNet V2 based
- **Accuracy**: 98% (as reported from training)
- **Speed**: ~150ms per prediction
- **Input**: 224x224 RGB images
- **Output**: Top-K predictions with confidence scores

## 🔐 Security Notes

- WebSocket endpoints can be authenticated by user_id
- API endpoints can use existing JWT authentication
- Model files should not be exposed publicly
- Consider rate limiting for production use

## 📈 Monitoring

Check model health:
```bash
curl http://localhost:8000/api/v1/psl-inference/health
```

View real-time logs:
```bash
# Backend logs will show:
# - Model loading status
# - WebSocket connections
# - Prediction requests
# - Performance metrics
```
