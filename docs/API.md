# ASLense API Documentation

## Overview
The ASLense API provides endpoints for ASL learning, practice, and user management.

**Base URL**: `http://localhost:8000/api/v1`

## Authentication

All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### Login
```http
POST /user/login
Content-Type: application/json

{
    "username": "user@example.com",
    "password": "password123"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "user": {
        "id": 1,
        "username": "user@example.com",
        "fullName": "John Doe",
        "role": "user"
    }
}
```

## Endpoints

### User Management

#### Get Current User
```http
GET /user/me
Authorization: Bearer <token>
```

#### Get User Dashboard
```http
GET /user/dashboard
Authorization: Bearer <token>
```

#### Get User Progress
```http
GET /user/progress
Authorization: Bearer <token>
```

### Practice & Learning

#### Predict from Video Upload
```http
POST /practice/predict-video
Authorization: Bearer <token>
Content-Type: multipart/form-data

video_file: <file>
target_word: "hello"
model_type: "mini"  # or "pro"
```

**Response:**
```json
{
    "target_word": "hello",
    "predictions": [
        {
            "word": "hello",
            "confidence": 0.95,
            "rank": 1
        },
        {
            "word": "hi",
            "confidence": 0.78,
            "rank": 2
        }
    ],
    "is_correct": true,
    "match_confidence": 0.95,
    "model_used": "mini"
}
```

#### Get Available Words
```http
GET /practice/available-words
```

#### Live Frame Prediction
```http
POST /practice/predict-frames
Authorization: Bearer <token>
Content-Type: application/json

{
    "frames": ["base64_encoded_image_1", "base64_encoded_image_2"],
    "target_word": "thank you",
    "model_type": "mini"
}
```

### Video Content

#### Get Videos
```http
GET /videos?category=basic&difficulty=beginner&limit=10
```

#### Get Video by ID
```http
GET /videos/{video_id}
```

#### Search Videos
```http
GET /videos/search/{word}
```

### Admin Endpoints

#### Get System Metrics
```http
GET /admin/metrics
Authorization: Bearer <admin-token>
```

#### Get All Users
```http
GET /admin/users?skip=0&limit=50
Authorization: Bearer <admin-token>
```

#### Get Video Statistics
```http
GET /admin/videos/stats
Authorization: Bearer <admin-token>
```

## Error Handling

### Error Response Format
```json
{
    "detail": "Error message",
    "status_code": 400,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Rate Limiting

API endpoints are rate limited:
- **General endpoints**: 100 requests per minute
- **Prediction endpoints**: 30 requests per minute
- **Admin endpoints**: 200 requests per minute

## Models

### Prediction Models
- **Mini Model**: Fast inference, good for real-time practice
- **Pro Model**: High accuracy, best for evaluation

### Data Models

#### User
```typescript
interface User {
    id: number;
    username: string;
    fullName: string;
    email: string;
    role: "user" | "admin";
    isActive: boolean;
    createdAt: string;
}
```

#### Video
```typescript
interface Video {
    id: number;
    word: string;
    title: string;
    description: string;
    videoPath: string;
    thumbnailPath: string;
    category: string;
    difficulty: string;
    duration: number;
}
```

#### Prediction
```typescript
interface Prediction {
    word: string;
    confidence: number;
    rank: number;
}
```

## WebSocket Endpoints

### Live Prediction
```javascript
const ws = new WebSocket("ws://localhost:8000/practice/live-predict");

// Send frame
ws.send(JSON.stringify({
    type: "frame",
    frame: "base64_encoded_image",
    target_word: "hello",
    model_type: "mini"
}));

// Receive predictions
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "predictions") {
        console.log(data.predictions);
    }
};
```

## SDK Examples

### Python
```python
import requests

# Login
response = requests.post("http://localhost:8000/api/v1/user/login", json={
    "username": "user@example.com",
    "password": "password"
})
token = response.json()["access_token"]

# Upload video for prediction
with open("sign_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/practice/predict-video",
        headers={"Authorization": f"Bearer {token}"},
        files={"video_file": f},
        data={"target_word": "hello", "model_type": "mini"}
    )
print(response.json())
```

### JavaScript
```javascript
// Login
const loginResponse = await fetch('/api/v1/user/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        username: 'user@example.com',
        password: 'password'
    })
});
const { access_token } = await loginResponse.json();

// Get user dashboard
const dashboardResponse = await fetch('/api/v1/user/dashboard', {
    headers: { 'Authorization': `Bearer ${access_token}` }
});
const dashboard = await dashboardResponse.json();
```
