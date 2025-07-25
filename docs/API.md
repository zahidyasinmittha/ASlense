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

### Contact Management

#### Submit Contact Form
```http
POST /contact/
Content-Type: application/json

{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "subject": "Feature Request",
    "message": "I would like to suggest a new feature..."
}
```

**Response:**
```json
{
    "id": 1,
    "name": "John Doe",
    "email": "john.doe@example.com",
    "subject": "Feature Request",
    "message": "I would like to suggest a new feature...",
    "status": "unread",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
}
```

#### Get Contact Messages (Admin Only)
```http
GET /contact/admin/messages?page=1&limit=10
Authorization: Bearer <admin-token>
```

**Response:**
```json
{
    "messages": [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john.doe@example.com",
            "subject": "Feature Request",
            "message": "I would like to suggest a new feature...",
            "status": "unread",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    ],
    "total": 1,
    "page": 1,
    "limit": 10,
    "pages": 1
}
```

#### Update Contact Message Status (Admin Only)
```http
PUT /contact/admin/messages/{message_id}/status
Authorization: Bearer <admin-token>
Content-Type: application/json

{
    "status": "read"
}
```

#### Delete Contact Message (Admin Only)
```http
DELETE /contact/admin/messages/{message_id}
Authorization: Bearer <admin-token>
```

#### Get Contact Statistics (Admin Only)
```http
GET /contact/admin/stats
Authorization: Bearer <admin-token>
```

**Response:**
```json
{
    "total_messages": 15,
    "unread_messages": 3,
    "read_messages": 10,
    "replied_messages": 2
}
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

#### Contact Message
```typescript
interface ContactMessage {
    id: number;
    name: string;
    email: string;
    subject: string;
    message: string;
    status: "unread" | "read" | "replied";
    created_at: string;
    updated_at: string;
}
```

#### Contact Statistics
```typescript
interface ContactStats {
    total_messages: number;
    unread_messages: number;
    read_messages: number;
    replied_messages: number;
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
