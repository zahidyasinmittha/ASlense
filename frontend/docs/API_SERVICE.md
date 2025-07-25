# API Service Documentation

This document explains how to use the centralized API service in the ASLense frontend application.

## Overview

The API service provides a centralized way to handle all backend API calls with the following benefits:
- Environment-based configuration (uses `VITE_BACKEND_BASEURL` from `.env`)
- Consistent error handling
- Type-safe API calls
- WebSocket utilities
- Custom hooks for common operations

## Files Structure

- `src/services/api.ts` - Main API service with endpoint configuration
- `src/services/apiHooks.ts` - React hooks for API operations
- `frontend/.env` - Environment configuration

## Usage

### 1. Basic API Configuration

```typescript
import { API_CONFIG } from '../services/api';

// Get base URLs
const baseUrl = API_CONFIG.BASE_URL; // http://localhost:8000 (from .env)
const wsUrl = API_CONFIG.WS_URL;     // ws://localhost:8000

// Get specific endpoints
const loginEndpoint = API_CONFIG.ENDPOINTS.AUTH.LOGIN; // '/auth/login'
```

### 2. Using API Functions

```typescript
import { authAPI, userAPI, translateAPI, contactAPI } from '../services/api';

// Authentication
const response = await authAPI.login({ username: 'user', password: 'pass' });
const userData = response.data;

// User operations
const progress = await userAPI.getProgress();

// Translation operations
const formData = new FormData();
formData.append('video', videoFile);
const result = await translateAPI.predictVideo(formData);

// Contact operations
const contactData = {
  name: 'John Doe',
  email: 'john@example.com',
  subject: 'Feature Request',
  message: 'I would like to suggest...'
};
const contactResponse = await contactAPI.submitContact(contactData);
```

### 3. Using Custom Hooks (Recommended)

```typescript
import { useTranslationAPI, useUserAPI } from '../services/apiHooks';

function MyComponent() {
  const { predictVideo, loading, error } = useTranslationAPI();
  const { getProgress } = useUserAPI();

  const handleVideoUpload = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('video', file);
      const result = await predictVideo(formData);
      console.log('Prediction:', result);
    } catch (err) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      {loading && <p>Processing...</p>}
      {error && <p>Error: {error}</p>}
      {/* Your component JSX */}
    </div>
  );
}
```

### 4. WebSocket Connections

```typescript
import { WebSocketAPI } from '../services/api';
import { useWebSocket } from '../services/apiHooks';

// Get WebSocket URL
const wsUrl = WebSocketAPI.getLiveTranslateUrl('mini', 'sentence');

// Using the hook
function LiveTranslation() {
  const { connect, send, status } = useWebSocket();

  useEffect(() => {
    const ws = connect(wsUrl, (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);
    });

    return () => ws?.close();
  }, []);

  const sendFrame = (frameData: string) => {
    send({ type: 'frame', frame: frameData });
  };

  return <div>Status: {status}</div>;
}
```

## Environment Configuration

Update your `frontend/.env` file:

```properties
# Frontend Environment Variables
VITE_BACKEND_BASEURL=http://localhost:8000
```

For production:
```properties
VITE_BACKEND_BASEURL=https://your-production-api.com
```

## Available API Modules

### AuthAPI
- `login(credentials)` - User login
- `register(userData)` - User registration  
- `refresh(token)` - Refresh access token
- `me()` - Get current user info

### UserAPI
- `getProgress()` - Get user progress
- `updateProfile(userData)` - Update user profile
- `getAchievements()` - Get user achievements

### TranslateAPI
- `predictVideo(formData)` - Predict ASL from video
- `startSession(data)` - Start translation session
- `endSession(sessionId, data)` - End translation session

### ContactAPI
- `submitContact(contactData)` - Submit contact form
- `getMessages(page, limit)` - Get contact messages (admin only)
- `updateMessageStatus(id, status)` - Update message status (admin only)
- `deleteMessage(id)` - Delete contact message (admin only)
- `getStats()` - Get contact statistics (admin only)

### PracticeAPI
- `predictVideo(formData)` - Practice video prediction
- `getModelsStatus()` - Get AI models status
- `getAvailableWords()` - Get available practice words

### WebSocketAPI
- `getLiveTranslateUrl(model, mode)` - Get translation WebSocket URL
- `getLivePracticeUrl(model, category)` - Get practice WebSocket URL

## Error Handling

All API calls include automatic error handling:

```typescript
try {
  const result = await translateAPI.predictVideo(formData);
} catch (error) {
  // Error is automatically formatted and user-friendly
  console.error(error.message);
}
```

## Migration from Old Code

### Before (hardcoded URLs):
```typescript
const response = await fetch('http://localhost:8000/translate/video-predict', {
  method: 'POST',
  body: formData,
});
```

### After (using API service):
```typescript
const response = await translateAPI.predictVideo(formData);
```

### Using hooks (even better):
```typescript
const { predictVideo } = useTranslationAPI();
const result = await predictVideo(formData);
```

## Benefits

1. **Environment-based URLs** - Automatically uses correct backend URL
2. **Consistent error handling** - All errors are properly formatted
3. **Type safety** - Better IDE support and fewer runtime errors
4. **Maintainability** - Single place to update API endpoints
5. **Reusability** - Hooks can be shared across components
6. **Testing** - Easier to mock API calls for testing

## Notes

- All API calls automatically include authentication headers when available
- WebSocket connections are handled separately but use the same URL configuration
- The service automatically handles token refresh for authenticated requests
- Error messages are user-friendly and consistent across the app
