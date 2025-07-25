# ASLense Development Guide

This guide provides detailed information for developers working on ASLense.

## 🏗️ Architecture Overview

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Models     │
│   (React TS)    │◄──►│   (FastAPI)     │◄──►│   (PyTorch)     │
│                 │    │                 │    │                 │
│ • Components    │    │ • API Routes    │    │ • Video Pred    │
│ • State Mgmt    │    │ • Business      │    │ • Frame Proc    │
│ • UI/UX         │    │ • Database      │    │ • Model Inf     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Database      │
                    │   (SQLite)      │
                    │                 │
                    │ • User Data     │
                    │ • Progress      │
                    │ • Videos        │
                    └─────────────────┘
```

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.104+
- **Database**: SQLite with SQLAlchemy ORM
- **Migrations**: Alembic
- **Authentication**: JWT tokens
- **AI/ML**: PyTorch, OpenCV
- **Testing**: pytest, httpx

#### Frontend
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: React Context + useReducer
- **HTTP Client**: Axios
- **Testing**: Vitest, React Testing Library

## 🔧 Development Environment

### Prerequisites
```bash
# System Requirements
Python 3.8+
Node.js 16+
Git
FFmpeg (for video processing)

# Optional but Recommended
Docker & Docker Compose
VS Code with extensions:
  - Python
  - TypeScript
  - ES7+ React/Redux/React-Native snippets
  - Tailwind CSS IntelliSense
```

### Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/ASlense.git
cd ASlense

# Setup backend
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup frontend
cd ../frontend
npm install

# Setup database
cd ../backend
alembic upgrade head

# Create .env file
cp .env.example .env
# Edit .env with your configuration
```

### Development Servers
```bash
# Terminal 1: Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Database (if needed)
cd backend
alembic upgrade head
```

## 📁 Detailed Project Structure

### Backend Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # App configuration
│   │   ├── database.py         # Database setup
│   │   └── security.py         # Auth utilities
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py          # API router aggregation
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── auth.py     # Authentication endpoints
│   │           ├── users.py    # User management
│   │           ├── videos.py   # Video content
│   │           ├── practice.py # Practice sessions
│   │           ├── learn.py    # Learning modules
│   │           └── translate.py # Translation features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py            # User models
│   │   ├── video.py           # Video models
│   │   └── progress.py        # Progress tracking
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py            # User Pydantic schemas
│   │   ├── video.py           # Video schemas
│   │   └── prediction.py      # Prediction schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py    # Authentication logic
│   │   ├── video_service.py   # Video processing
│   │   └── prediction_service.py # AI prediction logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── video_utils.py     # Video utilities
│   │   └── auth_utils.py      # Auth utilities
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py        # Test configuration
│       ├── test_auth.py       # Auth tests
│       ├── test_videos.py     # Video tests
│       └── test_prediction.py # Prediction tests
├── migrations/                # Alembic migrations
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── .env.example              # Environment template
```

### Frontend Structure
```
frontend/
├── src/
│   ├── main.tsx              # App entry point
│   ├── App.tsx               # Main app component
│   ├── index.css             # Global styles
│   ├── vite-env.d.ts         # Vite type definitions
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.tsx    # Reusable button
│   │   │   ├── Modal.tsx     # Modal component
│   │   │   ├── Loading.tsx   # Loading spinner
│   │   │   └── VideoPlayer.tsx # Video player
│   │   ├── layout/
│   │   │   ├── Header.tsx    # App header
│   │   │   ├── Footer.tsx    # App footer
│   │   │   └── Sidebar.tsx   # Navigation sidebar
│   │   └── forms/
│   │       ├── LoginForm.tsx # Login form
│   │       └── SignupForm.tsx # Signup form
│   ├── pages/
│   │   ├── Home.tsx          # Landing page
│   │   ├── Login.tsx         # Login page
│   │   ├── Dashboard.tsx     # User dashboard
│   │   ├── Practice.tsx      # Practice sessions
│   │   ├── Learn.tsx         # Learning modules
│   │   └── Translate.tsx     # Translation feature
│   ├── hooks/
│   │   ├── useAuth.tsx       # Authentication hook
│   │   ├── useVideos.tsx     # Video data hook
│   │   └── usePrediction.tsx # Prediction hook
│   ├── services/
│   │   ├── api.ts            # API client setup
│   │   ├── auth.ts           # Auth API calls
│   │   ├── videos.ts         # Video API calls
│   │   └── prediction.ts     # Prediction API calls
│   ├── types/
│   │   ├── auth.ts           # Auth type definitions
│   │   ├── video.ts          # Video type definitions
│   │   └── prediction.ts     # Prediction types
│   ├── context/
│   │   ├── AuthContext.tsx   # Auth state management
│   │   └── AppContext.tsx    # Global app state
│   └── utils/
│       ├── constants.ts      # App constants
│       ├── helpers.ts        # Utility functions
│       └── validation.ts     # Form validation
├── public/
│   ├── thumbnails/           # Video thumbnails
│   └── favicon.ico
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── .env.local
```

## 🔌 API Development

### Adding New Endpoints

1. **Create Schema** (if needed):
```python
# app/schemas/new_feature.py
from pydantic import BaseModel
from typing import Optional

class NewFeatureCreate(BaseModel):
    name: str
    description: Optional[str] = None

class NewFeatureResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
```

2. **Create Model** (if needed):
```python
# app/models/new_feature.py
from sqlalchemy import Column, Integer, String, DateTime
from app.core.database import Base

class NewFeature(Base):
    __tablename__ = "new_features"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

3. **Create Service**:
```python
# app/services/new_feature_service.py
from sqlalchemy.orm import Session
from app.models.new_feature import NewFeature
from app.schemas.new_feature import NewFeatureCreate

class NewFeatureService:
    def create(self, db: Session, feature_data: NewFeatureCreate) -> NewFeature:
        feature = NewFeature(**feature_data.dict())
        db.add(feature)
        db.commit()
        db.refresh(feature)
        return feature
```

4. **Create Endpoint**:
```python
# app/api/v1/endpoints/new_feature.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.new_feature_service import NewFeatureService
from app.schemas.new_feature import NewFeatureCreate, NewFeatureResponse

router = APIRouter()
service = NewFeatureService()

@router.post("/", response_model=NewFeatureResponse)
def create_feature(
    feature_data: NewFeatureCreate,
    db: Session = Depends(get_db)
):
    return service.create(db, feature_data)
```

5. **Register Router**:
```python
# app/api/v1/api.py
from app.api.v1.endpoints import new_feature

api_router.include_router(
    new_feature.router, 
    prefix="/new-features", 
    tags=["new-features"]
)
```

### API Testing
```python
# tests/test_new_feature.py
def test_create_feature(client, db):
    feature_data = {
        "name": "Test Feature",
        "description": "Test description"
    }
    response = client.post("/api/v1/new-features/", json=feature_data)
    assert response.status_code == 200
    assert response.json()["name"] == "Test Feature"
```

## 📧 Contact System Development

### Email Service Configuration

The contact system uses SMTP email service for sending notifications. Configure email settings in your environment:

```bash
# .env file
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_FROM=your-email@gmail.com
MAIL_FROM_NAME="ASLense Support"
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_STARTTLS=true
MAIL_SSL_TLS=false
```

### Contact Feature Components

#### Backend Components
1. **Models** (`backend/app/models/contact.py`):
   - ContactMessage model with SQLAlchemy
   - Status tracking (unread/read/replied)

2. **Schemas** (`backend/app/schemas/contact.py`):
   - ContactMessageCreate for form validation
   - ContactMessageResponse for API responses

3. **Services** (`backend/app/services/`):
   - `email_service.py`: SMTP email functionality
   - `contact_service.py`: Database operations

4. **API Endpoints** (`backend/app/api/v1/endpoints/contact.py`):
   - Public contact form submission
   - Admin contact management endpoints

#### Frontend Components
1. **Contact Form** (`frontend/src/pages/Contact.tsx`):
   - Form validation with error handling
   - Success/error notifications
   - Real-time API integration

2. **Admin Dashboard** (`frontend/src/pages/AdminDashboard.tsx`):
   - Contact messages management tab
   - Status filtering and updates
   - Message statistics display

### Development Workflow for Contact Features

1. **Database Changes**:
```bash
# Generate migration
cd backend
alembic revision --autogenerate -m "Add contact message table"

# Run migration
alembic upgrade head
```

2. **Testing Email Service**:
```python
# Test email functionality
python -c "
from app.services.email_service import EmailService
from app.core.config import settings

email_service = EmailService()
email_service.send_contact_form_email(
    'Test Subject',
    'test@example.com',
    'Test User',
    'Test message content'
)
"
```

3. **API Testing**:
```bash
# Test contact form submission
curl -X POST http://localhost:8000/api/v1/contact/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "email": "test@example.com",
    "subject": "Test Subject",
    "message": "Test message"
  }'
```

## 🎨 Frontend Development

### Component Development

1. **Create Component**:
```typescript
// src/components/ui/NewComponent.tsx
import React from 'react';
import { NewComponentProps } from '../../types/component';

const NewComponent: React.FC<NewComponentProps> = ({ 
  title, 
  onAction 
}) => {
  return (
    <div className="p-4 border rounded-lg">
      <h3 className="text-lg font-semibold">{title}</h3>
      <button 
        onClick={onAction}
        className="mt-2 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Action
      </button>
    </div>
  );
};

export default NewComponent;
```

2. **Add Types**:
```typescript
// src/types/component.ts
export interface NewComponentProps {
  title: string;
  onAction: () => void;
  variant?: 'primary' | 'secondary';
}
```

3. **Create Hook** (if needed):
```typescript
// src/hooks/useNewFeature.tsx
import { useState, useEffect } from 'react';
import { newFeatureService } from '../services/newFeature';

export const useNewFeature = () => {
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadFeatures = async () => {
    setLoading(true);
    try {
      const data = await newFeatureService.getAll();
      setFeatures(data);
    } catch (error) {
      console.error('Error loading features:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFeatures();
  }, []);

  return { features, loading, loadFeatures };
};
```

4. **Add Service**:
```typescript
// src/services/newFeature.ts
import { apiClient } from './api';
import { NewFeature, CreateNewFeatureRequest } from '../types/newFeature';

export const newFeatureService = {
  async getAll(): Promise<NewFeature[]> {
    const response = await apiClient.get('/new-features/');
    return response.data;
  },

  async create(data: CreateNewFeatureRequest): Promise<NewFeature> {
    const response = await apiClient.post('/new-features/', data);
    return response.data;
  }
};
```

## 🧪 Testing Strategy

### Backend Testing
```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import get_db, Base

@pytest.fixture
def client():
    # Test database setup
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)
```

### Frontend Testing
```typescript
// src/components/__tests__/NewComponent.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import NewComponent from '../ui/NewComponent';

describe('NewComponent', () => {
  it('renders title correctly', () => {
    const mockAction = vi.fn();
    render(<NewComponent title="Test Title" onAction={mockAction} />);
    
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('calls onAction when button clicked', () => {
    const mockAction = vi.fn();
    render(<NewComponent title="Test" onAction={mockAction} />);
    
    fireEvent.click(screen.getByText('Action'));
    expect(mockAction).toHaveBeenCalledTimes(1);
  });
});
```

## 🚀 Deployment

### Production Build
```bash
# Backend
cd backend
pip install -r requirements.txt
alembic upgrade head

# Frontend
cd frontend
npm run build
```

### Docker Deployment
```dockerfile
# Dockerfile.backend
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile.frontend
FROM node:16-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

### Environment Variables
```bash
# Backend .env
DATABASE_URL=sqlite:///./aslens.db
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=False

# Frontend .env.local
VITE_API_URL=http://localhost:8000/api/v1
VITE_APP_TITLE=ASLense
```

## 🔍 Debugging

### Backend Debugging
```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing video: {video_path}")
```

### Frontend Debugging
```typescript
// React DevTools
// Redux DevTools (if using Redux)
// Console debugging
console.log('Component state:', state);
console.table(data);
```

### Common Issues

1. **Import Errors**: Check Python path and virtual environment
2. **CORS Issues**: Verify CORS settings in FastAPI
3. **Database Issues**: Check connection and migrations
4. **Build Errors**: Clear node_modules and reinstall

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

## 🤝 Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Join our development community
- **Code Reviews**: Submit PRs for feedback

---

This guide is continuously updated. For the latest information, check the repository documentation.
