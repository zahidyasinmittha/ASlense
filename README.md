# 🤟 ASLense - Advanced Sign Language Learning Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **ASLense** is an AI-powered American Sign Language (ASL) learning platform that combines computer vision, machine learning, and modern web technologies to provide interactive, real-time sign language recognition and learning experiences.

## 🌟 Features

### 🎯 Core Features
- **Real-time ASL Recognition** - AI-powered sign language detection using computer vision
- **Interactive Learning** - Video-based lessons with progress tracking
- **Practice Mode** - Upload videos or use live camera for practice
- **Progress Analytics** - Detailed statistics and learning insights
- **Multi-Model Support** - Choose between accuracy (Pro) and speed (Mini) models
- **Admin Dashboard** - Complete system management and analytics
- **Contact System** - Integrated contact form with email notifications
- **User Support** - Direct communication channel with administrators

### 🤖 AI & Machine Learning
- **Ensemble Models** - Multiple AI models for high accuracy
- **Live Processing** - Real-time video frame analysis
- **Top-4 Predictions** - Shows multiple possible matches with confidence scores
- **Frame Aggregation** - Combines multiple frame predictions for better accuracy

### 📊 Analytics & Progress
- **XP System** - Gamified learning with experience points
- **Streak Tracking** - Daily practice streaks and longest streaks
- **Accuracy Metrics** - Detailed performance statistics
- **Achievement System** - Unlock achievements as you learn

### 📧 Communication & Support
- **Contact Form** - Users can submit inquiries and feedback
- **Email Notifications** - Automatic email alerts for new contact messages
- **Admin Message Management** - Complete contact message workflow
- **Status Tracking** - Mark messages as read, replied, or unread

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── app/
│   ├── api/v1/          # API routes and endpoints
│   ├── core/            # Configuration, security, database
│   ├── models/          # SQLAlchemy database models
│   ├── schemas/         # Pydantic validation schemas
│   ├── services/        # Business logic layer
│   ├── ai/              # AI/ML modules and inference
│   └── utils/           # Helper utilities
├── migrations/          # Database migrations
├── scripts/            # Utility scripts
└── tests/              # Backend tests
```

### Frontend (React + TypeScript)
```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/          # Page components
│   ├── hooks/          # Custom React hooks
│   ├── contexts/       # React context providers
│   ├── services/       # API service layer
│   ├── types/          # TypeScript definitions
│   └── utils/          # Frontend utilities
├── public/             # Static assets
└── tests/              # Frontend tests
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for version control

### 1. Clone Repository
```bash
git clone https://github.com/zahidyasinmittha/ASlense.git
cd ASlense
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings

# Initialize database
alembic upgrade head

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
# Navigate to frontend (new terminal)
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your settings

# Start development server
npm run dev
```

### 4. Access Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage

### For Learners
1. **Register/Login** - Create your account
2. **Explore Learn Section** - Browse video lessons
3. **Practice Mode** - Upload videos or use live camera
4. **Track Progress** - View your learning analytics
5. **Earn Achievements** - Complete challenges and unlock rewards
6. **Contact Support** - Submit questions or feedback through the contact form

### For Administrators
1. **Admin Dashboard** - Access comprehensive analytics
2. **User Management** - View and manage user accounts
3. **Content Management** - Upload and organize learning videos
4. **Contact Management** - View and respond to user messages
5. **System Analytics** - Monitor platform performance

## 🛠️ Development

### Project Structure
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization information.

### Key Technologies

#### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **Alembic** - Database migrations
- **PyTorch** - AI/ML framework
- **OpenCV** - Computer vision
- **JWT** - Authentication
- **SMTP Email** - Email notifications and communication
- **fastapi-mail** - Email service integration

#### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

### Code Quality
- **ESLint** + **Prettier** for frontend
- **Black** + **isort** for backend
- **TypeScript** strict mode
- **Comprehensive testing** with pytest and Jest

## 📚 API Documentation

### Authentication
```bash
# Login
POST /api/v1/user/login
{
  "username": "user@example.com",
  "password": "password"
}

# Get current user
GET /api/v1/user/me
Authorization: Bearer <token>
```

### Prediction
```bash
# Video prediction
POST /api/v1/practice/predict-video
Content-Type: multipart/form-data
- video_file: <video_file>
- target_word: "hello"
- model_type: "mini"
```

### Analytics
```bash
# User dashboard
GET /api/v1/user/dashboard
Authorization: Bearer <token>

# Admin metrics
GET /api/v1/admin/metrics
Authorization: Bearer <admin_token>
```

### Contact System
```bash
# Submit contact form
POST /api/v1/contact/
{
  "name": "John Doe",
  "email": "john@example.com",
  "subject": "Question about features",
  "message": "I would like to know more about..."
}

# Get contact messages (admin only)
GET /api/v1/contact/admin/messages
Authorization: Bearer <admin_token>

# Get contact statistics (admin only)
GET /api/v1/contact/admin/stats
Authorization: Bearer <admin_token>
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Integration Tests
```bash
# Run full test suite
npm run test:e2e
```

## 🚢 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Deployment
1. **Backend** - Deploy to Heroku, Railway, or VPS
2. **Frontend** - Deploy to Vercel, Netlify, or CDN
3. **Database** - PostgreSQL for production
4. **Storage** - AWS S3 for video files

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📋 Roadmap

### Version 2.0
- [ ] Mobile app development
- [ ] Multi-language support
- [ ] Advanced AI models
- [ ] Collaborative learning features
- [ ] Offline mode support

### Future Enhancements
- [ ] VR/AR integration
- [ ] Real-time multiplayer practice
- [ ] AI-powered personalized curriculum
- [ ] Social learning features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Development Lead** - [@zahidyasinmittha](https://github.com/zahidyasinmittha)
- **AI/ML Engineering** - ASL Recognition Models
- **Frontend Development** - React/TypeScript Interface
- **Backend Development** - FastAPI Services

## 🙏 Acknowledgments

- ASL community for guidance and feedback
- Open-source contributors and libraries
- AI/ML research community
- Beta testers and early adopters

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/zahidyasinmittha/ASlense/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zahidyasinmittha/ASlense/discussions)

---
## Model weights to download
- **ALL Model Weights** - [Drive Link](https://drive.google.com/drive/folders/1EFPTa3hPqEo2f15dHvCxpshaTn5cqDX-?usp=drive_link)

---
<div align="center">
  <strong>Made with ❤️ for the ASL learning community</strong>
</div>
