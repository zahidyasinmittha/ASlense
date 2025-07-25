# 📁 ASLense Project Structure

This document provides a comprehensive overview of the ASLense project structure, including backend and frontend organization, key files, and their purposes.

## 🏗️ Overall Architecture

```
ASlense/
├── 📁 backend/           # FastAPI backend application
├── 📁 frontend/          # React TypeScript frontend
├── 📁 docs/             # Project documentation
├── 📁 scripts/          # Utility scripts
├── 📄 README.md         # Main project documentation
├── 📄 docker-compose.yml # Docker configuration
└── 📄 .gitignore        # Git ignore rules
```

## 🐍 Backend Structure

### Core Application (`backend/app/`)

```
backend/
├── 📁 app/
│   ├── 📁 api/v1/                    # API routes and endpoints
│   │   ├── 📁 endpoints/
│   │   │   ├── 📄 auth.py           # Authentication endpoints
│   │   │   ├── 📄 user.py           # User management
│   │   │   ├── 📄 videos.py         # Video management
│   │   │   ├── 📄 learn.py          # Learning endpoints
│   │   │   ├── 📄 practice.py       # Practice mode
│   │   │   ├── 📄 translate.py      # Translation features
│   │   │   ├── 📄 admin.py          # Admin dashboard
│   │   │   ├── 📄 contact.py        # Contact form & messaging
│   │   │   ├── 📄 psl_alphabet.py   # PSL alphabet management
│   │   │   ├── 📄 psl_inference.py  # PSL AI inference
│   │   │   └── 📄 websocket_psl.py  # WebSocket connections
│   │   └── 📄 api.py                # Main API router
│   │
│   ├── 📁 core/                     # Core configuration
│   │   ├── 📄 config.py             # App settings & environment
│   │   ├── 📄 database.py           # Database connection
│   │   └── 📄 security.py           # Security utilities
│   │
│   ├── 📁 models/                   # Database models
│   │   ├── 📄 __init__.py           # Model exports
│   │   ├── 📄 user.py               # User models
│   │   ├── 📄 video.py              # Video models
│   │   ├── 📄 practice.py           # Practice session models
│   │   ├── 📄 contact.py            # Contact message models
│   │   └── 📄 psl_alphabet.py       # PSL alphabet models
│   │
│   ├── 📁 schemas/                  # Pydantic validation schemas
│   │   ├── 📄 user.py               # User data schemas
│   │   ├── 📄 video.py              # Video data schemas
│   │   ├── 📄 practice.py           # Practice data schemas
│   │   ├── 📄 contact.py            # Contact form schemas
│   │   └── 📄 admin.py              # Admin data schemas
│   │
│   ├── 📁 services/                 # Business logic layer
│   │   ├── 📄 user_service.py       # User operations
│   │   ├── 📄 email_service.py      # Email functionality
│   │   ├── 📄 contact_service.py    # Contact management
│   │   ├── 📄 psl_service.py        # PSL learning service
│   │   ├── 📄 psl_model_service.py  # PSL AI model service
│   │   └── 📄 llm_service.py        # LLM integration
│   │
│   ├── 📁 ai/                       # AI/ML modules
│   │   ├── 📁 models/               # AI model files
│   │   ├── 📁 inference/            # Inference engines
│   │   └── 📁 preprocessing/        # Data preprocessing
│   │
│   └── 📄 main.py                   # FastAPI application entry
│
├── 📁 migrations/                   # Database migrations (Alembic)
│   ├── 📁 versions/                 # Migration versions
│   ├── 📄 env.py                    # Migration environment
│   └── 📄 script.py.mako            # Migration template
│
├── 📁 psl models/                   # PSL AI model files
│   ├── 📄 label_encoder.pkl         # Label encoder
│   ├── 📄 psl_efficientnet_b0_model.pth  # EfficientNet model
│   └── 📄 psl_mobilenet_v2_model.pth     # MobileNet model
│
├── 📁 docs/                         # Backend documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 requirements-dev.txt          # Development dependencies
├── 📄 alembic.ini                   # Database migration config
└── 📄 aslens.db                     # SQLite database (development)
```

## ⚛️ Frontend Structure

### React Application (`frontend/src/`)

```
frontend/
├── 📁 src/
│   ├── 📁 components/               # Reusable UI components
│   │   ├── 📁 ui/                   # Base UI components
│   │   ├── 📁 forms/                # Form components
│   │   ├── 📁 navigation/           # Navigation components
│   │   └── 📁 charts/               # Chart/visualization components
│   │
│   ├── 📁 pages/                    # Page components
│   │   ├── 📄 Home.tsx              # Landing page
│   │   ├── 📄 Login.tsx             # User authentication
│   │   ├── 📄 Register.tsx          # User registration
│   │   ├── 📄 Dashboard.tsx         # User dashboard
│   │   ├── 📄 Learn.tsx             # Learning interface
│   │   ├── 📄 Practice.tsx          # Practice mode
│   │   ├── 📄 Profile.tsx           # User profile
│   │   ├── 📄 Contact.tsx           # Contact form page
│   │   ├── 📄 AdminDashboard.tsx    # Admin management
│   │   ├── 📄 PSLLearn.tsx          # PSL learning
│   │   └── 📄 Translate.tsx         # Translation page
│   │
│   ├── 📁 hooks/                    # Custom React hooks
│   │   ├── 📄 useAuth.tsx           # Authentication hook
│   │   ├── 📄 useApi.tsx            # API interaction hook
│   │   ├── 📄 useLocalStorage.tsx   # Local storage hook
│   │   └── 📄 useWebSocket.tsx      # WebSocket hook
│   │
│   ├── 📁 contexts/                 # React context providers
│   │   ├── 📄 AuthContext.tsx       # Authentication context
│   │   ├── 📄 ThemeContext.tsx      # Theme management
│   │   └── 📄 NotificationContext.tsx # Notifications
│   │
│   ├── 📁 services/                 # API service layer
│   │   ├── 📄 api.ts                # Main API client
│   │   ├── 📄 auth.ts               # Authentication API
│   │   ├── 📄 user.ts               # User management API
│   │   ├── 📄 video.ts              # Video API
│   │   ├── 📄 practice.ts           # Practice API
│   │   ├── 📄 contact.ts            # Contact form API
│   │   └── 📄 admin.ts              # Admin API
│   │
│   ├── 📁 types/                    # TypeScript definitions
│   │   ├── 📄 auth.ts               # Authentication types
│   │   ├── 📄 user.ts               # User data types
│   │   ├── 📄 video.ts              # Video data types
│   │   ├── 📄 practice.ts           # Practice data types
│   │   ├── 📄 contact.ts            # Contact form types
│   │   └── 📄 admin.ts              # Admin data types
│   │
│   ├── 📁 utils/                    # Utility functions
│   │   ├── 📄 formatting.ts         # Data formatting
│   │   ├── 📄 validation.ts         # Form validation
│   │   ├── 📄 constants.ts          # App constants
│   │   └── 📄 helpers.ts            # Helper functions
│   │
│   ├── 📄 App.tsx                   # Main App component
│   ├── 📄 main.tsx                  # React entry point
│   ├── 📄 index.css                 # Global styles
│   └── 📄 vite-env.d.ts             # Vite type definitions
│
├── 📁 public/                       # Static assets
│   ├── 📁 img/                      # Images
│   ├── 📁 thumbnails/               # Video thumbnails
│   ├── 📄 favicon.ico               # Favicon
│   └── 📄 camera-test.html          # Camera testing page
│
├── 📁 docs/                         # Frontend documentation
├── 📄 package.json                  # Node.js dependencies
├── 📄 vite.config.ts               # Vite configuration
├── 📄 tailwind.config.js           # Tailwind CSS config
├── 📄 tsconfig.json                # TypeScript config
└── 📄 eslint.config.js             # ESLint configuration
```

## 📚 Documentation Structure

```
docs/
├── 📄 API.md                        # API documentation
├── 📄 DEVELOPMENT.md                # Development guide
├── 📄 CONTRIBUTING.md               # Contribution guidelines
├── 📄 CHANGELOG.md                  # Version changelog
├── 📄 PSL_INTEGRATION.md            # PSL integration guide
└── 📁 api/                          # Auto-generated API docs
```

## 🔧 Configuration Files

### Backend Configuration
- **`alembic.ini`** - Database migration configuration
- **`requirements.txt`** - Production Python dependencies
- **`requirements-dev.txt`** - Development Python dependencies
- **`.env`** - Environment variables (not in repo)

### Frontend Configuration
- **`package.json`** - Node.js dependencies and scripts
- **`vite.config.ts`** - Build tool configuration
- **`tailwind.config.js`** - CSS framework configuration
- **`tsconfig.json`** - TypeScript compiler options
- **`eslint.config.js`** - Code linting rules

## 🗄️ Database Structure

### Core Tables
- **`users`** - User accounts and authentication
- **`videos`** - Learning video content
- **`practice_sessions`** - User practice data
- **`contact_messages`** - Contact form submissions
- **`psl_alphabet_entries`** - PSL alphabet learning content
- **`user_progress`** - Learning progress tracking

### Key Features by Module

#### Authentication & Users (`auth.py`, `user.py`)
- JWT-based authentication
- User registration and login
- Profile management
- Role-based access control (Admin/User)

#### Contact System (`contact.py`)
- Contact form submission
- Email notifications
- Admin message management
- Status tracking (unread/read/replied)

#### Learning System (`learn.py`, `practice.py`)
- Video-based lessons
- Practice session tracking
- Progress analytics
- Achievement system

#### Admin Dashboard (`admin.py`)
- User management
- Content management
- System analytics
- Contact message management

#### PSL Integration (`psl_*.py`)
- AI model management
- Real-time inference
- WebSocket communication
- Learning progress tracking

## 🔌 API Endpoints Structure

### Public Endpoints
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/contact/` - Submit contact form

### User Endpoints (Authentication Required)
- `GET /api/v1/user/me` - Get current user
- `GET /api/v1/user/dashboard` - User dashboard data
- `POST /api/v1/practice/predict-video` - Video prediction

### Admin Endpoints (Admin Role Required)
- `GET /api/v1/admin/stats` - System statistics
- `GET /api/v1/admin/users` - User management
- `GET /api/v1/contact/admin/messages` - Contact messages
- `GET /api/v1/contact/admin/stats` - Contact statistics

## 🚀 Development Workflow

### Backend Development
1. **Models** - Define database schema in `models/`
2. **Schemas** - Create Pydantic validation in `schemas/`
3. **Services** - Implement business logic in `services/`
4. **Endpoints** - Create API routes in `api/v1/endpoints/`
5. **Migration** - Generate database migration with Alembic

### Frontend Development
1. **Types** - Define TypeScript interfaces in `types/`
2. **Services** - Create API client functions in `services/`
3. **Components** - Build reusable UI in `components/`
4. **Pages** - Implement page components in `pages/`
5. **Integration** - Connect components with API services

## 📊 Data Flow

### Contact Form Flow
1. User submits contact form on frontend
2. Frontend sends POST request to `/api/v1/contact/`
3. Backend validates data and saves to database
4. Email service sends notification to admin
5. Admin views message in admin dashboard
6. Admin can mark as read/replied and manage status

### Practice Session Flow
1. User uploads video or uses live camera
2. Frontend sends video to `/api/v1/practice/predict-video`
3. Backend processes video through AI models
4. Results stored in practice_sessions table
5. Progress data updated and returned to frontend
6. Analytics updated for dashboard display

This structure ensures maintainable, scalable, and well-organized code across both backend and frontend components.
