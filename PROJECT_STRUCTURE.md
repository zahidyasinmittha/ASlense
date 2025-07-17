# ASLense Project Structure Guide

## 📁 Project Organization

This document outlines the recommended structure for the ASLense project to maintain clean, scalable, and maintainable code.

```
ASlense/
├── 📁 backend/                    # FastAPI Backend
│   ├── 📁 app/
│   │   ├── 📁 api/                # API layer
│   │   │   ├── 📁 v1/             # API version 1
│   │   │   │   ├── 📁 endpoints/  # Route handlers
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── 📁 core/               # Core functionality
│   │   │   ├── config.py          # Configuration
│   │   │   ├── security.py        # Authentication & Security
│   │   │   ├── database.py        # Database configuration
│   │   │   └── __init__.py
│   │   ├── 📁 models/             # Database models
│   │   │   ├── user.py
│   │   │   ├── video.py
│   │   │   ├── prediction.py
│   │   │   └── __init__.py
│   │   ├── 📁 schemas/            # Pydantic schemas
│   │   │   ├── user.py
│   │   │   ├── video.py
│   │   │   ├── prediction.py
│   │   │   └── __init__.py
│   │   ├── 📁 services/           # Business logic
│   │   │   ├── user_service.py
│   │   │   ├── video_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── ai_service.py
│   │   │   └── __init__.py
│   │   ├── 📁 utils/              # Utilities
│   │   │   ├── helpers.py
│   │   │   ├── validators.py
│   │   │   └── __init__.py
│   │   ├── 📁 ai/                 # AI/ML modules
│   │   │   ├── models/            # AI model files
│   │   │   ├── inference.py       # Inference logic
│   │   │   ├── preprocessing.py   # Data preprocessing
│   │   │   └── __init__.py
│   │   ├── main.py                # FastAPI app entry point
│   │   └── __init__.py
│   ├── 📁 migrations/             # Alembic migrations
│   ├── 📁 tests/                  # Backend tests
│   │   ├── 📁 unit/
│   │   ├── 📁 integration/
│   │   └── conftest.py
│   ├── 📁 scripts/                # Utility scripts
│   │   ├── setup_db.py
│   │   ├── seed_data.py
│   │   └── regenerate_thumbnails.py
│   ├── requirements.txt           # Python dependencies
│   ├── requirements-dev.txt       # Development dependencies
│   ├── .env.example              # Environment variables template
│   ├── alembic.ini               # Alembic configuration
│   └── README.md                 # Backend documentation
│
├── 📁 frontend/                   # React Frontend
│   ├── 📁 public/                 # Static assets
│   │   ├── 📁 thumbnails/         # Video thumbnails
│   │   ├── 📁 icons/              # App icons
│   │   └── index.html
│   ├── 📁 src/
│   │   ├── 📁 components/         # Reusable components
│   │   │   ├── 📁 ui/             # Basic UI components
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Card.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   └── index.ts
│   │   │   ├── 📁 forms/          # Form components
│   │   │   ├── 📁 layout/         # Layout components
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   ├── Footer.tsx
│   │   │   │   └── Layout.tsx
│   │   │   └── index.ts
│   │   ├── 📁 pages/              # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Practice.tsx
│   │   │   ├── Learn.tsx
│   │   │   ├── AdminDashboard.tsx
│   │   │   └── index.ts
│   │   ├── 📁 hooks/              # Custom React hooks
│   │   │   ├── useAuth.ts
│   │   │   ├── useAPI.ts
│   │   │   └── index.ts
│   │   ├── 📁 contexts/           # React contexts
│   │   │   ├── AuthContext.tsx
│   │   │   └── index.ts
│   │   ├── 📁 services/           # API services
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   ├── video.ts
│   │   │   └── index.ts
│   │   ├── 📁 utils/              # Utility functions
│   │   │   ├── constants.ts
│   │   │   ├── helpers.ts
│   │   │   ├── validators.ts
│   │   │   └── index.ts
│   │   ├── 📁 types/              # TypeScript types
│   │   │   ├── user.ts
│   │   │   ├── video.ts
│   │   │   ├── api.ts
│   │   │   └── index.ts
│   │   ├── 📁 styles/             # Global styles
│   │   │   ├── globals.css
│   │   │   ├── components.css
│   │   │   └── index.css
│   │   ├── App.tsx                # Main app component
│   │   ├── main.tsx               # Entry point
│   │   └── vite-env.d.ts
│   ├── 📁 tests/                  # Frontend tests
│   │   ├── 📁 components/
│   │   ├── 📁 pages/
│   │   └── setup.ts
│   ├── package.json               # Dependencies
│   ├── vite.config.ts            # Vite configuration
│   ├── tailwind.config.js        # Tailwind configuration
│   ├── tsconfig.json             # TypeScript configuration
│   ├── .env.example              # Environment variables
│   └── README.md                 # Frontend documentation
│
├── 📁 docs/                       # Documentation
│   ├── API.md                     # API documentation
│   ├── DEPLOYMENT.md              # Deployment guide
│   ├── CONTRIBUTING.md            # Contributing guidelines
│   └── ARCHITECTURE.md            # System architecture
│
├── 📁 scripts/                    # Project-wide scripts
│   ├── setup.sh                  # Project setup script
│   ├── deploy.sh                 # Deployment script
│   ├── check_video_encoding.py   # Video encoding checker
│   └── convert_videos.py         # Video converter
│
├── 📁 data/                       # Data files
│   ├── 📁 videos/                 # Video files
│   ├── 📁 models/                 # AI model files
│   └── 📁 samples/                # Sample data
│
├── .gitignore                     # Git ignore rules
├── .env.example                   # Global environment template
├── docker-compose.yml             # Docker composition
├── Dockerfile.backend             # Backend Docker file
├── Dockerfile.frontend            # Frontend Docker file
├── README.md                      # Main project documentation
└── LICENSE                       # Project license
```

## 🎯 Key Organization Principles

### 1. **Separation of Concerns**
- Each directory has a single responsibility
- Business logic separated from API routes
- UI components separated from business logic

### 2. **Modular Structure**
- Features organized into modules
- Reusable components in dedicated folders
- Clear import/export patterns

### 3. **Scalability**
- Easy to add new features
- Clear dependency management
- Testable code structure

### 4. **Developer Experience**
- Consistent naming conventions
- Clear file organization
- Comprehensive documentation

## 📝 Naming Conventions

### Files & Directories
- **kebab-case** for directories: `user-service/`
- **PascalCase** for React components: `Dashboard.tsx`
- **camelCase** for utilities: `apiHelper.ts`
- **snake_case** for Python files: `user_service.py`

### Variables & Functions
- **camelCase** for JavaScript/TypeScript
- **snake_case** for Python
- **PascalCase** for React components and classes
- **UPPER_CASE** for constants

## 🚀 Migration Plan

1. **Phase 1**: Reorganize backend structure
2. **Phase 2**: Reorganize frontend structure  
3. **Phase 3**: Update imports and dependencies
4. **Phase 4**: Update documentation
5. **Phase 5**: Add missing components and tests

## 📋 Benefits

- **Maintainability**: Easy to find and modify code
- **Scalability**: Structure supports growth
- **Team Collaboration**: Clear conventions for multiple developers
- **Testing**: Organized structure makes testing easier
- **Documentation**: Clear structure is self-documenting
