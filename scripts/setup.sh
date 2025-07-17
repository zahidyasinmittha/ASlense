# Setup Script for ASLense Project
#!/bin/bash

echo "🚀 Setting up ASLense Development Environment"
echo "=============================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed."
    exit 1
fi
echo "✅ npm found: $(npm --version)"

echo ""
echo "🔧 Setting up Backend..."
echo "========================"

# Backend setup
cd backend

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit backend/.env with your configuration"
fi

# Initialize database
echo "🗃️ Setting up database..."
alembic upgrade head

echo ""
echo "🎨 Setting up Frontend..."
echo "========================="

# Frontend setup
cd ../frontend

# Install dependencies
echo "📥 Installing Node.js dependencies..."
npm install

# Setup environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit frontend/.env with your configuration"
fi

# Build frontend for development
echo "🔨 Building frontend..."
npm run build

echo ""
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "🎯 Next Steps:"
echo "1. Edit backend/.env and frontend/.env files"
echo "2. Start backend: cd backend && source .venv/bin/activate && uvicorn app.main:app --reload"
echo "3. Start frontend: cd frontend && npm run dev"
echo "4. Visit http://localhost:5173 to see the application"
echo ""
echo "📚 For more information, see README.md"
echo "🐛 Report issues at: https://github.com/zahidyasinmittha/ASlense/issues"
