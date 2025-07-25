# ASLense Frontend

A modern React TypeScript application for ASL (American Sign Language) learning and practice with AI-powered real-time sign recognition.

## 🚀 Overview

The ASLense frontend is built with React 18 and TypeScript, providing an intuitive and interactive interface for:
- ASL learning through video lessons
- Real-time sign language practice with AI feedback
- Translation between ASL and text
- User progress tracking and analytics
- Contact form with admin management
- Responsive design for desktop and mobile

## 🛠️ Technology Stack

### Core Framework
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe JavaScript development
- **Vite** - Fast build tool and development server

### Styling & UI
- **Tailwind CSS** - Utility-first CSS framework
- **CSS Modules** - Scoped styling when needed
- **Responsive Design** - Mobile-first approach

### State Management
- **React Context** - Global state management
- **useReducer** - Complex state logic
- **Custom Hooks** - Reusable stateful logic

### API & Communication
- **Axios** - HTTP client for API calls
- **WebSocket** - Real-time communication for live predictions
- **Custom API Service** - Centralized API management

### Development Tools
- **ESLint** - Code linting and quality
- **Prettier** - Code formatting
- **Vitest** - Unit testing framework
- **React Testing Library** - Component testing

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/             # Base UI components
│   │   ├── forms/          # Form components
│   │   ├── navigation/     # Navigation components
│   │   └── charts/         # Chart/visualization components
│   │
│   ├── pages/              # Page components
│   │   ├── Home.tsx        # Landing page
│   │   ├── Login.tsx       # Authentication
│   │   ├── Dashboard.tsx   # User dashboard
│   │   ├── Learn.tsx       # Learning interface
│   │   ├── Practice.tsx    # Practice mode
│   │   ├── Contact.tsx     # Contact form
│   │   └── AdminDashboard.tsx # Admin panel
│   │
│   ├── hooks/              # Custom React hooks
│   │   ├── useAuth.tsx     # Authentication logic
│   │   ├── useApi.tsx      # API interaction
│   │   └── useWebSocket.tsx # WebSocket management
│   │
│   ├── contexts/           # React context providers
│   │   ├── AuthContext.tsx # Authentication state
│   │   └── ThemeContext.tsx # Theme management
│   │
│   ├── services/           # API and external services
│   │   ├── api.ts          # Main API client
│   │   ├── auth.ts         # Authentication API
│   │   ├── contact.ts      # Contact form API
│   │   └── websocket.ts    # WebSocket utilities
│   │
│   ├── types/              # TypeScript definitions
│   │   ├── auth.ts         # Authentication types
│   │   ├── user.ts         # User data types
│   │   ├── contact.ts      # Contact form types
│   │   └── api.ts          # API response types
│   │
│   └── utils/              # Utility functions
│       ├── formatting.ts   # Data formatting
│       ├── validation.ts   # Form validation
│       └── constants.ts    # App constants
│
├── public/                 # Static assets
│   ├── img/               # Images
│   ├── thumbnails/        # Video thumbnails
│   └── favicon.ico        # Favicon
│
├── docs/                  # Documentation
│   └── API_SERVICE.md     # API service guide
│
└── config files          # Configuration
    ├── package.json       # Dependencies
    ├── vite.config.ts     # Build configuration
    ├── tailwind.config.js # CSS framework
    ├── tsconfig.json      # TypeScript config
    └── eslint.config.js   # Linting rules
```

## 🚦 Getting Started

### Prerequisites
- **Node.js** 18+ (LTS recommended)
- **npm** or **yarn** package manager
- **Backend API** running on `http://localhost:8000`

### Installation

1. **Clone and navigate to frontend**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
# or
yarn install
```

3. **Environment setup**:
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
# VITE_BACKEND_BASEURL=http://localhost:8000
# VITE_WS_URL=ws://localhost:8000
```

4. **Start development server**:
```bash
npm run dev
# or
yarn dev
```

5. **Open in browser**:
Navigate to `http://localhost:5173`

## 🔧 Development Scripts

```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run type checking
npm run type-check

# Run linting
npm run lint

# Run linting with auto-fix
npm run lint:fix

# Run tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## 🏗️ Key Features Implementation

### Authentication System
- JWT-based authentication with automatic token refresh
- Protected routes with role-based access (user/admin)
- Persistent login state with localStorage
- Automatic logout on token expiration

### Real-time ASL Practice
- Live camera integration for sign practice
- WebSocket connection for real-time AI predictions
- Video upload and processing for practice sessions
- Progress tracking and performance analytics

### Contact System
- Contact form with validation and error handling
- Email notifications sent to admin
- Admin dashboard for contact management
- Message status tracking (unread/read/replied)

### Responsive Design
- Mobile-first responsive layout
- Touch-friendly interface for mobile devices
- Optimized performance across devices
- Accessible design following WCAG guidelines

## 🔌 API Integration

### Environment Configuration
```typescript
// .env file
VITE_BACKEND_BASEURL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### API Service Usage
```typescript
import { authAPI, contactAPI, practiceAPI } from './services/api';

// Authentication
const loginResponse = await authAPI.login({
  username: 'user@example.com',
  password: 'password123'
});

// Contact form submission
const contactResponse = await contactAPI.submitContact({
  name: 'John Doe',
  email: 'john@example.com',
  subject: 'Feature Request',
  message: 'I would like to suggest...'
});

// Practice video prediction
const formData = new FormData();
formData.append('video', videoFile);
formData.append('target_word', 'hello');
const predictionResponse = await practiceAPI.predictVideo(formData);
```

### WebSocket Integration
```typescript
import { useWebSocket } from './hooks/useWebSocket';

function LivePractice() {
  const { ws, isConnected, sendMessage } = useWebSocket(
    'ws://localhost:8000/practice/live-predict'
  );

  const sendFrame = (frameData: string) => {
    sendMessage({
      type: 'frame',
      frame: frameData,
      target_word: 'hello',
      model_type: 'mini'
    });
  };

  return (
    <div>
      {isConnected ? 'Connected' : 'Connecting...'}
    </div>
  );
}
```

## 🎨 Styling Guidelines

### Tailwind CSS Classes
```jsx
// Button styles
<button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
  Submit
</button>

// Card layout
<div className="bg-white shadow-lg rounded-lg p-6 mb-4">
  <h2 className="text-xl font-bold text-gray-800 mb-2">Card Title</h2>
  <p className="text-gray-600">Card content...</p>
</div>

// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Grid items */}
</div>
```

### Color Palette
- **Primary**: Blue tones (`blue-600`, `blue-700`)
- **Secondary**: Gray tones (`gray-600`, `gray-800`)
- **Success**: Green (`green-600`)
- **Warning**: Yellow (`yellow-600`)
- **Error**: Red (`red-600`)

## 🧪 Testing

### Component Testing
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { Contact } from '../pages/Contact';

test('contact form submission', async () => {
  render(<Contact />);
  
  fireEvent.change(screen.getByLabelText(/name/i), {
    target: { value: 'John Doe' }
  });
  
  fireEvent.click(screen.getByRole('button', { name: /submit/i }));
  
  expect(screen.getByText(/message sent/i)).toBeInTheDocument();
});
```

### API Testing
```typescript
import { contactAPI } from '../services/api';

test('contact API submission', async () => {
  const contactData = {
    name: 'Test User',
    email: 'test@example.com',
    subject: 'Test Subject',
    message: 'Test message'
  };
  
  const response = await contactAPI.submitContact(contactData);
  
  expect(response.status).toBe(200);
  expect(response.data.name).toBe('Test User');
});
```

## 🚀 Deployment

### Production Build
```bash
# Build for production
npm run build

# Files will be generated in dist/ folder
ls dist/
```

### Environment Variables for Production
```env
VITE_BACKEND_BASEURL=https://your-api-domain.com
VITE_WS_URL=wss://your-api-domain.com
```

### Deployment Options
- **Netlify**: Automatic deployment from Git
- **Vercel**: Zero-config deployment
- **AWS S3 + CloudFront**: Static hosting with CDN
- **Docker**: Containerized deployment

## 🔧 Performance Optimization

### Code Splitting
```typescript
// Lazy loading for routes
const Dashboard = lazy(() => import('./pages/Dashboard'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));

// Usage with Suspense
<Suspense fallback={<Loading />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/admin" element={<AdminDashboard />} />
  </Routes>
</Suspense>
```

### Bundle Analysis
```bash
# Analyze bundle size
npm run build
npx vite-bundle-analyzer dist/
```

## 🤝 Contributing

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with TypeScript
3. Add tests for new functionality
4. Run linting and type checking
5. Submit pull request with description

### Code Standards
- Use TypeScript for all new code
- Follow React hooks patterns
- Implement proper error handling
- Add unit tests for components
- Use semantic HTML and accessibility features

## 📚 Additional Resources

- **React Documentation**: https://react.dev/
- **TypeScript Handbook**: https://www.typescriptlang.org/docs/
- **Tailwind CSS Docs**: https://tailwindcss.com/docs
- **Vite Guide**: https://vitejs.dev/guide/
- **Testing Library**: https://testing-library.com/docs/react-testing-library/intro/

## 🐛 Troubleshooting

### Common Issues

**CORS Errors**:
```bash
# Ensure backend is running with proper CORS configuration
# Check VITE_BACKEND_BASEURL in .env file
```

**WebSocket Connection Issues**:
```bash
# Verify WebSocket URL in .env
# Check if backend WebSocket endpoints are accessible
```

**Build Errors**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Type Errors**:
```bash
# Run type checking
npm run type-check

# Update TypeScript definitions
npm update @types/*
```

## 📞 Support

For technical support or questions:
- Create an issue in the repository
- Contact the development team
- Check existing documentation in `/docs`

---

Built with ❤️ using React, TypeScript, and modern web technologies.
