# Contributing to ASLense

Thank you for your interest in contributing to ASLense! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions
- **Bug Reports** - Help us identify and fix issues
- **Feature Requests** - Suggest new functionality
- **Code Contributions** - Submit pull requests
- **Documentation** - Improve guides and API docs
- **Testing** - Help test new features and report issues

### Getting Started
1. Fork the repository
2. Clone your fork locally
3. Set up development environment (see README.md)
4. Create a feature branch
5. Make your changes
6. Test your changes
7. Submit a pull request

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Setup Commands
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ASlense.git
cd ASlense

# Run setup script
./scripts/setup.sh

# Or manual setup:
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd ../frontend
npm install
```

## 📝 Code Standards

### Python (Backend)
- **Style**: Follow PEP 8
- **Formatting**: Use Black for code formatting
- **Imports**: Use isort for import sorting
- **Type Hints**: Use type hints for function parameters and returns
- **Docstrings**: Use Google-style docstrings

Example:
```python
def predict_sign(
    video_path: str, 
    model_type: str = "mini"
) -> List[Dict[str, Any]]:
    """Predict ASL sign from video.
    
    Args:
        video_path: Path to the video file.
        model_type: Type of model to use ('mini' or 'pro').
        
    Returns:
        List of predictions with confidence scores.
        
    Raises:
        ValueError: If video_path is invalid.
    """
    # Implementation here
    pass
```

### TypeScript (Frontend)
- **Style**: Follow TypeScript best practices
- **Formatting**: Use Prettier
- **Components**: Use functional components with hooks
- **Props**: Define proper TypeScript interfaces
- **Naming**: Use PascalCase for components, camelCase for functions

Example:
```typescript
interface PredictionProps {
  targetWord: string;
  modelType: 'mini' | 'pro';
  onResult: (result: PredictionResult) => void;
}

const PredictionComponent: React.FC<PredictionProps> = ({
  targetWord,
  modelType,
  onResult
}) => {
  // Component implementation
  return <div>...</div>;
};
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Test Requirements
- **Unit Tests**: All new functions should have unit tests
- **Integration Tests**: API endpoints should have integration tests
- **E2E Tests**: Critical user flows should have end-to-end tests
- **Coverage**: Maintain >80% test coverage

## 🔄 Pull Request Process

### Before Submitting
1. **Run Tests**: Ensure all tests pass
2. **Code Quality**: Run linting and formatting tools
3. **Documentation**: Update relevant documentation
4. **Changelog**: Add entry to CHANGELOG.md if applicable

### PR Guidelines
- **Title**: Use clear, descriptive titles
- **Description**: Explain what changes were made and why
- **Screenshots**: Include screenshots for UI changes
- **Breaking Changes**: Clearly mark any breaking changes
- **Linked Issues**: Reference related issues

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

## Screenshots
(If applicable)

## Related Issues
Fixes #(issue number)
```

## 🐛 Bug Reports

### Bug Report Template
```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10, macOS 12]
- Browser: [e.g. Chrome 91, Firefox 89]
- Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 📁 Project Structure

When adding new features, follow the established project structure:

### Backend
- **Models**: Add to `app/models/`
- **Schemas**: Add to `app/schemas/`
- **Services**: Add to `app/services/`
- **API Routes**: Add to `app/api/v1/endpoints/`
- **Tests**: Add to `tests/`

### Frontend
- **Components**: Add to `src/components/`
- **Pages**: Add to `src/pages/`
- **Hooks**: Add to `src/hooks/`
- **Services**: Add to `src/services/`
- **Types**: Add to `src/types/`

## 🏷️ Commit Messages

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(auth): add JWT token refresh mechanism
fix(prediction): resolve video upload timeout issue
docs(api): update prediction endpoint documentation
```

## 🌟 Recognition

Contributors will be recognized in:
- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub Releases** - Release descriptions

## 📞 Getting Help

- **Discord**: Join our development Discord
- **Issues**: Open GitHub issues for questions
- **Discussions**: Use GitHub Discussions for broader topics
- **Email**: Contact maintainers directly

## 📋 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive experience for everyone.

### Our Standards
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement
Instances of abusive behavior may be reported to the project maintainers.

## 🙏 Thank You

Thank you for contributing to ASLense! Your efforts help make ASL learning more accessible to everyone.

---

For questions about this guide, please open an issue or reach out to the maintainers.
