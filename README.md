# Crypto Prediction & Real-Time Dashboard

Production-ready cryptocurrency price prediction system with machine learning capabilities, real-time data processing, user authentication, and interactive dashboard.

## Overview

A complete end-to-end solution for cryptocurrency price prediction featuring:

- **Advanced Data Analysis** - Comprehensive analysis of 104 cryptocurrencies with 150+ engineered features
- **FastAPI Backend** - RESTful API with authentication, real-time crypto data, and ML forecasting
- **PostgreSQL Database** - Persistent storage for user data, market data, and predictions
- **Redis Cache** - High-performance caching layer for frequently accessed crypto prices
- **React Dashboard** - Interactive frontend with real-time charts and user authentication
- **Machine Learning Pipeline** - LSTM/Transformer models for price prediction with MLflow tracking
- **User Authentication** - Secure signup/signin with Argon2 password hashing and JWT tokens
- **Real-time Crypto Data** - Live price feeds from CoinGecko with rate limiting and caching
- **Docker Infrastructure** - Fully containerized deployment with Docker Compose

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd crypto-prediction

# Start all services (backend, frontend, database, cache)
docker-compose up -d

# Database is automatically initialized with migrations
# Access the application:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

The application will be ready in about 1-2 minutes. You can create an account and start exploring real-time crypto prices immediately!

## Repository Structure

```text
/ (root)
├── backend/                   # FastAPI backend service
│   ├── app/
│   │   ├── main.py           # Application entry point
│   │   ├── config.py         # Configuration management
│   │   ├── db.py             # Database connection & ORM
│   │   ├── cache.py          # Redis cache manager
│   │   ├── api/              # API endpoints
│   │   ├── clients/          # External API clients
│   │   ├── services/         # Forecasting, ETL, caching, health
│   │   ├── models/           # SQLAlchemy models
│   │   └── utils/            # Connection pooling, circuit breaker
│   ├── migrations/           # Database migrations
│   ├── tests/                # Test suite (subset)
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile
│   └── README.md
│
├── data_analysis/             # Comprehensive crypto analysis
│   ├── crypto_data_analyzer.py     # Main data analysis engine
│   ├── feature_engineering.py  # 150+ technical indicators
│   └── statistical_analysis.py     # Advanced statistical analysis
│
├── data/                      # Data storage & processing
│   ├── features/             # Engineered feature datasets (104 cryptos)
│   ├── processed/            # Cleaned and processed data
│   ├── kaggle-raw/           # Raw Kaggle cryptocurrency dataset
│   ├── raw/demo/             # Sample parquet data
│   └── sources/              # Data source configs and tests
│
├── notebooks/                 # Interactive analysis notebooks
│   └── comprehensive_crypto_analysis.ipynb  # Main analysis notebook
│
├── analysis_results/         # Generated analysis outputs
│   ├── *.xlsx               # Statistical summaries and correlations
│   ├── *.csv                # Risk metrics and cluster assignments
│   ├── *.png                # Professional visualizations
│   └── *.json               # Analysis metadata
│
├── models/                    # ML training, configs, artifacts
│   ├── artifacts/            # Stored model binaries
│   ├── configs/              # Model configuration
│   └── src/                  # Model training code
│
├── frontend/                  # React dashboard
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── docs/                      # Documentation
│   └── reference/            # Reference docs and summaries
│
├── assets/                    # Binary assets
├── tests/                     # Root-level tests
│
├── .github/                   # GitHub workflows & templates
│   └── workflows/
│
├── docker-compose.yml         # Multi-container orchestration
└── README.md
```

## Documentation

- [Analysis Documentation](ANALYSIS_DOCUMENTATION.md) - Complete guide to understanding all analysis outputs and metrics
- [Quick Reference](docs/reference/QUICK_REFERENCE.md) - Getting started and key commands
- [Backend API Documentation](backend/README.md) - Detailed API documentation
- [Implementation Summary](docs/reference/IMPLEMENTATION_SUMMARY.md) - Architecture and feature overview

## Technology Stack

### Backend

- **FastAPI** - Modern Python web framework with automatic API documentation
- **SQLAlchemy** - ORM for database operations with async support
- **Alembic** - Database migration management
- **Pydantic** - Data validation and settings management
- **Argon2** - Secure password hashing (industry standard)
- **PyJWT** - JWT token generation and validation
- **Redis-py** - Redis client for caching and rate limiting
- **Slowapi** - Rate limiting middleware for FastAPI
- **HTTPX** - Modern HTTP client for external API calls

### Database & Caching

- **PostgreSQL 16** - Primary data store with Alpine Linux base
- **Redis 7** - High-performance caching and session storage

### Data Processing & ML

- **NumPy & Pandas** - Data manipulation and analysis
- **Polars** - Fast DataFrame library for large datasets
- **Statsmodels** - Statistical modeling and time series analysis
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **TensorFlow/Keras** - Deep learning models (LSTM/Transformer)
- **LightGBM/XGBoost** - Gradient boosting for ensemble models
- **MLflow** - Experiment tracking and model versioning
- **SHAP** - Model interpretability and feature importance
- **Plotly & Seaborn** - Advanced visualizations and statistical plots
- **TA-Lib** - Technical analysis indicators library
- **Feedparser** - RSS feed parsing for crypto news
- **BeautifulSoup4** - Web scraping for additional data sources

### Frontend

- **React 19** - Modern UI framework with concurrent features
- **TypeScript** - Type-safe JavaScript development
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Query** - Powerful data synchronization for React
- **Lightweight Charts** - Performant financial charting library
- **Lucide React** - Beautiful icon library

### DevOps & Infrastructure

- **Docker** - Containerization with multi-stage builds
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Web server and reverse proxy for frontend
- **GitHub Actions** - CI/CD pipelines

## Features

### ✅ Implemented Features

**Phase 1: Comprehensive Data Analysis (COMPLETED)**

- Complete analysis of 104 cryptocurrencies spanning 15+ years (2010-2026)
- Advanced feature engineering with 150+ technical indicators across 6 categories
- Statistical analysis including correlation matrices, PCA, and clustering
- Risk-return analysis with comprehensive metrics (Sharpe, Sortino, Calmar ratios)
- Time series decomposition and trend analysis
- Professional visualizations and interactive plots
- Comprehensive documentation for presentation and analysis interpretation

**Backend Infrastructure:**

- PostgreSQL database with optimized schema and migrations
- Redis caching layer with TTL management and rate limiting
- FastAPI server with comprehensive API endpoints
- Health check endpoints with database and cache monitoring
- Docker containerization with multi-stage builds

**Authentication & Security:**

- User registration and login with secure Argon2 password hashing
- JWT token-based authentication with access and refresh tokens
- Support for strong passwords from password managers (256+ characters)
- Secure session management and token validation

**Real-time Crypto Data:**

- Live cryptocurrency price feeds from CoinGecko API
- Intelligent rate limiting and caching to respect API limits
- Support for 30+ major cryptocurrencies in the UI (BTC, ETH, SOL, etc.)
- Error handling and graceful degradation

**Frontend Application:**

- React 19 dashboard with TypeScript and Tailwind CSS
- User authentication flow with signup/signin forms
- Real-time crypto price displays and charts
- Responsive design with modern UI components
- Integration with backend API using React Query
- Advanced charting and technical indicators in the UI

### 🚧 Phase 2: Machine Learning Pipeline (IN PROGRESS)

- [x] Feature engineering pipeline with 150+ indicators
- [x] Statistical analysis and data preprocessing
- [ ] LSTM/Transformer model implementation
- [ ] MLflow integration for experiment tracking
- [ ] Model training and validation pipeline
- [ ] Production model serving and inference

### 🚧 Additional In Progress Features

- [ ] WebSocket streaming integrated into UI (backend WS/SSE exists; hook present)
- [ ] Portfolio tracking and management (backend integration pending)
- [ ] Advanced analytics dashboard with real-time insights

### 📋 Planned Features

- [ ] Ensemble model combining multiple architectures
- [ ] Advanced backtesting and strategy evaluation
- [ ] Real-time model monitoring and alerting
- [ ] Mobile-responsive PWA capabilities
- [ ] Automated testing and CI/CD pipelines
- [ ] Production deployment on cloud platforms

## Database Schema

### Cryptocurrencies

Stores metadata about tracked cryptocurrencies (BTC, ETH, etc.)

### Market Data

OHLCV (Open, High, Low, Close, Volume) time-series data with indexes for performance.

### Predictions

Model predictions with confidence scores, actual prices, and error metrics.

For detailed schema documentation, see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md).

## Development

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+ (for frontend)

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd crypto-prediction

# Start services
docker-compose up -d

# Backend development
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend development
cd frontend
npm install
npm start
```

### Running Tests

```bash
# Backend tests
docker-compose exec backend pytest

# With coverage
docker-compose exec backend pytest --cov=app --cov-report=html
```

### Database Migrations

```bash
# Create migration
docker-compose exec backend alembic revision --autogenerate -m "Description"

# Apply migrations
docker-compose exec backend alembic upgrade head

# Rollback
docker-compose exec backend alembic downgrade -1
```

## API Endpoints

### Health Checks

- `GET /health` - Basic health check
- `GET /health/db` - Database connectivity
- `GET /health/cache` - Redis cache status
- `GET /health/detailed` - Comprehensive system health

### Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Status

**✅ PHASE 1 COMPLETED - Comprehensive Data Analysis**

Phase 1 is fully completed with:

- [x] Complete analysis of 104 cryptocurrencies (2010-2026)
- [x] Advanced feature engineering with 150+ technical indicators
- [x] Statistical analysis including correlation matrices and PCA
- [x] Risk-return analysis with professional visualizations
- [x] Time series decomposition and trend analysis
- [x] Comprehensive documentation for presentation
- [x] Interactive Jupyter notebook for exploration

**✅ INFRASTRUCTURE COMPLETED - Production-Ready Application**

The application infrastructure is fully functional with:

- [x] Complete Docker infrastructure with 4-service stack
- [x] PostgreSQL database with user authentication tables
- [x] Redis caching for performance optimization
- [x] FastAPI backend with comprehensive API endpoints
- [x] React frontend with modern UI and real-time updates
- [x] User authentication system with Argon2 security
- [x] Live cryptocurrency price feeds from CoinGecko
- [x] Rate limiting and error handling
- [x] Database migrations and health monitoring

**🚧 PHASE 2 IN PROGRESS - Machine Learning Pipeline**

- [ ] LSTM/Transformer model implementation
- [ ] MLflow integration for experiment tracking
- [ ] Production training pipeline with automation
- [ ] Advanced analytics dashboard
- [ ] Model performance monitoring and evaluation

**📋 FUTURE PHASES - Advanced Features**

- [ ] Real-time prediction serving
- [ ] Advanced portfolio optimization
- [ ] WebSocket streaming for live updates
- [ ] Mobile PWA capabilities
- [ ] Production deployment automation

## Acknowledgments

Built as a final project for BSc Computer Engineering program.
