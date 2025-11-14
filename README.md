# Crypto Prediction Monorepo

Production-ready cryptocurrency price prediction system with machine learning capabilities, real-time data processing, and interactive dashboard.

## Overview

A complete end-to-end solution for cryptocurrency price prediction featuring:

- **FastAPI Backend** - RESTful API for data management and model serving
- **PostgreSQL Database** - Persistent storage for historical and real-time market data
- **Redis Cache** - High-performance caching layer for frequently accessed data
- **Machine Learning Pipeline** - Model training, evaluation, and deployment
- **React Dashboard** - Interactive frontend for visualization and monitoring
- **Docker Infrastructure** - Containerized deployment with Docker Compose

## Quick Start

```bash
# Start all services
docker-compose up -d

# Initialize database
docker-compose exec backend alembic upgrade head

# Seed sample data
docker-compose exec backend python scripts/seed_data.py

# Access services
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

For detailed setup instructions, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Repository Structure

```text
/ (root)
├── backend/                   # FastAPI backend service
│   ├── app/
│   │   ├── main.py           # Application entry point
│   │   ├── config.py         # Configuration management
│   │   ├── database.py       # Database connection & ORM
│   │   ├── cache.py          # Redis cache manager
│   │   ├── api/              # API endpoints
│   │   │   └── health.py     # Health check endpoints
│   │   └── models/           # SQLAlchemy models
│   │       ├── cryptocurrency.py
│   │       ├── market_data.py
│   │       └── prediction.py
│   ├── alembic/              # Database migrations
│   ├── scripts/              # Utility scripts
│   │   ├── init-db.sql
│   │   └── seed_data.py
│   ├── tests/                # Test suite
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile
│   └── README.md
│
├── models/                    # ML training & notebooks
│   ├── notebooks/            # Jupyter notebooks for exploration
│   └── src/                  # Model training code
│
├── frontend/                  # React dashboard
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── data/                      # Data storage
│   ├── kaggle-raw/           # Raw datasets
│   └── processed/            # Processed datasets
│
├── infra/                     # Infrastructure as Code
│   └── ...                   # Terraform/K8s manifests
│
├── docs/                      # Documentation
│   ├── QUICKSTART.md         # Quick start guide
│   └── DATABASE_SETUP.md     # Database setup guide
│
├── .github/                   # GitHub workflows & templates
│   └── workflows/
│
├── docker-compose.yml         # Multi-container orchestration
└── README.md
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started in 5 minutes
- [Backend API Documentation](backend/README.md) - Detailed API documentation
- [Database Setup Guide](docs/DATABASE_SETUP.md) - PostgreSQL & Redis configuration

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database operations
- **Alembic** - Database migration management
- **Pydantic** - Data validation and settings
- **Redis-py** - Redis client for Python

### Database
- **PostgreSQL 16** - Primary data store
- **Redis 7** - Caching layer

### Machine Learning
- **PyTorch/TensorFlow** - Deep learning frameworks
- **scikit-learn** - Traditional ML algorithms
- **pandas/numpy** - Data manipulation

### Frontend
- **React** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Chart.js** - Data visualization

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **GitHub Actions** - CI/CD pipelines

## Features

### Current Implementation (SCRUM-2)
✅ PostgreSQL database with optimized schema
✅ Redis caching layer with TTL management
✅ Database migrations with Alembic
✅ Health check endpoints
✅ Connection pooling and error handling
✅ Comprehensive documentation
✅ Docker containerization
✅ Seed data scripts

### Planned Features
- [ ] Real-time data ingestion from CoinGecko/Binance APIs
- [ ] LSTM/Transformer models for price prediction
- [ ] Model training pipeline with MLflow
- [ ] Interactive React dashboard
- [ ] WebSocket support for real-time updates
- [ ] API authentication and rate limiting
- [ ] Automated testing and CI/CD
- [ ] Production deployment on AWS/GCP

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

## Contributing

This is a BSc Computer Engineering final project. Contributions, suggestions, and feedback are welcome!

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Project Status

**Current Sprint:** SCRUM-2 - Database Setup ✅ COMPLETED
- [x] PostgreSQL configuration
- [x] Redis setup
- [x] Database schema design
- [x] Migrations setup
- [x] Health monitoring
- [x] Documentation

**Next Sprint:** SCRUM-3 - Data Ingestion Pipeline
- [ ] API integration (CoinGecko/Binance)
- [ ] Real-time data collection
- [ ] Data validation and cleaning
- [ ] Automated data refresh

## License

[Your License Here]

## Contact

For questions or collaboration:
- GitHub Issues: [Create an issue]
- Email: [your-email@example.com]

## Acknowledgments

Built as a final project for BSc Computer Engineering program.
