# Crypto Prediction Backend

Production-ready FastAPI backend service for cryptocurrency prediction system with PostgreSQL and Redis integration.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Database Schema](#database-schema)
- [Setup Instructions](#setup-instructions)
- [Development](#development)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)

## Overview

The backend service provides:
- **RESTful API** for cryptocurrency data management
- **PostgreSQL database** for persistent storage of historical and real-time data
- **Redis cache** for high-performance data access
- **Health monitoring** endpoints for service observability
- **Database migrations** using Alembic
- **Docker support** for containerized deployment

## Architecture

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection & session management
│   ├── cache.py             # Redis cache manager
│   ├── api/                 # API route handlers
│   │   ├── __init__.py
│   │   └── health.py        # Health check endpoints
│   └── models/              # SQLAlchemy ORM models
│       ├── __init__.py
│       ├── cryptocurrency.py
│       ├── market_data.py
│       └── prediction.py
├── alembic/                 # Database migrations
│   ├── versions/
│   └── env.py
├── scripts/                 # Utility scripts
│   ├── init-db.sql
│   └── seed_data.py
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container definition
└── .env.example            # Environment variable template
```

## Database Schema

### Cryptocurrencies Table
Stores metadata about tracked cryptocurrencies.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| symbol | VARCHAR(10) | Trading symbol (BTC, ETH, etc.) |
| name | VARCHAR(100) | Full name |
| coin_gecko_id | VARCHAR(100) | CoinGecko API identifier |
| is_active | BOOLEAN | Active tracking flag |
| created_at | TIMESTAMP | Record creation time |
| updated_at | TIMESTAMP | Last update time |

**Indexes:**
- `idx_crypto_symbol_active` on (symbol, is_active)

### Market Data Table
Stores OHLCV (Open, High, Low, Close, Volume) time series data.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT | Primary key |
| cryptocurrency_id | INTEGER | FK to cryptocurrencies |
| timestamp | TIMESTAMP | Data point timestamp (UTC) |
| open_price | FLOAT | Opening price (USD) |
| high_price | FLOAT | Highest price (USD) |
| low_price | FLOAT | Lowest price (USD) |
| close_price | FLOAT | Closing price (USD) |
| volume | FLOAT | Trading volume |
| market_cap | FLOAT | Market capitalization (USD) |
| data_source | VARCHAR(50) | Data source identifier |
| created_at | TIMESTAMP | Record creation time |

**Indexes:**
- `idx_market_data_crypto_timestamp` (UNIQUE) on (cryptocurrency_id, timestamp)
- `idx_market_data_timestamp` on (timestamp)
- `idx_market_data_crypto_created` on (cryptocurrency_id, created_at)

**Constraints:**
- Price fields must be non-negative
- high_price >= low_price

### Predictions Table
Stores model predictions and performance metrics.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT | Primary key |
| cryptocurrency_id | INTEGER | FK to cryptocurrencies |
| prediction_timestamp | TIMESTAMP | When prediction was made |
| target_timestamp | TIMESTAMP | Future timestamp predicted |
| predicted_price | FLOAT | Predicted price (USD) |
| confidence_score | FLOAT | Model confidence (0-1) |
| model_name | VARCHAR(100) | Model identifier |
| model_version | VARCHAR(50) | Model version |
| features_used | JSON | Features used for prediction |
| actual_price | FLOAT | Actual price (filled later) |
| error | FLOAT | Prediction error percentage |
| created_at | TIMESTAMP | Record creation time |

**Indexes:**
- `idx_prediction_crypto_target` on (cryptocurrency_id, target_timestamp)
- `idx_prediction_crypto_prediction_time` on (cryptocurrency_id, prediction_timestamp)
- `idx_prediction_model` on (model_name, model_version)

**Constraints:**
- target_timestamp > prediction_timestamp
- confidence_score between 0 and 1

## Setup Instructions

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ (or use Docker)
- Redis 7+ (or use Docker)
- Docker & Docker Compose (optional but recommended)

### Option 1: Docker Setup (Recommended)

1. **Clone the repository and navigate to project root:**
   ```bash
   cd /path/to/crypto-prediction
   ```

2. **Create environment file:**
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your configuration
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

5. **Run database migrations:**
   ```bash
   docker-compose exec backend alembic upgrade head
   ```

6. **Seed initial data (optional):**
   ```bash
   docker-compose exec backend python scripts/seed_data.py
   ```

7. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Option 2: Local Development Setup

1. **Create Python virtual environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL:**
   ```bash
   # Using PostgreSQL CLI
   createdb crypto_prediction_db
   createuser crypto_user -P  # Set password when prompted
   psql -d crypto_prediction_db -f scripts/init-db.sql
   ```

4. **Set up Redis:**
   ```bash
   # Install and start Redis (platform-specific)
   redis-server
   ```

5. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your local database/redis URLs
   ```

6. **Run migrations:**
   ```bash
   alembic upgrade head
   ```

7. **Seed data (optional):**
   ```bash
   python scripts/seed_data.py
   ```

8. **Start the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Development

### Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback migration:
```bash
alembic downgrade -1
```

View migration history:
```bash
alembic history
```

### Cache Management

The Redis cache is used for:
- **Short TTL (5 min)**: Latest market prices
- **Medium TTL (30 min)**: Recent predictions
- **Long TTL (1 hour)**: Cryptocurrency metadata

Clear cache:
```python
from app.cache import cache_manager
cache_manager.flush_all()
```

### Code Quality

Format code:
```bash
black app/ --line-length 100
```

Lint code:
```bash
flake8 app/
```

Type checking:
```bash
mypy app/
```

## API Documentation

### Health Endpoints

#### GET /health
Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "service": "Crypto Prediction API",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00"
}
```

#### GET /health/db
Database connectivity check.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "latency_ms": 5.2,
  "timestamp": "2024-01-01T00:00:00"
}
```

#### GET /health/cache
Redis cache health check.

**Response:**
```json
{
  "status": "healthy",
  "connected": true,
  "latency_ms": 1.2,
  "used_memory_mb": 12.5,
  "connected_clients": 3,
  "uptime_seconds": 86400,
  "timestamp": "2024-01-01T00:00:00"
}
```

#### GET /health/detailed
Comprehensive health check of all services.

**Response:**
```json
{
  "status": "healthy",
  "service": {
    "name": "Crypto Prediction API",
    "version": "1.0.0",
    "environment": "production"
  },
  "database": {
    "status": "healthy",
    "connected": true,
    "latency_ms": 5.2
  },
  "cache": {
    "status": "healthy",
    "connected": true,
    "used_memory_mb": 12.5
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

### Quick Start

```bash
# Run all tests
docker-compose exec backend pytest

# Run with coverage
docker-compose exec backend pytest --cov=app --cov-report=html

# Run specific test file
docker-compose exec backend pytest tests/test_health.py

# Run with verbose output
docker-compose exec backend pytest -v

# Run by category
docker-compose exec backend pytest -m unit
docker-compose exec backend pytest -m integration
```

### Test Coverage

Current test coverage: **80%+**

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest --cov=app --cov-report=term-missing
```

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`) - Fast, isolated tests
- **Integration Tests** (`@pytest.mark.integration`) - Tests with dependencies
- **Database Tests** (`@pytest.mark.db`) - Database operations
- **Cache Tests** (`@pytest.mark.cache`) - Redis cache operations

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures and configuration
├── README.md             # Test suite documentation
│
├── unit/                 # Unit tests - fast, isolated
│   ├── conftest.py      # Auto-applies @unit marker
│   ├── test_models.py   # Database model tests (30+ tests)
│   ├── test_cache.py    # Cache manager tests (20+ tests)
│   └── test_config.py   # Configuration tests (15+ tests)
│
└── integration/          # Integration tests - with dependencies
    ├── conftest.py      # Auto-applies @integration marker
    └── test_health.py   # Health endpoint tests (15+ tests)
```

### CI/CD

Tests run automatically on every push via GitHub Actions:
- Backend tests with PostgreSQL and Redis
- Code quality checks (Black, Flake8, mypy)
- Database migration verification
- Security scanning
- Docker build verification
- Coverage reporting

See [docs/TESTING.md](../docs/TESTING.md) for comprehensive testing guide

## Deployment

### Environment Variables

Required environment variables for production:

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://host:6379/0
REDIS_MAX_CONNECTIONS=50

# Application
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=<generate-secure-random-string>

# API
CORS_ORIGINS=https://yourdomain.com
```

### Docker Deployment

1. **Build production image:**
   ```bash
   docker build -t crypto-prediction-backend:latest -f backend/Dockerfile backend/
   ```

2. **Run container:**
   ```bash
   docker run -d \
     -p 8000:8000 \
     --env-file backend/.env \
     --name crypto-backend \
     crypto-prediction-backend:latest
   ```

### Production Checklist

- [ ] Set `DEBUG=false` in environment
- [ ] Generate secure `SECRET_KEY`
- [ ] Configure proper `CORS_ORIGINS`
- [ ] Set up database connection pooling
- [ ] Configure Redis maxmemory and eviction policy
- [ ] Enable SSL/TLS for database connections
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Implement rate limiting
- [ ] Set up automated backups
- [ ] Configure health check monitoring

## Performance Tuning

### Database Optimization

1. **Connection Pooling:**
   - Default pool size: 20
   - Adjust based on concurrent load
   - Monitor connection usage

2. **Indexing:**
   - All foreign keys are indexed
   - Composite indexes for common queries
   - Regular VACUUM and ANALYZE

3. **Query Optimization:**
   - Use `.options()` for eager loading
   - Implement pagination for large datasets
   - Consider materialized views for analytics

### Cache Strategy

1. **Cache Patterns:**
   - Cache-aside for read-heavy operations
   - Write-through for critical data
   - TTL-based expiration

2. **Cache Keys:**
   - Use namespacing (e.g., `crypto:BTC:price`)
   - Implement cache warming for popular data
   - Monitor hit/miss ratios

## Troubleshooting

### Common Issues

**Database connection errors:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection
psql -h localhost -U crypto_user -d crypto_prediction_db
```

**Redis connection errors:**
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli ping
```

**Migration issues:**
```bash
# Check current version
alembic current

# View migration history
alembic history

# Reset to specific version
alembic downgrade <revision>
```

## Contributing

1. Create feature branch from `main`
2. Write tests for new features
3. Ensure all tests pass
4. Format code with Black
5. Submit pull request

## License

[Your License Here]

## Support

For issues and questions:
- Create an issue on GitHub
- Contact: [your-email@example.com]
