# Quick Start Guide

Get the Crypto Prediction system up and running in 5 minutes.

## Prerequisites

- Docker & Docker Compose installed
- Git installed
- 4GB RAM available
- Ports 3000, 5432, 6379, 8000 available

## Step 1: Clone and Configure

```bash
# Clone repository (if not already done)
cd /path/to/crypto-prediction

# Create environment file
cp backend/.env.example backend/.env

# (Optional) Edit .env with your preferred settings
# For development, the defaults work fine
nano backend/.env
```

## Step 2: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# This will start:
# - PostgreSQL (port 5432)
# - Redis (port 6379)
# - Backend API (port 8000)
# - Frontend (port 3000)
```

## Step 3: Initialize Database

```bash
# Wait for services to be healthy (about 30 seconds)
docker-compose ps

# Run database migrations
docker-compose exec backend alembic upgrade head

# Seed sample data (optional but recommended)
docker-compose exec backend python scripts/seed_data.py
```

## Step 4: Verify Installation

```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Test database connectivity
curl http://localhost:8000/health/db

# Test Redis cache
curl http://localhost:8000/health/cache

# Comprehensive health check
curl http://localhost:8000/health/detailed
```

## Step 5: Access Services

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend Dashboard**: http://localhost:3000
- **PostgreSQL**: localhost:5432 (user: crypto_user, db: crypto_prediction_db)
- **Redis**: localhost:6379

## What's Next?

### Explore the API

Visit http://localhost:8000/docs to see interactive API documentation with Swagger UI.

### View Sample Data

```bash
# Connect to database
docker exec -it crypto-prediction-postgres psql -U crypto_user -d crypto_prediction_db

# View cryptocurrencies
SELECT * FROM cryptocurrencies;

# View recent market data
SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 10;

# Exit psql
\q
```

### Check Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Stop Services

```bash
# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (WARNING: deletes data)
docker-compose down -v
```

## Common Issues

### Port Already in Use

If you see "port is already allocated" errors:

```bash
# Check what's using the port
lsof -i :8000  # Replace with the problematic port

# Either stop that service or change the port in docker-compose.yml
```

### Services Not Starting

```bash
# Check service logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Rebuild if needed
docker-compose up -d --build
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker exec -it crypto-prediction-postgres psql -U crypto_user -d crypto_prediction_db
```

### Cache Connection Issues

```bash
# Verify Redis is running
docker-compose ps redis

# Check Redis logs
docker-compose logs redis

# Test connection
docker exec -it crypto-prediction-redis redis-cli ping
```

## Development Workflow

### Making Code Changes

The backend code is mounted as a volume, so changes are reflected immediately with hot-reload:

```bash
# Edit files in backend/app/
# The server will automatically reload
```

### Running Migrations

```bash
# Create new migration after model changes
docker-compose exec backend alembic revision --autogenerate -m "Description"

# Apply migration
docker-compose exec backend alembic upgrade head
```

### Accessing Services Directly

```bash
# Backend shell
docker-compose exec backend bash

# PostgreSQL CLI
docker exec -it crypto-prediction-postgres psql -U crypto_user -d crypto_prediction_db

# Redis CLI
docker exec -it crypto-prediction-redis redis-cli
```

## Next Steps

1. Read [DATABASE_SETUP.md](./DATABASE_SETUP.md) for detailed database documentation
2. Read [backend/README.md](../backend/README.md) for API details
3. Start building your ML models in the `models/` directory
4. Integrate with external data sources (CoinGecko, Binance, etc.)
5. Develop your frontend dashboard

## Need Help?

- Check the [main README](../README.md) for project overview
- Review [backend documentation](../backend/README.md)
- Check [database setup guide](./DATABASE_SETUP.md)
- Create an issue on GitHub
