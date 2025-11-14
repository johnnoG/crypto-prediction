# Database Setup Guide

Complete guide for setting up PostgreSQL and Redis for the Crypto Prediction system.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [PostgreSQL Setup](#postgresql-setup)
4. [Redis Setup](#redis-setup)
5. [Database Schema](#database-schema)
6. [Cache Strategy](#cache-strategy)
7. [Monitoring](#monitoring)
8. [Backup & Recovery](#backup--recovery)

## Overview

This project uses:
- **PostgreSQL 16** for persistent storage of cryptocurrency data
- **Redis 7** for high-performance caching

Both services are containerized using Docker for easy deployment and management.

## Quick Start

The fastest way to get started is using Docker Compose:

```bash
# From project root
docker-compose up -d postgres redis

# Wait for services to be healthy
docker-compose ps

# Run migrations
docker-compose exec backend alembic upgrade head

# Seed sample data (optional)
docker-compose exec backend python scripts/seed_data.py
```

## PostgreSQL Setup

### Architecture

The PostgreSQL database stores three main types of data:

1. **Cryptocurrency metadata** - Symbol, name, and identifiers
2. **Market data (OHLCV)** - Historical and real-time price data
3. **Predictions** - Model predictions and performance metrics

### Database Configuration

**Connection Settings:**
- Host: `localhost`
- Port: `5432`
- Database: `crypto_prediction_db`
- User: `crypto_user`
- Password: `crypto_pass` (change in production!)

**Performance Settings:**
```ini
# Included in docker-compose.yml
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 128MB
```

### Schema Details

#### 1. Cryptocurrencies Table

**Purpose:** Store metadata about tracked cryptocurrencies

**Key Features:**
- Unique constraint on `symbol`
- Soft-delete capability via `is_active` flag
- Automatic timestamp management

**Example Data:**
```sql
INSERT INTO cryptocurrencies (symbol, name, coin_gecko_id, is_active)
VALUES ('BTC', 'Bitcoin', 'bitcoin', true);
```

#### 2. Market Data Table

**Purpose:** Store time-series OHLCV data

**Key Features:**
- Unique constraint on `(cryptocurrency_id, timestamp)` prevents duplicates
- Check constraints ensure data integrity (prices >= 0, high >= low)
- Partitioning-ready design for large datasets

**Example Data:**
```sql
INSERT INTO market_data (
    cryptocurrency_id, timestamp,
    open_price, high_price, low_price, close_price,
    volume, market_cap, data_source
)
VALUES (
    1, '2024-01-01 00:00:00',
    45000, 45500, 44800, 45200,
    1000000000, 900000000000, 'coingecko'
);
```

**Performance Considerations:**
- Use BIGINT for `id` to handle billions of records
- Indexes on `timestamp` and `(cryptocurrency_id, timestamp)` for fast queries
- Consider partitioning by time range for historical data

#### 3. Predictions Table

**Purpose:** Store model predictions and track accuracy

**Key Features:**
- Links predictions to actual market data for validation
- Stores model metadata for reproducibility
- JSON field for flexible feature storage

**Example Data:**
```sql
INSERT INTO predictions (
    cryptocurrency_id,
    prediction_timestamp, target_timestamp,
    predicted_price, confidence_score,
    model_name, model_version
)
VALUES (
    1,
    '2024-01-01 00:00:00', '2024-01-02 00:00:00',
    46000, 0.85,
    'LSTM-v1', '1.0.0'
);
```

### Data Relationships

```
cryptocurrencies (1) ──< (M) market_data
cryptocurrencies (1) ──< (M) predictions
```

### Indexing Strategy

**Primary Indexes (Auto-created):**
- Primary keys on all tables
- Unique indexes on `cryptocurrencies.symbol` and `coin_gecko_id`
- Unique index on `(cryptocurrency_id, timestamp)` in market_data

**Secondary Indexes:**
```sql
-- Fast lookups by timestamp range
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);

-- Fast filtering by active status
CREATE INDEX idx_crypto_symbol_active ON cryptocurrencies(symbol, is_active);

-- Prediction queries
CREATE INDEX idx_prediction_crypto_target
    ON predictions(cryptocurrency_id, target_timestamp);
```

### Migrations with Alembic

**Initialize Alembic (already done):**
```bash
alembic init alembic
```

**Create a new migration:**
```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "Add new column"

# Or create empty migration
alembic revision -m "Custom migration"
```

**Apply migrations:**
```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade <revision>

# Downgrade one step
alembic downgrade -1
```

**Check current version:**
```bash
alembic current
```

**View migration history:**
```bash
alembic history --verbose
```

### Maintenance

**Regular Tasks:**

1. **Vacuum and Analyze (weekly):**
   ```sql
   VACUUM ANALYZE market_data;
   VACUUM ANALYZE predictions;
   ```

2. **Update Statistics:**
   ```sql
   ANALYZE cryptocurrencies;
   ANALYZE market_data;
   ```

3. **Check Index Health:**
   ```sql
   SELECT schemaname, tablename, indexname, idx_scan
   FROM pg_stat_user_indexes
   WHERE idx_scan = 0 AND indexname NOT LIKE 'pg_%';
   ```

4. **Monitor Table Sizes:**
   ```sql
   SELECT
       schemaname,
       tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables
   WHERE schemaname = 'public'
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   ```

## Redis Setup

### Architecture

Redis is used as a caching layer to:
- Reduce database load for frequently accessed data
- Improve API response times
- Store temporary computation results

### Configuration

**Connection Settings:**
- Host: `localhost`
- Port: `6379`
- Database: `0` (default)

**Memory Settings:**
```conf
maxmemory 256mb
maxmemory-policy allkeys-lru  # Evict least recently used keys
```

**Persistence Settings:**
```conf
save 60 1000              # Save if 1000 keys changed in 60 seconds
appendonly yes            # Enable AOF for durability
```

### Cache Strategy

#### Cache TTL Tiers

1. **Short (5 minutes)** - Frequently updated data
   - Latest cryptocurrency prices
   - Real-time market snapshots
   - Active trading indicators

2. **Medium (30 minutes)** - Moderately stable data
   - Recent predictions
   - Aggregated statistics
   - User session data

3. **Long (1 hour)** - Relatively static data
   - Cryptocurrency metadata
   - Historical data summaries
   - Model configurations

#### Key Naming Convention

Use hierarchical naming with colons:

```
{domain}:{entity}:{id}:{attribute}

Examples:
crypto:BTC:price
crypto:ETH:metadata
market:BTC:ohlcv:latest
prediction:BTC:1h:latest
stats:daily:2024-01-01
```

#### Cache Patterns

**1. Cache-Aside (Lazy Loading):**
```python
def get_crypto_price(symbol: str):
    # Try cache first
    cache_key = f"crypto:{symbol}:price"
    cached = cache_manager.get(cache_key)

    if cached:
        return cached

    # Cache miss - fetch from DB
    price = db.query(MarketData).filter_by(symbol=symbol).first()

    # Store in cache
    cache_manager.set(cache_key, price, ttl=300)  # 5 min

    return price
```

**2. Write-Through:**
```python
def update_crypto_price(symbol: str, price: float):
    # Update database
    db.execute(update(MarketData).values(price=price))
    db.commit()

    # Update cache immediately
    cache_key = f"crypto:{symbol}:price"
    cache_manager.set(cache_key, price, ttl=300)
```

**3. Cache Invalidation:**
```python
def invalidate_crypto_cache(symbol: str):
    # Invalidate all related cache keys
    pattern = f"crypto:{symbol}:*"
    cache_manager.delete_pattern(pattern)
```

### Monitoring Redis

**Check Memory Usage:**
```bash
redis-cli info memory
```

**Monitor Commands:**
```bash
redis-cli monitor
```

**Check Hit Rate:**
```bash
redis-cli info stats | grep keyspace
```

**List Keys by Pattern:**
```bash
redis-cli keys "crypto:*"
```

**Get TTL:**
```bash
redis-cli ttl "crypto:BTC:price"
```

## Monitoring

### Health Checks

**Database Health:**
```bash
curl http://localhost:8000/health/db
```

**Cache Health:**
```bash
curl http://localhost:8000/health/cache
```

**Comprehensive Check:**
```bash
curl http://localhost:8000/health/detailed
```

### Metrics to Monitor

**PostgreSQL:**
- Connection pool usage
- Query execution time
- Table sizes
- Index hit ratio
- Dead tuples count

**Redis:**
- Memory usage
- Eviction rate
- Hit/miss ratio
- Connected clients
- Commands per second

### Alerting Thresholds

**Database:**
- Connection pool > 80% used
- Query time > 1 second
- Dead tuples > 10% of live tuples

**Cache:**
- Memory > 90% used
- Hit rate < 70%
- Evictions > 100/second

## Backup & Recovery

### PostgreSQL Backup

**Automated Backup (recommended):**
```bash
# Add to crontab: daily at 2 AM
0 2 * * * docker exec crypto-prediction-postgres pg_dump \
    -U crypto_user crypto_prediction_db | \
    gzip > /backups/crypto_db_$(date +\%Y\%m\%d).sql.gz
```

**Manual Backup:**
```bash
# Full database
docker exec crypto-prediction-postgres pg_dump \
    -U crypto_user crypto_prediction_db > backup.sql

# Specific table
docker exec crypto-prediction-postgres pg_dump \
    -U crypto_user -t market_data crypto_prediction_db > market_data_backup.sql
```

**Restore:**
```bash
docker exec -i crypto-prediction-postgres psql \
    -U crypto_user crypto_prediction_db < backup.sql
```

### Redis Backup

**Automated Backup:**
Redis automatically creates `dump.rdb` based on save configuration.

**Manual Backup:**
```bash
# Trigger save
redis-cli BGSAVE

# Copy RDB file
docker cp crypto-prediction-redis:/data/dump.rdb ./redis_backup.rdb
```

**Restore:**
```bash
# Stop Redis
docker-compose stop redis

# Replace RDB file
docker cp redis_backup.rdb crypto-prediction-redis:/data/dump.rdb

# Start Redis
docker-compose start redis
```

### Disaster Recovery Plan

1. **Daily automated backups** of PostgreSQL to remote storage
2. **Hourly Redis snapshots** for cache reconstruction
3. **Point-in-time recovery** capability for critical data
4. **Test restore procedures** monthly
5. **Document recovery steps** and RTO/RPO requirements

## Performance Tuning

### Database Optimization

**Connection Pooling:**
```python
# Adjust based on load
pool_size = 20          # Base connections
max_overflow = 10       # Additional connections
pool_timeout = 30       # Wait time for connection
```

**Query Optimization:**
```python
# Use indexes
db.query(MarketData).filter(
    MarketData.cryptocurrency_id == crypto_id,
    MarketData.timestamp >= start_date
).all()

# Eager loading to avoid N+1
db.query(Cryptocurrency).options(
    joinedload(Cryptocurrency.market_data)
).all()

# Pagination for large results
db.query(MarketData).limit(100).offset(0)
```

### Cache Optimization

**Batch Operations:**
```python
# Use pipeline for multiple operations
pipe = redis_client.pipeline()
for key, value in data.items():
    pipe.set(key, value, ex=300)
pipe.execute()
```

**Connection Pooling:**
```python
pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50
)
redis_client = redis.Redis(connection_pool=pool)
```

## Troubleshooting

### Common Issues

**Problem: Connection refused to PostgreSQL**
```bash
# Check if running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify connection
docker exec -it crypto-prediction-postgres psql -U crypto_user -d crypto_prediction_db
```

**Problem: Redis out of memory**
```bash
# Check memory usage
redis-cli info memory

# Check eviction policy
redis-cli config get maxmemory-policy

# Increase memory limit
redis-cli config set maxmemory 512mb
```

**Problem: Slow queries**
```sql
-- Enable query logging
ALTER DATABASE crypto_prediction_db SET log_min_duration_statement = 1000;

-- Check slow queries
SELECT * FROM pg_stat_statements
ORDER BY total_time DESC LIMIT 10;
```

## Best Practices

1. **Use transactions** for multi-step operations
2. **Implement retry logic** for transient failures
3. **Monitor cache hit rates** and adjust TTL accordingly
4. **Regular maintenance** - vacuum, analyze, reindex
5. **Test backups** regularly
6. **Use prepared statements** to prevent SQL injection
7. **Implement rate limiting** on expensive queries
8. **Set up alerting** for critical metrics
9. **Document schema changes** in migrations
10. **Use connection pooling** appropriately

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/16/)
- [Redis Documentation](https://redis.io/docs/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [FastAPI Database Guide](https://fastapi.tiangolo.com/tutorial/sql-databases/)
