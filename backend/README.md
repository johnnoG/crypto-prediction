# Backend README

Crypto Forecast & Real‑Time Dashboard backend built with FastAPI. This document covers what’s implemented, external tools/data sources, and the available API endpoints.

## What’s Implemented

- Real-time crypto pricing + market data with cache-first access and graceful fallbacks.
- Forecasting pipeline (baseline/ARIMA + optional ML models) with caching and backtesting metadata.
- News aggregation (real-time + stored) with sentiment, trending, and source stats.
- Crypto “dashboard” aggregator that pulls from multiple free sources.
- Streaming endpoints (WebSocket + SSE) for live price updates.
- Authentication (JWT) + OAuth (Google) for user accounts.
- User features: alerts and watchlists.
- Health/metrics/monitoring endpoints.
- Rate limit monitoring, batching statistics, and optimization recommendations.
- Admin endpoints for cache control and DB status.

## Tech Stack

- FastAPI + Uvicorn
- SQLAlchemy + Alembic
- Redis (optional) + in-memory fallback
- slowapi (rate limiting)
- Prometheus metrics
- Sentry (error tracking)
- APScheduler (ETL scheduling)
- TensorFlow ≥ 2.15 + LightGBM ≥ 4.1 + scikit-learn ≥ 1.3 (ML inference)
- PyArrow ≥ 14 (parquet feature data reading)

## External Tools / Data Sources

Clients live in `backend/app/clients`:

- CoinGecko API (prices, market data, OHLC)
- CoinCap fallback (used in forecasts when rate-limited)
- CryptoCompare
- Binance
- CryptoPanic
- Firecrawl (article crawling)
- Twitter via Nitter
- RSS/news crawling (CoinDesk, CoinTelegraph, Decrypt, The Block)
- Crypto social aggregators (Reddit + CryptoPanic + RSS)

## Services Overview

Service modules live in `backend/app/services`:

- `prices_service`: prices + market data with cache integration
- `smart_cache_service`: scheduled cache refresh + cached JSON snapshots
- `forecast_service`: baseline/ARIMA forecasting with backtesting metadata
- `ml_forecast_service`: **full ML inference pipeline** — loads all 5 trained model types (DLinear, TCN, LSTM, Transformer, LightGBM) via `models/artifacts/manifest.json`, applies coin-specific RobustScaler preprocessing, runs inference from parquet feature data, and converts log-return predictions back to prices. Uses `importlib.util.spec_from_file_location` to avoid the `backend/app/models/` namespace collision.
- `news_service` / `realtime_news_service`: crawl + real-time aggregation
- `crypto_data_service`: dashboard aggregator for prices/news/sentiment
- `feature_engineering` + `data_aggregator`: OHLCV + technical features
- `etl_scheduler`: job scheduling + status
- `rate_limit_manager` + `request_batcher`: API usage tracking + batching insights
- `health_monitor`: system health + trends
- `prometheus_metrics`: metrics export
- `oauth_service`: OAuth flows
- `sentry_config`: error tracking

### ML Serving Prerequisites

Before ML forecasts are available, run:

```bash
python3 models/src/build_manifest.py
```

This generates:
- `models/artifacts/manifest.json` — maps each coin to its available models and artifact timestamps
- `models/artifacts/preprocessing/{COIN}/feature_names.json` — selected feature list
- `models/artifacts/preprocessing/{COIN}/feature_scaler.joblib` — fitted RobustScaler for features
- `models/artifacts/preprocessing/{COIN}/target_scaler.joblib` — fitted RobustScaler for log-return target

The manifest is re-read on each request, so rerunning `build_manifest.py` after retraining automatically activates new artifacts.

## API Endpoints

Base API prefix is `/api` unless noted. OpenAPI docs live at `/docs` and `/redoc`.

### System & Monitoring (no `/api` prefix)

- `GET /health` — basic health
- `GET /health/quick` — quick health with timestamp
- `GET /health/api` — tests external API connectivity
- `GET /cache/smart` — smart cache status
- `GET /health/detailed` — component health
- `GET /health/trends` — health trends
- `GET /health/components/{component_name}` — component details
- `GET /metrics` — Prometheus metrics

### Auth

- `POST /api/auth/signup`
- `POST /api/auth/signin`
- `POST /api/auth/refresh`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `PUT /api/auth/me`
- `PUT /api/auth/me/password`
- `DELETE /api/auth/me`
- `GET /api/auth/verify-token`
- `GET /api/auth/oauth/{provider}`
- `GET /api/auth/oauth/{provider}/callback`
- `POST /api/auth/oauth/{provider}/mobile`

### Prices & Market

- `GET /api/prices` — price snapshot (cache JSON file)
- `GET /api/prices/market` — market snapshot (cache JSON file)
- `GET /api/prices/history/{coin_id}` — OHLC history
- `GET /api/market/data` — market data (cache-first)
- `GET /api/quick/prices` — emergency cache-only prices
- `GET /api/quick/market` — emergency cache-only market data

### Forecasts

- `GET /api/forecasts` — forecasts for assets
- `GET /api/forecasts/models` — available models
- `GET /api/forecasts/performance` — model performance metadata

### News

- `GET /api/news` — list news (real-time or stored)
- `POST /api/news/refresh` — crawl URL via Firecrawl
- `GET /api/news/sources` — configured sources
- `GET /api/news/trending` — trending topics
- `GET /api/news/sentiment` — market sentiment
- `GET /api/news/stats` — news pipeline stats

### Crypto Aggregation

- `GET /api/crypto/dashboard`
- `GET /api/crypto/news`
- `GET /api/crypto/sentiment`
- `GET /api/crypto/twitter`
- `GET /api/crypto/prices`
- `GET /api/crypto/market-summary`
- `GET /api/crypto/real-social`
- `GET /api/crypto/sources`

### Features & ETL

- `GET /api/features/ohlcv/{symbol}`
- `GET /api/features/price/{symbol}`
- `GET /api/features/prices`
- `GET /api/features/technical/{symbol}`
- `GET /api/features/feature-sets`
- `GET /api/features/etl/jobs`
- `GET /api/features/etl/jobs/{job_id}`
- `POST /api/features/etl/jobs/{job_id}/run`
- `POST /api/features/etl/jobs/{job_id}/enable`
- `POST /api/features/etl/jobs/{job_id}/disable`
- `GET /api/features/etl/status`
- `GET /api/features/data-sources/health`
- `POST /api/features/etl/setup-default`

### Streaming

- `WS /api/stream/ws` — WebSocket updates
- `GET /api/stream/sse` — Server‑Sent Events updates
- `GET /api/stream/snapshot` — one‑shot snapshot

### User Features

- `POST /api/alerts`
- `GET /api/alerts`
- `GET /api/alerts/{alert_id}`
- `DELETE /api/alerts/{alert_id}`
- `GET /api/alerts/crypto/{crypto_symbol}`

- `POST /api/watchlist`
- `GET /api/watchlist`
- `GET /api/watchlist/{watchlist_id}`
- `PUT /api/watchlist/{watchlist_id}`
- `DELETE /api/watchlist/{watchlist_id}`
- `DELETE /api/watchlist/crypto/{crypto_symbol}`

### Admin & Ops

- `GET /api/cache/status`
- `POST /api/cache/clear`
- `GET /api/cache/stats`
- `GET /api/db/status`
- `POST /api/admin/cache/refresh`
- `GET /api/admin/cache/invalidate`
- `GET /api/rate-limit/status`
- `GET /api/rate-limit/status/{api_name}`
- `GET /api/rate-limit/batching/stats`
- `GET /api/rate-limit/recommendations`
- `POST /api/rate-limit/reset-stats`

## Database Models

Key SQLAlchemy models in `backend/app/models`:

- Users + auth
- Watchlists
- Alerts
- Assets + OHLCV
- News sources + news articles

Migrations live in `backend/migrations`.

## Configuration (Env Vars)

Loaded via `backend/app/config.py` (Pydantic settings). Common vars:

- `DATABASE_URL`
- `REDIS_URL`
- `SECRET_KEY`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `REFRESH_TOKEN_EXPIRE_DAYS`
- `COINGECKO_API_KEY`
- `CRYPTOPANIC_API_KEY`
- `FIRECRAWL_API_KEY`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `RATE_LIMIT_*` (see `config.py` for full list)

## Run Locally

From `backend/`:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docker:

```bash
docker build -t crypto-backend .
docker run -p 8000:8000 crypto-backend
```

## Docker Compose (backend service)

The backend container mounts the models directory and sets ML-specific limits:

```yaml
volumes:
  - ./backend:/app
  - ./data:/data
  - ./models:/models      # Required for ML inference
deploy:
  resources:
    limits:
      memory: 4g          # TF model loading requires headroom
environment:
  - TF_CPP_MIN_LOG_LEVEL=2
  - TF_FORCE_CPU_ALLOW_GROWTH=true
```

## Notes

- Prices/market endpoints currently read from JSON cache files in `backend/app/data/cache`.
- Redis is optional; if `REDIS_URL` is not set or unavailable, in-memory cache is used.
- Forecast ML models are optional and only available if `models/artifacts/manifest.json` exists and ML dependencies are installed. If unavailable, the forecast endpoint falls back to ARIMA/statistical models.
- For ML models, the forecast API uses a 5-second CoinGecko fast-fetch timeout and falls back to prices embedded in the parquet feature data if CoinGecko is slow or rate-limited.
- Available ML model types for `GET /api/forecasts?model=<type>`: `lightgbm`, `lstm`, `transformer`, `tcn`, `dlinear`, `ml_ensemble`.
