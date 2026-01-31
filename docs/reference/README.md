## Crypto Forecast & Real‑Time Dashboard

A clean, professional crypto web app to:

- Forecast crypto prices
- Show real‑time market prices from reliable APIs
  - Use real CoinGecko data and cache responses; serve cached data when rate‑limited or offline
- Aggregate crypto news via MCP Firecrawl and social sentiment
- Provide a modern UI inspired by Coinbase

This README is the source of truth for goals, requirements, and next steps.

---

### Product Goals

- **Accurate forecasting**: Start with strong baselines; graduate to ML/Deep Learning when it clearly adds value
- **Real‑time pricing**: Low‑latency streaming where available; resilient fallbacks via cached snapshots
- **News intelligence**: Crawl, normalize, de‑duplicate and classify crypto news (MCP/Firecrawl + sentiment + topics)
- **Delightful UX**: Coinbase‑like polish, accessible, responsive, dark mode by default

### Non‑Functional Requirements

- **Data integrity**: Prefer official/established sources; avoid mock data in production
- **Resilience**: Cache all external responses; serve cached data when upstream is unavailable
- **Windows‑friendly**: Document PowerShell commands (no command chaining with `&&`)
- **Simplicity first**: Clear code, typed where sensible, minimal magic
- **No Docker (by default)**: Prefer native setup; containerization optional later

---

### Tech Stack

- **Frontend**: React + Vite + TypeScript + Tailwind CSS + shadcn/ui; TanStack Query; React Router
- **Backend**: FastAPI (Python), Uvicorn
- **Clients**: `httpx` with retries/backoff and explicit timeout/429 handling
- **Streaming**: WebSocket price feeds; Server‑Sent Events fallback
- **Data Sources**: CoinGecko REST (spot/market), optional exchange feeds later
- **Storage/Cache**: PostgreSQL (historical), Redis (cache, SWR); SQLite fallback for dev cache
- **Forecasting**: Prophet/ARIMA baselines → LightGBM/XGBoost → optional LSTM/TFT
- **MCP News**: Firecrawl MCP + REST fallback; sentiment (VADER/TextBlob → FinBERT/DistilRoBERTa)
- **Orchestration/Jobs**: APScheduler
- **Config**: `pydantic-settings`
- **Testing**: `pytest`, `@testing-library/react` + `vitest`

---

### High‑Level Architecture

1. Backend FastAPI
   - REST endpoints for `/prices`, `/forecasts`, `/news`, `/health`, `/cache/status`
   - Streaming endpoint `/stream` (WebSocket/SSE)
   - Background jobs for periodic fetch + cache refresh (APScheduler)
2. Data layer
   - HTTP clients (CoinGecko, Firecrawl, optional exchanges)
   - Unified caching (write‑through + TTL + SWR, graceful stale reads)
   - PostgreSQL for historical time‑series + news/sentiment
3. Forecasting service
   - Historical data ingestion + feature engineering
   - Model training/evaluation + versioned artifacts
   - Inference endpoint for on‑demand predictions
4. News pipeline (MCP Firecrawl)
   - Crawl configured sources; normalize, dedup, enrich (sentiment/topics); persist + cache
5. Frontend
   - Coinbase‑style UI: price ticker, watchlists, asset detail, charts, news feed, forecasts

---

### Repository Structure

```
/ (root)
├── backend/                   # FastAPI service + business logic
│   ├── app/
│   │   ├── api/
│   │   ├── clients/
│   │   ├── services/
│   │   ├── tasks/
│   │   └── models/
│   └── tests/
│
├── frontend/                  # React dashboard (scaffold later)
│   ├── src/
│   ├── public/
│   └── tests/
│
├── data/                      # raw + processed datasets
│   ├── raw/
│   └── processed/
│
├── models/                    # training scripts, notebooks, checkpoints
│   ├── notebooks/
│   └── src/
│
├── infra/
│   └── mcp/
│       └── firecrawl/         # runtime setup, scripts, docs
│
├── .github/
│   └── workflows/
│
├── .cursor/
│   └── rules
├── .cursorrules               # mirror of .cursor/rules for compatibility
├── README.md
└── requirements.txt
```

---

### Environment & Configuration

- `COINGECKO_BASE_URL` (default: `https://api.coingecko.com/api/v3`)
- `COINGECKO_API_KEY` (optional; header `x-cg-pro-api-key`)
- `DATABASE_URL` (Postgres), e.g., `postgresql+psycopg2://user:pass@localhost:5432/crypto`
- `REDIS_URL` (e.g., `redis://localhost:6379`)
- `ALLOWED_ORIGINS` for CORS
- Firecrawl MCP + REST:
  - `FIRECRAWL_BASE_URL=https://api.firecrawl.dev`
  - `FIRECRAWL_API_KEY=...`
  - `MCP_FIRECRAWL_PORT=7355`
  - `NEWS_MAX_CONTENT_BYTES=1048576`

Add to `.env.example`:
```bash
COINGECKO_BASE_URL=https://api.coingecko.com/api/v3
COINGECKO_API_KEY=
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/crypto
REDIS_URL=redis://localhost:6379
ALLOWED_ORIGINS=*
FIRECRAWL_BASE_URL=https://api.firecrawl.dev
FIRECRAWL_API_KEY=
MCP_FIRECRAWL_PORT=7355
NEWS_MAX_CONTENT_BYTES=1048576
```

---

### API Rate Limiting (slowapi)

All public endpoints now include per-IP rate limiting via `slowapi`. Responses include standard `X-RateLimit-*` headers plus `Retry-After` when throttled.

| Endpoint / Group            | Limit (per IP) |
|-----------------------------|----------------|
| `/prices`                  | 60 requests/min |
| `/prices/market`           | 45 requests/min |
| `/prices/history/{id}`     | 30 requests/min |
| `/forecasts/*`             | 20 requests/min |
| `/news` (GET routes)       | 30 requests/min |
| `/news/refresh`            | 5 requests/min  |
| `/cache/smart`             | 30 requests/min |
| All other endpoints        | 120 requests/min (default) |

Customize limits via the `RATE_LIMIT_*` variables in `.env`.

---

### Local Development (Windows PowerShell)

1) Create venv
```
py -3 -m venv .venv
```
2) Activate
```
.\.venv\Scripts\Activate.ps1
```
3) Install backend deps
```
pip install -r requirements.txt
```
4) Configure environment
```
# Copy .env.example to .env and set values; or set for the session:
$env:COINGECKO_API_KEY = "YOUR_KEY"
$env:REDIS_URL = "redis://localhost:6379"
$env:DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/crypto"
$env:FIRECRAWL_API_KEY = "YOUR_KEY"
```
5) Run backend
```
python backend/app/main.py
```
6) Optional: Redis on Windows
```
# Option A: Redis for Windows port → https://github.com/tporadowski/redis/releases
# Option B: Memurai Community → https://www.memurai.com/get-memurai
```
7) Optional: Firecrawl MCP runtime (see infra/mcp/firecrawl/README.md)
```
winget install OpenJS.NodeJS.LTS
# clone firecrawl mcp under infra/mcp/firecrawl, npm install, then
npm run start
```

Note: On Windows, avoid chaining commands with `&&`. Run them line‑by‑line.

---

### EPIC: Data Foundation (SCRUM‑1 → 4)

- SCRUM‑1: Config + pipeline setup
  - `pydantic-settings` config; `.env` support; structured logging
  - Data lake folders: `data/raw`, `data/processed`; optional DVC init
  - Redis cache utility and `/cache/status` endpoint
  - Acceptance: health endpoints OK; basic write/read to Redis; config loads from `.env`

- SCRUM‑2: PostgreSQL + Redis setup
  - Postgres schema: assets, ohlcv, features; Alembic init + base migration
  - Indices on `(asset_id, ts)` and `(ts)`
  - Acceptance: `alembic upgrade head` succeeds; DB healthcheck OK

- SCRUM‑3: Kaggle ingestion
  - `data/ingestion/kaggle_ingest.py` (Polars/Pandas), idempotent upserts
  - Dedupe; schema mapping; unit tests for parsers
  - Acceptance: 1M+ rows ingested; indices effective

- SCRUM‑4: ETL features (high‑volume)
  - Polars/DuckDB (Spark optional later); features: returns, log returns, RSI/MACD, vol, VWAP
  - APScheduler jobs with jitter; retry/backoff
  - Acceptance: feature tables materialized with versioning; backfills run

---

### EPIC: MCP News via Firecrawl (SCRUM‑5A → 5G)

- SCRUM‑5A: Firecrawl MCP runtime bootstrap
  - `infra/mcp/firecrawl/` with `README.md`, `start.ps1`, `.env.example`
  - Verify local run; sample crawl to `data/raw/news/`
  - Acceptance: MCP server responds; JSON dumps created

- SCRUM‑5B: News schema + migrations
  - Tables: `news_sources`, `news_articles`, `news_sentiment`, `news_topics`, `news_fetch_log`
  - Indices: `url_hash` unique, `content_hash`, `published_at`, `(source_id, published_at)`
  - Acceptance: migrations apply; constraints prevent duplicates

- SCRUM‑5C: Backend adapter + service
  - `adapters/firecrawl_client.py` (httpx retries/backoff)
  - `services/news_service.py` normalization + dedupe (SHA256 `url_hash`/`content_hash`)
  - `api/news.py`: `GET /news`, `POST /news/refresh` (admin)
  - Cache raw + normalized in Redis; return cached on failure
  - Acceptance: endpoints return fresh or cached data; unit tests pass

- SCRUM‑5D: Source seeding + scheduling
  - `data/sources/news_sources.yaml` allowlist (CoinDesk, CoinTelegraph, The Block, etc.)
  - APScheduler jobs with jitter; incremental crawl since `last_checked_at`
  - Acceptance: jobs run, logs tracked, retries/backoff honored

- SCRUM‑5E: Dedup + NLP enrichment
  - Sentiment: VADER/TextBlob → FinBERT/DistilRoBERTa later
  - Topics: zero‑shot labels [regulation, exchange, hacks, adoption, markets]
  - Entities: symbol mapping (BTC, ETH, SOL...)
  - Acceptance: articles enriched and linked to assets (if present)

- SCRUM‑5F: Admin + Observability
  - `GET /news/sources`, `GET /news/stats` + logs/metrics
  - Acceptance: ops visibility, requeue flow available

- SCRUM‑5G: MCP validation + troubleshooting
  - Timeouts, retries, rate limits; size limits via `NEWS_MAX_CONTENT_BYTES`
  - REST fallback path from backend when MCP is unstable
  - Acceptance: end‑to‑end ingest via MCP; REST fallback proven

---

### EPIC: Modeling & Forecasting (SCRUM‑6 → 10)

- SCRUM‑6/7: Preprocessing + feature engineering
  - Rolling windows; leakage checks; train/val/test splits
- SCRUM‑8: Baselines
  - Naive, SMA/EMA, ARIMA/Prophet; metrics: MAPE, RMSE, sMAPE
- SCRUM‑9: Advanced models
  - LightGBM/XGBoost; optional LSTM/TFT when justified
- SCRUM‑10: Backtesting
  - Walk‑forward evaluation; slippage assumptions; artifacted reports
- Acceptance: advanced models beat baselines on out‑of‑sample with fixed seeds

---

### EPIC: Backend & APIs (SCRUM‑11 → 15)

- Endpoints
  - `/prices` (CoinGecko) with Redis cache + SWR fallback
  - `/forecasts` (model outputs + confidence)
  - `/stream` (WS/SSE) with cached snapshot fallback
  - `/news` (queryable), `/health`, `/cache/status`
- Internals
  - `clients/` (`httpx` + retries/backoff), `services/`, `models/`
  - Security: API key for admin; CORS; basic rate limiting
- Acceptance: contract tests; 429/timeouts return cached data

---

### EPIC: Frontend UI (SCRUM‑16 → 19)

- SCRUM‑16: Scaffold Vite + React + TS + Tailwind + shadcn/ui
- SCRUM‑17: Core screens
  - Home with price ticker + top movers; watchlist
  - Asset detail: chart (Highcharts/Recharts), news sidebar
  - Forecasts view: model predictions, confidence bands
- SCRUM‑18: Visualizations
  - Charts for prices, volumes, volatility; news sentiment overlays
- SCRUM‑19: Settings + alerts UI
  - User preferences (theme, watchlist), alert rules configuration
- State & Data
  - TanStack Query for server state; WebSocket/SSE for `/stream`
  - Typed API clients; no ad‑hoc fetches
- Testing: `vitest` + `@testing-library/react`
- Acceptance: responsive, dark‑mode first, a11y basics

---

### EPIC: Notifications & Alerts (SCRUM‑20 → 23)

- Triggers: price/volatility thresholds, forecast deviation, sentiment spikes
- Channels: Twilio SMS/Email; Firebase push
- Acceptance: rules in DB; cooldowns; at‑least‑once delivery with retries

---

### EPIC: Security, Testing, Optimization (SCRUM‑24 → 27)

- Security: secrets mgmt, request validation, audit logs
- Testing: unit + integration + contract; load tests (Locust/k6)
- Perf: Redis key design; query optimization; profiling hot paths
- Acceptance: meet SLOs (e.g., 500 rps cached `/prices`, p95 latency target)

---

### EPIC: Deployment & Monitoring (SCRUM‑28 → 31)

- CI/CD: GitHub Actions (lint, test, build, migrations, deploy gates)
- Cloud: minimal VM/PaaS; managed Postgres/Redis if available
- Observability: logs, metrics, traces; uptime checks; dashboards
- Acceptance: safe deploys (blue/green or canary); alerting on SLO breaches

---

### Task Board (Living Checklist)

- [x] Repo flattened; rules set up
- [ ] SCRUM‑2: Postgres + Redis ready; Alembic init; healthchecks
- [ ] SCRUM‑3: Kaggle ingestion script + tests
- [ ] SCRUM‑4: Feature ETL + scheduler
- [ ] SCRUM‑5A→5G: Firecrawl MCP news pipeline end‑to‑end
- [ ] SCRUM‑6→10: Models baseline → advanced → backtesting
- [ ] SCRUM‑11→15: Backend endpoints `/prices`, `/forecasts`, `/stream`, `/news`
- [ ] SCRUM‑16→19: Frontend scaffold + UI
- [ ] SCRUM‑20→23: Alerts system
- [ ] SCRUM‑24→27: Security, testing, perf
- [ ] SCRUM‑28→31: CI/CD, deployment, monitoring

---

### Decision Log

- Use real CoinGecko data; always cache; serve cached when rate‑limited/unavailable
- Prefer native Windows dev setup; no Docker by default
- Single‑repo monorepo for backend + frontend + MCP adapters
- MCP Firecrawl is a first‑class path with REST fallback for resiliency

---

### Immediate Next Steps

1) Initialize Alembic; create base schema (assets, ohlcv, features, news tables)
2) Add Redis health and cache utility; expose `/cache/status`
3) Implement CoinGecko client with retries/backoff + Redis caching; wire `/prices`
4) Scaffold Firecrawl MCP runtime folder and example ingestion; add `/news` endpoints
5) Begin Kaggle ingestion and feature ETL; schedule with APScheduler
6) Scaffold frontend after core APIs stabilize


