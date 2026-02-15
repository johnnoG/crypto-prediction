# Crypto Prediction & Real-Time Dashboard

Production-grade cryptocurrency price prediction system combining deep learning, gradient boosting, and ensemble methods with a real-time trading dashboard. Trained on 104 cryptocurrencies with 150+ engineered features.

## Overview

An end-to-end ML pipeline and web application for multi-horizon cryptocurrency price forecasting:

- **Five Model Architectures** — DLinear baseline, Temporal Convolutional Network (TCN), Bidirectional LSTM with attention, Transformer with causal masking, and LightGBM gradient boosting, combined through a CV-stacked ensemble with market regime detection
- **150+ Engineered Features** — Technical indicators, momentum oscillators, volatility metrics, volume analysis, and statistical features for 104 cryptocurrencies
- **Production Training Pipeline** — Automated training with chronological splits, RobustScaler preprocessing, multi-step forecasting (1/7/30 day horizons), and 10 diagnostic visualizations per training run
- **MLflow Experiment Tracking** — Full experiment logging with model versioning, metric tracking, and artifact management
- **FastAPI Backend** — RESTful API with JWT authentication, real-time crypto data from CoinGecko/Binance, WebSocket streaming, and ML inference endpoints
- **React Dashboard** — Interactive frontend with live price charts, forecast panels, news aggregation, portfolio tracking, and alert management
- **Docker Infrastructure** — Containerized deployment with PostgreSQL, Redis, and Nginx

## Quick Start

```bash
# Clone and start all services
git clone <repository-url>
cd crypto-prediction
docker-compose up -d

# Access the application:
#   Frontend:  http://localhost:3000
#   API:       http://localhost:8000
#   API Docs:  http://localhost:8000/docs
```

### Train Models (GPU recommended: Colab T4/A100 or Apple Silicon with Metal)

```bash
# Full production training for BTC and ETH (5 models + ensemble)
python3 models/src/train_production.py --crypto BTC,ETH

# Quick validation run
python3 models/src/train_production.py --crypto BTC --epochs 5 --no-ensemble

# With Optuna hyperparameter tuning (all 5 models)
python3 models/src/train_production.py --crypto BTC --tune --tune-trials 20 \
    --tune-models dlinear,tcn,lstm,transformer,lightgbm

# Full professional pipeline (tuning + walk-forward + both coins)
python3 models/src/train_production.py --crypto BTC,ETH --tune --tune-trials 20 \
    --tune-models dlinear,tcn,lstm,transformer,lightgbm --walk-forward --epochs 150
```

Training outputs:
- Model weights saved to `models/artifacts/`
- 10 diagnostic PNG plots saved to `models/src/training_output/`
- JSON training report with metrics, uncertainty analysis, ensemble evaluation
- MLflow experiment logs with interactive training curves and model cards

## Repository Structure

```
crypto-prediction/
├── backend/                          # FastAPI backend service
│   ├── app/
│   │   ├── main.py                   # Application entry point
│   │   ├── api/                      # REST endpoints (auth, crypto, forecasts, health, ...)
│   │   ├── clients/                  # External API clients (CoinGecko, Binance, news)
│   │   ├── services/                 # Business logic (forecasting, ETL, caching)
│   │   ├── models/                   # SQLAlchemy ORM models
│   │   └── utils/                    # Connection pooling, circuit breaker
│   ├── migrations/                   # Alembic database migrations
│   └── tests/
│
├── models/                           # ML models, training, and deployment
│   ├── MODEL_ARCHITECTURE_AND_TRAINING.md  # Detailed model documentation
│   ├── artifacts/                    # Saved model weights and metadata
│   │   ├── enhanced_lstm/
│   │   ├── transformer/
│   │   ├── lightgbm/
│   │   └── advanced_ensemble/
│   └── src/
│       ├── models/                   # Model implementations
│       │   ├── dlinear_model.py      # DLinear trend/seasonal decomposition baseline
│       │   ├── tcn_model.py          # Temporal Convolutional Network (dilated causal)
│       │   ├── enhanced_lstm.py      # Bidirectional LSTM + attention + residual + L2
│       │   ├── transformer_model.py  # Multi-head self-attention transformer
│       │   ├── lightgbm_model.py     # Gradient boosting with summary statistics
│       │   └── advanced_ensemble.py  # CV-stacked ensemble with regime detection
│       ├── training/                 # Training infrastructure
│       │   ├── hyperopt_pipeline.py  # Optuna hyperparameter optimization
│       │   ├── production_pipeline.py # End-to-end training pipeline
│       │   └── mlflow_integration.py # MLflow experiment tracking
│       ├── pipelines/
│       │   └── enhanced_training_pipeline.py  # Walk-forward validation
│       ├── deployment/
│       │   └── deployment_manager.py # Blue-green deployment with rollback
│       ├── ab_testing/
│       │   └── ab_test_manager.py    # Champion-challenger A/B testing
│       ├── monitoring/
│       │   └── performance_monitor.py # Drift detection and alerting
│       ├── mlflow_advanced/
│       │   └── experiment_manager.py # Advanced MLflow management
│       ├── visualization/            # Training dashboards and plots
│       │   ├── training_dashboard.py
│       │   ├── training_monitor.py
│       │   ├── hyperopt_dashboard.py
│       │   ├── model_inspector.py
│       │   └── launch_dashboards.py
│       ├── train_production.py       # Main production training script
│       └── phase3_integration.py     # MLflow + deployment integration demo
│
├── data/
│   ├── features/                     # Engineered feature parquets (104 cryptos)
│   ├── processed/                    # Cleaned OHLCV data (CSV + Parquet)
│   ├── kaggle-raw/                   # Original Kaggle dataset
│   └── sources/                      # Data source configs
│
├── data_analysis/                    # Analysis engine
│   ├── crypto_data_analyzer.py       # Main data analysis pipeline
│   ├── feature_engineering.py        # 150+ technical indicators
│   └── statistical_analysis.py       # PCA, clustering, risk metrics
│
├── frontend/                         # React 19 + TypeScript dashboard
│   └── src/
│       ├── components/               # UI components (charts, auth, pages)
│       ├── contexts/                 # Auth state management
│       ├── hooks/                    # Data fetching hooks
│       └── lib/                      # API client, utilities
│
├── notebooks/                        # Jupyter analysis notebooks
├── docs/                             # Reference documentation
├── docker-compose.yml
├── requirements.txt                  # Python dependencies (122 packages)
└── README.md
```

## Machine Learning Pipeline

### Architecture Overview

```
Raw Parquet Data (5600+ rows, 150+ features per crypto)
    |
    v
Chronological Split --> Feature Selection (top 80 by mutual info, train only)
    |
    v
Cross-Asset Features (ETH indicators for BTC) --> Log-Return Targets
    |
    v
RobustScaler (fit on train only) --> Data Augmentation (jitter + scale, 3x)
    |
    v
Sliding Window Sequences (60 days lookback)
    |
    v
+---------+--------+------------------+--------------------+------------------+
| DLinear |  TCN   |   Enhanced LSTM  |    Transformer     |    LightGBM      |
| Trend/  | Causal | Bidirectional    | Multi-head         | Summary stats    |
| Seasonal| Conv   | Attention + L2   | Self-attention     | (5 per feature)  |
+---------+--------+------------------+--------------------+------------------+
    |         |          |                    |                    |
    v         v          v                    v                    v
+--------------------------------------------------------------------+
|               CV-Stacked Ensemble (5 models)                       |
|  OOF meta-learner + Market regime detection + Online adaptation    |
+--------------------------------------------------------------------+
    |
    v
Multi-Horizon Predictions: 1-day, 7-day, 30-day
    |
    v
Visualizations (10 PNG plots) + Model Artifacts + MLflow Logs
```

For detailed model documentation including architecture diagrams, hyperparameter rationale, and training methodology, see [Model Architecture & Training](models/MODEL_ARCHITECTURE_AND_TRAINING.md).

### Training Visualizations

Each training run generates 10 diagnostic plots:

| Plot | Description |
|------|-------------|
| `loss_curves.png` | Train vs validation loss for each model |
| `metrics_progression.png` | Per-horizon loss progression over training |
| `learning_rates.png` | LR schedules (warmup, decay, plateau) |
| `attention_heatmap.png` | LSTM attention weights over timesteps |
| `feature_importance.png` | Top 30 LightGBM features |
| `model_comparison.png` | RMSE/MAE bars across all models |
| `predictions_vs_actual.png` | Scatter plots with R-squared |
| `residual_analysis.png` | Residual distributions |
| `ensemble_weights.png` | Model contribution weights |
| `training_summary.png` | Multi-panel overview |

## Technology Stack

### Machine Learning

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow/Keras (DLinear, TCN, LSTM, Transformer) |
| Gradient Boosting | LightGBM, XGBoost |
| Hyperparameter Tuning | Optuna (Bayesian optimization) |
| Experiment Tracking | MLflow (logging, model registry) |
| Feature Engineering | Pandas, NumPy, TA-Lib, Statsmodels |
| Preprocessing | Scikit-learn (RobustScaler, metrics) |
| Interpretability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly, Dash |

### Backend

| Component | Technology |
|-----------|------------|
| Framework | FastAPI with async support |
| Database | PostgreSQL 16 + SQLAlchemy + Alembic |
| Cache | Redis 7 |
| Auth | Argon2 password hashing + JWT tokens |
| HTTP Client | HTTPX (async) |
| Rate Limiting | SlowAPI |
| External APIs | CoinGecko, Binance, CryptoCompare |

### Frontend

| Component | Technology |
|-----------|------------|
| Framework | React 19 + TypeScript |
| Build Tool | Vite |
| Styling | Tailwind CSS |
| Data Fetching | React Query |
| Charts | Lightweight Charts (TradingView) |
| Icons | Lucide React |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker + Docker Compose |
| Web Server | Nginx (reverse proxy) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Sentry |

## Data

The system processes data for 104 cryptocurrencies spanning 2010-2026:

- **Raw Data**: Historical OHLCV from Kaggle and exchange APIs
- **Processed Data**: Cleaned and normalized time series (CSV + Parquet)
- **Feature Data**: 150+ engineered features per cryptocurrency including:
  - Price-derived (returns, momentum, spreads)
  - Moving averages (SMA/EMA at multiple periods, MACD)
  - Momentum oscillators (RSI, Stochastic, Williams %R)
  - Volatility (Bollinger Bands, ATR, Keltner Channels)
  - Volume indicators (OBV, VPT, volume ratios)
  - Statistical (skewness, kurtosis, z-scores)
  - Time-based (cyclical day/month/quarter encoding)
  - Regime detection (bull/bear, volatility regimes)

## Development

### Prerequisites

- Python 3.11+ with pip
- Node.js 18+
- Docker & Docker Compose
- GPU recommended for model training (Google Colab T4/A100 or Apple Silicon Mac with Metal)

### Local Setup

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install && npm run dev

# ML environment
pip install -r requirements.txt  # Root requirements.txt

# ML environment with Apple Silicon GPU (requires Python 3.12)
python3.12 -m venv tf-gpu-env && source tf-gpu-env/bin/activate
pip install tensorflow==2.18 tensorflow-metal
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d  # PostgreSQL + Redis
```

### Running Tests

```bash
docker-compose exec backend pytest --cov=app --cov-report=html
```

### Database Migrations

```bash
docker-compose exec backend alembic revision --autogenerate -m "description"
docker-compose exec backend alembic upgrade head
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | System health check |
| `GET /health/detailed` | Database + cache + service health |
| `POST /auth/signup` | User registration |
| `POST /auth/signin` | JWT authentication |
| `GET /api/crypto/prices` | Live cryptocurrency prices |
| `GET /api/forecasts/{symbol}` | ML price forecasts |
| `GET /api/market/overview` | Market summary |
| `WS /api/stream` | Real-time price WebSocket |

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

## Project Status

### Completed

- **Phase 1: Data Analysis** — 104 cryptocurrencies analyzed, 150+ features engineered, statistical analysis (PCA, clustering, risk metrics), professional visualizations
- **Phase 2: ML Models** — DLinear, TCN, LSTM, Transformer, LightGBM, and CV-stacked Ensemble (5 models) with multi-step forecasting, log-return targets, data augmentation, cross-asset features, and uncertainty quantification
- **Phase 3: MLflow & Deployment** — Experiment tracking, model versioning, blue-green deployment, A/B testing framework, performance monitoring, training dashboards
- **Infrastructure** — Docker stack, PostgreSQL, Redis, FastAPI, React dashboard, authentication, real-time data feeds
- **BTC & ETH Production Training** — 5 model architectures (DLinear, TCN, LSTM, Transformer, LightGBM), Optuna tuning, CV-stacked ensemble, MC dropout uncertainty, cross-asset features, 10 diagnostic visualizations per run
- **BTC Overfitting Fixes** — Log-return targets, mutual information feature selection (train-only), reduced model capacity, L2 regularization, data augmentation, summary statistics for LightGBM, CV-stacked ensemble meta-learner

### In Progress

- Expanding training to top-20 coins by data availability (LTC, XRP, DOGE, BNB, SOL, ADA, LINK, DOT)
- WebSocket streaming integration in frontend
- Advanced analytics dashboard with live predictions

### Planned

- Real-time model serving with FastAPI inference endpoint
- Portfolio optimization with trained models
- Automated retraining pipeline with drift detection
- Mobile-responsive PWA
- Cloud deployment (AWS/GCP)

## Documentation

- [Model Architecture & Training](models/MODEL_ARCHITECTURE_AND_TRAINING.md) — Detailed documentation of all model architectures, hyperparameters, training methodology, and ensemble strategy
- [Analysis Documentation](ANALYSIS_DOCUMENTATION.md) — Guide to analysis outputs and metrics
- [Backend API](backend/README.md) — API endpoint documentation
- [Quick Reference](docs/reference/QUICK_REFERENCE.md) — Getting started guide

## Acknowledgments

Built as a final project for BSc Computer Engineering program.
