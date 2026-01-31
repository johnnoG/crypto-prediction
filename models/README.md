# Cryptocurrency ML Forecasting Models

Production-grade machine learning models for cryptocurrency price prediction.

## Structure

```
models/
├── src/                           # Source code
│   ├── data/                      # Data loading and feature engineering
│   │   ├── data_loader.py         # Fetch & cache historical data
│   │   └── feature_engineering.py # 50+ technical indicators & features
│   ├── models/                    # ML model implementations
│   │   ├── lightgbm_model.py      # LightGBM gradient boosting
│   │   ├── lstm_model.py          # LSTM deep learning
│   │   ├── ensemble.py            # Hybrid ensemble (LGB + LSTM)
│   │   └── model_registry.py      # Model versioning & deployment
│   ├── training/                  # Training pipeline
│   │   ├── train_pipeline.py      # Training orchestration
│   │   └── backtester.py          # Walk-forward validation
│   └── evaluation/                # Model evaluation
│       └── metrics.py             # MAPE, RMSE, Sharpe, etc.
├── notebooks/                     # Jupyter notebooks for experimentation
├── artifacts/                     # Saved models
│   ├── lightgbm/                  # LightGBM models
│   ├── lstm/                      # LSTM models
│   └── ensemble/                  # Ensemble models
└── configs/                       # Model configurations
    └── model_config.yaml          # Hyperparameters
```

## Quick Start

### 1. Install Dependencies

```powershell
pip install lightgbm xgboost tensorflow scikit-learn optuna pandas numpy
```

### 2. Train a Model

```python
import asyncio
from models.src.training.train_pipeline import train_model_for_crypto

# Train LightGBM model for Bitcoin
model = asyncio.run(train_model_for_crypto(
    crypto_id='bitcoin',
    model_type='lightgbm',
    days=365,
    save_model=True
))
```

### 3. Make Predictions

```python
from models.src.models.lightgbm_model import LightGBMForecaster

# Load trained model
model = LightGBMForecaster()
model.load_model('models/artifacts/lightgbm/v1.0.0_<timestamp>.pkl')

# Prepare features and predict
predictions = model.predict(features)
```

## Models

### LightGBM (Traditional ML)
- **Speed**: Fast (~1s training, <10ms inference)
- **Accuracy**: High (MAPE typically 5-8%)
- **Features**: Uses 50+ engineered features
- **Best for**: Short-term predictions (1-7 days)

### LSTM (Deep Learning)
- **Speed**: Slower (~5min training, ~50ms inference)
- **Accuracy**: Very High (MAPE typically 4-7%)
- **Features**: Learns temporal patterns automatically
- **Best for**: Medium-term predictions (7-14 days)

### Hybrid Ensemble
- **Speed**: Medium
- **Accuracy**: Best (MAPE typically 3-6%)
- **Method**: Combines LightGBM + LSTM with meta-learning
- **Best for**: Production deployment

## Feature Engineering

The system engineers 50+ features including:

**Technical Indicators:**
- RSI (multiple periods)
- MACD, Signal, Histogram
- Bollinger Bands & position
- ATR, Stochastic, CCI, Williams %R
- Multiple moving averages (SMA, EMA)

**Price Features:**
- Returns (1d, 3d, 7d, 14d, 30d)
- Momentum & acceleration
- Volatility measures
- Z-scores

**Volume Features:**
- Volume momentum
- Volume ratios
- OBV (On-Balance Volume)
- Price-volume correlation

**Time Features:**
- Day of week, month, quarter
- Weekend indicators
- Cyclical encodings

## Backtesting

Walk-forward validation simulates real-world deployment:

```python
from models.src.training.backtester import ForecastBacktester

backtester = ForecastBacktester(
    initial_train_size=200,
    test_size=30,
    step_size=30,
    retrain_frequency=30
)

results = backtester.run_walk_forward_validation(data, model_trainer_fn)
aggregate_metrics = backtester.calculate_aggregate_metrics()
```

## Model Registry

Track and manage model versions:

```python
from models.src.models.model_registry import model_registry

# Register a model
model_id = model_registry.register_model(
    model_type='lightgbm',
    version='1.0.0',
    crypto_id='bitcoin',
    model_path='path/to/model.pkl',
    metrics={'mape': 5.2, 'rmse': 0.03},
    config={},
    status='production'
)

# Get production model
prod_model = model_registry.get_production_model('bitcoin', 'lightgbm')
```

## Performance Targets

- **MAPE < 5%** for 1-day predictions
- **MAPE < 10%** for 7-day predictions
- **Beat naive baseline by 30%+**
- **Directional accuracy > 60%**
- **R² score > 0.80**

## Notes

- Models are retrained weekly for optimal performance
- Feature importance is tracked to optimize feature sets
- Confidence intervals based on validation performance
- All predictions cached for 5 minutes
- Fallback to technical analysis if ML models unavailable

