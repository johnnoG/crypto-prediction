# Getting Started with ML-Powered Forecasting

## Quick Start (5 Minutes)

### Step 1: Install ML Dependencies

The basic ML libraries (recommended for start):

```powershell
pip install lightgbm scikit-learn pandas numpy matplotlib joblib
```

**Optional (for full ML stack):**
```powershell
pip install tensorflow optuna xgboost ta pyarrow
```

**Optional (for monitoring):**
```powershell
pip install prometheus-client sentry-sdk psutil
```

### Step 2: Train Your First Model

Navigate to the models directory and run the training script:

```powershell
cd models
python train_all_models.py
```

This will train LightGBM models for Bitcoin, Ethereum, Solana, Cardano, and BNB.

**Expected time:** 5-10 minutes for all 5 cryptos

### Step 3: Verify Models Trained

Check that models were saved:

```powershell
dir artifacts\lightgbm
```

You should see `.pkl` files with timestamps.

### Step 4: Use ML Models in API

Restart your backend (the ML models will be automatically detected):

```powershell
python main.py
```

Then test the ML forecasts:

```bash
# Test LightGBM model
curl "http://127.0.0.1:8000/forecasts?ids=bitcoin&days=7&model=lightgbm"

# Check available models
curl "http://127.0.0.1:8000/forecasts/models"
```

### Step 5: Monitor Health

Check backend health:

```bash
# Detailed health check
curl "http://127.0.0.1:8000/health/detailed"

# Connection pool status
curl "http://127.0.0.1:8000/health/components/connection_pool"

# Circuit breaker status
curl "http://127.0.0.1:8000/health/components/circuit_breakers"
```

## Training Custom Models

### Train a Single Crypto

```python
import asyncio
from models.src.training.train_pipeline import train_model_for_crypto

# Train LightGBM
model = asyncio.run(train_model_for_crypto(
    crypto_id='bitcoin',
    model_type='lightgbm',
    days=365,  # 1 year of history
    save_model=True
))

print(f"Model saved! Metrics: {model.metadata['metrics']}")
```

### Run Backtesting

```python
import asyncio
from models.src.data.data_loader import data_loader
from models.src.data.feature_engineering import FeatureEngineer
from models.src.training.backtester import run_quick_backtest

# Load data
data = asyncio.run(data_loader.load_crypto_data('bitcoin', days=365))

# Engineer features
engineer = FeatureEngineer()
data_features = engineer.engineer_features(data)

# Run backtest
results = run_quick_backtest(data_features, model_type='lightgbm')
print(f"Backtest MAPE: {results['mape']['mean']:.2f}%")
```

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Fastest Results** | `baseline` | No training needed, <1ms inference |
| **Best Accuracy** | `ml_ensemble` | Hybrid approach, MAPE 3-6% |
| **Good Balance** | `lightgbm` | Fast training/inference, MAPE 5-8% |
| **Research** | `lstm` | Deep learning, best for patterns |

## Frontend Integration

The frontend already supports ML models! Just select a different model in the Forecast Panel UI or the dropdown will automatically show ML models once trained.

## Troubleshooting

### Problem: "ML models not available"
**Solution:** Install dependencies and train models:
```powershell
pip install lightgbm scikit-learn pandas
cd models
python train_all_models.py
```

### Problem: "Connection pool errors"
**Solution:** The new connection pool manager should prevent this. Check health:
```bash
curl http://127.0.0.1:8000/health/detailed
```

### Problem: "TensorFlow too large to install"
**Solution:** Start with LightGBM only (it's faster and smaller):
```powershell
pip install lightgbm scikit-learn
# Skip TensorFlow for now
```

### Problem: "Model training takes too long"
**Solution:** Reduce data period:
```python
model = asyncio.run(train_model_for_crypto(
    'bitcoin',
    'lightgbm',
    days=180  # Use 6 months instead of 12
))
```

## Feature Highlights

### 50+ Engineered Features
The system automatically creates:
- **Technical Indicators:** RSI, MACD, Bollinger Bands, ATR, Stochastic, CCI
- **Moving Averages:** SMA/EMA (7, 14, 21, 50, 100, 200 periods)
- **Price Features:** Returns, momentum, volatility, z-scores
- **Volume Features:** OBV, volume ratios, price-volume correlation
- **Time Features:** Day of week, seasonality, cyclical encodings

### Walk-Forward Validation
Simulates real-world deployment:
- Retrain every 30 days
- Test on next 30 days
- Calculate metrics for each fold
- Compare against naive baseline

### Model Registry
Track and manage models:
- Version control for all models
- Production/staging/development status
- Performance comparison
- One-click promotion to production

## Production Deployment Checklist

- [ ] Train models for all major cryptos
- [ ] Run backtests and validate MAPE < 10%
- [ ] Register models in model registry
- [ ] Promote best models to production
- [ ] Set up Prometheus monitoring
- [ ] Configure Sentry error tracking
- [ ] Set up weekly retraining schedule
- [ ] Monitor model performance
- [ ] Set up alerts for degradation

## Support

For issues or questions:
1. Check health endpoint: `/health/detailed`
2. Review logs in terminal
3. Check model registry: `/forecasts/models`
4. Verify dependencies: `pip list | grep -E "lightgbm|tensorflow|sklearn"`

Congratulations! Your backend now has professional-grade ML forecasting capabilities! 🚀

