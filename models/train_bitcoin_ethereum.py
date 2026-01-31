"""
Simple Training Script for Bitcoin and Ethereum

Trains LightGBM models using the backend's existing infrastructure.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend" / "app"
sys.path.insert(0, str(backend_path))

# Import backend modules
from clients.coingecko_client import CoinGeckoClient
from cache import AsyncCache

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import joblib
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("ERROR: LightGBM not installed!")
    print("Run: py -m pip install lightgbm scikit-learn pandas numpy")
    sys.exit(1)


async def fetch_historical_data(crypto_id: str, days: int = 365):
    """Fetch historical OHLC data from CoinGecko"""
    print(f"  📥 Fetching {days} days of data for {crypto_id}...")
    
    client = CoinGeckoClient(timeout_seconds=15.0)
    
    try:
        ohlc_data = await client.get_coin_ohlc_by_id(
            coin_id=crypto_id,
            vs_currency="usd",
            days=days
        )
        
        await client.close()
        
        if not ohlc_data or len(ohlc_data) == 0:
            raise ValueError(f"No data received for {crypto_id}")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        
        print(f"  ✅ Fetched {len(df)} data points")
        return df
        
    except Exception as e:
        print(f"  ❌ Error fetching data: {e}")
        if client:
            await client.close()
        raise


def engineer_basic_features(df):
    """Create basic technical features"""
    print(f"  🔧 Engineering features...")
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    for period in [7, 14, 30]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility_7d'] = df['returns'].rolling(window=7).std()
    df['volatility_30d'] = df['returns'].rolling(window=30).std()
    
    # Momentum
    df['momentum_7d'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14d'] = df['close'] / df['close'].shift(14) - 1
    
    # Price lags
    for lag in [1, 3, 7]:
        df[f'price_lag_{lag}'] = df['close'].shift(lag)
    
    # Drop NaN
    df = df.dropna()
    
    print(f"  ✅ Created {len(df.columns)} features, {len(df)} samples")
    return df


def train_lightgbm_model(crypto_id: str, df):
    """Train LightGBM model"""
    print(f"  🎓 Training LightGBM model...")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'close']
    X = df[feature_cols].values
    y = df['close'].values
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
    )
    
    print(f"  📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Train model
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
    test_mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
    
    print(f"  📈 Train RMSE: {train_rmse:.2f}")
    print(f"  📈 Test RMSE: {test_rmse:.2f}")
    print(f"  📈 Test MAPE: {test_mape:.2f}%")
    
    # Save model
    model_dir = Path("models/artifacts/lightgbm")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{crypto_id}_v1.0.0_{timestamp}.pkl"
    
    joblib.dump(model, model_path)
    print(f"  💾 Model saved to: {model_path}")
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'model_path': str(model_path)
    }


async def train_crypto(crypto_id: str):
    """Complete training pipeline for one crypto"""
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {crypto_id.upper()}")
    print(f"{'='*70}")
    
    try:
        # 1. Fetch data
        df = await fetch_historical_data(crypto_id, days=365)
        
        # 2. Engineer features
        df_features = engineer_basic_features(df)
        
        # 3. Train model
        results = train_lightgbm_model(crypto_id, df_features)
        
        print(f"\n✅ {crypto_id.upper()} training SUCCESS!")
        return {'status': 'success', **results}
        
    except Exception as e:
        print(f"\n❌ {crypto_id.upper()} training FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


async def main():
    """Train Bitcoin and Ethereum"""
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM not installed!")
        return
    
    print("\n" + "=" * 70)
    print("ML MODEL TRAINING - BITCOIN & ETHEREUM")
    print("=" * 70)
    print("\n📚 This will:")
    print("   • Fetch 365 days of historical data")
    print("   • Engineer technical features (RSI, MACD, MA, etc.)")
    print("   • Train LightGBM models")
    print("   • Save models to artifacts/")
    print("\n⏱️  Estimated time: 3-5 minutes\n")
    
    # Train both
    btc_result = await train_crypto('bitcoin')
    eth_result = await train_crypto('ethereum')
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    results = {'bitcoin': btc_result, 'ethereum': eth_result}
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    
    print(f"\n✅ Successful: {successful}/2")
    
    if successful > 0:
        print("\n🎉 Models trained and saved!")
        print("\n📝 Next steps:")
        print("   1. Models are ready to use")
        print("   2. Restart backend to load them")
        print("   3. Use ML forecasts in API:")
        print("      curl \"http://127.0.0.1:8000/forecasts?ids=bitcoin&model=lightgbm\"")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

