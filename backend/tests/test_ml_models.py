"""
Tests for ML Forecasting Models

Tests feature engineering, model training, and predictions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add models to path
models_path = Path(__file__).parent.parent.parent / "models" / "src"
sys.path.insert(0, str(models_path))

from data.feature_engineering import FeatureEngineer, calculate_technical_features_quick
from evaluation.metrics import MetricsCalculator, ForecastMetrics


class TestFeatureEngineering:
    """Test feature engineering module"""
    
    def test_price_features(self):
        """Test price-based feature generation"""
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'price': np.random.randn(100).cumsum() + 100,
            'timestamp': dates
        }).set_index('timestamp')
        
        engineer = FeatureEngineer()
        features = engineer._add_price_features(prices)
        
        # Check returns calculated
        assert 'returns' in features.columns
        assert 'log_returns' in features.columns
        assert 'momentum_7d' in features.columns
        
        # Check no NaN in recent data (after warm-up period)
        recent_features = features.iloc[-50:]
        assert recent_features['returns'].notna().sum() > 45
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        prices = pd.DataFrame({
            'price': np.random.randn(200).cumsum() + 1000,
            'timestamp': dates
        }).set_index('timestamp')
        
        engineer = FeatureEngineer()
        features = engineer._add_technical_indicators(prices)
        
        # Check key indicators exist
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
        assert 'bb_position' in features.columns
        assert 'sma_50' in features.columns
        
        # Check RSI in valid range (0-100)
        rsi_values = features['rsi_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_quick_technical_features(self):
        """Test quick feature calculation for inference"""
        prices = list(np.random.randn(100).cumsum() + 1000)
        
        features = calculate_technical_features_quick(prices)
        
        assert 'rsi_14' in features
        assert 'macd' in features
        assert 'sma_50' in features
        assert 'volatility_30d' in features
        
        # Check reasonable values
        assert 0 <= features['rsi_14'] <= 100
        assert features['sma_50'] > 0


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_mape_calculation(self):
        """Test MAPE metric"""
        calc = MetricsCalculator()
        
        y_true = np.array([100, 110, 105, 115, 120])
        y_pred = np.array([102, 108, 107, 113, 122])
        
        mape = calc.calculate_mape(y_true, y_pred)
        
        # MAPE should be reasonable
        assert 0 <= mape <= 100
        assert mape < 10  # Less than 10% error
    
    def test_directional_accuracy(self):
        """Test directional accuracy metric"""
        calc = MetricsCalculator()
        
        # Perfect directional predictions
        y_true = np.array([100, 105, 103, 110, 115])
        y_pred = np.array([101, 106, 104, 111, 116])
        
        dir_acc = calc.calculate_directional_accuracy(y_true, y_pred)
        
        # Should be high since all directions match
        assert dir_acc > 75
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        calc = MetricsCalculator()
        
        # Positive returns with low volatility
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
        
        sharpe = calc.calculate_sharpe_ratio(returns)
        
        # Should be positive for positive returns
        assert sharpe > 0
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        calc = MetricsCalculator()
        
        # Prices with a drawdown
        prices = np.array([100, 110, 105, 95, 100, 105])
        
        max_dd = calc.calculate_max_drawdown(prices)
        
        # Should be negative (it's a drawdown)
        assert max_dd < 0
        # Roughly 13.6% drawdown from 110 to 95
        assert -15 < max_dd < -10


class TestLightGBM:
    """Test LightGBM model (if available)"""
    
    @pytest.mark.skipif(not True, reason="LightGBM tests run conditionally")
    def test_model_training(self):
        """Test basic LightGBM training"""
        try:
            from models.lightgbm_model import LightGBMForecaster
            
            # Create synthetic data
            X_train = np.random.randn(100, 10)
            y_train = np.random.randn(100)
            X_val = np.random.randn(20, 10)
            y_val = np.random.randn(20)
            
            model = LightGBMForecaster()
            metrics = model.train(X_train, y_train, X_val, y_val)
            
            # Check metrics returned
            assert 'train_rmse' in metrics
            assert metrics['train_rmse'] > 0
            
            # Check predictions work
            predictions = model.predict(X_val)
            assert len(predictions) == len(y_val)
        
        except ImportError:
            pytest.skip("LightGBM not installed")


class TestDataLoader:
    """Test data loader"""
    
    @pytest.mark.asyncio
    async def test_data_splitting(self):
        """Test train/val/test split"""
        from data.data_loader import CryptoDataLoader
        
        loader = CryptoDataLoader(
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        
        # Create sample data
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D')
        }).set_index('timestamp')
        
        train, val, test = loader.train_val_test_split(df, target_column='close')
        
        # Check split sizes
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        
        # Check no overlap
        assert set(train.index) & set(val.index) == set()
        assert set(train.index) & set(test.index) == set()
        assert set(val.index) & set(test.index) == set()
    
    def test_sequence_creation(self):
        """Test LSTM sequence creation"""
        from data.data_loader import CryptoDataLoader
        
        loader = CryptoDataLoader()
        
        data = np.random.randn(100, 1)
        X, y = loader.create_sequences(data, sequence_length=10, forecast_horizon=1)
        
        # Check shapes
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10  # sequence length
        assert X.shape[0] == 100 - 10  # sequences created


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

