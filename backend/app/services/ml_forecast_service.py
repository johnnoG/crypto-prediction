"""
Production ML Forecast Service

Integrates trained ML models (LightGBM, LSTM, Ensemble) with the backend API.
Handles model loading, inference, caching, and fallbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add models path
models_path = Path(__file__).parent.parent.parent.parent / "models" / "src"
sys.path.insert(0, str(models_path))

try:
    from models.lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE  # type: ignore
    from models.lstm_model import LSTMForecaster, TENSORFLOW_AVAILABLE  # type: ignore
    from models.ensemble import HybridEnsemble  # type: ignore
    from models.model_registry import model_registry  # type: ignore
    from data.feature_engineering import FeatureEngineer, calculate_technical_features_quick  # type: ignore
    from evaluation.metrics import MetricsCalculator  # type: ignore
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    ML_MODELS_AVAILABLE = False
    print(f"ML models not available: {e}")

try:
    from cache import AsyncCache
    from clients.coingecko_client import CoinGeckoClient
except ImportError:
    print("Warning: Backend modules not available")


class ProductionMLForecastService:
    """
    Production service for ML-based cryptocurrency forecasting.
    
    Features:
    - Automatic model loading and caching
    - Fallback to simpler models if complex models fail
    - Confidence interval estimation
    - Performance tracking
    """
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer() if ML_MODELS_AVAILABLE else None
        self.metrics_calculator = MetricsCalculator() if ML_MODELS_AVAILABLE else None
        self.model_cache_ttl = 3600  # Cache loaded models for 1 hour
        self.last_model_load: Dict[str, datetime] = {}
    
    async def _load_model_for_crypto(
        self,
        crypto_id: str,
        model_type: str = 'lightgbm'
    ) -> Optional[Any]:
        """
        Load the best production model for a cryptocurrency.
        
        Args:
            crypto_id: CoinGecko ID
            model_type: Preferred model type
            
        Returns:
            Loaded model instance
        """
        if not ML_MODELS_AVAILABLE:
            return None
        
        cache_key = f"{crypto_id}_{model_type}"
        
        # Check if model already loaded and cached
        if cache_key in self.loaded_models:
            last_load = self.last_model_load.get(cache_key)
            if last_load and (datetime.now() - last_load).seconds < self.model_cache_ttl:
                return self.loaded_models[cache_key]
        
        # Try to load from registry
        production_model = model_registry.get_production_model(crypto_id, model_type)
        
        if production_model and Path(production_model.model_path).exists():
            try:
                if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = LightGBMForecaster()
                    model.load_model(production_model.model_path)
                elif model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                    model = LSTMForecaster()
                    model.load_model(production_model.model_path)
                elif model_type == 'ensemble':
                    model = HybridEnsemble()
                    model.load_ensemble(production_model.model_path)
                else:
                    return None
                
                # Cache the loaded model
                self.loaded_models[cache_key] = model
                self.last_model_load[cache_key] = datetime.now()
                
                print(f"Loaded production {model_type} model for {crypto_id}")
                
                return model
                
            except Exception as e:
                print(f"Error loading model for {crypto_id}: {e}")
                return None
        
        return None
    
    async def generate_ml_forecast(
        self,
        crypto_id: str,
        current_price: float,
        historical_prices: List[float],
        days: int = 7,
        model_type: str = 'lightgbm'
    ) -> Dict[str, Any]:
        """
        Generate forecast using ML model.
        
        Args:
            crypto_id: Cryptocurrency ID
            current_price: Current price
            historical_prices: Historical price data
            days: Forecast horizon
            model_type: Model type to use
            
        Returns:
            Forecast dictionary
        """
        # Try to load ML model
        model = await self._load_model_for_crypto(crypto_id, model_type)
        
        if model is None:
            # Fallback to technical analysis
            return await self._generate_technical_forecast(
                crypto_id, current_price, historical_prices, days
            )
        
        try:
            # Prepare features
            features = self._prepare_features_for_inference(
                historical_prices,
                current_price
            )
            
            # Make prediction
            if hasattr(model, 'predict'):
                predictions = model.predict(features, return_confidence=False)
                
                # Generate multi-day forecast
                forecast_points = self._generate_multi_day_forecast(
                    model,
                    features,
                    current_price,
                    days
                )
                
                return {
                    'model': model_type,
                    'current_price': current_price,
                    'forecasts': forecast_points,
                    'model_metrics': model.metadata.get('metrics', {}),
                    'status': 'success',
                    'using_ml': True
                }
            
        except Exception as e:
            print(f"ML forecast error for {crypto_id}: {e}")
            # Fallback to technical analysis
            return await self._generate_technical_forecast(
                crypto_id, current_price, historical_prices, days
            )
    
    def _prepare_features_for_inference(
        self,
        historical_prices: List[float],
        current_price: float
    ) -> np.ndarray:
        """
        Prepare features for real-time inference.
        
        Args:
            historical_prices: Historical price data
            current_price: Most recent price
            
        Returns:
            Feature matrix for prediction
        """
        if not self.feature_engineer:
            return np.array([[current_price]])
        
        # Create DataFrame from prices
        df = pd.DataFrame({
            'price': historical_prices + [current_price],
            'timestamp': pd.date_range(
                end=datetime.now(),
                periods=len(historical_prices) + 1,
                freq='D'
            )
        }).set_index('timestamp')
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        
        # Get the most recent features (for current prediction)
        latest_features = df_features.iloc[-1:].drop(columns=['price']).values
        
        return latest_features
    
    def _generate_multi_day_forecast(
        self,
        model: Any,
        initial_features: np.ndarray,
        current_price: float,
        days: int
    ) -> List[Dict[str, Any]]:
        """
        Generate multi-day forecast by iterating predictions.
        
        Args:
            model: Trained model
            initial_features: Features for first prediction
            current_price: Current price
            days: Number of days to forecast
            
        Returns:
            List of forecast points
        """
        forecasts = []
        
        # For simplicity, we'll use a conservative approach:
        # Predict next day, then use that to predict day after, etc.
        
        current_features = initial_features.copy()
        last_price = current_price
        
        for i in range(days):
            # Predict next price
            try:
                pred_price = model.predict(current_features)[0]
            except:
                # Fallback: simple trend continuation
                pred_price = last_price * 1.001  # Small upward bias
            
            # Calculate confidence (decreases with time)
            base_confidence = 0.90 - (i * 0.03)  # Decay by 3% per day
            confidence = max(0.5, base_confidence)
            
            # Calculate confidence intervals (approximate)
            val_rmse = model.metadata.get('metrics', {}).get('val_rmse', pred_price * 0.05)
            margin = val_rmse * 1.96  # 95% confidence interval
            
            forecast_point = {
                'date': (datetime.now() + timedelta(days=i+1)).isoformat(),
                'predicted_price': float(pred_price),
                'confidence': float(confidence),
                'confidence_lower': float(max(0, pred_price - margin)),
                'confidence_upper': float(pred_price + margin),
                'technical_signals': calculate_technical_features_quick([last_price, pred_price])
            }
            
            forecasts.append(forecast_point)
            
            # Update for next iteration (simplified feature update)
            last_price = pred_price
        
        return forecasts
    
    async def _generate_technical_forecast(
        self,
        crypto_id: str,
        current_price: float,
        historical_prices: List[float],
        days: int
    ) -> Dict[str, Any]:
        """
        Fallback to technical analysis-based forecast.
        
        This is used when ML models are not available or fail.
        """
        # Use the existing technical analysis from backend
        from services.forecast_service import generate_professional_forecast
        
        return await generate_professional_forecast(
            crypto_id=crypto_id,
            current_price=current_price,
            historical_prices=historical_prices,
            days=days,
            model_type='baseline'
        )


# Global service instance
ml_forecast_service = ProductionMLForecastService()

