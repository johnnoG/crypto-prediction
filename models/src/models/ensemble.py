"""
Hybrid Ensemble Model for Cryptocurrency Forecasting

Combines multiple models (LightGBM + LSTM) using:
- Weighted averaging
- Meta-learning (stacking)
- Dynamic weight adjustment based on recent performance

Kaggle competition winning technique.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import StackingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")

from .lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE
from .lstm_model import LSTMForecaster, TENSORFLOW_AVAILABLE


class HybridEnsemble:
    """
    Hybrid ensemble combining traditional ML (LightGBM) and deep learning (LSTM).
    
    Uses meta-learning to optimally combine predictions from different model types.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/ensemble"
    ):
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Base models
        self.lgb_model: Optional[LightGBMForecaster] = None
        self.lstm_model: Optional[LSTMForecaster] = None
        
        # Meta-learner for stacking
        self.meta_model: Optional[Any] = None
        self.use_meta_learner = self.config.get('use_meta_learner', True)
        
        # Weights for simple weighted averaging
        self.lgb_weight = self.config.get('lgb_weight', 0.6)
        self.lstm_weight = self.config.get('lstm_weight', 0.4)
        
        # Performance tracking for dynamic weighting
        self.performance_history: Dict[str, List[float]] = {
            'lgb_rmse': [],
            'lstm_rmse': [],
            'ensemble_rmse': []
        }
        
        self.metadata: Dict[str, Any] = {}
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default ensemble configuration"""
        return {
            'use_meta_learner': True,
            'meta_model_type': 'ridge',  # 'ridge', 'lasso', or 'elasticnet'
            'lgb_weight': 0.6,
            'lstm_weight': 0.4,
            'dynamic_weighting': True,  # Adjust weights based on recent performance
            'performance_window': 10  # Window for calculating recent performance
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble model.
        
        Trains both LightGBM and LSTM, then trains meta-learner on their predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            
        Returns:
            Training metrics
        """
        print("Training Hybrid Ensemble Model...")
        print("=" * 60)
        
        metrics = {}
        
        # 1. Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n[1/3] Training LightGBM base model...")
            self.lgb_model = LightGBMForecaster()
            lgb_metrics = self.lgb_model.train(
                X_train, y_train, X_val, y_val, feature_names
            )
            metrics['lgb'] = lgb_metrics
            print(f"LightGBM trained - Val RMSE: {lgb_metrics.get('val_rmse', 'N/A')}")
        else:
            print("Skipping LightGBM (not available)")
        
        # 2. Train LSTM (requires 3D input)
        if TENSORFLOW_AVAILABLE:
            print("\n[2/3] Training LSTM base model...")
            
            # Reshape for LSTM if needed
            if len(X_train.shape) == 2:
                # Assume sequence_length from config
                seq_len = self.config.get('sequence_length', 60)
                n_features = X_train.shape[1]
                
                # Reshape (n_samples, n_features) -> (n_samples, seq_len, n_features)
                # This is simplified; in practice you'd use proper sequence creation
                if X_train.shape[0] >= seq_len:
                    X_train_lstm = X_train.reshape(-1, seq_len, n_features // seq_len)
                    if X_val is not None:
                        X_val_lstm = X_val.reshape(-1, seq_len, n_features // seq_len)
                    else:
                        X_val_lstm = None
                else:
                    print("Not enough data for LSTM sequences, skipping LSTM")
                    X_train_lstm = None
            else:
                X_train_lstm = X_train
                X_val_lstm = X_val
            
            if X_train_lstm is not None:
                self.lstm_model = LSTMForecaster()
                lstm_metrics = self.lstm_model.train(
                    X_train_lstm, y_train, X_val_lstm, y_val, verbose=0
                )
                metrics['lstm'] = lstm_metrics
                print(f"LSTM trained - Val RMSE: {lstm_metrics.get('val_rmse', 'N/A')}")
            else:
                print("LSTM training skipped - insufficient data")
        else:
            print("Skipping LSTM (TensorFlow not available)")
        
        # 3. Train meta-learner (stacking)
        if self.use_meta_learner and self.lgb_model and self.lstm_model:
            print("\n[3/3] Training meta-learner...")
            
            # Get predictions from base models on training data
            lgb_train_pred = self.lgb_model.predict(X_train)
            lstm_train_pred = self.lstm_model.predict(X_train_lstm if X_train_lstm is not None else X_train)
            
            # Stack predictions
            meta_features = np.column_stack([lgb_train_pred, lstm_train_pred])
            
            # Train meta-model
            if not SKLEARN_AVAILABLE:
                print("scikit-learn not available, using simple averaging")
                self.meta_model = None
            else:
                if self.config['meta_model_type'] == 'ridge':
                    self.meta_model = Ridge(alpha=1.0)
                elif self.config['meta_model_type'] == 'lasso':
                    self.meta_model = Lasso(alpha=0.1)
                else:
                    self.meta_model = Ridge(alpha=1.0)
                
                self.meta_model.fit(meta_features, y_train)
                
                # Get meta-model weights
                if hasattr(self.meta_model, 'coef_'):
                    coefs = self.meta_model.coef_
                    print(f"Meta-model weights: LGB={coefs[0]:.3f}, LSTM={coefs[1]:.3f}")
            
            # Validate ensemble
            if X_val is not None and y_val is not None:
                ensemble_pred = self.predict(X_val, X_val_lstm if X_val_lstm is not None else X_val)
                ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_pred) ** 2))
                metrics['ensemble'] = {'val_rmse': ensemble_rmse}
                print(f"Ensemble Val RMSE: {ensemble_rmse:.4f}")
        
        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics,
            'models_used': {
                'lightgbm': self.lgb_model is not None,
                'lstm': self.lstm_model is not None,
                'meta_learner': self.meta_model is not None
            }
        }
        
        print("\n" + "=" * 60)
        print("Ensemble training complete!")
        
        return metrics
    
    def predict(
        self,
        X_lgb: np.ndarray,
        X_lstm: Optional[np.ndarray] = None,
        return_confidence: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X_lgb: Features for LightGBM
            X_lstm: Sequences for LSTM (if different from X_lgb)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        confidence_intervals = []
        
        # Get LightGBM predictions
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(X_lgb, return_confidence=False)
            predictions.append(lgb_pred)
            
            if return_confidence:
                _, lgb_conf = self.lgb_model.predict(X_lgb, return_confidence=True)
                confidence_intervals.append(lgb_conf)
        
        # Get LSTM predictions
        if self.lstm_model:
            X_lstm_input = X_lstm if X_lstm is not None else X_lgb
            lstm_pred = self.lstm_model.predict(X_lstm_input, return_confidence=False)
            predictions.append(lstm_pred)
            
            if return_confidence:
                _, lstm_conf = self.lstm_model.predict(X_lstm_input, return_confidence=True, num_simulations=50)
                confidence_intervals.append(lstm_conf)
        
        if len(predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # Combine predictions
        if self.use_meta_learner and self.meta_model:
            # Use meta-learner
            meta_features = np.column_stack(predictions)
            ensemble_pred = self.meta_model.predict(meta_features)
        else:
            # Use weighted averaging
            weights = [self.lgb_weight, self.lstm_weight][:len(predictions)]
            weights = np.array(weights) / sum(weights)  # Normalize
            
            ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        if return_confidence:
            # Combine confidence intervals (use average)
            if len(confidence_intervals) > 0:
                avg_lower = np.mean([ci[:, 0] for ci in confidence_intervals], axis=0)
                avg_upper = np.mean([ci[:, 1] for ci in confidence_intervals], axis=0)
                combined_confidence = np.column_stack([avg_lower, avg_upper])
                return ensemble_pred, combined_confidence
        
        return ensemble_pred
    
    def update_weights_dynamically(
        self,
        y_true: np.ndarray,
        lgb_pred: np.ndarray,
        lstm_pred: np.ndarray
    ) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            y_true: Actual values
            lgb_pred: LightGBM predictions
            lstm_pred: LSTM predictions
        """
        # Calculate recent RMSE for each model
        lgb_rmse = np.sqrt(np.mean((y_true - lgb_pred) ** 2))
        lstm_rmse = np.sqrt(np.mean((y_true - lstm_pred) ** 2))
        
        # Store in history
        self.performance_history['lgb_rmse'].append(lgb_rmse)
        self.performance_history['lstm_rmse'].append(lstm_rmse)
        
        # Keep only recent history
        window = self.config.get('performance_window', 10)
        self.performance_history['lgb_rmse'] = self.performance_history['lgb_rmse'][-window:]
        self.performance_history['lstm_rmse'] = self.performance_history['lstm_rmse'][-window:]
        
        # Calculate average recent performance
        avg_lgb_rmse = np.mean(self.performance_history['lgb_rmse'])
        avg_lstm_rmse = np.mean(self.performance_history['lstm_rmse'])
        
        # Inverse error weighting (better model gets higher weight)
        total_inverse_error = (1 / avg_lgb_rmse) + (1 / avg_lstm_rmse)
        
        self.lgb_weight = (1 / avg_lgb_rmse) / total_inverse_error
        self.lstm_weight = (1 / avg_lstm_rmse) / total_inverse_error
        
        print(f"Updated weights: LGB={self.lgb_weight:.3f}, LSTM={self.lstm_weight:.3f}")
    
    def get_model_contributions(self, X_lgb: np.ndarray, X_lstm: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions to understand contributions.
        
        Args:
            X_lgb: Features for LightGBM
            X_lstm: Sequences for LSTM
            
        Returns:
            Dictionary with predictions from each model
        """
        contributions = {}
        
        if self.lgb_model:
            contributions['lightgbm'] = self.lgb_model.predict(X_lgb)
        
        if self.lstm_model:
            X_lstm_input = X_lstm if X_lstm is not None else X_lgb
            contributions['lstm'] = self.lstm_model.predict(X_lstm_input)
        
        if len(contributions) > 0:
            # Calculate ensemble prediction
            if self.use_meta_learner and self.meta_model:
                meta_features = np.column_stack(list(contributions.values()))
                contributions['ensemble'] = self.meta_model.predict(meta_features)
            else:
                weights = [self.lgb_weight, self.lstm_weight][:len(contributions)]
                weights = np.array(weights) / sum(weights)
                contributions['ensemble'] = sum(w * p for w, p in zip(weights, contributions.values()))
        
        return contributions
    
    def save_ensemble(self, version: str = "1.0.0") -> Path:
        """
        Save ensemble (all models + meta-learner).
        
        Args:
            version: Model version
            
        Returns:
            Path to ensemble directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_name = f"v{version}_{timestamp}"
        ensemble_dir = self.model_dir / ensemble_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        if self.lgb_model:
            lgb_path = ensemble_dir / "lightgbm.pkl"
            joblib.dump(self.lgb_model, lgb_path)
        
        if self.lstm_model and self.lstm_model.model:
            lstm_path = ensemble_dir / "lstm.h5"
            self.lstm_model.model.save(lstm_path)
        
        # Save meta-learner
        if self.meta_model:
            meta_path = ensemble_dir / "meta_model.pkl"
            joblib.dump(self.meta_model, meta_path)
        
        # Save weights and metadata
        metadata = {
            'version': version,
            'timestamp': timestamp,
            'config': self.config,
            'weights': {
                'lightgbm': float(self.lgb_weight),
                'lstm': float(self.lstm_weight)
            },
            'performance_history': {
                k: [float(x) for x in v]
                for k, v in self.performance_history.items()
            },
            'metadata': self.metadata
        }
        
        metadata_path = ensemble_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Ensemble saved to {ensemble_dir}")
        
        return ensemble_dir
    
    def load_ensemble(self, ensemble_dir: str | Path) -> None:
        """
        Load ensemble from disk.
        
        Args:
            ensemble_dir: Path to ensemble directory
        """
        ensemble_dir = Path(ensemble_dir)
        
        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble not found: {ensemble_dir}")
        
        # Load LightGBM
        lgb_path = ensemble_dir / "lightgbm.pkl"
        if lgb_path.exists() and LIGHTGBM_AVAILABLE:
            self.lgb_model = joblib.load(lgb_path)
            print("LightGBM model loaded")
        
        # Load LSTM
        lstm_path = ensemble_dir / "lstm.h5"
        if lstm_path.exists() and TENSORFLOW_AVAILABLE:
            from tensorflow import keras
            self.lstm_model = LSTMForecaster()
            self.lstm_model.model = keras.models.load_model(lstm_path)
            print("LSTM model loaded")
        
        # Load meta-learner
        meta_path = ensemble_dir / "meta_model.pkl"
        if meta_path.exists() and SKLEARN_AVAILABLE:
            self.meta_model = joblib.load(meta_path)
            print("Meta-learner loaded")
        
        # Load metadata
        metadata_path = ensemble_dir / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', self.config)
                self.lgb_weight = data.get('weights', {}).get('lightgbm', 0.6)
                self.lstm_weight = data.get('weights', {}).get('lstm', 0.4)
                self.performance_history = data.get('performance_history', {})
                self.metadata = data.get('metadata', {})
        
        print(f"Ensemble loaded from {ensemble_dir}")
    
    def evaluate_ensemble(
        self,
        X_test_lgb: np.ndarray,
        y_test: np.ndarray,
        X_test_lstm: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance and compare with base models.
        
        Args:
            X_test_lgb: Test features for LightGBM
            y_test: Test targets
            X_test_lstm: Test sequences for LSTM
            
        Returns:
            Evaluation metrics
        """
        results = {}
        
        # LightGBM performance
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(X_test_lgb)
            lgb_rmse = np.sqrt(np.mean((y_test - lgb_pred) ** 2))
            lgb_mape = np.mean(np.abs((y_test - lgb_pred) / y_test)) * 100
            results['lightgbm'] = {'rmse': lgb_rmse, 'mape': lgb_mape}
        
        # LSTM performance
        if self.lstm_model:
            X_test_lstm_input = X_test_lstm if X_test_lstm is not None else X_test_lgb
            lstm_pred = self.lstm_model.predict(X_test_lstm_input)
            lstm_rmse = np.sqrt(np.mean((y_test - lstm_pred) ** 2))
            lstm_mape = np.mean(np.abs((y_test - lstm_pred) / y_test)) * 100
            results['lstm'] = {'rmse': lstm_rmse, 'mape': lstm_mape}
        
        # Ensemble performance
        ensemble_pred = self.predict(X_test_lgb, X_test_lstm)
        ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))
        ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        results['ensemble'] = {'rmse': ensemble_rmse, 'mape': ensemble_mape}
        
        # Calculate improvement
        if self.lgb_model:
            improvement = (lgb_rmse - ensemble_rmse) / lgb_rmse * 100
            results['improvement_over_lgb'] = improvement
            print(f"Ensemble improves over LightGBM by {improvement:.2f}%")
        
        return results


class AdaptiveEnsemble(HybridEnsemble):
    """
    Adaptive ensemble that adjusts weights based on market conditions.
    
    Different models perform better in different market regimes:
    - Trending markets: LSTM often better
    - Ranging markets: LightGBM often better
    """
    
    def detect_market_regime(self, recent_prices: np.ndarray) -> str:
        """
        Detect current market regime.
        
        Args:
            recent_prices: Recent price history
            
        Returns:
            'trending', 'ranging', or 'volatile'
        """
        if len(recent_prices) < 20:
            return 'unknown'
        
        # Calculate trend strength
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # ADX-like calculation (trend strength)
        price_range = np.max(recent_prices) - np.min(recent_prices)
        avg_price = np.mean(recent_prices)
        trend_strength = price_range / avg_price
        
        # Volatility
        volatility = np.std(returns)
        
        if volatility > 0.05:  # 5% daily volatility
            return 'volatile'
        elif trend_strength > 0.10:  # 10% range
            return 'trending'
        else:
            return 'ranging'
    
    def predict_adaptive(
        self,
        X_lgb: np.ndarray,
        recent_prices: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions with regime-adjusted weights.
        
        Args:
            X_lgb: Features for LightGBM
            recent_prices: Recent prices for regime detection
            X_lstm: Sequences for LSTM
            
        Returns:
            Predictions
        """
        regime = self.detect_market_regime(recent_prices)
        
        # Adjust weights based on regime
        if regime == 'trending':
            # LSTM better at capturing trends
            self.lstm_weight = 0.6
            self.lgb_weight = 0.4
        elif regime == 'ranging':
            # LightGBM better at mean reversion
            self.lgb_weight = 0.7
            self.lstm_weight = 0.3
        else:  # volatile or unknown
            # Equal weights
            self.lgb_weight = 0.5
            self.lstm_weight = 0.5
        
        print(f"Market regime: {regime} | Weights: LGB={self.lgb_weight:.2f}, LSTM={self.lstm_weight:.2f}")
        
        return self.predict(X_lgb, X_lstm)

