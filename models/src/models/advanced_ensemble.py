"""
Advanced Multi-Model Ensemble for Cryptocurrency Forecasting

Sophisticated ensemble combining:
- Transformer (attention-based temporal modeling)
- Enhanced LSTM (bidirectional with attention)
- LightGBM (gradient boosting for feature interactions)

Features:
- Dynamic weight adjustment based on market conditions
- Market regime detection (trending/ranging/volatile)
- Uncertainty propagation across models
- Meta-learning with advanced stacking
- Online learning for continuous adaptation
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
from collections import defaultdict

try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")

# Import our models
from .transformer_model import TransformerForecaster, TENSORFLOW_AVAILABLE
from .enhanced_lstm import EnhancedLSTMForecaster
from .lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators.

    Identifies market states:
    - Trending (strong directional movement)
    - Ranging (sideways movement)
    - Volatile (high uncertainty)
    - Transition (changing regime)
    """

    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window

    def detect_regime(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Detect current market regime from price history.

        Args:
            prices: Recent price history

        Returns:
            Regime information dictionary
        """
        if len(prices) < self.lookback_window:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'metrics': {}
            }

        # Calculate various regime indicators
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.diff(np.log(prices))

        # Trend strength indicators
        trend_strength = self._calculate_trend_strength(prices)
        mean_reversion = self._calculate_mean_reversion(returns)
        volatility_regime = self._calculate_volatility_regime(log_returns)

        # Range detection
        range_strength = self._calculate_range_strength(prices)

        # Combine indicators
        regime_scores = {
            'trending': trend_strength,
            'ranging': range_strength,
            'volatile': volatility_regime,
            'mean_reverting': mean_reversion
        }

        # Determine primary regime
        primary_regime = max(regime_scores.items(), key=lambda x: x[1])

        return {
            'regime': primary_regime[0],
            'confidence': primary_regime[1],
            'scores': regime_scores,
            'metrics': {
                'volatility': np.std(returns),
                'trend_slope': self._calculate_trend_slope(prices),
                'range_ratio': (np.max(prices) - np.min(prices)) / np.mean(prices)
            }
        }

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression slope and R²"""
        if len(prices) < 5:
            return 0.0

        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]

        # Calculate R² for trend fit
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Combine slope significance and R²
        normalized_slope = abs(slope) / np.mean(prices)
        trend_strength = r_squared * (normalized_slope * 100)

        return min(trend_strength, 1.0)

    def _calculate_mean_reversion(self, returns: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        if len(returns) < 5:
            return 0.0

        # Autocorrelation of returns (negative indicates mean reversion)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Convert to 0-1 scale (more negative = higher mean reversion)
        mean_reversion_score = max(0, -autocorr)

        return mean_reversion_score

    def _calculate_volatility_regime(self, log_returns: np.ndarray) -> float:
        """Calculate volatility regime indicator"""
        if len(log_returns) < 5:
            return 0.0

        # Rolling volatility
        rolling_vol = np.std(log_returns)

        # Compare to historical volatility (using full history)
        historical_vol = np.std(log_returns)

        # High volatility ratio indicates volatile regime
        if historical_vol > 0:
            vol_ratio = rolling_vol / historical_vol
            # Normalize to 0-1 scale
            volatility_score = min(vol_ratio / 2, 1.0)
        else:
            volatility_score = 0.0

        return volatility_score

    def _calculate_range_strength(self, prices: np.ndarray) -> float:
        """Calculate ranging market strength"""
        if len(prices) < 5:
            return 0.0

        # Check for horizontal resistance/support levels
        price_range = np.max(prices) - np.min(prices)
        mean_price = np.mean(prices)

        if mean_price == 0:
            return 0.0

        # Low range relative to mean suggests ranging
        range_ratio = price_range / mean_price

        # Calculate time spent near range boundaries
        upper_threshold = np.max(prices) * 0.95
        lower_threshold = np.min(prices) * 1.05

        boundary_time = np.sum((prices >= upper_threshold) | (prices <= lower_threshold))
        boundary_ratio = boundary_time / len(prices)

        # Combine indicators (low range + high boundary time = ranging)
        ranging_score = boundary_ratio * (1 - min(range_ratio * 10, 1.0))

        return ranging_score

    def _calculate_trend_slope(self, prices: np.ndarray) -> float:
        """Calculate normalized trend slope"""
        if len(prices) < 2:
            return 0.0

        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize by mean price
        return slope / np.mean(prices) if np.mean(prices) > 0 else 0.0


class AdvancedEnsemble:
    """
    Advanced ensemble combining Transformer, Enhanced LSTM, and LightGBM.

    Features sophisticated model combination with regime-aware weighting,
    uncertainty quantification, and online learning capabilities.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/advanced_ensemble"
    ):
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Base models
        self.transformer: Optional[TransformerForecaster] = None
        self.enhanced_lstm: Optional[EnhancedLSTMForecaster] = None
        self.lightgbm: Optional[LightGBMForecaster] = None

        # Meta-learner for advanced stacking
        self.meta_model: Optional[Any] = None
        self.regime_detector = MarketRegimeDetector(
            lookback_window=self.config.get('regime_lookback', 20)
        )

        # Dynamic weights
        self.model_weights = {
            'transformer': self.config.get('transformer_weight', 0.4),
            'lstm': self.config.get('lstm_weight', 0.35),
            'lightgbm': self.config.get('lightgbm_weight', 0.25)
        }

        # Performance tracking
        self.performance_history = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))

        # Online learning parameters
        self.learning_rate = self.config.get('online_learning_rate', 0.01)
        self.adaptation_window = self.config.get('adaptation_window', 50)

        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default ensemble configuration"""
        return {
            # Model weights
            'transformer_weight': 0.4,
            'lstm_weight': 0.35,
            'lightgbm_weight': 0.25,

            # Meta-learning
            'use_meta_learner': True,
            'meta_model_type': 'elastic_net',  # 'ridge', 'elastic_net', 'random_forest'
            'meta_alpha': 0.1,
            'meta_l1_ratio': 0.5,

            # Regime-based weighting
            'use_regime_weighting': True,
            'regime_lookback': 20,
            'regime_weight_sensitivity': 0.3,

            # Dynamic adaptation
            'use_online_learning': True,
            'online_learning_rate': 0.01,
            'adaptation_window': 50,
            'min_confidence_threshold': 0.3,

            # Uncertainty quantification
            'uncertainty_method': 'ensemble_variance',  # 'monte_carlo', 'ensemble_variance'
            'mc_samples': 50,

            # Multi-horizon forecasting
            'horizons': [1, 7, 30],
            'horizon_weights': [1.0, 0.8, 0.6]
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        price_history: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train advanced ensemble model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names for LightGBM
            price_history: Price history for regime detection

        Returns:
            Training metrics
        """
        print("Training Advanced Multi-Model Ensemble...")
        print("=" * 60)

        metrics = {}
        trained_models = []

        # 1. Train Transformer
        if TENSORFLOW_AVAILABLE:
            print("\\n[1/4] Training Transformer model...")
            try:
                self.transformer = TransformerForecaster()

                # Prepare sequences for Transformer
                X_train_seq, y_train_seq = self.transformer.prepare_sequences(
                    np.column_stack([y_train.reshape(-1, 1), X_train])
                )

                if X_val is not None:
                    X_val_seq, y_val_seq = self.transformer.prepare_sequences(
                        np.column_stack([y_val.reshape(-1, 1), X_val])
                    )
                else:
                    X_val_seq, y_val_seq = None, None

                transformer_metrics = self.transformer.train(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0
                )
                metrics['transformer'] = transformer_metrics
                trained_models.append('transformer')
                print(f"✅ Transformer trained successfully")

            except Exception as e:
                print(f"❌ Transformer training failed: {e}")
                self.transformer = None
        else:
            print("⚠️ Skipping Transformer (TensorFlow not available)")

        # 2. Train Enhanced LSTM
        if TENSORFLOW_AVAILABLE:
            print("\\n[2/4] Training Enhanced LSTM model...")
            try:
                self.enhanced_lstm = EnhancedLSTMForecaster()

                # Prepare sequences for LSTM
                X_train_lstm, y_train_lstm = self.enhanced_lstm.prepare_sequences(
                    np.column_stack([y_train.reshape(-1, 1), X_train])
                )

                if X_val is not None:
                    X_val_lstm, y_val_lstm = self.enhanced_lstm.prepare_sequences(
                        np.column_stack([y_val.reshape(-1, 1), X_val])
                    )
                else:
                    X_val_lstm, y_val_lstm = None, None

                lstm_metrics = self.enhanced_lstm.train(
                    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, verbose=0
                )
                metrics['enhanced_lstm'] = lstm_metrics
                trained_models.append('enhanced_lstm')
                print(f"✅ Enhanced LSTM trained successfully")

            except Exception as e:
                print(f"❌ Enhanced LSTM training failed: {e}")
                self.enhanced_lstm = None
        else:
            print("⚠️ Skipping Enhanced LSTM (TensorFlow not available)")

        # 3. Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\\n[3/4] Training LightGBM model...")
            try:
                self.lightgbm = LightGBMForecaster()
                lgb_metrics = self.lightgbm.fit(
                    X_train, y_train, X_val, y_val, feature_names
                )
                metrics['lightgbm'] = lgb_metrics
                trained_models.append('lightgbm')
                print(f"✅ LightGBM trained successfully")

            except Exception as e:
                print(f"❌ LightGBM training failed: {e}")
                self.lightgbm = None
        else:
            print("⚠️ Skipping LightGBM (LightGBM not available)")

        if not trained_models:
            raise RuntimeError("No models were successfully trained!")

        # 4. Train Meta-Learner
        if self.config['use_meta_learner'] and len(trained_models) >= 2:
            print("\\n[4/4] Training meta-learner...")
            self._train_meta_learner(X_train, y_train, X_val, y_val, price_history)

        # Initialize regime-based weights
        if price_history is not None and self.config['use_regime_weighting']:
            self._initialize_regime_weights(price_history)

        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics,
            'trained_models': trained_models,
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val) if X_val is not None else 0
        }

        print("\\n" + "=" * 60)
        print("🎉 Advanced ensemble training complete!")
        print(f"📊 Trained models: {', '.join(trained_models)}")

        return metrics

    def _train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        price_history: Optional[np.ndarray] = None
    ) -> None:
        """Train meta-learner for stacking"""
        try:
            # Get base model predictions
            base_predictions = self._get_base_predictions(X_train, for_training=True)

            if len(base_predictions) < 2:
                print("⚠️ Need at least 2 base models for meta-learning")
                return

            # Stack predictions
            stacked_features = np.column_stack(list(base_predictions.values()))

            # Add regime features if available
            if price_history is not None and len(price_history) >= len(y_train):
                regime_features = self._extract_regime_features(
                    price_history[-len(y_train):]
                )
                stacked_features = np.column_stack([stacked_features, regime_features])

            # Create meta-learner
            if self.config['meta_model_type'] == 'elastic_net':
                self.meta_model = ElasticNet(
                    alpha=self.config['meta_alpha'],
                    l1_ratio=self.config['meta_l1_ratio'],
                    random_state=42
                )
            elif self.config['meta_model_type'] == 'random_forest':
                self.meta_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            else:  # ridge
                self.meta_model = Ridge(alpha=self.config['meta_alpha'])

            # Train meta-learner
            self.meta_model.fit(stacked_features, y_train)

            # Print meta-learner coefficients if available
            if hasattr(self.meta_model, 'coef_'):
                coef_names = list(base_predictions.keys())
                if price_history is not None:
                    coef_names.extend(['regime_trend', 'regime_volatility', 'regime_mean_reversion'])

                print("📈 Meta-learner coefficients:")
                for name, coef in zip(coef_names, self.meta_model.coef_):
                    print(f"   {name}: {coef:.4f}")

            print("✅ Meta-learner trained successfully")

        except Exception as e:
            print(f"❌ Meta-learner training failed: {e}")
            self.meta_model = None

    def _get_base_predictions(
        self,
        X: np.ndarray,
        for_training: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get predictions from all available base models"""
        predictions = {}

        # Transformer predictions
        if self.transformer is not None:
            try:
                if for_training:
                    # Need to create sequences
                    dummy_targets = np.zeros((X.shape[0], 1))
                    X_seq, _ = self.transformer.prepare_sequences(
                        np.column_stack([dummy_targets, X])
                    )
                    if len(X_seq) > 0:
                        pred = self.transformer.predict(X_seq)
                        if isinstance(pred, dict):
                            # Multi-horizon, take first horizon
                            pred = list(pred.values())[0]
                        predictions['transformer'] = pred.flatten()
                else:
                    pred = self.transformer.predict(X)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    predictions['transformer'] = pred.flatten()
            except Exception as e:
                print(f"Warning: Transformer prediction failed: {e}")

        # Enhanced LSTM predictions
        if self.enhanced_lstm is not None:
            try:
                if for_training:
                    dummy_targets = np.zeros((X.shape[0], 1))
                    X_seq, _ = self.enhanced_lstm.prepare_sequences(
                        np.column_stack([dummy_targets, X])
                    )
                    if len(X_seq) > 0:
                        pred = self.enhanced_lstm.predict(X_seq)
                        if isinstance(pred, dict):
                            pred = list(pred.values())[0]
                        predictions['enhanced_lstm'] = pred.flatten()
                else:
                    pred = self.enhanced_lstm.predict(X)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    predictions['enhanced_lstm'] = pred.flatten()
            except Exception as e:
                print(f"Warning: Enhanced LSTM prediction failed: {e}")

        # LightGBM predictions
        if self.lightgbm is not None:
            try:
                pred = self.lightgbm.predict(X)
                predictions['lightgbm'] = pred.flatten()
            except Exception as e:
                print(f"Warning: LightGBM prediction failed: {e}")

        return predictions

    def _extract_regime_features(self, price_history: np.ndarray) -> np.ndarray:
        """Extract regime-based features for meta-learning"""
        regime_info = self.regime_detector.detect_regime(price_history)

        features = [
            regime_info['metrics'].get('trend_slope', 0),
            regime_info['metrics'].get('volatility', 0),
            regime_info['scores'].get('mean_reverting', 0)
        ]

        return np.array(features).reshape(1, -1)

    def _initialize_regime_weights(self, price_history: np.ndarray) -> None:
        """Initialize weights based on current market regime"""
        regime_info = self.regime_detector.detect_regime(price_history)
        regime = regime_info['regime']

        # Adjust weights based on regime
        if regime == 'trending':
            # Transformer and LSTM better at capturing trends
            self.model_weights.update({
                'transformer': 0.45,
                'enhanced_lstm': 0.40,
                'lightgbm': 0.15
            })
        elif regime == 'ranging':
            # LightGBM better at mean reversion
            self.model_weights.update({
                'transformer': 0.25,
                'enhanced_lstm': 0.30,
                'lightgbm': 0.45
            })
        elif regime == 'volatile':
            # More balanced approach in volatile markets
            self.model_weights.update({
                'transformer': 0.35,
                'enhanced_lstm': 0.35,
                'lightgbm': 0.30
            })

        print(f"🎯 Initialized weights for {regime} market:")
        for model, weight in self.model_weights.items():
            print(f"   {model}: {weight:.3f}")

    def predict(
        self,
        X: np.ndarray,
        price_history: Optional[np.ndarray] = None,
        return_confidence: bool = False,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """
        Make ensemble predictions.

        Args:
            X: Input features
            price_history: Recent price history for regime detection
            return_confidence: Whether to return confidence intervals
            return_individual: Whether to return individual model predictions

        Returns:
            Ensemble predictions (and optionally confidence intervals)
        """
        # Get base model predictions
        base_predictions = self._get_base_predictions(X, for_training=False)

        if not base_predictions:
            raise ValueError("No trained models available for prediction")

        # Update weights based on current regime
        if price_history is not None and self.config['use_regime_weighting']:
            self._update_regime_weights(price_history)

        # Combine predictions
        if self.meta_model is not None:
            # Use meta-learner
            stacked_features = np.column_stack(list(base_predictions.values()))

            # Add regime features
            if price_history is not None:
                regime_features = self._extract_regime_features(price_history)
                # Broadcast regime features to match batch size
                regime_features = np.repeat(regime_features, len(stacked_features), axis=0)
                stacked_features = np.column_stack([stacked_features, regime_features])

            ensemble_pred = self.meta_model.predict(stacked_features)
        else:
            # Use weighted averaging
            ensemble_pred = self._weighted_average(base_predictions)

        if return_individual:
            base_predictions['ensemble'] = ensemble_pred
            return base_predictions

        if return_confidence:
            # Calculate ensemble uncertainty
            confidence_intervals = self._calculate_uncertainty(base_predictions, ensemble_pred)
            return ensemble_pred, confidence_intervals

        return ensemble_pred

    def _update_regime_weights(self, price_history: np.ndarray) -> None:
        """Update model weights based on current market regime"""
        regime_info = self.regime_detector.detect_regime(price_history)
        regime = regime_info['regime']
        confidence = regime_info['confidence']

        # Only update if confidence is high enough
        if confidence < self.config['min_confidence_threshold']:
            return

        # Get regime-specific weights
        regime_weights = self._get_regime_weights(regime)

        # Interpolate between current and regime weights
        sensitivity = self.config['regime_weight_sensitivity']

        for model in self.model_weights:
            if model in regime_weights:
                current = self.model_weights[model]
                target = regime_weights[model]
                self.model_weights[model] = (
                    current * (1 - sensitivity) + target * sensitivity
                )

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model in self.model_weights:
                self.model_weights[model] /= total_weight

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get optimal weights for a given market regime"""
        regime_weights = {
            'trending': {
                'transformer': 0.45,
                'enhanced_lstm': 0.40,
                'lightgbm': 0.15
            },
            'ranging': {
                'transformer': 0.25,
                'enhanced_lstm': 0.30,
                'lightgbm': 0.45
            },
            'volatile': {
                'transformer': 0.35,
                'enhanced_lstm': 0.35,
                'lightgbm': 0.30
            },
            'mean_reverting': {
                'transformer': 0.20,
                'enhanced_lstm': 0.30,
                'lightgbm': 0.50
            }
        }

        return regime_weights.get(regime, self.model_weights)

    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted average of predictions"""
        total_weight = 0
        weighted_sum = None

        for model, pred in predictions.items():
            if model in self.model_weights:
                weight = self.model_weights[model]
                total_weight += weight

                if weighted_sum is None:
                    weighted_sum = weight * pred
                else:
                    weighted_sum += weight * pred

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Equal weights if no weights defined
            return np.mean(list(predictions.values()), axis=0)

    def _calculate_uncertainty(
        self,
        base_predictions: Dict[str, np.ndarray],
        ensemble_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate uncertainty intervals for ensemble predictions"""
        if self.config['uncertainty_method'] == 'ensemble_variance':
            # Use variance across models as uncertainty measure
            pred_array = np.array(list(base_predictions.values()))
            pred_std = np.std(pred_array, axis=0)

            # Create confidence intervals (assuming normal distribution)
            lower_bound = ensemble_pred - 1.96 * pred_std
            upper_bound = ensemble_pred + 1.96 * pred_std

            return np.column_stack([lower_bound, upper_bound])

        elif self.config['uncertainty_method'] == 'monte_carlo':
            # Use Monte Carlo from individual models if available
            # This is a simplified implementation
            # In practice, you'd collect MC samples from each model

            # For now, use ensemble variance as fallback
            return self._calculate_uncertainty(base_predictions, ensemble_pred)

        else:
            # Default: simple standard error
            pred_array = np.array(list(base_predictions.values()))
            pred_std = np.std(pred_array, axis=0)
            se = pred_std / np.sqrt(len(base_predictions))

            lower_bound = ensemble_pred - 1.96 * se
            upper_bound = ensemble_pred + 1.96 * se

            return np.column_stack([lower_bound, upper_bound])

    def update_online(
        self,
        X_new: np.ndarray,
        y_true: np.ndarray,
        price_history: Optional[np.ndarray] = None
    ) -> None:
        """
        Online learning update based on new observations.

        Args:
            X_new: New features
            y_true: True targets
            price_history: Recent price history
        """
        if not self.config['use_online_learning']:
            return

        try:
            # Get current predictions
            base_predictions = self._get_base_predictions(X_new)
            ensemble_pred = self.predict(X_new, price_history)

            # Calculate errors
            errors = {}
            for model, pred in base_predictions.items():
                if len(pred) == len(y_true):
                    error = mean_squared_error(y_true, pred)
                    errors[model] = error

            # Update performance history
            for model, error in errors.items():
                self.performance_history[model].append(error)

                # Keep only recent history
                if len(self.performance_history[model]) > self.adaptation_window:
                    self.performance_history[model] = self.performance_history[model][-self.adaptation_window:]

            # Update weights based on recent performance
            self._update_weights_from_performance()

            print(f"📊 Online learning update complete. Updated weights:")
            for model, weight in self.model_weights.items():
                print(f"   {model}: {weight:.4f}")

        except Exception as e:
            print(f"⚠️ Online learning update failed: {e}")

    def _update_weights_from_performance(self) -> None:
        """Update model weights based on recent performance"""
        if not self.performance_history:
            return

        # Calculate recent performance scores (lower error = higher score)
        performance_scores = {}
        for model in self.model_weights:
            if model in self.performance_history and self.performance_history[model]:
                recent_errors = self.performance_history[model][-10:]  # Last 10 observations
                avg_error = np.mean(recent_errors)
                # Convert error to score (lower error = higher score)
                performance_scores[model] = 1.0 / (1.0 + avg_error)

        if not performance_scores:
            return

        # Update weights using exponential moving average
        lr = self.learning_rate
        total_score = sum(performance_scores.values())

        for model in self.model_weights:
            if model in performance_scores:
                target_weight = performance_scores[model] / total_score
                current_weight = self.model_weights[model]
                self.model_weights[model] = (1 - lr) * current_weight + lr * target_weight

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model in self.model_weights:
                self.model_weights[model] /= total_weight

    def evaluate_ensemble(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        price_history: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive ensemble evaluation.

        Args:
            X_test: Test features
            y_test: Test targets
            price_history: Price history for regime analysis

        Returns:
            Evaluation metrics
        """
        results = {}

        # Get individual model predictions
        individual_preds = self.predict(X_test, price_history, return_individual=True)

        if not isinstance(individual_preds, dict):
            print("Warning: Could not get individual predictions")
            return {}

        ensemble_pred = individual_preds.pop('ensemble', None)

        # Evaluate each model
        for model, pred in individual_preds.items():
            if len(pred) == len(y_test):
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

                results[f'{model}_rmse'] = rmse
                results[f'{model}_mae'] = mae
                results[f'{model}_mape'] = mape

        # Evaluate ensemble
        if ensemble_pred is not None and len(ensemble_pred) == len(y_test):
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

            results['ensemble_rmse'] = ensemble_rmse
            results['ensemble_mae'] = ensemble_mae
            results['ensemble_mape'] = ensemble_mape

            # Calculate improvement over best individual model
            individual_rmses = [results[k] for k in results.keys() if 'rmse' in k and k != 'ensemble_rmse']
            if individual_rmses:
                best_individual_rmse = min(individual_rmses)
                improvement = (best_individual_rmse - ensemble_rmse) / best_individual_rmse * 100
                results['improvement_pct'] = improvement

                print(f"🎯 Ensemble improves over best individual model by {improvement:.2f}%")

        return results

    def save_ensemble(self, version: str = "1.0.0") -> Path:
        """
        Save complete ensemble.

        Args:
            version: Model version

        Returns:
            Path to ensemble directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_name = f"advanced_ensemble_v{version}_{timestamp}"
        ensemble_dir = self.model_dir / ensemble_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save individual models
        saved_models = []

        if self.transformer is not None:
            try:
                transformer_path = self.transformer.save_model(version)
                saved_models.append(('transformer', str(transformer_path)))
            except Exception as e:
                print(f"Warning: Failed to save transformer: {e}")

        if self.enhanced_lstm is not None:
            try:
                lstm_path = self.enhanced_lstm.save_model(version)
                saved_models.append(('enhanced_lstm', str(lstm_path)))
            except Exception as e:
                print(f"Warning: Failed to save enhanced LSTM: {e}")

        if self.lightgbm is not None:
            try:
                lgb_path = ensemble_dir / "lightgbm_model.pkl"
                joblib.dump(self.lightgbm, lgb_path)
                saved_models.append(('lightgbm', str(lgb_path)))
            except Exception as e:
                print(f"Warning: Failed to save LightGBM: {e}")

        # Save meta-learner
        if self.meta_model is not None:
            try:
                meta_path = ensemble_dir / "meta_model.pkl"
                joblib.dump(self.meta_model, meta_path)
            except Exception as e:
                print(f"Warning: Failed to save meta-model: {e}")

        # Save ensemble metadata
        ensemble_metadata = {
            'version': version,
            'timestamp': timestamp,
            'config': self.config,
            'model_weights': self.model_weights,
            'saved_models': saved_models,
            'performance_history': {
                k: list(v) for k, v in self.performance_history.items()
            },
            'metadata': self.metadata
        }

        metadata_path = ensemble_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)

        print(f"💾 Advanced ensemble saved to {ensemble_dir}")
        return ensemble_dir

    def load_ensemble(self, ensemble_dir: str | Path) -> None:
        """
        Load complete ensemble.

        Args:
            ensemble_dir: Path to ensemble directory
        """
        ensemble_dir = Path(ensemble_dir)

        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble not found: {ensemble_dir}")

        # Load metadata
        metadata_path = ensemble_dir / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', self.config)
                self.model_weights = data.get('model_weights', self.model_weights)
                performance_hist = data.get('performance_history', {})
                self.performance_history = defaultdict(list, {
                    k: v for k, v in performance_hist.items()
                })
                self.metadata = data.get('metadata', {})
                saved_models = data.get('saved_models', [])

        print(f"📁 Loading advanced ensemble from {ensemble_dir}")

        # Load individual models
        for model_type, model_path in saved_models:
            try:
                if model_type == 'transformer' and TENSORFLOW_AVAILABLE:
                    self.transformer = TransformerForecaster()
                    self.transformer.load_model(model_path)
                    print("✅ Transformer loaded")

                elif model_type == 'enhanced_lstm' and TENSORFLOW_AVAILABLE:
                    self.enhanced_lstm = EnhancedLSTMForecaster()
                    self.enhanced_lstm.load_model(model_path)
                    print("✅ Enhanced LSTM loaded")

                elif model_type == 'lightgbm':
                    self.lightgbm = joblib.load(model_path)
                    print("✅ LightGBM loaded")

            except Exception as e:
                print(f"⚠️ Failed to load {model_type}: {e}")

        # Load meta-learner
        meta_path = ensemble_dir / "meta_model.pkl"
        if meta_path.exists():
            try:
                self.meta_model = joblib.load(meta_path)
                print("✅ Meta-learner loaded")
            except Exception as e:
                print(f"⚠️ Failed to load meta-learner: {e}")

        print(f"🎉 Advanced ensemble loaded successfully")


def create_ensemble_from_config(config_path: str) -> AdvancedEnsemble:
    """
    Create ensemble from configuration file.

    Args:
        config_path: Path to configuration JSON file

    Returns:
        Configured ensemble
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    return AdvancedEnsemble(config=config)