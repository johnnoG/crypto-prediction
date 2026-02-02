"""
LightGBM Model for Cryptocurrency Prediction
Provides gradient boosting baseline for ensemble methods
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import mlflow
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMForecaster:
    """LightGBM-based cryptocurrency price forecasting model"""

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 model_dir: str = "models/artifacts/lightgbm"):
        """
        Initialize LightGBM forecaster

        Args:
            config: Model configuration parameters
            model_dir: Directory to save model artifacts
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")

        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.models = {}  # One model per prediction horizon
        self.feature_names = []
        self.is_fitted = False

        # Validation history
        self.validation_history = []
        self.feature_importance = {}

    def _default_config(self) -> Dict[str, Any]:
        """Default LightGBM configuration"""
        return {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'prediction_horizons': [1, 7, 30],
            'early_stopping_rounds': 10
        }

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare features for LightGBM (flatten sequences)

        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)

        Returns:
            Flattened features of shape (n_samples, sequence_length * n_features)
        """
        if len(X.shape) == 3:
            # Flatten sequence data for traditional ML
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(n_samples, seq_len * n_features)

            # Create feature names if not set
            if not self.feature_names:
                self.feature_names = [f'feature_{i}_{j}' for i in range(seq_len) for j in range(n_features)]

            return X_flat
        return X

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train LightGBM models for each prediction horizon

        Args:
            X: Training features of shape (n_samples, sequence_length, n_features)
            y: Training targets of shape (n_samples, n_horizons)
            validation_data: Optional validation data
            **kwargs: Additional training parameters

        Returns:
            Training history dictionary
        """
        logger.info("Starting LightGBM training...")

        # Prepare features
        X_flat = self._prepare_features(X)

        # Prepare validation data if provided
        X_val_flat = None
        y_val = None
        if validation_data is not None:
            X_val_flat = self._prepare_features(validation_data[0])
            y_val = validation_data[1]

        # Get prediction horizons
        horizons = self.config.get('prediction_horizons', [1, 7, 30])
        n_horizons = len(horizons)

        # Ensure y has correct shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if y.shape[1] != n_horizons:
            logger.warning(f"Target shape {y.shape} doesn't match horizons {n_horizons}. Using first column.")
            y = y[:, :1]
            horizons = horizons[:1]

        # Train model for each horizon
        history = {'train_mae': [], 'val_mae': [], 'feature_importance': {}}

        for i, horizon in enumerate(horizons):
            logger.info(f"Training model for {horizon}-day horizon...")

            # Prepare target for this horizon
            y_horizon = y[:, i] if y.shape[1] > i else y[:, 0]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_flat, label=y_horizon, feature_name=self.feature_names)

            valid_sets = [train_data]
            valid_names = ['train']

            if X_val_flat is not None and y_val is not None:
                y_val_horizon = y_val[:, i] if y_val.shape[1] > i else y_val[:, 0]
                val_data = lgb.Dataset(X_val_flat, label=y_val_horizon,
                                     feature_name=self.feature_names, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('valid')

            # Train model
            model = lgb.train(
                params=self.config,
                train_set=train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                num_boost_round=self.config.get('n_estimators', 100),
                callbacks=[
                    lgb.early_stopping(self.config.get('early_stopping_rounds', 10)),
                    lgb.log_evaluation(period=10)
                ]
            )

            self.models[f'horizon_{horizon}'] = model

            # Calculate metrics
            train_pred = model.predict(X_flat)
            train_mae = np.mean(np.abs(train_pred - y_horizon))
            history['train_mae'].append(train_mae)

            if X_val_flat is not None:
                val_pred = model.predict(X_val_flat)
                val_mae = np.mean(np.abs(val_pred - y_val_horizon))
                history['val_mae'].append(val_mae)

            # Store feature importance
            importance = model.feature_importance(importance_type='gain')
            history['feature_importance'][f'horizon_{horizon}'] = dict(zip(self.feature_names, importance))

            logger.info(f"Horizon {horizon} - Train MAE: {train_mae:.4f}")
            if X_val_flat is not None:
                logger.info(f"Horizon {horizon} - Val MAE: {val_mae:.4f}")

        self.is_fitted = True
        self.validation_history = history

        logger.info("LightGBM training completed")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for all horizons

        Args:
            X: Input features

        Returns:
            Predictions for all horizons
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_flat = self._prepare_features(X)

        predictions = []
        horizons = self.config.get('prediction_horizons', [1, 7, 30])

        for horizon in horizons:
            model_key = f'horizon_{horizon}'
            if model_key in self.models:
                pred = self.models[model_key].predict(X_flat)
                predictions.append(pred)
            else:
                # Fallback to first model if specific horizon not available
                pred = list(self.models.values())[0].predict(X_flat)
                predictions.append(pred)

        return np.column_stack(predictions)

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using quantile regression

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X)

        # Simple uncertainty estimation based on feature importance variance
        uncertainties = []
        X_flat = self._prepare_features(X)

        for horizon in self.config.get('prediction_horizons', [1, 7, 30]):
            model_key = f'horizon_{horizon}'
            if model_key in self.models:
                model = self.models[model_key]

                # Use prediction variance as uncertainty proxy
                # This is a simplified approach - in practice, you'd use quantile regression
                base_pred = model.predict(X_flat)

                # Add small noise and predict multiple times for uncertainty estimation
                uncertainty_preds = []
                for _ in range(10):
                    noise = np.random.normal(0, 0.01, X_flat.shape)
                    noisy_pred = model.predict(X_flat + noise)
                    uncertainty_preds.append(noisy_pred)

                uncertainty = np.std(uncertainty_preds, axis=0)
                uncertainties.append(uncertainty)
            else:
                uncertainties.append(np.ones(len(X_flat)) * 0.1)  # Default uncertainty

        uncertainties = np.column_stack(uncertainties)
        return predictions, uncertainties

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Input features
            y: True targets

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)

        # Ensure y has correct shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Calculate metrics for each horizon
        metrics = {}
        horizons = self.config.get('prediction_horizons', [1, 7, 30])

        for i, horizon in enumerate(horizons):
            if i < predictions.shape[1] and i < y.shape[1]:
                pred_horizon = predictions[:, i]
                y_horizon = y[:, i]

                mae = np.mean(np.abs(pred_horizon - y_horizon))
                mse = np.mean((pred_horizon - y_horizon) ** 2)
                rmse = np.sqrt(mse)

                metrics[f'mae_horizon_{horizon}'] = mae
                metrics[f'mse_horizon_{horizon}'] = mse
                metrics[f'rmse_horizon_{horizon}'] = rmse

        # Overall metrics
        metrics['mae'] = np.mean([v for k, v in metrics.items() if 'mae_horizon' in k])
        metrics['mse'] = np.mean([v for k, v in metrics.items() if 'mse_horizon' in k])
        metrics['rmse'] = np.mean([v for k, v in metrics.items() if 'rmse_horizon' in k])

        return metrics

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models"""
        return self.validation_history.get('feature_importance', {})

    def save_model(self, save_path: str):
        """
        Save LightGBM models and configuration

        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_file = save_path / f"{model_name}.txt"
            model.save_model(str(model_file))

        # Save configuration and metadata
        config_file = save_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'model_files': list(self.models.keys())
            }, f, indent=2)

        # Save validation history
        history_file = save_path / "validation_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.validation_history, f, indent=2, default=str)

        logger.info(f"LightGBM model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load LightGBM models and configuration

        Args:
            load_path: Path to load model from
        """
        load_path = Path(load_path)

        # Load configuration
        config_file = load_path / "config.json"
        with open(config_file, 'r') as f:
            saved_data = json.load(f)

        self.config = saved_data['config']
        self.feature_names = saved_data['feature_names']
        self.is_fitted = saved_data['is_fitted']

        # Load models
        self.models = {}
        for model_name in saved_data['model_files']:
            model_file = load_path / f"{model_name}.txt"
            self.models[model_name] = lgb.Booster(model_file=str(model_file))

        # Load validation history
        history_file = load_path / "validation_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.validation_history = json.load(f)

        logger.info(f"LightGBM model loaded from {load_path}")

    def create_feature_importance_plot(self) -> Dict[str, Any]:
        """
        Create feature importance visualization data

        Returns:
            Dictionary with plot data for visualization
        """
        if not self.validation_history.get('feature_importance'):
            return {}

        plot_data = {}
        for horizon, importance in self.validation_history['feature_importance'].items():
            # Get top 20 most important features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

            plot_data[horizon] = {
                'features': [item[0] for item in sorted_features],
                'importance': [item[1] for item in sorted_features]
            }

        return plot_data

if __name__ == "__main__":
    # Example usage
    if LIGHTGBM_AVAILABLE:
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(1000, 60, 10)  # (samples, timesteps, features)
        y = np.random.randn(1000, 3)       # (samples, horizons)

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Initialize and train model
        config = {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'prediction_horizons': [1, 7, 30]
        }

        model = LightGBMForecaster(config=config)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val))

        # Make predictions
        predictions = model.predict(X_val)
        print(f"Predictions shape: {predictions.shape}")

        # Evaluate
        metrics = model.evaluate(X_val, y_val)
        print("Evaluation metrics:", metrics)

        # Get feature importance
        importance = model.get_feature_importance()
        print("Feature importance available for:", list(importance.keys()))
    else:
        print("LightGBM not available. Install with: pip install lightgbm")