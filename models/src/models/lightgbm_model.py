"""
LightGBM Model for Cryptocurrency Price Forecasting

Kaggle competition-winning approach:
- Gradient boosting with proper hyperparameters
- Walk-forward validation
- Feature importance tracking
- Early stopping to prevent overfitting
"""

from __future__ import annotations

import asyncio
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


class LightGBMForecaster:
    """
    LightGBM-based cryptocurrency price forecaster.
    
    Uses gradient boosting to learn from technical indicators
    and price patterns. Fast training and inference.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/lightgbm"
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.feature_importances: Optional[np.ndarray] = None
        self.training_history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        self.metadata: Dict[str, Any] = {}
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default hyperparameters tuned for crypto forecasting"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'n_estimators': 100,
            'verbose': -1,
            'random_state': 42
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 10
    ) -> Dict[str, float]:
        """
        Train LightGBM model with validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features
            early_stopping_rounds: Rounds for early stopping
            
        Returns:
            Training metrics dictionary
        """
        print("Training LightGBM model...")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Initialize model
        self.model = lgb.LGBMRegressor(**self.config)
        
        # Prepare validation data
        eval_set = []
        eval_names = []
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
        else:
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=10)
            ] if X_val is not None else None
        )
        
        # Store feature importances
        self.feature_importances = self.model.feature_importances_
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        
        metrics = {'train_rmse': train_rmse}
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            metrics['val_rmse'] = val_rmse
        
        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val) if X_val is not None else 0,
            'n_features': X_train.shape[1],
            'config': self.config,
            'metrics': metrics
        }
        
        print(f"Training complete. Train RMSE: {train_rmse:.4f}")
        if 'val_rmse' in metrics:
            print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with optional confidence intervals.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Predictions (and confidence intervals if requested)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        
        if return_confidence:
            # Estimate confidence based on validation performance
            # In practice, you'd use quantile regression or ensemble variance
            std_error = self.metadata.get('metrics', {}).get('val_rmse', 0) * 1.96
            confidence_intervals = np.column_stack([
                predictions - std_error,
                predictions + std_error
            ])
            return predictions, confidence_intervals
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importances is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, version: str = "1.0.0") -> Path:
        """
        Save model and metadata to disk.
        
        Args:
            version: Model version string
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"v{version}_{timestamp}.pkl"
        metadata_filename = f"v{version}_{timestamp}_metadata.json"
        
        model_path = self.model_dir / model_filename
        metadata_path = self.model_dir / metadata_filename
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            **self.metadata,
            'version': version,
            'feature_names': self.feature_names,
            'model_path': str(model_path)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: str | Path) -> None:
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.parent / model_path.name.replace('.pkl', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
        
        print(f"Model loaded from {model_path}")
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        initial_train_size: int = 200,
        step_size: int = 30,
        forecast_horizon: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward validation (time-series cross-validation).
        
        This simulates real-world usage where model is retrained periodically.
        
        Args:
            data: Full dataset
            target_column: Target variable
            initial_train_size: Initial training set size
            step_size: How many samples to add each iteration
            forecast_horizon: How many steps ahead to predict
            
        Returns:
            List of validation results
        """
        results = []
        n_samples = len(data)
        
        for train_end in range(initial_train_size, n_samples - forecast_horizon, step_size):
            # Split data
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:train_end + forecast_horizon]
            
            # Get features and targets
            X_train = train_data.drop(columns=[target_column]).values
            y_train = train_data[target_column].values
            X_test = test_data.drop(columns=[target_column]).values
            y_test = test_data[target_column].values
            
            # Train model
            self.train(X_train, y_train)
            
            # Predict
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results.append({
                'train_end_idx': train_end,
                'test_start_idx': train_end,
                'test_end_idx': train_end + forecast_horizon,
                'rmse': rmse,
                'mape': mape,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            })
            
            print(f"Fold {len(results)}: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
        return results


# Convenience function
def train_lightgbm_for_crypto(
    crypto_id: str,
    days: int = 365,
    save_model: bool = True
) -> LightGBMForecaster:
    """
    Train a LightGBM model for a specific cryptocurrency.
    
    Args:
        crypto_id: CoinGecko ID
        days: Historical data period
        save_model: Whether to save trained model
        
    Returns:
        Trained LightGBMForecaster instance
    """
    from data.data_loader import data_loader
    from data.feature_engineering import FeatureEngineer
    
    # Load data
    data = asyncio.run(data_loader.load_crypto_data(crypto_id, days))
    
    # Engineer features
    engineer = FeatureEngineer()
    data_with_features = engineer.engineer_features(data)
    
    # Prepare for training
    X, y = engineer.prepare_features_for_training(data_with_features, target_column='close')
    
    # Split data
    train_df, val_df, test_df = data_loader.train_val_test_split(data_with_features)
    
    X_train = train_df.drop(columns=['close']).values
    y_train = train_df['close'].values
    X_val = val_df.drop(columns=['close']).values
    y_val = val_df['close'].values
    
    # Train model
    model = LightGBMForecaster()
    model.train(X_train, y_train, X_val, y_val, feature_names=engineer.feature_names)
    
    # Save if requested
    if save_model:
        model.save_model(version="1.0.0")
    
    return model

