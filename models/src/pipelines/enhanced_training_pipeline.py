"""
Enhanced Production Training Pipeline

Advanced training pipeline featuring:
- Walk-Forward Cross-Validation with expanding windows
- Time-series aware validation strategies
- Automated retraining workflows with smart scheduling
- Advanced feature engineering and selection
- Model ensemble with dynamic weighting
- Performance monitoring and drift detection
- Professional logging and visualization
"""

from __future__ import annotations

import os
import json
import time
import uuid
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import our modules
from ..models.transformer_model import TransformerForecaster
from ..models.enhanced_lstm import EnhancedLSTMForecaster
from ..models.lightgbm_model import LightGBMForecaster
from ..models.advanced_ensemble import AdvancedEnsemble
from ..mlflow_advanced.experiment_manager import AdvancedExperimentManager, create_experiment_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Walk-forward validation configuration"""
    min_train_size: int = 1000  # Minimum training set size
    max_train_size: Optional[int] = None  # Maximum training set size (None = expanding)
    validation_size: int = 100  # Size of each validation set
    step_size: int = 50  # Step size for walk-forward
    purged_size: int = 0  # Purged samples between train and validation
    expanding_window: bool = True  # Use expanding vs rolling window
    embargo_size: int = 0  # Embargo period to prevent data leakage


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration"""
    enable_feature_selection: bool = True
    selection_method: str = "mutual_info"  # mutual_info, f_test, correlation
    max_features: Optional[int] = None
    min_feature_importance: float = 0.001
    enable_feature_creation: bool = True
    technical_indicators: List[str] = None
    lag_features: List[int] = None
    rolling_windows: List[int] = None

    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = ["sma", "ema", "rsi", "macd", "bollinger"]
        if self.lag_features is None:
            self.lag_features = [1, 2, 3, 5, 10]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]


@dataclass
class RetrainingConfig:
    """Automated retraining configuration"""
    enable_auto_retrain: bool = True
    retrain_frequency: str = "daily"  # daily, weekly, monthly, adaptive
    performance_threshold: float = 0.05  # Retrain if performance degrades by this amount
    drift_threshold: float = 0.1  # Data drift threshold
    min_retrain_interval: int = 24  # Minimum hours between retraining
    max_retrain_attempts: int = 3  # Maximum retrain attempts before alert


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training pipeline configuration"""
    experiment_name: str
    model_types: List[str] = None
    validation_config: ValidationConfig = None
    feature_config: FeatureEngineeringConfig = None
    retrain_config: RetrainingConfig = None
    enable_ensemble: bool = True
    ensemble_weights: Optional[Dict[str, float]] = None
    use_advanced_optimization: bool = True
    monitoring_enabled: bool = True
    artifact_storage: str = "models/artifacts/enhanced"

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["transformer", "lstm", "lightgbm"]
        if self.validation_config is None:
            self.validation_config = ValidationConfig()
        if self.feature_config is None:
            self.feature_config = FeatureEngineeringConfig()
        if self.retrain_config is None:
            self.retrain_config = RetrainingConfig()


class WalkForwardValidator:
    """Walk-forward cross-validation for time series"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def create_splits(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward validation splits"""
        splits = []
        n_samples = len(data)

        # Calculate initial training end
        train_end = self.config.min_train_size

        while train_end + self.config.validation_size + self.config.purged_size <= n_samples:
            # Training indices
            if self.config.expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - (self.config.max_train_size or self.config.min_train_size))

            train_indices = np.arange(train_start, train_end)

            # Validation indices (with purge and embargo)
            val_start = train_end + self.config.purged_size
            val_end = val_start + self.config.validation_size

            if val_end + self.config.embargo_size <= n_samples:
                val_indices = np.arange(val_start, val_end)
                splits.append((train_indices, val_indices))

            # Move forward
            train_end += self.config.step_size

        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits

    def validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, List[float]]:
        """Perform walk-forward validation"""

        # Convert to DataFrame for easier handling
        data_df = pd.DataFrame(X, columns=feature_names)
        data_df['target'] = y

        splits = self.create_splits(data_df)

        fold_scores = defaultdict(list)
        predictions = []
        actuals = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold + 1}/{len(splits)}")

            # Prepare training and validation data
            train_X = X[train_idx]
            train_y = y[train_idx]
            val_X = X[val_idx]
            val_y = y[val_idx]

            try:
                # Train model on fold
                if hasattr(model, 'train'):
                    # Prepare sequences for deep learning models
                    if hasattr(model, 'prepare_sequences'):
                        # Combine features with target for sequence preparation
                        train_data = np.column_stack([train_y.reshape(-1, 1), train_X])
                        val_data = np.column_stack([val_y.reshape(-1, 1), val_X])

                        X_train_seq, y_train_seq = model.prepare_sequences(train_data)
                        X_val_seq, y_val_seq = model.prepare_sequences(val_data)

                        if len(X_train_seq) > 0 and len(X_val_seq) > 0:
                            model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0)
                            pred = model.predict(X_val_seq)

                            if isinstance(pred, dict):
                                pred = list(pred.values())[0]  # Take first prediction horizon
                            pred = pred.flatten()
                            actual = y_val_seq
                        else:
                            continue  # Skip if no sequences generated
                    else:
                        # Traditional ML models
                        model.train(train_X, train_y, val_X, val_y, feature_names)
                        pred = model.predict(val_X).flatten()
                        actual = val_y
                else:
                    # Fallback for sklearn-style models
                    model.fit(train_X, train_y)
                    pred = model.predict(val_X).flatten()
                    actual = val_y

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual, pred))
                mae = mean_absolute_error(actual, pred)

                # Calculate MAPE safely
                try:
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                    mape = mape if np.isfinite(mape) else float('inf')
                except:
                    mape = float('inf')

                # Calculate R²
                try:
                    r2 = r2_score(actual, pred)
                    r2 = r2 if np.isfinite(r2) else -float('inf')
                except:
                    r2 = -float('inf')

                fold_scores['rmse'].append(rmse)
                fold_scores['mae'].append(mae)
                fold_scores['mape'].append(mape)
                fold_scores['r2'].append(r2)

                predictions.extend(pred)
                actuals.extend(actual)

                logger.debug(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            except Exception as e:
                logger.warning(f"Error in fold {fold + 1}: {e}")
                continue

        # Calculate overall metrics
        if predictions and actuals:
            overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
            overall_mae = mean_absolute_error(actuals, predictions)

            try:
                overall_mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
                overall_mape = overall_mape if np.isfinite(overall_mape) else float('inf')
            except:
                overall_mape = float('inf')

            try:
                overall_r2 = r2_score(actuals, predictions)
                overall_r2 = overall_r2 if np.isfinite(overall_r2) else -float('inf')
            except:
                overall_r2 = -float('inf')

            fold_scores['overall_rmse'] = [overall_rmse]
            fold_scores['overall_mae'] = [overall_mae]
            fold_scores['overall_mape'] = [overall_mape]
            fold_scores['overall_r2'] = [overall_r2]

        return dict(fold_scores)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for time series"""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.feature_selector = None
        self.selected_features = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        logger.info("Engineering features...")

        # Make a copy to avoid modifying original
        engineered_df = df.copy()

        if self.config.enable_feature_creation:
            # Add lag features
            engineered_df = self._add_lag_features(engineered_df)

            # Add rolling window features
            engineered_df = self._add_rolling_features(engineered_df)

            # Add technical indicators
            engineered_df = self._add_technical_indicators(engineered_df)

            # Add time-based features
            engineered_df = self._add_time_features(engineered_df)

        # Handle missing values
        engineered_df = self._handle_missing_values(engineered_df)

        logger.info(f"Feature engineering completed. Features: {len(engineered_df.columns)} -> {len(engineered_df.columns)}")
        return engineered_df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            for lag in self.config.lag_features:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            for window in self.config.rolling_windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window).max()

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        if 'close' not in df.columns:
            return df

        close = df['close']

        if 'sma' in self.config.technical_indicators:
            # Simple Moving Average
            for window in [10, 20, 50]:
                df[f'sma_{window}'] = close.rolling(window).mean()

        if 'ema' in self.config.technical_indicators:
            # Exponential Moving Average
            for window in [10, 20, 50]:
                df[f'ema_{window}'] = close.ewm(span=window).mean()

        if 'rsi' in self.config.technical_indicators:
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        if 'macd' in self.config.technical_indicators:
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        if 'bollinger' in self.config.technical_indicators:
            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Forward fill price data
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

        # Interpolate numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(df[col].median())

        return df

    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select best features"""
        if not self.config.enable_feature_selection:
            return X, feature_names

        logger.info("Selecting features...")

        # Remove features with very low variance
        variances = np.var(X, axis=0)
        high_var_mask = variances > 1e-10
        X_filtered = X[:, high_var_mask]
        features_filtered = [feature_names[i] for i, mask in enumerate(high_var_mask) if mask]

        if len(features_filtered) == 0:
            logger.warning("No features passed variance filter")
            return X, feature_names

        # Apply feature selection method
        if self.config.selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k='all')
        elif self.config.selection_method == "f_test":
            selector = SelectKBest(score_func=f_regression, k='all')
        else:
            # Correlation-based selection
            target_corr = np.abs([np.corrcoef(X_filtered[:, i], y)[0, 1] for i in range(X_filtered.shape[1])])
            feature_importance = target_corr

            # Select top features
            if self.config.max_features:
                top_indices = np.argsort(feature_importance)[-self.config.max_features:]
            else:
                top_indices = np.where(feature_importance >= self.config.min_feature_importance)[0]

            X_selected = X_filtered[:, top_indices]
            features_selected = [features_filtered[i] for i in top_indices]

            self.selected_features = features_selected
            logger.info(f"Selected {len(features_selected)} features using correlation")
            return X_selected, features_selected

        # Fit selector and transform
        try:
            X_selected = selector.fit_transform(X_filtered, y)
            feature_scores = selector.scores_

            # Get selected feature indices
            if self.config.max_features:
                top_indices = np.argsort(feature_scores)[-self.config.max_features:]
            else:
                top_indices = np.where(feature_scores >= np.percentile(feature_scores, 75))[0]

            X_selected = X_filtered[:, top_indices]
            features_selected = [features_filtered[i] for i in top_indices]

            self.selected_features = features_selected
            logger.info(f"Selected {len(features_selected)} features using {self.config.selection_method}")
            return X_selected, features_selected

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return X_filtered, features_filtered


class DriftDetector:
    """Detect data and concept drift"""

    def __init__(self):
        self.reference_stats = {}
        self.drift_threshold = 0.1

    def set_reference(self, X: np.ndarray, feature_names: List[str]):
        """Set reference statistics for drift detection"""
        self.reference_stats = {
            'means': np.mean(X, axis=0),
            'stds': np.std(X, axis=0),
            'feature_names': feature_names
        }

    def detect_drift(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Detect drift in new data"""
        if not self.reference_stats:
            return {'drift_detected': False, 'reason': 'No reference set'}

        # Calculate current statistics
        current_means = np.mean(X_new, axis=0)
        current_stds = np.std(X_new, axis=0)

        # Calculate drift metrics
        mean_drift = np.abs((current_means - self.reference_stats['means']) /
                           (self.reference_stats['stds'] + 1e-8))

        # Check for drift
        drift_features = []
        for i, drift in enumerate(mean_drift):
            if drift > self.drift_threshold:
                feature_name = self.reference_stats['feature_names'][i] if i < len(self.reference_stats['feature_names']) else f"feature_{i}"
                drift_features.append((feature_name, drift))

        drift_detected = len(drift_features) > 0

        return {
            'drift_detected': drift_detected,
            'drift_score': float(np.max(mean_drift)),
            'drift_features': drift_features,
            'overall_drift': float(np.mean(mean_drift))
        }


class EnhancedTrainingPipeline:
    """Enhanced production training pipeline with advanced features"""

    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.experiment_manager = create_experiment_manager(
            experiment_name=config.experiment_name,
            model_type="ensemble",
            description="Enhanced training pipeline with walk-forward validation"
        )

        # Initialize components
        self.validator = WalkForwardValidator(config.validation_config)
        self.feature_engineer = AdvancedFeatureEngineer(config.feature_config)
        self.drift_detector = DriftDetector()

        # Training state
        self.trained_models = {}
        self.validation_results = {}
        self.ensemble_model = None

        # Create artifact directory
        Path(config.artifact_storage).mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Load and prepare training data"""
        logger.info(f"Loading data from {data_path}")

        # Load data
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Engineer features
        df = self.feature_engineer.engineer_features(df)

        # Prepare feature matrix
        target_col = 'close'
        feature_cols = [col for col in df.columns if col != target_col]

        # Remove rows with missing target
        df = df.dropna(subset=[target_col])

        logger.info(f"Data prepared: {len(df)} samples, {len(feature_cols)} features")

        return df, feature_cols

    def train_models(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Train all models with walk-forward validation"""
        logger.info("Starting enhanced training pipeline")

        target_col = 'close'

        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values

        # Feature selection
        X_selected, selected_features = self.feature_engineer.select_features(X, y, feature_cols)

        # Set reference for drift detection
        self.drift_detector.set_reference(X_selected, selected_features)

        training_results = {}

        with self.experiment_manager.start_run("enhanced_training"):
            # Log experiment configuration
            self.experiment_manager.log_params_batch(asdict(self.config))

            # Train each model type
            for model_type in self.config.model_types:
                logger.info(f"Training {model_type} with walk-forward validation")

                try:
                    # Create model instance
                    model = self._create_model(model_type)

                    # Perform walk-forward validation
                    validation_scores = self.validator.validate_model(
                        model, X_selected, y, selected_features
                    )

                    # Train final model on full dataset
                    final_model = self._create_model(model_type)

                    if hasattr(final_model, 'train'):
                        if hasattr(final_model, 'prepare_sequences'):
                            # Deep learning models
                            full_data = np.column_stack([y.reshape(-1, 1), X_selected])
                            X_seq, y_seq = final_model.prepare_sequences(full_data)
                            if len(X_seq) > 0:
                                final_model.train(X_seq, y_seq, verbose=1)
                        else:
                            # Traditional ML models
                            final_model.train(X_selected, y, feature_names=selected_features)
                    else:
                        final_model.fit(X_selected, y)

                    # Store results
                    model_result = {
                        'model': final_model,
                        'validation_scores': validation_scores,
                        'selected_features': selected_features
                    }

                    training_results[model_type] = model_result
                    self.trained_models[model_type] = final_model
                    self.validation_results[model_type] = validation_scores

                    # Log validation metrics
                    for metric, values in validation_scores.items():
                        if values and len(values) > 0:
                            mean_score = np.mean(values)
                            std_score = np.std(values)

                            self.experiment_manager.log_metrics_batch({
                                f"{model_type}_{metric}_mean": mean_score,
                                f"{model_type}_{metric}_std": std_score
                            })

                    # Log feature importance if available
                    if hasattr(final_model, 'get_feature_importance'):
                        try:
                            importance = final_model.get_feature_importance()
                            if len(importance) == len(selected_features):
                                self.experiment_manager.log_feature_importance(
                                    selected_features, importance,
                                    title=f"{model_type} Feature Importance"
                                )
                        except Exception as e:
                            logger.warning(f"Could not log feature importance for {model_type}: {e}")

                    logger.info(f"✅ {model_type} training completed")

                except Exception as e:
                    logger.error(f"❌ {model_type} training failed: {e}")
                    continue

            # Create ensemble if enabled
            if self.config.enable_ensemble and len(training_results) > 1:
                ensemble_result = self._create_ensemble(training_results, X_selected, y)
                if ensemble_result:
                    training_results['ensemble'] = ensemble_result

            # Generate comprehensive report
            report = self._generate_training_report(training_results)

            # Log training curves and visualizations
            self._log_training_visualizations(training_results)

            # Save models
            self._save_models(training_results)

        logger.info("Enhanced training pipeline completed")
        return training_results

    def _create_model(self, model_type: str) -> Any:
        """Create model instance based on type"""
        if model_type == 'transformer':
            config = {
                'n_heads': 8,
                'd_model': 256,
                'n_layers': 4,
                'dropout': 0.1,
                'sequence_length': 60,
                'prediction_horizons': [1, 7, 30]
            }
            return TransformerForecaster(config=config)

        elif model_type == 'lstm':
            config = {
                'lstm_units': [128, 64],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'sequence_length': 60,
                'prediction_horizons': [1, 7, 30],
                'use_attention': True,
                'use_residual': True
            }
            return EnhancedLSTMForecaster(config=config)

        elif model_type == 'lightgbm':
            config = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            return LightGBMForecaster(config=config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_ensemble(self, training_results: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Create ensemble model from trained models"""
        try:
            logger.info("Creating ensemble model")

            # Extract models
            models = {name: result['model'] for name, result in training_results.items()}

            # Create ensemble
            ensemble_config = {
                'models': list(models.keys()),
                'voting_strategy': 'weighted',
                'weights': self.config.ensemble_weights
            }

            ensemble = AdvancedEnsemble(config=ensemble_config)

            # Add pre-trained models to ensemble
            for name, model in models.items():
                ensemble.add_model(name, model)

            # Train ensemble meta-learner if supported
            if hasattr(ensemble, 'train'):
                feature_names = training_results[list(training_results.keys())[0]]['selected_features']
                ensemble.train(X, y, feature_names=feature_names)

            # Validate ensemble
            ensemble_scores = self.validator.validate_model(
                ensemble, X, y, training_results[list(training_results.keys())[0]]['selected_features']
            )

            self.ensemble_model = ensemble

            return {
                'model': ensemble,
                'validation_scores': ensemble_scores,
                'selected_features': training_results[list(training_results.keys())[0]]['selected_features']
            }

        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return None

    def _generate_training_report(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'models_trained': list(training_results.keys()),
            'validation_summary': {},
            'feature_analysis': {},
            'recommendations': []
        }

        # Validation summary
        for model_name, result in training_results.items():
            scores = result['validation_scores']
            summary = {}

            for metric, values in scores.items():
                if values and len(values) > 0:
                    summary[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }

            report['validation_summary'][model_name] = summary

        # Feature analysis
        if training_results:
            first_result = list(training_results.values())[0]
            selected_features = first_result['selected_features']

            report['feature_analysis'] = {
                'total_features': len(selected_features),
                'selected_features': selected_features[:10],  # Top 10
                'feature_engineering_config': asdict(self.config.feature_config)
            }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(training_results)

        # Save report
        report_path = Path(self.config.artifact_storage) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.experiment_manager.log_artifact(str(report_path), "reports")

        return report

    def _generate_recommendations(self, training_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if not training_results:
            return ["No models were successfully trained. Check data quality and model configurations."]

        # Compare model performance
        model_performances = {}
        for model_name, result in training_results.items():
            scores = result['validation_scores']
            if 'overall_rmse' in scores:
                model_performances[model_name] = scores['overall_rmse'][0]

        if model_performances:
            best_model = min(model_performances, key=model_performances.get)
            recommendations.append(f"Best performing model: {best_model} (RMSE: {model_performances[best_model]:.4f})")

            # Performance threshold check
            threshold = 0.05  # 5% improvement threshold
            worst_performance = max(model_performances.values())
            best_performance = min(model_performances.values())

            if (worst_performance - best_performance) / best_performance > threshold:
                recommendations.append("Significant performance differences detected. Consider ensemble methods.")

        # Feature analysis recommendations
        if training_results:
            first_result = list(training_results.values())[0]
            n_features = len(first_result['selected_features'])

            if n_features > 100:
                recommendations.append("High number of features detected. Consider additional feature selection.")
            elif n_features < 10:
                recommendations.append("Low number of features. Consider feature engineering or relaxing selection criteria.")

        return recommendations

    def _log_training_visualizations(self, training_results: Dict[str, Any]):
        """Log training visualizations"""
        try:
            # Create validation scores comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['RMSE Comparison', 'MAE Comparison', 'R² Comparison', 'Validation Scores Distribution'],
                vertical_spacing=0.1
            )

            # RMSE comparison
            for model_name, result in training_results.items():
                scores = result['validation_scores']
                if 'rmse' in scores and scores['rmse']:
                    fig.add_trace(
                        go.Box(y=scores['rmse'], name=model_name, boxpoints='outliers'),
                        row=1, col=1
                    )

            # MAE comparison
            for model_name, result in training_results.items():
                scores = result['validation_scores']
                if 'mae' in scores and scores['mae']:
                    fig.add_trace(
                        go.Box(y=scores['mae'], name=model_name, boxpoints='outliers'),
                        row=1, col=2
                    )

            # R² comparison
            for model_name, result in training_results.items():
                scores = result['validation_scores']
                if 'r2' in scores and scores['r2']:
                    r2_scores = [score for score in scores['r2'] if np.isfinite(score)]
                    if r2_scores:
                        fig.add_trace(
                            go.Box(y=r2_scores, name=model_name, boxpoints='outliers'),
                            row=2, col=1
                        )

            fig.update_layout(
                title="Model Validation Results Comparison",
                height=800,
                showlegend=False
            )

            # Save and log
            viz_file = "validation_comparison.html"
            fig.write_html(viz_file)
            self.experiment_manager.log_artifact(viz_file, "visualizations")
            os.unlink(viz_file)

        except Exception as e:
            logger.warning(f"Failed to create training visualizations: {e}")

    def _save_models(self, training_results: Dict[str, Any]):
        """Save trained models"""
        try:
            for model_name, result in training_results.items():
                model = result['model']
                model_path = Path(self.config.artifact_storage) / f"{model_name}_model"

                if hasattr(model, 'save'):
                    model.save(str(model_path))
                else:
                    # Fallback to pickle
                    import pickle
                    with open(f"{model_path}.pkl", 'wb') as f:
                        pickle.dump(model, f)

                logger.info(f"Saved {model_name} model to {model_path}")

        except Exception as e:
            logger.warning(f"Failed to save models: {e}")

    def detect_drift_and_retrain(self, new_data_path: str) -> Dict[str, Any]:
        """Detect drift in new data and trigger retraining if needed"""
        try:
            # Load new data
            df_new, feature_cols = self.load_and_prepare_data(new_data_path)

            # Prepare data
            X_new = df_new[feature_cols].values

            # Select same features as training
            if hasattr(self, 'feature_engineer') and self.feature_engineer.selected_features:
                feature_indices = [i for i, col in enumerate(feature_cols) if col in self.feature_engineer.selected_features]
                X_new = X_new[:, feature_indices]

            # Detect drift
            drift_result = self.drift_detector.detect_drift(X_new)

            # Log drift detection
            logger.info(f"Drift detection result: {drift_result}")

            # Trigger retraining if drift detected
            if drift_result['drift_detected'] and self.config.retrain_config.enable_auto_retrain:
                logger.info("Drift detected, triggering retraining...")

                # Combine old and new data
                # This is a simplified approach - in production, you'd want more sophisticated data management
                retrain_results = self.train_models(df_new, feature_cols)

                return {
                    'drift_detected': True,
                    'drift_result': drift_result,
                    'retrain_triggered': True,
                    'retrain_results': retrain_results
                }

            return {
                'drift_detected': drift_result['drift_detected'],
                'drift_result': drift_result,
                'retrain_triggered': False
            }

        except Exception as e:
            logger.error(f"Drift detection and retraining failed: {e}")
            return {
                'error': str(e),
                'drift_detected': False,
                'retrain_triggered': False
            }


def create_enhanced_pipeline(
    experiment_name: str,
    model_types: Optional[List[str]] = None,
    **kwargs
) -> EnhancedTrainingPipeline:
    """Create an enhanced training pipeline instance"""
    config = EnhancedTrainingConfig(
        experiment_name=experiment_name,
        model_types=model_types,
        **kwargs
    )

    return EnhancedTrainingPipeline(config)


if __name__ == "__main__":
    # Example usage
    pipeline = create_enhanced_pipeline(
        experiment_name="crypto_enhanced_training",
        model_types=["lightgbm", "lstm"],
        enable_ensemble=True
    )

    # Train models
    results = pipeline.train_models("data/crypto_features.parquet", [])

    print("Enhanced training pipeline completed!")
    print(f"Trained models: {list(results.keys())}")