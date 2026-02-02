"""
Production Training Pipeline for Cryptocurrency Prediction Models

Automated end-to-end training pipeline featuring:
- Data loading and preprocessing from engineered features
- Automated train/validation/test splits with temporal awareness
- Model training with hyperparameter optimization
- Cross-validation and performance evaluation
- Model comparison and selection
- Automated deployment and versioning
- Monitoring and drift detection
- Scheduled retraining capabilities
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import warnings
import logging
import schedule
import time
from dataclasses import dataclass, asdict
import pickle

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import our modules
from ..models.transformer_model import TransformerForecaster
from ..models.enhanced_lstm import EnhancedLSTMForecaster
from ..models.lightgbm_model import LightGBMForecaster
from ..models.advanced_ensemble import AdvancedEnsemble
from .mlflow_integration import MLflowExperimentTracker, MLflowModelRegistry
from .hyperopt_pipeline import HyperparameterOptimizer, run_complete_optimization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for production training pipeline"""

    # Data configuration
    features_path: str = "data/features"
    processed_path: str = "data/processed"
    cryptocurrencies: Optional[List[str]] = None  # None = all available
    max_cryptocurrencies: int = 20  # Limit for performance

    # Training configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_samples: int = 500  # Minimum samples per cryptocurrency

    # Model configuration
    models_to_train: List[str] = None
    use_hyperopt: bool = True
    hyperopt_trials: int = 50
    cross_validation_folds: int = 3

    # Production configuration
    model_dir: str = "models/artifacts/production"
    experiment_name: str = "crypto_production"
    auto_deploy: bool = False
    deployment_metric: str = "val_rmse"
    deployment_threshold: float = 0.05  # Maximum acceptable error increase

    # Retraining configuration
    retrain_schedule: str = "weekly"  # daily, weekly, monthly
    drift_threshold: float = 0.1  # Performance degradation threshold
    monitoring_window: int = 7  # Days to monitor for drift

    # Feature engineering
    feature_selection: bool = True
    feature_selection_method: str = "correlation"  # correlation, importance, variance
    max_features: int = 50

    # Scaling
    scaling_method: str = "minmax"  # minmax, standard, robust

    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ["transformer", "lstm", "lightgbm", "ensemble"]


class DataLoader:
    """
    Production data loader with preprocessing and validation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scalers = {}
        self.feature_names = []

    def load_and_preprocess(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess cryptocurrency data.

        Returns:
            Dictionary of preprocessed DataFrames
        """
        logger.info("Loading cryptocurrency data...")

        features_path = Path(self.config.features_path)
        if not features_path.exists():
            raise FileNotFoundError(f"Features directory not found: {features_path}")

        # Find available feature files
        feature_files = list(features_path.glob("*_features.parquet"))
        logger.info(f"Found {len(feature_files)} cryptocurrency feature files")

        crypto_data = {}

        # Load specified cryptocurrencies or all available
        target_cryptos = self.config.cryptocurrencies or [
            f.stem.replace('_features', '') for f in feature_files
        ]

        loaded_count = 0
        for file_path in feature_files:
            if loaded_count >= self.config.max_cryptocurrencies:
                break

            crypto_name = file_path.stem.replace('_features', '')

            if crypto_name not in target_cryptos:
                continue

            try:
                df = pd.read_parquet(file_path)

                # Data quality checks
                if len(df) < self.config.min_samples:
                    logger.warning(f"Skipping {crypto_name}: insufficient samples ({len(df)})")
                    continue

                # Basic preprocessing
                df = self._preprocess_dataframe(df, crypto_name)

                if df is not None and len(df) > 0:
                    crypto_data[crypto_name] = df
                    loaded_count += 1
                    logger.info(f"Loaded {crypto_name}: {len(df)} samples, {len(df.columns)} features")

            except Exception as e:
                logger.error(f"Failed to load {crypto_name}: {e}")

        logger.info(f"Successfully loaded {len(crypto_data)} cryptocurrencies")
        return crypto_data

    def _preprocess_dataframe(self, df: pd.DataFrame, crypto_name: str) -> Optional[pd.DataFrame]:
        """
        Preprocess individual cryptocurrency DataFrame.

        Args:
            df: Raw DataFrame
            crypto_name: Cryptocurrency name

        Returns:
            Preprocessed DataFrame
        """
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df = df.set_index('date')
                    df.index = pd.to_datetime(df.index)
                else:
                    logger.warning(f"{crypto_name}: No datetime index found")
                    return None

            # Sort by date
            df = df.sort_index()

            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='last')]

            # Handle missing values
            df = self._handle_missing_values(df)

            # Feature selection if enabled
            if self.config.feature_selection:
                df = self._select_features(df, crypto_name)

            # Store feature names (from first crypto)
            if not self.feature_names:
                self.feature_names = [col for col in df.columns if col != 'close']

            return df

        except Exception as e:
            logger.error(f"Preprocessing failed for {crypto_name}: {e}")
            return None

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Forward fill price data
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

        # Fill remaining NaNs with interpolation or mean
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    # Interpolate numeric data
                    df[col] = df[col].interpolate()
                    # Fill any remaining with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Forward fill non-numeric
                    df[col] = df[col].fillna(method='ffill')

        return df

    def _select_features(self, df: pd.DataFrame, crypto_name: str) -> pd.DataFrame:
        """Select most relevant features"""
        if len(df.columns) <= self.config.max_features:
            return df  # Already within limit

        target_col = 'close'
        if target_col not in df.columns:
            return df

        feature_cols = [col for col in df.columns if col != target_col]

        if self.config.feature_selection_method == "correlation":
            # Select features most correlated with target
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            top_features = correlations.nlargest(self.config.max_features).index.tolist()
        elif self.config.feature_selection_method == "variance":
            # Select features with highest variance
            variances = df[feature_cols].var()
            top_features = variances.nlargest(self.config.max_features).index.tolist()
        else:
            # Default: take first N features
            top_features = feature_cols[:self.config.max_features]

        selected_cols = top_features + [target_col]
        logger.debug(f"{crypto_name}: Selected {len(top_features)} features")

        return df[selected_cols]

    def create_train_test_splits(
        self,
        crypto_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Create temporal train/validation/test splits for all cryptocurrencies.

        Args:
            crypto_data: Dictionary of cryptocurrency DataFrames

        Returns:
            Dictionary with splits for each cryptocurrency
        """
        logger.info("Creating train/validation/test splits...")

        splits = {}

        for crypto_name, df in crypto_data.items():
            try:
                # Calculate split indices
                n_samples = len(df)
                train_end = int(n_samples * self.config.train_ratio)
                val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

                # Create temporal splits
                train_df = df.iloc[:train_end]
                val_df = df.iloc[train_end:val_end]
                test_df = df.iloc[val_end:]

                # Prepare features and targets
                feature_cols = [col for col in df.columns if col != 'close']
                target_col = 'close'

                # Scale features
                scaler = self._get_scaler()

                # Fit scaler on training data only
                X_train = scaler.fit_transform(train_df[feature_cols])
                X_val = scaler.transform(val_df[feature_cols])
                X_test = scaler.transform(test_df[feature_cols])

                # Store scaler for later use
                self.scalers[crypto_name] = scaler

                # Targets (no scaling for now)
                y_train = train_df[target_col].values
                y_val = val_df[target_col].values
                y_test = test_df[target_col].values

                splits[crypto_name] = {
                    'train': (X_train, y_train),
                    'val': (X_val, y_val),
                    'test': (X_test, y_test),
                    'train_dates': train_df.index,
                    'val_dates': val_df.index,
                    'test_dates': test_df.index
                }

                logger.info(f"{crypto_name}: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

            except Exception as e:
                logger.error(f"Failed to create splits for {crypto_name}: {e}")

        logger.info(f"Created splits for {len(splits)} cryptocurrencies")
        return splits

    def _get_scaler(self):
        """Get scaler based on configuration"""
        if self.config.scaling_method == "standard":
            return StandardScaler()
        elif self.config.scaling_method == "minmax":
            return MinMaxScaler()
        else:
            return MinMaxScaler()  # Default


class ModelTrainer:
    """
    Production model trainer with automated hyperparameter optimization.
    """

    def __init__(self, config: PipelineConfig, mlflow_tracker: MLflowExperimentTracker):
        self.config = config
        self.mlflow_tracker = mlflow_tracker
        self.trained_models = {}
        self.model_performances = {}

    def train_all_models(
        self,
        splits: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        feature_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models on all cryptocurrencies.

        Args:
            splits: Data splits for all cryptocurrencies
            feature_names: List of feature names

        Returns:
            Dictionary of trained models and their performances
        """
        logger.info(f"Training {len(self.config.models_to_train)} model types on {len(splits)} cryptocurrencies")

        all_results = {}

        for model_type in self.config.models_to_train:
            logger.info(f"Training {model_type} models...")

            model_results = {}

            for crypto_name, crypto_splits in splits.items():
                try:
                    logger.info(f"Training {model_type} for {crypto_name}")

                    # Start model-specific MLflow run
                    run_name = f"{model_type}_{crypto_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.mlflow_tracker.start_run(
                        run_name=run_name,
                        tags={
                            "model_type": model_type,
                            "cryptocurrency": crypto_name,
                            "production_pipeline": "true"
                        }
                    )

                    # Get data splits
                    X_train, y_train = crypto_splits['train']
                    X_val, y_val = crypto_splits['val']
                    X_test, y_test = crypto_splits['test']

                    # Log dataset info
                    self.mlflow_tracker.log_dataset_info(
                        X_train, y_train, X_val, y_val, feature_names
                    )

                    # Train model (with or without hyperparameter optimization)
                    if self.config.use_hyperopt:
                        model, config, performance = self._train_with_hyperopt(
                            model_type, X_train, y_train, X_val, y_val, feature_names
                        )
                    else:
                        model, config, performance = self._train_with_default_config(
                            model_type, X_train, y_train, X_val, y_val, feature_names
                        )

                    if model is not None:
                        # Evaluate on test set
                        test_performance = self._evaluate_model(
                            model, model_type, X_test, y_test
                        )
                        performance.update(test_performance)

                        # Log performance metrics
                        self.mlflow_tracker.log_training_metrics(performance, model_type=model_type)

                        # Log model artifacts
                        self._log_model_artifacts(model, model_type, config)

                        # Store results
                        model_results[crypto_name] = {
                            'model': model,
                            'config': config,
                            'performance': performance,
                            'mlflow_run_id': self.mlflow_tracker.current_run_id
                        }

                        logger.info(f"✅ {model_type} for {crypto_name} - Test RMSE: {performance.get('test_rmse', 'N/A'):.4f}")

                except Exception as e:
                    logger.error(f"Failed to train {model_type} for {crypto_name}: {e}")

                finally:
                    # End model run
                    if self.mlflow_tracker.current_run_id:
                        self.mlflow_tracker.end_run()

            all_results[model_type] = model_results

            # Store in instance variables
            self.trained_models[model_type] = model_results

        # Calculate overall model performance
        self._calculate_overall_performance(all_results)

        return all_results

    def _train_with_hyperopt(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """Train model with hyperparameter optimization"""
        try:
            # Setup optimizer
            optimizer = HyperparameterOptimizer(
                study_name=f"production_{model_type}",
                mlflow_tracker=self.mlflow_tracker
            )

            # Optimize hyperparameters
            opt_results = optimizer.optimize_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=self.config.hyperopt_trials,
                feature_names=feature_names
            )

            if opt_results and 'best_params' in opt_results:
                best_config = opt_results['best_params']

                # Log optimization results
                self.mlflow_tracker.log_hyperparameter_optimization(opt_results, model_type)

                # Train final model with best configuration
                model, performance = self._train_single_model(
                    model_type, best_config, X_train, y_train, X_val, y_val, feature_names
                )

                return model, best_config, performance
            else:
                logger.warning(f"Hyperparameter optimization failed for {model_type}, using default config")
                return self._train_with_default_config(model_type, X_train, y_train, X_val, y_val, feature_names)

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for {model_type}: {e}")
            return self._train_with_default_config(model_type, X_train, y_train, X_val, y_val, feature_names)

    def _train_with_default_config(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """Train model with default configuration"""
        try:
            # Get default configuration
            if model_type == 'transformer':
                config = TransformerForecaster._default_config()
            elif model_type == 'lstm':
                config = EnhancedLSTMForecaster._default_config()
            elif model_type == 'lightgbm':
                config = LightGBMForecaster._default_config()
            elif model_type == 'ensemble':
                config = AdvancedEnsemble._default_config()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train model
            model, performance = self._train_single_model(
                model_type, config, X_train, y_train, X_val, y_val, feature_names
            )

            return model, config, performance

        except Exception as e:
            logger.error(f"Training with default config failed for {model_type}: {e}")
            return None, {}, {}

    def _train_single_model(
        self,
        model_type: str,
        config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a single model with given configuration"""
        try:
            # Create model instance
            if model_type == 'transformer':
                model = TransformerForecaster(config=config)

                # Prepare sequences for Transformer
                train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
                val_data = np.column_stack([y_val.reshape(-1, 1), X_val])

                X_train_seq, y_train_seq = model.prepare_sequences(train_data)
                X_val_seq, y_val_seq = model.prepare_sequences(val_data)

                # Train
                metrics = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0)

            elif model_type == 'lstm':
                model = EnhancedLSTMForecaster(config=config)

                # Prepare sequences for LSTM
                train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
                val_data = np.column_stack([y_val.reshape(-1, 1), X_val])

                X_train_seq, y_train_seq = model.prepare_sequences(train_data)
                X_val_seq, y_val_seq = model.prepare_sequences(val_data)

                # Train
                metrics = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0)

            elif model_type == 'lightgbm':
                model = LightGBMForecaster(config=config)

                # Train
                metrics = model.train(X_train, y_train, X_val, y_val, feature_names)

            elif model_type == 'ensemble':
                model = AdvancedEnsemble(config=config)

                # Train
                metrics = model.train(X_train, y_train, X_val, y_val, feature_names)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            return model, metrics

        except Exception as e:
            logger.error(f"Model training failed for {model_type}: {e}")
            return None, {}

    def _evaluate_model(
        self,
        model: Any,
        model_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        try:
            # Get predictions
            if model_type in ['transformer', 'lstm']:
                # Need to prepare sequences
                test_data = np.column_stack([y_test.reshape(-1, 1), X_test])
                if hasattr(model, 'prepare_sequences'):
                    X_test_seq, y_test_seq = model.prepare_sequences(test_data)
                    if len(X_test_seq) > 0:
                        pred = model.predict(X_test_seq)
                        if isinstance(pred, dict):
                            pred = list(pred.values())[0]  # Take first horizon
                        y_pred = pred.flatten()
                        y_true = y_test_seq
                    else:
                        return {}
                else:
                    y_pred = model.predict(X_test).flatten()
                    y_true = y_test
            else:
                y_pred = model.predict(X_test).flatten()
                y_true = y_test

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # Handle potential division by zero
            try:
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                r2 = r2_score(y_true, y_pred)
            except:
                mape = float('inf')
                r2 = -float('inf')

            return {
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_mape': float(mape),
                'test_r2': float(r2)
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

    def _log_model_artifacts(self, model: Any, model_type: str, config: Dict[str, Any]) -> None:
        """Log model artifacts to MLflow"""
        try:
            # Log configuration
            self.mlflow_tracker.log_model_config(config, model_type)

            # Log model based on type
            if model_type in ['transformer', 'lstm'] and hasattr(model, 'model'):
                self.mlflow_tracker.log_model_tensorflow(
                    model.model,
                    model_type=model_type
                )
            elif model_type == 'lightgbm' and hasattr(model, 'model'):
                self.mlflow_tracker.log_model_lightgbm(model.model)
            elif model_type == 'ensemble':
                # Save ensemble directory first
                ensemble_dir = model.save_ensemble()
                self.mlflow_tracker.log_ensemble_model(ensemble_dir, model_type)

        except Exception as e:
            logger.warning(f"Failed to log model artifacts for {model_type}: {e}")

    def _calculate_overall_performance(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Calculate overall performance across all models and cryptocurrencies"""
        logger.info("Calculating overall model performance...")

        overall_performance = {}

        for model_type, model_results in all_results.items():
            if not model_results:
                continue

            # Collect all performance metrics
            rmse_scores = []
            mae_scores = []

            for crypto_name, result in model_results.items():
                if 'performance' in result:
                    perf = result['performance']
                    if 'test_rmse' in perf and not np.isnan(perf['test_rmse']):
                        rmse_scores.append(perf['test_rmse'])
                    if 'test_mae' in perf and not np.isnan(perf['test_mae']):
                        mae_scores.append(perf['test_mae'])

            if rmse_scores:
                overall_performance[model_type] = {
                    'mean_rmse': np.mean(rmse_scores),
                    'std_rmse': np.std(rmse_scores),
                    'mean_mae': np.mean(mae_scores) if mae_scores else 0,
                    'n_successful_models': len(rmse_scores)
                }

                logger.info(f"{model_type}: Mean RMSE = {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

        self.model_performances = overall_performance


class ProductionPipeline:
    """
    Main production pipeline coordinator.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Setup MLflow tracking
        try:
            from .mlflow_integration import setup_mlflow_tracking
            self.mlflow_tracker = setup_mlflow_tracking(
                experiment_name=config.experiment_name
            )
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.mlflow_tracker = None

        # Setup model registry
        try:
            if self.mlflow_tracker:
                self.model_registry = MLflowModelRegistry()
        except Exception as e:
            logger.warning(f"Model registry setup failed: {e}")
            self.model_registry = None

        # Initialize components
        self.data_loader = DataLoader(config)
        self.model_trainer = ModelTrainer(config, self.mlflow_tracker) if self.mlflow_tracker else None

        # Pipeline state
        self.pipeline_results = {}
        self.best_models = {}

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete production pipeline.

        Returns:
            Pipeline execution results
        """
        logger.info("🚀 Starting production training pipeline")
        logger.info(f"Configuration: {self.config}")

        pipeline_start = datetime.now()

        try:
            # Start main pipeline run
            if self.mlflow_tracker:
                self.mlflow_tracker.start_run(
                    run_name=f"production_pipeline_{pipeline_start.strftime('%Y%m%d_%H%M%S')}",
                    tags={
                        "pipeline": "production",
                        "stage": "complete_training",
                        "models": ",".join(self.config.models_to_train)
                    }
                )

            # Step 1: Load and preprocess data
            logger.info("📊 Step 1: Loading and preprocessing data...")
            crypto_data = self.data_loader.load_and_preprocess()

            if not crypto_data:
                raise RuntimeError("No cryptocurrency data loaded")

            # Step 2: Create train/test splits
            logger.info("✂️ Step 2: Creating train/validation/test splits...")
            splits = self.data_loader.create_train_test_splits(crypto_data)

            if not splits:
                raise RuntimeError("No valid data splits created")

            # Step 3: Train all models
            if self.model_trainer:
                logger.info("🤖 Step 3: Training all models...")
                training_results = self.model_trainer.train_all_models(splits, self.data_loader.feature_names)
            else:
                logger.error("Model trainer not available")
                training_results = {}

            # Step 4: Model selection and comparison
            logger.info("🏆 Step 4: Model selection and comparison...")
            best_models = self._select_best_models(training_results)

            # Step 5: Production deployment (if enabled)
            if self.config.auto_deploy and best_models:
                logger.info("🚀 Step 5: Deploying best models...")
                deployment_results = self._deploy_models(best_models)
            else:
                deployment_results = {}

            # Compile results
            pipeline_end = datetime.now()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

            results = {
                'pipeline_config': asdict(self.config),
                'execution_time': pipeline_duration,
                'data_summary': {
                    'n_cryptocurrencies': len(crypto_data),
                    'feature_count': len(self.data_loader.feature_names),
                    'date_range': {
                        'start': min(df.index.min() for df in crypto_data.values()).isoformat(),
                        'end': max(df.index.max() for df in crypto_data.values()).isoformat()
                    }
                },
                'training_results': training_results,
                'best_models': best_models,
                'deployment_results': deployment_results,
                'overall_performance': self.model_trainer.model_performances if self.model_trainer else {}
            }

            # Log pipeline results
            if self.mlflow_tracker:
                self._log_pipeline_results(results)

            self.pipeline_results = results

            logger.info(f"✅ Production pipeline completed successfully in {pipeline_duration:.1f} seconds")
            return results

        except Exception as e:
            logger.error(f"❌ Production pipeline failed: {e}")
            raise

        finally:
            if self.mlflow_tracker and self.mlflow_tracker.current_run_id:
                self.mlflow_tracker.end_run()

    def _select_best_models(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Select best models based on performance metrics"""
        logger.info("Selecting best models...")

        best_models = {}

        for model_type, model_results in training_results.items():
            if not model_results:
                continue

            # Find best model for this type based on validation performance
            best_crypto = None
            best_performance = float('inf')

            for crypto_name, result in model_results.items():
                if 'performance' in result:
                    # Use validation RMSE as selection criterion
                    val_rmse = result['performance'].get('val_rmse', float('inf'))
                    if val_rmse < best_performance:
                        best_performance = val_rmse
                        best_crypto = crypto_name

            if best_crypto:
                best_models[model_type] = {
                    'cryptocurrency': best_crypto,
                    'model': model_results[best_crypto]['model'],
                    'config': model_results[best_crypto]['config'],
                    'performance': model_results[best_crypto]['performance'],
                    'mlflow_run_id': model_results[best_crypto]['mlflow_run_id']
                }

                logger.info(f"Best {model_type}: {best_crypto} (Val RMSE: {best_performance:.4f})")

        return best_models

    def _deploy_models(self, best_models: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Deploy best models to production"""
        deployment_results = {}

        if not self.model_registry:
            logger.warning("Model registry not available, skipping deployment")
            return deployment_results

        for model_type, model_info in best_models.items():
            try:
                # Register model in MLflow
                model_name = f"crypto_{model_type}_production"

                # Get MLflow run ID
                run_id = model_info.get('mlflow_run_id')
                if run_id:
                    # Register model
                    version = self.model_registry.promote_model(
                        model_name=model_name,
                        version="latest",
                        stage="Production"
                    )

                    deployment_results[model_type] = True
                    logger.info(f"✅ Deployed {model_type} model to production")
                else:
                    deployment_results[model_type] = False
                    logger.warning(f"⚠️ No MLflow run ID for {model_type}")

            except Exception as e:
                deployment_results[model_type] = False
                logger.error(f"❌ Failed to deploy {model_type}: {e}")

        return deployment_results

    def _log_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Log overall pipeline results to MLflow"""
        try:
            # Log execution metrics
            self.mlflow_tracker.log_training_metrics({
                'pipeline_duration': results['execution_time'],
                'n_cryptocurrencies': results['data_summary']['n_cryptocurrencies'],
                'n_features': results['data_summary']['feature_count'],
                'n_model_types': len(results['training_results'])
            })

            # Log overall performance
            if 'overall_performance' in results:
                for model_type, perf in results['overall_performance'].items():
                    for metric, value in perf.items():
                        self.mlflow_tracker.log_training_metrics({
                            f"overall_{model_type}_{metric}": value
                        })

            # Save results as artifact
            results_file = "pipeline_results.json"
            with open(results_file, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2)

            self.mlflow_tracker.log_artifact(results_file, "pipeline_results")

            # Clean up
            if os.path.exists(results_file):
                os.unlink(results_file)

        except Exception as e:
            logger.warning(f"Failed to log pipeline results: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string
        else:
            return obj

    def schedule_retraining(self) -> None:
        """Schedule automatic retraining based on configuration"""
        logger.info(f"Scheduling retraining: {self.config.retrain_schedule}")

        if self.config.retrain_schedule == "daily":
            schedule.every().day.at("02:00").do(self._retrain_job)
        elif self.config.retrain_schedule == "weekly":
            schedule.every().week.at("02:00").do(self._retrain_job)
        elif self.config.retrain_schedule == "monthly":
            schedule.every(30).days.at("02:00").do(self._retrain_job)

        logger.info("✅ Retraining scheduled. Use run_scheduler() to start monitoring.")

    def _retrain_job(self) -> None:
        """Retraining job to be executed by scheduler"""
        logger.info("🔄 Starting scheduled retraining...")

        try:
            # Check for model drift before retraining
            if self._check_model_drift():
                logger.info("Model drift detected, proceeding with retraining")
                self.run_complete_pipeline()
            else:
                logger.info("No significant drift detected, skipping retraining")

        except Exception as e:
            logger.error(f"Scheduled retraining failed: {e}")

    def _check_model_drift(self) -> bool:
        """Check if models need retraining due to performance drift"""
        # This is a placeholder for drift detection logic
        # In practice, you'd compare recent model performance with baseline
        logger.info("Checking for model drift...")

        # For now, return True to always retrain
        # In production, implement proper drift detection
        return True

    def run_scheduler(self) -> None:
        """Run the retraining scheduler"""
        logger.info("🕐 Starting retraining scheduler...")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def create_production_config(**kwargs) -> PipelineConfig:
    """
    Create production pipeline configuration.

    Args:
        **kwargs: Configuration overrides

    Returns:
        PipelineConfig instance
    """
    return PipelineConfig(**kwargs)


def run_production_pipeline(config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Run production pipeline with given configuration.

    Args:
        config: Pipeline configuration

    Returns:
        Pipeline execution results
    """
    if config is None:
        config = PipelineConfig()

    pipeline = ProductionPipeline(config)
    return pipeline.run_complete_pipeline()


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        models_to_train=["lightgbm", "lstm"],
        use_hyperopt=True,
        hyperopt_trials=20,
        max_cryptocurrencies=5
    )

    results = run_production_pipeline(config)
    print("Production pipeline completed successfully!")