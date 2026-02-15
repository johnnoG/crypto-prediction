"""
Production Training Script for Cryptocurrency Price Prediction

Trains LSTM, Transformer, LightGBM, and Ensemble models on real feature data,
generates rich matplotlib visualizations, saves models, and logs to MLflow.

Advanced features (behind optional flags):
  --tune              Optuna Bayesian hyperparameter optimization before training
  --walk-forward      Walk-forward cross-validation after training
  --no-uncertainty    Disable Monte Carlo dropout uncertainty quantification
  --no-advanced-mlflow  Disable interactive MLflow training curves & model cards

Usage:
    # Standard training
    python models/src/train_production.py --crypto BTC,ETH

    # Quick test run
    python models/src/train_production.py --crypto BTC --epochs 5 --no-ensemble

    # Full professional pipeline with hyperparameter tuning
    python models/src/train_production.py --crypto BTC --tune --tune-trials 20 --walk-forward

    # Tune LightGBM only (fast)
    python models/src/train_production.py --crypto BTC --tune --tune-trials 10 --tune-models lightgbm
"""

import os
import sys
import argparse
import logging
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add current directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(current_dir))

# Set MLflow tracking URI before imports
mlflow_dir = current_dir / "mlruns_production"
mlflow_dir.mkdir(exist_ok=True)
os.environ['MLFLOW_TRACKING_URI'] = f"file://{mlflow_dir}"

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models
from models.enhanced_lstm import EnhancedLSTMForecaster
from models.transformer_model import TransformerForecaster
from models.lightgbm_model import LightGBMForecaster
from models.advanced_ensemble import AdvancedEnsemble

# Hyperparameter optimization (optional — requires optuna)
try:
    from training.hyperopt_pipeline import HyperparameterOptimizer
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Walk-forward validation (optional)
try:
    from pipelines.enhanced_training_pipeline import WalkForwardValidator, ValidationConfig
    WALKFORWARD_AVAILABLE = True
except ImportError:
    WALKFORWARD_AVAILABLE = False

import tensorflow as tf


class BatchProgressCallback(tf.keras.callbacks.Callback):
    """Prints per-batch progress so you can see training is moving."""

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_start = time.time()
        self.total_batches = None

    def on_train_begin(self, logs=None):
        total_samples = self.params.get('samples') or self.params.get('steps', 0) * (self.params.get('batch_size', 32))
        self.total_batches = self.params.get('steps', '?')
        logger.info(f"Training started — {self.params.get('epochs')} epochs, {self.total_batches} batches/epoch")

    def on_batch_end(self, batch, logs=None):
        if batch > 0 and batch % 20 == 0:
            loss = logs.get('loss', 0)
            elapsed = time.time() - self.epoch_start
            print(f"  Epoch {self.epoch+1} | batch {batch}/{self.total_batches} | "
                  f"loss: {loss:.4f} | {elapsed:.0f}s elapsed", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        val_loss = logs.get('val_loss', None)
        loss = logs.get('loss', 0)
        msg = f"Epoch {epoch+1} done in {elapsed:.1f}s — loss: {loss:.4f}"
        if val_loss is not None:
            msg += f", val_loss: {val_loss:.4f}"
        logger.info(msg)

# Optional MLflow
try:
    from mlflow_advanced.experiment_manager import create_experiment_manager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Production training configuration."""
    # Data
    crypto: str = "BTC"
    features_dir: str = str(project_root / "data" / "features")

    # Splits (chronological)
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Sequence / model params
    sequence_length: int = 60
    epochs: int = 150
    batch_size: int = 64
    max_features: int = 60

    # Ensemble toggle
    train_ensemble: bool = True

    # Output
    output_dir: str = str(current_dir / "training_output")
    artifacts_dir: str = str(current_dir.parent / "artifacts")

    # Hyperparameter tuning (--tune)
    tune: bool = False
    tune_trials: int = 20
    tune_timeout: int = 3600
    tune_models: str = "lstm,transformer,lightgbm"

    # Walk-forward validation (--walk-forward)
    walk_forward: bool = False
    wf_n_splits: int = 5
    wf_min_train_size: int = 1000
    wf_expanding: bool = True

    # Uncertainty quantification (on by default)
    uncertainty: bool = True
    mc_samples: int = 50

    # Advanced MLflow (on by default)
    advanced_mlflow: bool = True


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

class ProductionDataLoader:
    """Load parquet feature data and prepare for training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_scaler: Optional[RobustScaler] = None
        self.target_scaler: Optional[RobustScaler] = None
        self.feature_names: List[str] = []

    # -- public API ----------------------------------------------------------

    def load_and_prepare(self) -> Dict[str, Any]:
        """Full pipeline: load → clean → select → scale → split → sequence."""
        df = self._load_parquet()
        df = self._clean(df)
        feature_cols = self._select_features(df)
        self.feature_names = feature_cols

        # Chronological split indices
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

        # Scale (fit on train only)
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

        X_train = self.feature_scaler.fit_transform(train_df[feature_cols].values)
        X_val = self.feature_scaler.transform(val_df[feature_cols].values)
        X_test = self.feature_scaler.transform(test_df[feature_cols].values)

        y_train_raw = train_df['close'].values
        y_val_raw = val_df['close'].values
        y_test_raw = test_df['close'].values

        y_train = self.target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
        y_val = self.target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
        y_test = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

        # Build combined arrays for prepare_sequences (target as col 0)
        combined_train = np.column_stack([y_train.reshape(-1, 1), X_train])
        combined_val = np.column_stack([y_val.reshape(-1, 1), X_val])
        combined_test = np.column_stack([y_test.reshape(-1, 1), X_test])

        return {
            'combined_train': combined_train,
            'combined_val': combined_val,
            'combined_test': combined_test,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'y_train_raw': y_train_raw, 'y_val_raw': y_val_raw, 'y_test_raw': y_test_raw,
            'feature_names': feature_cols,
            'dates_train': train_df.index,
            'dates_val': val_df.index,
            'dates_test': test_df.index,
        }

    # -- internals -----------------------------------------------------------

    def _load_parquet(self) -> pd.DataFrame:
        path = Path(self.config.features_dir) / f"{self.config.crypto}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {self.config.crypto}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop ticker column if present
        if 'ticker' in df.columns:
            df = df.drop(columns=['ticker'])

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # Drop columns with >50% NaN
        thresh = len(df) * 0.5
        df = df.dropna(axis=1, thresh=int(thresh))

        # Forward-fill then drop remaining NaN rows
        df = df.ffill()
        df = df.dropna()

        logger.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select top features by absolute correlation with close."""
        if 'close' not in df.columns:
            raise ValueError("'close' column not found in data")

        candidates = [c for c in df.columns if c != 'close']
        corrs = df[candidates].corrwith(df['close']).abs().sort_values(ascending=False)
        selected = corrs.head(self.config.max_features).index.tolist()
        logger.info(f"Selected {len(selected)} features (top correlation with close)")
        return selected


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

class ProductionTrainer:
    """Trains all models and collects results."""

    def __init__(self, config: TrainingConfig, tuned_configs: Optional[Dict] = None):
        self.config = config
        self.tuned_configs = tuned_configs or {}
        self.results: Dict[str, Any] = {}

    def train_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train LSTM, Transformer, LightGBM, and optionally Ensemble."""

        # --- LSTM ---
        logger.info("=" * 60)
        logger.info("[1/4] Training Enhanced LSTM...")
        logger.info("=" * 60)
        self.results['lstm'] = self._train_lstm(data)

        # --- Transformer ---
        logger.info("=" * 60)
        logger.info("[2/4] Training Transformer...")
        logger.info("=" * 60)
        self.results['transformer'] = self._train_transformer(data)

        # --- LightGBM ---
        logger.info("=" * 60)
        logger.info("[3/4] Training LightGBM...")
        logger.info("=" * 60)
        self.results['lightgbm'] = self._train_lightgbm(data)

        # --- Ensemble ---
        if self.config.train_ensemble:
            logger.info("=" * 60)
            logger.info("[4/4] Training Advanced Ensemble...")
            logger.info("=" * 60)
            self.results['ensemble'] = self._train_ensemble(data)

        return self.results

    # -- individual trainers -------------------------------------------------

    def _train_lstm(self, data: Dict) -> Dict[str, Any]:
        seq_len = self.config.sequence_length
        artifact_dir = str(Path(self.config.artifacts_dir) / "enhanced_lstm")

        lstm_config = {
            'sequence_length': seq_len,
            'n_features': data['combined_train'].shape[1],
            'lstm_units': [128, 64, 32],        # smaller to reduce overfitting
            'dense_units': [64, 32],             # smaller dense head
            'dropout_rate': 0.4,                 # higher dropout (was 0.25)
            'recurrent_dropout': 0.0,            # must be 0 for CuDNN/Metal GPU kernels
            'learning_rate': 0.0005,             # lower initial LR (was 0.001)
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'early_stopping_patience': 20,
            'gradient_clip': 1.0,
            'use_bidirectional': True,
            'use_attention': True,
            'use_residual': True,
            'use_layer_norm': True,
            'multi_step': [1, 7, 30],
            'teacher_forcing_ratio': 0.5,
            'mc_samples': 100,
        }

        # Apply tuned hyperparameters if available
        if 'lstm' in self.tuned_configs:
            tuned = self.tuned_configs['lstm']
            for key in ('lstm_units', 'dense_units', 'dropout_rate', 'recurrent_dropout',
                        'learning_rate', 'use_bidirectional', 'use_attention',
                        'use_residual', 'gradient_clip', 'batch_size'):
                if key in tuned:
                    lstm_config[key] = tuned[key]
            logger.info(f"  Applied tuned hyperparameters for LSTM: {list(tuned.keys())}")

        model = EnhancedLSTMForecaster(config=lstm_config, model_dir=artifact_dir)

        # Prepare sequences
        X_train_seq, y_train_seq = model.prepare_sequences(data['combined_train'])
        X_val_seq, y_val_seq = model.prepare_sequences(data['combined_val'])
        X_test_seq, y_test_seq = model.prepare_sequences(data['combined_test'])

        t0 = time.time()
        metrics = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=1)
        train_time = time.time() - t0

        # Keras history dict
        history = model.training_history.history if model.training_history else {}

        # Test predictions
        test_preds = model.model.predict(X_test_seq, verbose=0)

        return {
            'model': model,
            'metrics': metrics,
            'history': history,
            'train_time': train_time,
            'config': lstm_config,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'test_preds': test_preds,
        }

    def _train_transformer(self, data: Dict) -> Dict[str, Any]:
        seq_len = self.config.sequence_length
        artifact_dir = str(Path(self.config.artifacts_dir) / "transformer")

        transformer_config = {
            'sequence_length': seq_len,
            'n_features': data['combined_train'].shape[1],
            'd_model': 128,                      # smaller (was 256) — less overfitting
            'num_heads': 4,                      # match d_model reduction (was 8)
            'ff_dim': 512,                       # smaller (was 1024)
            'num_layers': 3,                     # fewer layers (was 4)
            'dropout_rate': 0.25,                # higher dropout (was 0.1)
            'learning_rate': 0.0001,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'early_stopping_patience': 15,
            'warmup_steps': 500,                 # faster warmup (was 1000) for ~115 steps/epoch
            'multi_step': [1, 7, 30],
            'use_causal_mask': True,
        }

        # Apply tuned hyperparameters if available
        if 'transformer' in self.tuned_configs:
            tuned = self.tuned_configs['transformer']
            for key in ('d_model', 'num_heads', 'ff_dim', 'num_layers', 'dropout_rate',
                        'learning_rate', 'warmup_steps', 'batch_size'):
                if key in tuned:
                    transformer_config[key] = tuned[key]
            logger.info(f"  Applied tuned hyperparameters for Transformer: {list(tuned.keys())}")

        model = TransformerForecaster(config=transformer_config, model_dir=artifact_dir)

        X_train_seq, y_train_seq = model.prepare_sequences(data['combined_train'])
        X_val_seq, y_val_seq = model.prepare_sequences(data['combined_val'])
        X_test_seq, y_test_seq = model.prepare_sequences(data['combined_test'])

        t0 = time.time()
        metrics = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=1)
        train_time = time.time() - t0

        history = model.training_history.history if model.training_history else {}

        test_preds = model.model.predict(X_test_seq, verbose=0)

        return {
            'model': model,
            'metrics': metrics,
            'history': history,
            'train_time': train_time,
            'config': transformer_config,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'test_preds': test_preds,
        }

    def _train_lightgbm(self, data: Dict) -> Dict[str, Any]:
        artifact_dir = str(Path(self.config.artifacts_dir) / "lightgbm")

        lgbm_config = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 8,
            'learning_rate': 0.03,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'prediction_horizons': [1, 7, 30],
            'early_stopping_rounds': 50,
        }

        # Apply tuned hyperparameters if available
        if 'lightgbm' in self.tuned_configs:
            tuned = self.tuned_configs['lightgbm']
            for key in ('num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction',
                        'min_child_samples', 'reg_alpha', 'reg_lambda', 'max_depth',
                        'n_estimators', 'bagging_freq'):
                if key in tuned:
                    lgbm_config[key] = tuned[key]
            logger.info(f"  Applied tuned hyperparameters for LightGBM: {list(tuned.keys())}")

        model = LightGBMForecaster(config=lgbm_config, model_dir=artifact_dir)

        # LightGBM needs 3D input (it flattens internally) and multi-horizon y
        seq_len = self.config.sequence_length
        horizons = [1, 7, 30]
        max_h = max(horizons)

        def build_lgbm_data(combined: np.ndarray):
            """Build (X_3d, y_multi) for LightGBM from combined array."""
            target = combined[:, 0]  # first col is close
            features = combined[:, 1:]  # rest are features
            n = len(target)
            X_list, y_list = [], []
            for i in range(seq_len, n - max_h):
                X_list.append(features[i - seq_len:i])
                y_row = [target[i + h] for h in horizons]
                y_list.append(y_row)
            return np.array(X_list), np.array(y_list)

        X_train_lgb, y_train_lgb = build_lgbm_data(data['combined_train'])
        X_val_lgb, y_val_lgb = build_lgbm_data(data['combined_val'])
        X_test_lgb, y_test_lgb = build_lgbm_data(data['combined_test'])

        t0 = time.time()
        history = model.fit(X_train_lgb, y_train_lgb, validation_data=(X_val_lgb, y_val_lgb))
        train_time = time.time() - t0

        # Test predictions
        test_preds = model.predict(X_test_lgb)

        return {
            'model': model,
            'metrics': history,
            'history': history,
            'train_time': train_time,
            'config': lgbm_config,
            'X_test': X_test_lgb,
            'y_test': y_test_lgb,
            'test_preds': test_preds,
        }

    def _train_ensemble(self, data: Dict) -> Dict[str, Any]:
        artifact_dir = str(Path(self.config.artifacts_dir) / "advanced_ensemble")

        ensemble_config = AdvancedEnsemble._default_config()
        ensemble_config.update({
            'sequence_length': self.config.sequence_length,
            'horizons': [1, 7, 30],
        })

        model = AdvancedEnsemble(config=ensemble_config, model_dir=artifact_dir)

        # Reuse already-trained models instead of training from scratch
        # This saves hours of redundant training
        trained_models = []
        if 'transformer' in self.results:
            model.transformer = self.results['transformer']['model']
            trained_models.append('transformer')
        if 'lstm' in self.results:
            model.enhanced_lstm = self.results['lstm']['model']
            trained_models.append('enhanced_lstm')
        if 'lightgbm' in self.results:
            model.lightgbm = self.results['lightgbm']['model']
            trained_models.append('lightgbm')

        model.trained_models = trained_models
        price_history = data['y_train_raw']

        t0 = time.time()

        # Only train meta-learner and regime weights (skip sub-model training)
        if model.config['use_meta_learner'] and len(trained_models) >= 2:
            logger.info("  Training meta-learner on pre-trained model predictions...")
            try:
                model._train_meta_learner(
                    data['X_train'], data['y_train'],
                    data['X_val'], data['y_val'],
                    price_history,
                )
            except Exception as e:
                logger.warning(f"  Meta-learner training failed: {e}")

        if price_history is not None and model.config['use_regime_weighting']:
            model._initialize_regime_weights(price_history)

        model.metadata = {
            'trained_at': datetime.now().isoformat(),
            'config': ensemble_config,
            'trained_models': trained_models,
            'reused_pretrained': True,
        }

        train_time = time.time() - t0

        return {
            'model': model,
            'metrics': {},
            'train_time': train_time,
            'config': ensemble_config,
        }


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(
    config: TrainingConfig,
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Run Optuna hyperparameter optimization for selected models.

    Returns dict mapping model_type -> best_params.
    """
    if not HYPEROPT_AVAILABLE:
        logger.warning("Hyperopt not available (optuna not installed). Skipping tuning.")
        logger.warning("Install with: pip install optuna")
        return {}

    logger.info("=" * 60)
    logger.info("  HYPERPARAMETER OPTIMIZATION")
    logger.info(f"  Trials per model: {config.tune_trials}")
    logger.info(f"  Timeout per model: {config.tune_timeout}s")
    logger.info("=" * 60)

    optimizer = HyperparameterOptimizer(
        study_name=f"tune_{config.crypto}",
    )

    # Build sequenced data for the optimizer (LSTM/Transformer need sequences)
    # The optimizer's ObjectiveFunction handles this internally via model.prepare_sequences()
    # But it expects flat X, y arrays — pass the combined train/val arrays
    models_to_tune = [m.strip().lower() for m in config.tune_models.split(',')]
    tuned_configs = {}
    tuning_results = {}

    for model_type in models_to_tune:
        logger.info(f"\n  Tuning {model_type} ({config.tune_trials} trials)...")
        try:
            result = optimizer.optimize_model(
                model_type=model_type,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                n_trials=config.tune_trials,
                timeout=config.tune_timeout,
                feature_names=data['feature_names'],
                optimization_metric="rmse",
                cv_folds=3,
            )
            if result and 'best_params' in result:
                tuned_configs[model_type] = result['best_params']
                tuning_results[model_type] = {
                    'best_value': result.get('best_value'),
                    'best_trial': result.get('best_trial'),
                    'n_trials': result.get('n_trials'),
                }
                logger.info(f"  Best {model_type} RMSE: {result.get('best_value', 'N/A'):.4f}")
                logger.info(f"  Best params: {result['best_params']}")
        except Exception as e:
            logger.warning(f"  Tuning failed for {model_type}: {e}")

    logger.info(f"\n  Tuning complete. Tuned models: {list(tuned_configs.keys())}")
    return tuned_configs


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def run_walk_forward_validation(
    config: TrainingConfig,
    data: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Run walk-forward cross-validation on trained models.

    Returns dict mapping model_name -> {metric: [fold_scores]}.
    """
    if not WALKFORWARD_AVAILABLE:
        logger.warning("Walk-forward validation not available. Skipping.")
        return {}

    logger.info("=" * 60)
    logger.info("  WALK-FORWARD VALIDATION")
    logger.info(f"  Splits: {config.wf_n_splits}, Expanding: {config.wf_expanding}")
    logger.info("=" * 60)

    val_config = ValidationConfig(
        min_train_size=config.wf_min_train_size,
        validation_size=200,
        step_size=100,
        expanding_window=config.wf_expanding,
    )
    validator = WalkForwardValidator(val_config)

    # Combine all scaled data back for walk-forward splits
    X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])
    y_all = np.concatenate([data['y_train'], data['y_val'], data['y_test']])

    wf_results = {}

    for model_name in ('lstm', 'transformer', 'lightgbm'):
        if model_name not in results:
            continue

        model = results[model_name]['model']
        logger.info(f"\n  Walk-forward validation for {model_name}...")

        try:
            fold_scores = validator.validate_model(
                model, X_all, y_all, data['feature_names']
            )
            wf_results[model_name] = fold_scores

            for metric in ('rmse', 'mae', 'r2'):
                if metric in fold_scores and fold_scores[metric]:
                    mean_val = np.mean(fold_scores[metric])
                    std_val = np.std(fold_scores[metric])
                    logger.info(f"    {model_name} {metric}: {mean_val:.4f} +/- {std_val:.4f}")
        except Exception as e:
            logger.warning(f"    Walk-forward failed for {model_name}: {e}")

    return wf_results


# ---------------------------------------------------------------------------
# Uncertainty Quantification
# ---------------------------------------------------------------------------

def run_uncertainty_quantification(
    results: Dict[str, Any],
    data: Dict[str, Any],
    config: TrainingConfig,
) -> Dict[str, Dict[str, Any]]:
    """Run MC dropout / predict_proba for confidence intervals.

    Returns dict mapping model_name -> uncertainty metrics.
    """
    logger.info("=" * 60)
    logger.info("  UNCERTAINTY QUANTIFICATION")
    logger.info(f"  MC samples: {config.mc_samples}")
    logger.info("=" * 60)

    uq_results = {}

    # LSTM and Transformer: Monte Carlo dropout
    for model_name in ('lstm', 'transformer'):
        if model_name not in results:
            continue

        model = results[model_name]['model']
        X_test = results[model_name]['X_test']

        try:
            preds, confidence = model.predict(
                X_test, return_confidence=True, num_simulations=config.mc_samples
            )

            # Compute mean CI width from first horizon
            if isinstance(confidence, dict):
                first_key = list(confidence.keys())[0]
                ci = confidence[first_key]
            elif isinstance(confidence, list):
                ci = confidence[0]
            else:
                ci = confidence

            width = ci[:, 1] - ci[:, 0]
            mean_width = float(np.mean(width))

            uq_results[model_name] = {
                'mean_ci_width': mean_width,
                'median_ci_width': float(np.median(width)),
                'mc_samples': config.mc_samples,
            }
            logger.info(f"  {model_name}: mean 95%% CI width = {mean_width:.4f}")
        except Exception as e:
            logger.warning(f"  Uncertainty failed for {model_name}: {e}")

    # LightGBM: predict_proba uncertainty
    if 'lightgbm' in results:
        model = results['lightgbm']['model']
        X_test = results['lightgbm']['X_test']

        try:
            preds, uncertainties = model.predict_proba(X_test)
            mean_unc = float(np.mean(uncertainties))
            uq_results['lightgbm'] = {
                'mean_uncertainty': mean_unc,
                'median_uncertainty': float(np.median(uncertainties)),
            }
            logger.info(f"  lightgbm: mean uncertainty = {mean_unc:.4f}")
        except Exception as e:
            logger.warning(f"  Uncertainty failed for lightgbm: {e}")

    return uq_results


# ---------------------------------------------------------------------------
# Ensemble Evaluation
# ---------------------------------------------------------------------------

def run_ensemble_evaluation(
    results: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Comprehensive ensemble evaluation with per-model metrics."""
    if 'ensemble' not in results:
        return {}

    logger.info("=" * 60)
    logger.info("  ENSEMBLE EVALUATION")
    logger.info("=" * 60)

    ensemble_model = results['ensemble']['model']

    try:
        eval_metrics = ensemble_model.evaluate_ensemble(
            X_test=data['X_test'],
            y_test=data['y_test'],
            price_history=data['y_test_raw'],
        )

        # Log ensemble weights
        weights = getattr(ensemble_model, 'model_weights', {})
        logger.info(f"  Final ensemble weights: {weights}")

        if 'improvement_pct' in eval_metrics:
            logger.info(f"  Ensemble improvement over best individual: "
                        f"{eval_metrics['improvement_pct']:.2f}%")

        return {
            'metrics': eval_metrics,
            'weights': {k: float(v) for k, v in weights.items()} if weights else {},
        }
    except Exception as e:
        logger.warning(f"  Ensemble evaluation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Visualization Generation
# ---------------------------------------------------------------------------

class TrainingVisualizer:
    """Generates and saves all training plots as PNG files."""

    def __init__(self, output_dir: Path, crypto: str):
        self.output_dir = output_dir
        self.crypto = crypto
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files: List[str] = []

    def generate_all(self, results: Dict[str, Any], data: Dict[str, Any]):
        """Generate all visualizations."""
        logger.info(f"Generating visualizations in {self.output_dir}")

        self._plot_loss_curves(results)
        self._plot_metrics_progression(results)
        self._plot_learning_rates(results)
        self._plot_attention_heatmap(results, data)
        self._plot_feature_importance(results)
        self._plot_model_comparison(results)
        self._plot_predictions_vs_actual(results)
        self._plot_residual_analysis(results)
        self._plot_ensemble_weights(results)
        self._plot_training_summary(results)

        logger.info(f"Generated {len(self.generated_files)} visualizations")
        return self.generated_files

    def _save(self, fig, name: str):
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.generated_files.append(str(path))
        logger.info(f"  Saved: {name}")

    # -- 1. Loss Curves ------------------------------------------------------

    def _plot_loss_curves(self, results: Dict):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{self.crypto} — Training & Validation Loss', fontsize=14, fontweight='bold')

        for idx, (model_name, ax) in enumerate(zip(['lstm', 'transformer', 'lightgbm'], axes)):
            if model_name not in results:
                ax.set_visible(False)
                continue

            r = results[model_name]
            h = r.get('history', {})

            if model_name in ('lstm', 'transformer'):
                loss = h.get('loss', [])
                val_loss = h.get('val_loss', [])
                if loss:
                    epochs = range(1, len(loss) + 1)
                    ax.plot(epochs, loss, label='Train Loss', linewidth=2)
                if val_loss:
                    ax.plot(epochs, val_loss, label='Val Loss', linewidth=2, linestyle='--')
            else:
                # LightGBM: history has train_mae / val_mae lists
                train_mae = h.get('train_mae', [])
                val_mae = h.get('val_mae', [])
                if train_mae:
                    ax.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', linewidth=2)
                if val_mae:
                    ax.plot(range(1, len(val_mae) + 1), val_mae, label='Val MAE', linewidth=2, linestyle='--')

            ax.set_title(model_name.upper())
            ax.set_xlabel('Epoch / Round')
            ax.set_ylabel('Loss / MAE')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, 'loss_curves.png')

    # -- 2. Metrics Progression ----------------------------------------------

    def _plot_metrics_progression(self, results: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.crypto} — Metrics Progression', fontsize=14, fontweight='bold')

        metric_pairs = [
            ('loss', 'val_loss', 'Total Loss'),
            ('output_1d_loss', 'val_output_1d_loss', '1-Day Loss'),
            ('output_7d_loss', 'val_output_7d_loss', '7-Day Loss'),
            ('output_30d_loss', 'val_output_30d_loss', '30-Day Loss'),
        ]

        for ax, (train_key, val_key, title) in zip(axes.flat, metric_pairs):
            has_data = False
            for model_name in ('lstm', 'transformer'):
                if model_name not in results:
                    continue
                h = results[model_name].get('history', {})
                train_vals = h.get(train_key, [])
                val_vals = h.get(val_key, [])
                if train_vals:
                    ax.plot(train_vals, label=f'{model_name} train', linewidth=1.5)
                    has_data = True
                if val_vals:
                    ax.plot(val_vals, label=f'{model_name} val', linewidth=1.5, linestyle='--')
                    has_data = True

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            if has_data:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, 'metrics_progression.png')

    # -- 3. Learning Rate Schedules ------------------------------------------

    def _plot_learning_rates(self, results: Dict):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{self.crypto} — Learning Rate Schedules', fontsize=14, fontweight='bold')

        has_data = False
        for model_name in ('lstm', 'transformer'):
            if model_name not in results:
                continue
            h = results[model_name].get('history', {})
            # TF 2.18/Keras 3.x uses 'learning_rate', older versions use 'lr'
            lr = h.get('lr', h.get('learning_rate', []))
            if lr:
                ax.plot(range(1, len(lr) + 1), lr, label=model_name.upper(), linewidth=2)
                has_data = True

        if has_data:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR history recorded', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

        fig.tight_layout()
        self._save(fig, 'learning_rates.png')

    # -- 4. Attention Heatmap ------------------------------------------------

    def _plot_attention_heatmap(self, results: Dict, data: Dict):
        if 'lstm' not in results:
            return

        lstm_model = results['lstm']['model']
        X_test = results['lstm'].get('X_test')

        if X_test is None or len(X_test) == 0:
            return

        try:
            # Get attention weights from a sample
            sample = X_test[:1]
            attention_weights = lstm_model.analyze_attention_weights(sample)

            if attention_weights is not None and hasattr(attention_weights, 'shape'):
                fig, ax = plt.subplots(figsize=(12, 4))
                if attention_weights.ndim == 1:
                    ax.bar(range(len(attention_weights)), attention_weights, color='steelblue', alpha=0.8)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Attention Weight')
                else:
                    sns.heatmap(attention_weights, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Weight'})
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Sample')

                ax.set_title(f'{self.crypto} — LSTM Attention Weights', fontsize=14, fontweight='bold')
                fig.tight_layout()
                self._save(fig, 'attention_heatmap.png')
        except Exception as e:
            logger.warning(f"Could not generate attention heatmap: {e}")

    # -- 5. Feature Importance -----------------------------------------------

    def _plot_feature_importance(self, results: Dict):
        if 'lightgbm' not in results:
            return

        lgbm_model = results['lightgbm']['model']

        try:
            raw_importance = lgbm_model.get_feature_importance()
            if not raw_importance:
                return

            # Aggregate across horizons (get_feature_importance returns per-horizon dicts)
            aggregated = {}
            for horizon_key, feat_dict in raw_importance.items():
                if isinstance(feat_dict, dict):
                    for feat, val in feat_dict.items():
                        aggregated[feat] = aggregated.get(feat, 0) + val
                else:
                    # Already flat {feature: value}
                    aggregated[horizon_key] = feat_dict

            if not aggregated:
                return

            # Take top 30
            sorted_imp = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:30]
            names, values = zip(*sorted_imp)

            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = range(len(names))
            ax.barh(y_pos, values, color='steelblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{self.crypto} — LightGBM Feature Importance (Top 30)',
                         fontsize=14, fontweight='bold')
            fig.tight_layout()
            self._save(fig, 'feature_importance.png')
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")

    # -- 6. Model Comparison -------------------------------------------------

    def _plot_model_comparison(self, results: Dict):
        """Bar chart comparing RMSE/MAE across models and horizons."""
        horizons = ['1d', '7d', '30d']
        model_names = [m for m in ('lstm', 'transformer', 'lightgbm') if m in results]

        if not model_names:
            return

        # Collect test-set metrics per model per horizon
        comparison = {m: {} for m in model_names}

        for model_name in model_names:
            r = results[model_name]
            y_test = r.get('y_test', {})
            test_preds = r.get('test_preds')

            if test_preds is None:
                continue

            if isinstance(y_test, dict):
                # Multi-output model (LSTM / Transformer)
                for i, h in enumerate(horizons):
                    key = f'output_{h}'
                    if key not in y_test:
                        continue
                    y_true = y_test[key]
                    # test_preds can be dict (named outputs) or list (positional)
                    if isinstance(test_preds, dict):
                        y_pred_raw = test_preds.get(key)
                        if y_pred_raw is None:
                            continue
                        y_pred = np.asarray(y_pred_raw).flatten()[:len(y_true)]
                    else:
                        pred_list = test_preds if isinstance(test_preds, list) else [test_preds]
                        if i >= len(pred_list):
                            continue
                        y_pred = np.asarray(pred_list[i]).flatten()[:len(y_true)]
                    comparison[model_name][h] = {
                        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        'mae': float(mean_absolute_error(y_true, y_pred)),
                    }
            elif isinstance(y_test, np.ndarray) and y_test.ndim == 2:
                # LightGBM multi-horizon
                preds = test_preds if isinstance(test_preds, np.ndarray) else np.array(test_preds)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                for i, h in enumerate(horizons):
                    if i < y_test.shape[1] and i < preds.shape[1]:
                        comparison[model_name][h] = {
                            'rmse': float(np.sqrt(mean_squared_error(y_test[:, i], preds[:, i]))),
                            'mae': float(mean_absolute_error(y_test[:, i], preds[:, i])),
                        }

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.crypto} — Model Comparison', fontsize=14, fontweight='bold')

        for ax, metric in zip(axes, ['rmse', 'mae']):
            x = np.arange(len(horizons))
            width = 0.25
            for i, model_name in enumerate(model_names):
                vals = [comparison[model_name].get(h, {}).get(metric, 0) for h in horizons]
                ax.bar(x + i * width, vals, width, label=model_name.upper(), alpha=0.85)
            ax.set_xticks(x + width)
            ax.set_xticklabels(horizons)
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        self._save(fig, 'model_comparison.png')

    # -- 7. Prediction vs Actual ---------------------------------------------

    def _plot_predictions_vs_actual(self, results: Dict):
        model_names = [m for m in ('lstm', 'transformer', 'lightgbm') if m in results]
        n_models = len(model_names)
        if n_models == 0:
            return

        fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{self.crypto} — Predictions vs Actual', fontsize=14, fontweight='bold')
        horizons = ['1d', '7d', '30d']

        for row, model_name in enumerate(model_names):
            r = results[model_name]
            y_test = r.get('y_test')
            test_preds = r.get('test_preds')
            if test_preds is None:
                continue

            for col, h in enumerate(horizons):
                ax = axes[row, col]
                y_true, y_pred = self._extract_horizon(y_test, test_preds, h, col)

                if y_true is not None and y_pred is not None:
                    n = min(len(y_true), len(y_pred))
                    y_true, y_pred = y_true[:n], y_pred[:n]
                    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
                    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
                    r2 = r2_score(y_true, y_pred)
                    ax.set_title(f'{model_name.upper()} {h} (R²={r2:.3f})', fontsize=10)
                else:
                    ax.set_title(f'{model_name.upper()} {h} (no data)')

                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, 'predictions_vs_actual.png')

    def _extract_horizon(self, y_test, test_preds, horizon_label, idx):
        """Extract (y_true, y_pred) for a given horizon from model outputs."""
        if isinstance(y_test, dict):
            key = f'output_{horizon_label}'
            if key not in y_test:
                return None, None
            y_true = y_test[key]
            # test_preds can be: list of arrays, dict of arrays, or single array
            if isinstance(test_preds, dict):
                y_pred_raw = test_preds.get(key)
                if y_pred_raw is None:
                    return None, None
                y_pred = np.asarray(y_pred_raw).flatten()
            else:
                pred_list = test_preds if isinstance(test_preds, list) else [test_preds]
                if idx < len(pred_list):
                    y_pred = np.asarray(pred_list[idx]).flatten()
                else:
                    return None, None
        elif isinstance(y_test, np.ndarray) and y_test.ndim == 2:
            if idx >= y_test.shape[1]:
                return None, None
            y_true = y_test[:, idx]
            preds = test_preds if isinstance(test_preds, np.ndarray) else np.array(test_preds)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if idx >= preds.shape[1]:
                return None, None
            y_pred = preds[:, idx]
        else:
            return None, None
        return y_true, y_pred

    # -- 8. Residual Analysis ------------------------------------------------

    def _plot_residual_analysis(self, results: Dict):
        model_names = [m for m in ('lstm', 'transformer', 'lightgbm') if m in results]
        if not model_names:
            return

        fig, axes = plt.subplots(len(model_names), 2, figsize=(12, 5 * len(model_names)))
        if len(model_names) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{self.crypto} — Residual Analysis (1-Day Horizon)',
                     fontsize=14, fontweight='bold')

        for row, model_name in enumerate(model_names):
            r = results[model_name]
            y_true, y_pred = self._extract_horizon(r.get('y_test'), r.get('test_preds'), '1d', 0)

            if y_true is None or y_pred is None:
                continue

            n = min(len(y_true), len(y_pred))
            residuals = y_true[:n] - y_pred[:n]

            # Histogram
            axes[row, 0].hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
            axes[row, 0].axvline(0, color='red', linestyle='--')
            axes[row, 0].set_title(f'{model_name.upper()} — Residual Distribution')
            axes[row, 0].set_xlabel('Residual')

            # Residuals vs Predicted
            axes[row, 1].scatter(y_pred[:n], residuals, alpha=0.3, s=10, color='steelblue')
            axes[row, 1].axhline(0, color='red', linestyle='--')
            axes[row, 1].set_title(f'{model_name.upper()} — Residuals vs Predicted')
            axes[row, 1].set_xlabel('Predicted')
            axes[row, 1].set_ylabel('Residual')

        fig.tight_layout()
        self._save(fig, 'residual_analysis.png')

    # -- 9. Ensemble Weights -------------------------------------------------

    def _plot_ensemble_weights(self, results: Dict):
        if 'ensemble' not in results:
            return

        ensemble = results['ensemble']['model']
        weights = getattr(ensemble, 'model_weights', None)

        if weights is None or not isinstance(weights, dict):
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(weights.keys())
        vals = list(weights.values())
        colors = sns.color_palette("husl", len(names))

        ax.bar(names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Weight')
        ax.set_title(f'{self.crypto} — Ensemble Model Weights', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2 if vals else 1)

        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)

        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        self._save(fig, 'ensemble_weights.png')

    # -- 10. Training Summary ------------------------------------------------

    def _plot_training_summary(self, results: Dict):
        model_names = [m for m in ('lstm', 'transformer', 'lightgbm', 'ensemble') if m in results]
        if not model_names:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.crypto} — Training Summary', fontsize=16, fontweight='bold')

        # Panel 1: Training times
        ax = axes[0, 0]
        times = [results[m].get('train_time', 0) for m in model_names]
        colors = sns.color_palette("husl", len(model_names))
        ax.bar([m.upper() for m in model_names], times, color=colors, alpha=0.85)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time')
        for i, t in enumerate(times):
            ax.text(i, t + 0.5, f'{t:.1f}s', ha='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 2: Final training loss (LSTM & Transformer)
        ax = axes[0, 1]
        for model_name in ('lstm', 'transformer'):
            if model_name not in results:
                continue
            h = results[model_name].get('history', {})
            loss = h.get('loss', [])
            val_loss = h.get('val_loss', [])
            if loss:
                ax.plot(loss, label=f'{model_name} train', linewidth=1.5)
            if val_loss:
                ax.plot(val_loss, label=f'{model_name} val', linewidth=1.5, linestyle='--')
        ax.set_title('Loss Convergence')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 3: 1-Day RMSE comparison
        ax = axes[1, 0]
        rmse_vals = []
        rmse_names = []
        for model_name in ('lstm', 'transformer', 'lightgbm'):
            if model_name not in results:
                continue
            r = results[model_name]
            y_true, y_pred = self._extract_horizon(r.get('y_test'), r.get('test_preds'), '1d', 0)
            if y_true is not None and y_pred is not None:
                n = min(len(y_true), len(y_pred))
                rmse = np.sqrt(mean_squared_error(y_true[:n], y_pred[:n]))
                rmse_vals.append(rmse)
                rmse_names.append(model_name.upper())

        if rmse_vals:
            colors_bar = sns.color_palette("husl", len(rmse_names))
            ax.bar(rmse_names, rmse_vals, color=colors_bar, alpha=0.85)
            for i, v in enumerate(rmse_vals):
                ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
        ax.set_title('Test RMSE (1-Day)')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 4: Epochs trained
        ax = axes[1, 1]
        epoch_info = []
        epoch_names = []
        for model_name in ('lstm', 'transformer'):
            if model_name not in results:
                continue
            h = results[model_name].get('history', {})
            n_epochs = len(h.get('loss', []))
            epoch_info.append(n_epochs)
            epoch_names.append(model_name.upper())

        if epoch_info:
            colors_e = sns.color_palette("husl", len(epoch_names))
            ax.bar(epoch_names, epoch_info, color=colors_e, alpha=0.85)
            for i, v in enumerate(epoch_info):
                ax.text(i, v + 0.5, str(v), ha='center', fontsize=11)
        ax.set_title('Epochs Trained (early stopping)')
        ax.set_ylabel('Epochs')
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        self._save(fig, 'training_summary.png')


# ---------------------------------------------------------------------------
# Model Saving
# ---------------------------------------------------------------------------

def save_models(results: Dict[str, Any], config: TrainingConfig):
    """Save all trained models to artifacts directory."""
    logger.info("Saving models...")

    if 'lstm' in results:
        try:
            results['lstm']['model'].save_model("production")
            logger.info("  Saved LSTM model")
        except Exception as e:
            logger.warning(f"  Failed to save LSTM: {e}")

    if 'transformer' in results:
        try:
            results['transformer']['model'].save_model("production")
            logger.info("  Saved Transformer model")
        except Exception as e:
            logger.warning(f"  Failed to save Transformer: {e}")

    if 'lightgbm' in results:
        try:
            lgbm_dir = Path(config.artifacts_dir) / "lightgbm"
            results['lightgbm']['model'].save_model(str(lgbm_dir))
            logger.info("  Saved LightGBM model")
        except Exception as e:
            logger.warning(f"  Failed to save LightGBM: {e}")

    if 'ensemble' in results:
        try:
            results['ensemble']['model'].save_ensemble("production")
            logger.info("  Saved Ensemble model")
        except Exception as e:
            logger.warning(f"  Failed to save Ensemble: {e}")


# ---------------------------------------------------------------------------
# Training Report
# ---------------------------------------------------------------------------

def _extract_horizon_standalone(y_test, test_preds, horizon_label, idx):
    """Extract (y_true, y_pred) arrays for a given horizon from model outputs."""
    if isinstance(y_test, dict):
        key = f'output_{horizon_label}'
        if key not in y_test:
            return None, None
        y_true = np.asarray(y_test[key]).flatten()
        if isinstance(test_preds, dict):
            y_pred_raw = test_preds.get(key)
            if y_pred_raw is None:
                return None, None
            y_pred = np.asarray(y_pred_raw).flatten()
        else:
            pred_list = test_preds if isinstance(test_preds, list) else [test_preds]
            if idx < len(pred_list):
                y_pred = np.asarray(pred_list[idx]).flatten()
            else:
                return None, None
    elif isinstance(y_test, np.ndarray) and y_test.ndim == 2:
        if idx >= y_test.shape[1]:
            return None, None
        y_true = y_test[:, idx]
        preds = test_preds if isinstance(test_preds, np.ndarray) else np.array(test_preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if idx >= preds.shape[1]:
            return None, None
        y_pred = preds[:, idx]
    else:
        return None, None
    return y_true, y_pred


def generate_report(
    results: Dict[str, Any],
    config: TrainingConfig,
    output_dir: Path,
    uq_results: Optional[Dict] = None,
    wf_results: Optional[Dict] = None,
    ensemble_eval: Optional[Dict] = None,
    tuning_results: Optional[Dict] = None,
):
    """Generate a comprehensive JSON training report."""
    report = {
        'crypto': config.crypto,
        'timestamp': datetime.now().isoformat(),
        'pipeline_flags': {
            'tune': config.tune,
            'walk_forward': config.walk_forward,
            'uncertainty': config.uncertainty,
            'advanced_mlflow': config.advanced_mlflow,
        },
        'config': {
            'sequence_length': config.sequence_length,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'max_features': config.max_features,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
        },
        'models': {},
    }

    for model_name, r in results.items():
        entry = {
            'train_time_seconds': r.get('train_time', 0),
        }
        metrics = r.get('metrics', {})
        if isinstance(metrics, dict):
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    clean_metrics[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    # LightGBM stores per-horizon metrics as lists
                    clean_metrics[k] = v
            entry['metrics'] = clean_metrics

        # Directional accuracy: % of timesteps where predicted direction matches actual
        y_test = r.get('y_test')
        test_preds = r.get('test_preds')
        if y_test is not None and test_preds is not None:
            dir_acc = {}
            for idx, horizon in enumerate(['1d', '7d', '30d']):
                y_true, y_pred = _extract_horizon_standalone(y_test, test_preds, horizon, idx)
                if y_true is not None and y_pred is not None and len(y_true) > 1:
                    n = min(len(y_true), len(y_pred))
                    actual_dir = np.sign(np.diff(y_true[:n]))
                    pred_dir = np.sign(np.diff(y_pred[:n]))
                    mask = actual_dir != 0  # exclude flat days
                    if mask.sum() > 0:
                        dir_acc[horizon] = float(np.mean(actual_dir[mask] == pred_dir[mask]))
            if dir_acc:
                entry['directional_accuracy'] = dir_acc

        report['models'][model_name] = entry

    # Hyperparameter tuning results
    if tuning_results:
        report['hyperparameter_tuning'] = {}
        for model_name, info in tuning_results.items():
            report['hyperparameter_tuning'][model_name] = {
                k: v for k, v in info.items() if isinstance(v, (int, float, str))
            }

    # Uncertainty analysis
    if uq_results:
        report['uncertainty_analysis'] = {}
        for model_name, uq in uq_results.items():
            report['uncertainty_analysis'][model_name] = {
                k: v for k, v in uq.items() if isinstance(v, (int, float, str))
            }

    # Walk-forward validation
    if wf_results:
        report['walk_forward_validation'] = {}
        for model_name, scores in wf_results.items():
            report['walk_forward_validation'][model_name] = {
                metric: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n_folds': len(values),
                }
                for metric, values in scores.items()
                if isinstance(values, list) and len(values) > 0
            }

    # Ensemble evaluation
    if ensemble_eval and 'metrics' in ensemble_eval:
        report['ensemble_evaluation'] = {
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in ensemble_eval['metrics'].items()},
            'weights': ensemble_eval.get('weights', {}),
        }

    report_path = output_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Training report saved: {report_path}")
    return report


# ---------------------------------------------------------------------------
# MLflow Integration
# ---------------------------------------------------------------------------

def log_to_mlflow(
    results: Dict[str, Any],
    config: TrainingConfig,
    viz_files: List[str],
    uq_results: Optional[Dict] = None,
    wf_results: Optional[Dict] = None,
    ensemble_eval: Optional[Dict] = None,
    tuning_results: Optional[Dict] = None,
):
    """Log training run to MLflow with advanced features."""
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available, skipping logging")
        return

    try:
        manager = create_experiment_manager(
            experiment_name=f"production_{config.crypto}",
            model_type="production_training",
            description=f"Production training for {config.crypto}",
            tracking_uri=f"file://{mlflow_dir}",
        )

        run_name = f"train_{config.crypto}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with manager.start_run(run_name):
            # Standard params
            manager.log_params_batch({
                'crypto': config.crypto,
                'sequence_length': config.sequence_length,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'max_features': config.max_features,
                'tune_enabled': config.tune,
                'walk_forward_enabled': config.walk_forward,
                'uncertainty_enabled': config.uncertainty,
            })

            # Per-model metrics and training times
            for model_name, r in results.items():
                metrics = r.get('metrics', {})
                if isinstance(metrics, dict):
                    flat = {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            flat[f'{model_name}_{k}'] = v
                    if flat:
                        manager.log_metrics_batch(flat)

                train_time = r.get('train_time', 0)
                manager.log_metric(f'{model_name}_train_time', train_time)

            # Advanced MLflow features
            if config.advanced_mlflow:
                # Interactive training curves
                for model_name in ('lstm', 'transformer'):
                    if model_name in results:
                        history = results[model_name].get('history', {})
                        if history and hasattr(manager, 'log_training_curves'):
                            try:
                                manager.log_training_curves(
                                    history,
                                    title=f"{config.crypto} {model_name.upper()} Training",
                                )
                            except Exception as e:
                                logger.debug(f"Training curves logging failed: {e}")

                # Feature importance (LightGBM)
                if 'lightgbm' in results and hasattr(manager, 'log_feature_importance'):
                    try:
                        lgbm = results['lightgbm']['model']
                        importance = lgbm.get_feature_importance()
                        if importance:
                            first_horizon = list(importance.values())[0]
                            if isinstance(first_horizon, dict):
                                names = list(first_horizon.keys())[:30]
                                values = list(first_horizon.values())[:30]
                                manager.log_feature_importance(
                                    names, values,
                                    title=f"{config.crypto} LightGBM Feature Importance",
                                )
                    except Exception as e:
                        logger.debug(f"Feature importance logging failed: {e}")

                # Model card
                if hasattr(manager, 'create_model_card'):
                    try:
                        model_info = {
                            'name': f"{config.crypto} Production Ensemble",
                            'description': f"Multi-model ensemble for {config.crypto} price prediction",
                            'dataset': f"{config.crypto}_features.parquet",
                            'architecture': 'LSTM + Transformer + LightGBM + Ensemble',
                            'training_time': sum(
                                r.get('train_time', 0) for r in results.values()
                            ),
                            'metrics': {},
                        }
                        for mn, r in results.items():
                            m = r.get('metrics', {})
                            if isinstance(m, dict):
                                for k, v in m.items():
                                    if isinstance(v, (int, float)):
                                        model_info['metrics'][f'{mn}_{k}'] = v
                        manager.create_model_card(model_info)
                    except Exception as e:
                        logger.debug(f"Model card creation failed: {e}")

            # Uncertainty results
            if uq_results:
                for model_name, uq in uq_results.items():
                    for k, v in uq.items():
                        if isinstance(v, (int, float)):
                            manager.log_metric(f'{model_name}_uq_{k}', v)

            # Walk-forward results
            if wf_results:
                for model_name, scores in wf_results.items():
                    for metric, values in scores.items():
                        if isinstance(values, list) and len(values) > 0:
                            manager.log_metric(
                                f'wf_{model_name}_{metric}_mean', float(np.mean(values))
                            )
                            manager.log_metric(
                                f'wf_{model_name}_{metric}_std', float(np.std(values))
                            )

            # Ensemble evaluation
            if ensemble_eval and 'metrics' in ensemble_eval:
                for k, v in ensemble_eval['metrics'].items():
                    if isinstance(v, (int, float)):
                        manager.log_metric(f'ensemble_eval_{k}', v)

            # Log visualization artifacts
            for viz_file in viz_files:
                try:
                    if hasattr(manager, 'log_artifact'):
                        manager.log_artifact(viz_file, "visualizations")
                except Exception:
                    pass

        logger.info("MLflow logging complete")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_single_crypto(config: TrainingConfig):
    """Full training pipeline for a single cryptocurrency."""
    logger.info(f"{'='*60}")
    logger.info(f"  TRAINING: {config.crypto}")
    logger.info(f"  Flags: tune={config.tune}, walk_forward={config.walk_forward}, "
                f"uncertainty={config.uncertainty}")
    logger.info(f"{'='*60}")

    # 1. Load data
    loader = ProductionDataLoader(config)
    data = loader.load_and_prepare()

    # 2. Hyperparameter tuning (optional)
    tuned_configs = {}
    tuning_results = None
    if config.tune:
        tuned_configs = run_hyperparameter_tuning(config, data)
        # Store tuning metadata for report
        tuning_results = tuned_configs

    # 3. Train models (with optional tuned configs)
    trainer = ProductionTrainer(config, tuned_configs=tuned_configs)
    results = trainer.train_all(data)

    # 4. Walk-forward validation (optional)
    wf_results = None
    if config.walk_forward:
        wf_results = run_walk_forward_validation(config, data, results)

    # 5. Uncertainty quantification (on by default)
    uq_results = None
    if config.uncertainty:
        uq_results = run_uncertainty_quantification(results, data, config)

    # 6. Ensemble evaluation (always, if ensemble trained)
    ensemble_eval = None
    if 'ensemble' in results:
        ensemble_eval = run_ensemble_evaluation(results, data)

    # 7. Save models FIRST (never lose trained weights to a viz crash)
    save_models(results, config)

    # 8. Generate visualizations (non-fatal — wrapped in try/except)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = Path(config.output_dir) / f"{config.crypto}_{timestamp}"
    visualizer = TrainingVisualizer(viz_dir, config.crypto)
    viz_files = []
    try:
        viz_files = visualizer.generate_all(results, data)
    except Exception as e:
        logger.warning(f"Visualization generation failed (models already saved): {e}")

    # 9. Generate enhanced report (non-fatal)
    try:
        report = generate_report(
            results, config, viz_dir,
            uq_results=uq_results,
            wf_results=wf_results,
            ensemble_eval=ensemble_eval,
            tuning_results=tuning_results,
        )
        # Log directional accuracy summary
        for model_name, model_info in report.get('models', {}).items():
            da = model_info.get('directional_accuracy')
            if da:
                parts = [f"{h}: {v:.1%}" for h, v in da.items()]
                logger.info(f"  {model_name} directional accuracy — {', '.join(parts)}")
    except Exception as e:
        logger.warning(f"Report generation failed (models already saved): {e}")

    # 10. Log to MLflow (non-fatal)
    try:
        log_to_mlflow(
            results, config, viz_files,
            uq_results=uq_results,
            wf_results=wf_results,
            ensemble_eval=ensemble_eval,
            tuning_results=tuning_results,
        )
    except Exception as e:
        logger.warning(f"MLflow logging failed (models already saved): {e}")

    logger.info(f"Training complete for {config.crypto}")
    logger.info(f"Visualizations: {viz_dir}")
    logger.info(f"Artifacts: {config.artifacts_dir}")

    return results, viz_dir


def main():
    parser = argparse.ArgumentParser(description="Production Crypto Model Training")
    parser.add_argument("--crypto", type=str, default="BTC",
                        help="Comma-separated crypto tickers (default: BTC)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Skip ensemble training")
    parser.add_argument("--output-dir", type=str,
                        default=str(current_dir / "training_output"))

    # Hyperparameter tuning
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter optimization before training")
    parser.add_argument("--tune-trials", type=int, default=20,
                        help="Optuna trials per model (default: 20)")
    parser.add_argument("--tune-timeout", type=int, default=3600,
                        help="Max seconds for tuning per model (default: 3600)")
    parser.add_argument("--tune-models", type=str, default="lstm,transformer,lightgbm",
                        help="Comma-separated models to tune (default: lstm,transformer,lightgbm)")

    # Walk-forward validation
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward cross-validation after training")
    parser.add_argument("--wf-splits", type=int, default=5,
                        help="Walk-forward splits (default: 5)")
    parser.add_argument("--wf-min-train", type=int, default=1000,
                        help="Min training samples per fold (default: 1000)")
    parser.add_argument("--wf-rolling", action="store_true",
                        help="Use rolling window instead of expanding (default: expanding)")

    # Uncertainty & reporting
    parser.add_argument("--no-uncertainty", action="store_true",
                        help="Disable MC dropout uncertainty quantification")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="MC dropout samples for uncertainty (default: 50)")
    parser.add_argument("--no-advanced-mlflow", action="store_true",
                        help="Use basic MLflow logging instead of advanced")

    args = parser.parse_args()

    cryptos = [c.strip().upper() for c in args.crypto.split(',')]

    for crypto in cryptos:
        config = TrainingConfig(
            crypto=crypto,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            max_features=args.max_features,
            train_ensemble=not args.no_ensemble,
            output_dir=args.output_dir,
            tune=args.tune,
            tune_trials=args.tune_trials,
            tune_timeout=args.tune_timeout,
            tune_models=args.tune_models,
            walk_forward=args.walk_forward,
            wf_n_splits=args.wf_splits,
            wf_min_train_size=args.wf_min_train,
            wf_expanding=not args.wf_rolling,
            uncertainty=not args.no_uncertainty,
            mc_samples=args.mc_samples,
            advanced_mlflow=not args.no_advanced_mlflow,
        )

        try:
            results, viz_dir = train_single_crypto(config)
            print(f"\n{'='*60}")
            print(f"  {crypto} TRAINING COMPLETE")
            print(f"  Plots saved to: {viz_dir}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Training failed for {crypto}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
