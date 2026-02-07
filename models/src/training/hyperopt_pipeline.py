"""
Optuna Hyperparameter Optimization Pipeline

Advanced Bayesian optimization for cryptocurrency prediction models:
- Automated hyperparameter tuning for all model types
- Multi-objective optimization (accuracy + speed)
- Pruning for efficient resource usage
- MLflow integration for experiment tracking
- Model-specific search spaces
- Cross-validation with time-series awareness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime
import json
import warnings

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")

# MLflowCallback moved to optuna-integration in optuna 4.x
try:
    from optuna.integration import MLflowCallback
    OPTUNA_MLFLOW_AVAILABLE = True
except ImportError:
    OPTUNA_MLFLOW_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our models (try relative imports first, fall back to absolute)
try:
    from ..models.transformer_model import TransformerForecaster, TENSORFLOW_AVAILABLE
    from ..models.enhanced_lstm import EnhancedLSTMForecaster
    from ..models.lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE
    from ..models.advanced_ensemble import AdvancedEnsemble
    from .mlflow_integration import MLflowExperimentTracker
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models.transformer_model import TransformerForecaster, TENSORFLOW_AVAILABLE
    from models.enhanced_lstm import EnhancedLSTMForecaster
    from models.lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE
    from models.advanced_ensemble import AdvancedEnsemble
    from training.mlflow_integration import MLflowExperimentTracker


class OptimizationObjective:
    """
    Objective function for Optuna optimization.

    Supports different model types and optimization strategies.
    """

    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int = 3,
        optimization_metric: str = "rmse",
        feature_names: Optional[List[str]] = None,
        mlflow_tracker: Optional[MLflowExperimentTracker] = None
    ):
        self.model_type = model_type.lower()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        self.optimization_metric = optimization_metric
        self.feature_names = feature_names
        self.mlflow_tracker = mlflow_tracker

        # Time series cross-validation
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Validation for model type
        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Validate that model type is supported"""
        supported_models = ['transformer', 'lstm', 'lightgbm', 'ensemble']

        if self.model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Check availability
        if self.model_type in ['transformer', 'lstm'] and not TENSORFLOW_AVAILABLE:
            raise RuntimeError(f"TensorFlow not available for {self.model_type}")

        if self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (lower is better)
        """
        try:
            # Start nested MLflow run if tracker available
            nested_run = False
            if self.mlflow_tracker:
                run_name = f"{self.model_type}_trial_{trial.number}"
                self.mlflow_tracker.start_run(
                    run_name=run_name,
                    tags={"optimization_trial": str(trial.number)},
                    nested=True
                )
                nested_run = True

            # Get hyperparameters for this trial
            if self.model_type == 'transformer':
                config = self._suggest_transformer_params(trial)
            elif self.model_type == 'lstm':
                config = self._suggest_lstm_params(trial)
            elif self.model_type == 'lightgbm':
                config = self._suggest_lightgbm_params(trial)
            elif self.model_type == 'ensemble':
                config = self._suggest_ensemble_params(trial)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Perform cross-validation
            cv_scores = self._cross_validate(config, trial)

            # Calculate final objective value
            objective_value = np.mean(cv_scores)

            # Log to MLflow if available
            if self.mlflow_tracker:
                self.mlflow_tracker.log_model_config(config, self.model_type)
                self.mlflow_tracker.log_training_metrics({
                    f"cv_{self.optimization_metric}": objective_value,
                    "cv_std": float(np.std(cv_scores))
                })

            return objective_value

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Return a large value to indicate failure
            return float('inf')

        finally:
            # End nested run
            if nested_run and self.mlflow_tracker:
                self.mlflow_tracker.end_run()

    def _suggest_transformer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Transformer model"""
        return {
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 12]),
            'ff_dim': trial.suggest_categorical('ff_dim', [1024, 2048, 3072]),
            'num_layers': trial.suggest_int('num_layers', 3, 8),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'warmup_steps': trial.suggest_int('warmup_steps', 500, 2000),
            'epochs': 50,  # Fixed for optimization speed
            'early_stopping_patience': 10
        }

    def _suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Enhanced LSTM model"""
        # Number of LSTM layers
        n_layers = trial.suggest_int('n_lstm_layers', 1, 3)
        lstm_units = []
        for i in range(n_layers):
            units = trial.suggest_categorical(f'lstm_units_{i}', [32, 64, 128, 256])
            lstm_units.append(units)

        # Dense layers
        n_dense = trial.suggest_int('n_dense_layers', 1, 3)
        dense_units = []
        for i in range(n_dense):
            units = trial.suggest_categorical(f'dense_units_{i}', [16, 32, 64, 128])
            dense_units.append(units)

        return {
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.2),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 2.0),
            'epochs': 50,
            'early_stopping_patience': 10
        }

    def _suggest_lightgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for LightGBM model"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'random_state': 42,
            'verbose': -1
        }

    def _suggest_ensemble_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Ensemble model"""
        return {
            'transformer_weight': trial.suggest_float('transformer_weight', 0.1, 0.6),
            'lstm_weight': trial.suggest_float('lstm_weight', 0.1, 0.6),
            'lightgbm_weight': trial.suggest_float('lightgbm_weight', 0.1, 0.6),
            'use_meta_learner': trial.suggest_categorical('use_meta_learner', [True, False]),
            'meta_model_type': trial.suggest_categorical('meta_model_type', ['ridge', 'elastic_net', 'random_forest']),
            'meta_alpha': trial.suggest_float('meta_alpha', 0.01, 1.0, log=True),
            'meta_l1_ratio': trial.suggest_float('meta_l1_ratio', 0.1, 0.9),
            'use_regime_weighting': trial.suggest_categorical('use_regime_weighting', [True, False]),
            'regime_weight_sensitivity': trial.suggest_float('regime_weight_sensitivity', 0.1, 0.5),
            'use_online_learning': trial.suggest_categorical('use_online_learning', [True, False]),
            'online_learning_rate': trial.suggest_float('online_learning_rate', 0.001, 0.1, log=True)
        }

    def _cross_validate(self, config: Dict[str, Any], trial: optuna.Trial) -> List[float]:
        """
        Perform time-series cross-validation.

        Args:
            config: Model configuration
            trial: Optuna trial for pruning

        Returns:
            List of CV scores
        """
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.X_train)):
            try:
                X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]

                # Train model with current config
                score = self._train_and_evaluate(
                    config, X_fold_train, y_fold_train, X_fold_val, y_fold_val
                )

                cv_scores.append(score)

                # Report intermediate value for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                print(f"Fold {fold} failed: {e}")
                # Use a large score to indicate failure
                cv_scores.append(float('inf'))

        return cv_scores

    def _train_and_evaluate(
        self,
        config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Train model and evaluate performance.

        Args:
            config: Model configuration
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Evaluation score
        """
        if self.model_type == 'transformer':
            return self._evaluate_transformer(config, X_train, y_train, X_val, y_val)
        elif self.model_type == 'lstm':
            return self._evaluate_lstm(config, X_train, y_train, X_val, y_val)
        elif self.model_type == 'lightgbm':
            return self._evaluate_lightgbm(config, X_train, y_train, X_val, y_val)
        elif self.model_type == 'ensemble':
            return self._evaluate_ensemble(config, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _evaluate_transformer(self, config, X_train, y_train, X_val, y_val) -> float:
        """Evaluate Transformer model"""
        model = TransformerForecaster(config=config)

        # Prepare sequences
        train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
        val_data = np.column_stack([y_val.reshape(-1, 1), X_val])

        X_train_seq, y_train_seq = model.prepare_sequences(train_data)
        X_val_seq, y_val_seq = model.prepare_sequences(val_data)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return float('inf')

        # Train
        model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0)

        # Predict
        pred = model.predict(X_val_seq)
        if isinstance(pred, dict):
            pred = list(pred.values())[0]  # Take first horizon

        # Calculate score
        return self._calculate_score(y_val_seq, pred.flatten())

    def _evaluate_lstm(self, config, X_train, y_train, X_val, y_val) -> float:
        """Evaluate Enhanced LSTM model"""
        model = EnhancedLSTMForecaster(config=config)

        # Prepare sequences
        train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
        val_data = np.column_stack([y_val.reshape(-1, 1), X_val])

        X_train_seq, y_train_seq = model.prepare_sequences(train_data)
        X_val_seq, y_val_seq = model.prepare_sequences(val_data)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return float('inf')

        # Train
        model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=0)

        # Predict
        pred = model.predict(X_val_seq)
        if isinstance(pred, dict):
            pred = list(pred.values())[0]

        # Calculate score
        return self._calculate_score(y_val_seq, pred.flatten())

    def _evaluate_lightgbm(self, config, X_train, y_train, X_val, y_val) -> float:
        """Evaluate LightGBM model"""
        model = LightGBMForecaster(config=config)

        # Train
        model.train(X_train, y_train, X_val, y_val, self.feature_names)

        # Predict
        pred = model.predict(X_val)

        # Calculate score
        return self._calculate_score(y_val, pred)

    def _evaluate_ensemble(self, config, X_train, y_train, X_val, y_val) -> float:
        """Evaluate Ensemble model"""
        # Normalize weights
        total_weight = config['transformer_weight'] + config['lstm_weight'] + config['lightgbm_weight']
        config['transformer_weight'] /= total_weight
        config['lstm_weight'] /= total_weight
        config['lightgbm_weight'] /= total_weight

        ensemble = AdvancedEnsemble(config=config)

        # Train (simplified for optimization)
        try:
            ensemble.train(X_train, y_train, X_val, y_val, self.feature_names)

            # Predict
            pred = ensemble.predict(X_val)

            # Calculate score
            return self._calculate_score(y_val, pred)

        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")
            return float('inf')

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate evaluation score based on optimization metric"""
        try:
            if len(y_pred) != len(y_true):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            if self.optimization_metric == 'rmse':
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif self.optimization_metric == 'mae':
                return float(mean_absolute_error(y_true, y_pred))
            elif self.optimization_metric == 'mape':
                return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            else:
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        except Exception as e:
            print(f"Score calculation failed: {e}")
            return float('inf')


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization coordinator.

    Manages Optuna studies with MLflow integration.
    """

    def __init__(
        self,
        study_name: str = "crypto_optimization",
        storage_url: Optional[str] = None,
        mlflow_tracker: Optional[MLflowExperimentTracker] = None
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed")

        self.study_name = study_name
        self.storage_url = storage_url
        self.mlflow_tracker = mlflow_tracker

        # Study configuration
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    def optimize_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        optimization_metric: str = "rmse",
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.

        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            feature_names: Feature names
            optimization_metric: Metric to optimize
            cv_folds: Number of CV folds

        Returns:
            Optimization results
        """
        print(f"🎯 Starting hyperparameter optimization for {model_type}")
        print(f"📊 Configuration: {n_trials} trials, {cv_folds}-fold CV, {optimization_metric} metric")

        # Create objective function
        objective = OptimizationObjective(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cv_folds=cv_folds,
            optimization_metric=optimization_metric,
            feature_names=feature_names,
            mlflow_tracker=self.mlflow_tracker
        )

        # Create study
        study_name = f"{self.study_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Lower is better for RMSE/MAE/MAPE
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage_url
        )

        # Setup MLflow callback if available
        callbacks = []
        if self.mlflow_tracker and MLFLOW_AVAILABLE:
            # Start main optimization run
            self.mlflow_tracker.start_run(
                run_name=f"{model_type}_optimization",
                tags={
                    "optimization": "true",
                    "model_type": model_type,
                    "n_trials": str(n_trials)
                }
            )

            if OPTUNA_MLFLOW_AVAILABLE:
                mlflow_callback = MLflowCallback(
                    tracking_uri=self.mlflow_tracker.tracking_uri,
                    metric_name=optimization_metric
                )
                callbacks.append(mlflow_callback)

        try:
            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=True
            )

            # Collect results
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'study_name': study_name,
                'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
            }

            # Log optimization results to MLflow
            if self.mlflow_tracker:
                self.mlflow_tracker.log_hyperparameter_optimization(results, model_type)

                # Log best configuration
                self.mlflow_tracker.log_model_config(study.best_params, f"{model_type}_best")

            print(f"🏆 Optimization complete!")
            print(f"   Best {optimization_metric}: {study.best_value:.4f}")
            print(f"   Best trial: {study.best_trial.number}")
            print(f"   Total trials: {len(study.trials)}")

            return results

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {}

        finally:
            # End main optimization run
            if self.mlflow_tracker and self.mlflow_tracker.current_run_id:
                self.mlflow_tracker.end_run()

    def optimize_ensemble_weights(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        n_trials: int = 200,
        metric: str = "rmse"
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights given base model predictions.

        Args:
            base_predictions: Dictionary of model predictions
            y_true: True target values
            n_trials: Number of optimization trials
            metric: Optimization metric

        Returns:
            Optimal weights
        """
        def objective(trial):
            # Suggest weights for each model
            weights = {}
            for model_name in base_predictions.keys():
                weights[model_name] = trial.suggest_float(f"weight_{model_name}", 0.0, 1.0)

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                # Equal weights if all are zero
                n_models = len(base_predictions)
                weights = {k: 1.0/n_models for k in base_predictions.keys()}

            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(y_true, dtype=float)
            for model_name, pred in base_predictions.items():
                ensemble_pred += weights[model_name] * pred

            # Calculate objective
            if metric == "rmse":
                return np.sqrt(mean_squared_error(y_true, ensemble_pred))
            elif metric == "mae":
                return mean_absolute_error(y_true, ensemble_pred)
            else:
                return np.sqrt(mean_squared_error(y_true, ensemble_pred))

        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract optimal weights
        optimal_weights = {}
        for model_name in base_predictions.keys():
            optimal_weights[model_name] = study.best_params[f"weight_{model_name}"]

        # Normalize final weights
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}

        print(f"🎯 Optimal ensemble weights:")
        for model, weight in optimal_weights.items():
            print(f"   {model}: {weight:.4f}")
        print(f"   Best {metric}: {study.best_value:.4f}")

        return optimal_weights

    def multi_objective_optimization(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Multi-objective optimization (accuracy vs speed).

        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials
            feature_names: Feature names

        Returns:
            Pareto-optimal solutions
        """
        def objective(trial):
            # Get model configuration
            if model_type.lower() == 'transformer':
                config = OptimizationObjective(model_type, X_train, y_train, X_val, y_val)._suggest_transformer_params(trial)
            elif model_type.lower() == 'lstm':
                config = OptimizationObjective(model_type, X_train, y_train, X_val, y_val)._suggest_lstm_params(trial)
            elif model_type.lower() == 'lightgbm':
                config = OptimizationObjective(model_type, X_train, y_train, X_val, y_val)._suggest_lightgbm_params(trial)
            else:
                raise ValueError(f"Multi-objective optimization not implemented for {model_type}")

            # Train and evaluate
            start_time = datetime.now()

            try:
                objective_func = OptimizationObjective(
                    model_type, X_train, y_train, X_val, y_val,
                    feature_names=feature_names
                )
                accuracy = objective_func._train_and_evaluate(config, X_train, y_train, X_val, y_val)

                training_time = (datetime.now() - start_time).total_seconds()

                return accuracy, training_time

            except Exception:
                return float('inf'), float('inf')

        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize', 'minimize'],  # Minimize both error and time
            sampler=self.sampler
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract Pareto front
        pareto_solutions = []
        for trial in study.best_trials:
            solution = {
                'trial_number': trial.number,
                'params': trial.params,
                'accuracy': trial.values[0],
                'training_time': trial.values[1]
            }
            pareto_solutions.append(solution)

        print(f"🎯 Found {len(pareto_solutions)} Pareto-optimal solutions")

        return {
            'pareto_solutions': pareto_solutions,
            'study_name': study.study_name,
            'n_trials': len(study.trials)
        }


def run_complete_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]] = None,
    models_to_optimize: List[str] = None,
    n_trials_per_model: int = 100,
    output_dir: str = "optimization_results"
) -> Dict[str, Dict[str, Any]]:
    """
    Run complete hyperparameter optimization for all models.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        feature_names: Feature names
        models_to_optimize: List of models to optimize
        n_trials_per_model: Number of trials per model
        output_dir: Output directory for results

    Returns:
        Optimization results for all models
    """
    if models_to_optimize is None:
        models_to_optimize = ['transformer', 'lstm', 'lightgbm', 'ensemble']

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup MLflow tracking
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        try:
            from .mlflow_integration import setup_mlflow_tracking
            mlflow_tracker = setup_mlflow_tracking(
                experiment_name="crypto_hyperopt"
            )
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")

    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="crypto_complete_optimization",
        mlflow_tracker=mlflow_tracker
    )

    results = {}

    for model_type in models_to_optimize:
        print(f"\\n{'='*60}")
        print(f"🎯 Optimizing {model_type.upper()} model")
        print(f"{'='*60}")

        try:
            model_results = optimizer.optimize_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=n_trials_per_model,
                feature_names=feature_names,
                optimization_metric="rmse",
                cv_folds=3
            )

            results[model_type] = model_results

            # Save individual results
            result_file = output_path / f"{model_type}_optimization_results.json"
            with open(result_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            print(f"✅ {model_type} optimization complete")

        except Exception as e:
            print(f"❌ {model_type} optimization failed: {e}")
            results[model_type] = {"error": str(e)}

    # Save combined results
    combined_file = output_path / "complete_optimization_results.json"
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\n🎉 Complete optimization finished!")
    print(f"📁 Results saved to {output_path}")

    return results