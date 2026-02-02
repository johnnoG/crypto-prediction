"""
MLflow Integration for Cryptocurrency Prediction Models

Comprehensive experiment tracking and model management:
- Experiment tracking for all model types
- Automated hyperparameter logging
- Model registry with versioning
- Performance comparison dashboards
- Artifact management (models, plots, datasets)
- Model deployment utilities
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pickle
import tempfile
import shutil

try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.sklearn
    import mlflow.lightgbm
    from mlflow.models.signature import infer_signature
    from mlflow.types.schema import Schema, ColSpec
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class MLflowExperimentTracker:
    """
    Comprehensive MLflow experiment tracking for cryptocurrency models.

    Features:
    - Automatic experiment setup
    - Model-specific parameter logging
    - Performance metrics tracking
    - Artifact management
    - Model registry integration
    """

    def __init__(
        self,
        experiment_name: str = "crypto_prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed")

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlflow_tracking"
        self.artifact_location = artifact_location

        # Setup MLflow
        self._setup_mlflow()

        # Experiment and run tracking
        self.current_run_id: Optional[str] = None
        self.experiment_id: Optional[str] = None

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking and experiment"""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                if self.artifact_location:
                    self.experiment_id = mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=self.artifact_location
                    )
                else:
                    self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.experiment_name)
            print(f"✅ MLflow experiment '{self.experiment_name}' ready (ID: {self.experiment_id})")

        except Exception as e:
            print(f"❌ MLflow setup failed: {e}")
            raise

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"crypto_model_{timestamp}"

        # Default tags
        default_tags = {
            "model_type": "cryptocurrency_prediction",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }

        if tags:
            default_tags.update(tags)

        # Start run
        run = mlflow.start_run(
            run_name=run_name,
            tags=default_tags,
            nested=nested
        )

        self.current_run_id = run.info.run_id
        print(f"🚀 Started MLflow run: {run_name} (ID: {self.current_run_id})")

        return self.current_run_id

    def end_run(self) -> None:
        """End current MLflow run"""
        if self.current_run_id:
            mlflow.end_run()
            print(f"✅ Ended MLflow run: {self.current_run_id}")
            self.current_run_id = None

    def log_model_config(self, config: Dict[str, Any], model_type: str) -> None:
        """
        Log model configuration parameters.

        Args:
            config: Model configuration dictionary
            model_type: Type of model (transformer, lstm, lightgbm, ensemble)
        """
        try:
            # Flatten nested config for MLflow
            flat_config = self._flatten_dict(config, prefix=f"{model_type}_")

            # Log parameters
            for key, value in flat_config.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)

            print(f"📊 Logged {len(flat_config)} parameters for {model_type}")

        except Exception as e:
            print(f"⚠️ Failed to log config: {e}")

    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        model_type: str = ""
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch
            model_type: Model type prefix
        """
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_key = f"{model_type}_{metric_name}" if model_type else metric_name
                    mlflow.log_metric(metric_key, value, step=step)

            print(f"📈 Logged {len(metrics)} metrics")

        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")

    def log_training_history(
        self,
        history: Dict[str, List[float]],
        model_type: str = ""
    ) -> None:
        """
        Log complete training history.

        Args:
            history: Training history dictionary
            model_type: Model type prefix
        """
        try:
            for metric_name, values in history.items():
                for epoch, value in enumerate(values):
                    metric_key = f"{model_type}_{metric_name}" if model_type else metric_name
                    mlflow.log_metric(metric_key, float(value), step=epoch)

            print(f"📊 Logged training history for {len(history)} metrics")

        except Exception as e:
            print(f"⚠️ Failed to log training history: {e}")

    def log_dataset_info(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Log dataset information.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
        """
        try:
            # Dataset shape information
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            if X_val is not None:
                mlflow.log_param("val_samples", X_val.shape[0])

            # Feature statistics
            train_stats = {
                "feature_mean": float(np.mean(X_train)),
                "feature_std": float(np.std(X_train)),
                "feature_min": float(np.min(X_train)),
                "feature_max": float(np.max(X_train))
            }

            target_stats = {
                "target_mean": float(np.mean(y_train)),
                "target_std": float(np.std(y_train)),
                "target_min": float(np.min(y_train)),
                "target_max": float(np.max(y_train))
            }

            for key, value in {**train_stats, **target_stats}.items():
                mlflow.log_metric(key, value)

            # Log feature names if provided
            if feature_names:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(feature_names, f)
                    mlflow.log_artifact(f.name, "features")
                os.unlink(f.name)

            print("📊 Logged dataset information")

        except Exception as e:
            print(f"⚠️ Failed to log dataset info: {e}")

    def log_model_tensorflow(
        self,
        model,
        model_type: str = "tensorflow",
        signature = None,
        input_example = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Log TensorFlow model to MLflow.

        Args:
            model: TensorFlow/Keras model
            model_type: Model type
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry

        Returns:
            Model URI
        """
        try:
            model_info = mlflow.tensorflow.log_model(
                tf_saved_model_dir=None,
                tf_meta_graph_tags=None,
                tf_signature_def_key=None,
                model=model,
                artifact_path=model_type,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )

            print(f"🤖 Logged {model_type} model to MLflow")
            return model_info.model_uri

        except Exception as e:
            print(f"⚠️ Failed to log TensorFlow model: {e}")
            return ""

    def log_model_sklearn(
        self,
        model,
        model_type: str = "sklearn",
        signature = None,
        input_example = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Log scikit-learn model to MLflow.

        Args:
            model: Scikit-learn model
            model_type: Model type
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry

        Returns:
            Model URI
        """
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_type,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )

            print(f"🤖 Logged {model_type} model to MLflow")
            return model_info.model_uri

        except Exception as e:
            print(f"⚠️ Failed to log sklearn model: {e}")
            return ""

    def log_model_lightgbm(
        self,
        model,
        signature = None,
        input_example = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Log LightGBM model to MLflow.

        Args:
            model: LightGBM model
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry

        Returns:
            Model URI
        """
        try:
            model_info = mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="lightgbm",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )

            print(f"🤖 Logged LightGBM model to MLflow")
            return model_info.model_uri

        except Exception as e:
            print(f"⚠️ Failed to log LightGBM model: {e}")
            return ""

    def log_ensemble_model(
        self,
        ensemble_dir: Path,
        model_type: str = "ensemble",
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Log ensemble model artifacts.

        Args:
            ensemble_dir: Path to ensemble directory
            model_type: Model type
            registered_model_name: Name for model registry

        Returns:
            Artifact URI
        """
        try:
            # Log entire ensemble directory
            mlflow.log_artifacts(str(ensemble_dir), artifact_path=model_type)

            print(f"🤖 Logged ensemble model artifacts")

            # If registering, create a custom model
            if registered_model_name:
                return self._register_ensemble_model(
                    ensemble_dir, registered_model_name
                )

            return f"runs:/{self.current_run_id}/{model_type}"

        except Exception as e:
            print(f"⚠️ Failed to log ensemble model: {e}")
            return ""

    def log_prediction_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str = "",
        confidence_intervals: Optional[np.ndarray] = None
    ) -> None:
        """
        Log prediction visualization plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Model type for naming
            confidence_intervals: Confidence intervals (optional)
        """
        if not PLOTTING_AVAILABLE:
            return

        try:
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Actual vs Predicted scatter
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Time series plot (if sequential)
            axes[1, 0].plot(y_true, label='Actual', linewidth=2)
            axes[1, 0].plot(y_pred, label='Predicted', linewidth=2)

            if confidence_intervals is not None:
                axes[1, 0].fill_between(
                    range(len(y_pred)),
                    confidence_intervals[:, 0],
                    confidence_intervals[:, 1],
                    alpha=0.3, label='Confidence Interval'
                )

            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Time Series Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Error distribution
            axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Residual Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'{model_type} Model Predictions', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save and log plot
            plot_path = f"{model_type}_predictions.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path, "plots")
            plt.close()

            # Clean up
            if os.path.exists(plot_path):
                os.unlink(plot_path)

            print(f"📊 Logged prediction plots for {model_type}")

        except Exception as e:
            print(f"⚠️ Failed to log plots: {e}")

    def log_feature_importance(
        self,
        feature_importance: Dict[str, float],
        model_type: str = ""
    ) -> None:
        """
        Log and plot feature importance.

        Args:
            feature_importance: Feature importance dictionary
            model_type: Model type for naming
        """
        if not PLOTTING_AVAILABLE:
            return

        try:
            # Log as metrics
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)

            # Create importance plot
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())

            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            top_features = [features[i] for i in sorted_idx[:20]]  # Top 20
            top_importances = [importances[i] for i in sorted_idx[:20]]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importances)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Importance')
            plt.title(f'{model_type} Feature Importance (Top 20)')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Save and log plot
            plot_path = f"{model_type}_feature_importance.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path, "plots")
            plt.close()

            # Clean up
            if os.path.exists(plot_path):
                os.unlink(plot_path)

            print(f"📊 Logged feature importance for {model_type}")

        except Exception as e:
            print(f"⚠️ Failed to log feature importance: {e}")

    def log_hyperparameter_optimization(
        self,
        study_results: Dict[str, Any],
        model_type: str = ""
    ) -> None:
        """
        Log hyperparameter optimization results.

        Args:
            study_results: Optuna study results
            model_type: Model type
        """
        try:
            # Log best parameters
            if 'best_params' in study_results:
                for param, value in study_results['best_params'].items():
                    mlflow.log_param(f"best_{param}", value)

            # Log optimization metrics
            if 'best_value' in study_results:
                mlflow.log_metric(f"{model_type}_best_objective", study_results['best_value'])

            if 'n_trials' in study_results:
                mlflow.log_param(f"{model_type}_n_trials", study_results['n_trials'])

            # Log optimization history if available
            if 'optimization_history' in study_results:
                for trial, value in enumerate(study_results['optimization_history']):
                    mlflow.log_metric(f"{model_type}_trial_objective", value, step=trial)

            print(f"🎯 Logged hyperparameter optimization results for {model_type}")

        except Exception as e:
            print(f"⚠️ Failed to log hyperparameter optimization: {e}")

    def compare_models(
        self,
        run_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model runs.

        Args:
            run_ids: List of MLflow run IDs
            metrics: List of metrics to compare

        Returns:
            Comparison DataFrame
        """
        try:
            if metrics is None:
                metrics = ['rmse', 'mae', 'mape']

            client = mlflow.tracking.MlflowClient()
            comparison_data = []

            for run_id in run_ids:
                run = client.get_run(run_id)

                row = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'model_type': run.data.tags.get('model_type', 'Unknown'),
                    'start_time': run.info.start_time,
                    'duration': run.info.end_time - run.info.start_time if run.info.end_time else None
                }

                # Add metrics
                for metric in metrics:
                    # Try different metric name variations
                    metric_value = None
                    for metric_name in run.data.metrics:
                        if metric in metric_name.lower():
                            metric_value = run.data.metrics[metric_name]
                            break
                    row[metric] = metric_value

                comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)

            print(f"📊 Compared {len(run_ids)} model runs")
            return comparison_df

        except Exception as e:
            print(f"⚠️ Failed to compare models: {e}")
            return pd.DataFrame()

    def register_best_model(
        self,
        experiment_id: str,
        metric_name: str = "val_rmse",
        model_name: str = "crypto_prediction_model",
        ascending: bool = True
    ) -> Optional[str]:
        """
        Register the best model from an experiment.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric to optimize
            model_name: Name for registered model
            ascending: Whether lower is better

        Returns:
            Model version if successful
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # Search for runs in the experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
            )

            if runs.empty:
                print("No runs found in experiment")
                return None

            # Find best run
            if metric_name in runs.columns:
                best_run = runs.loc[runs[metric_name].idxmin() if ascending else runs[metric_name].idxmax()]
            else:
                print(f"Metric '{metric_name}' not found")
                return None

            best_run_id = best_run['run_id']

            # Try to register the model
            try:
                result = mlflow.register_model(
                    model_uri=f"runs:/{best_run_id}/model",
                    name=model_name
                )

                print(f"🏆 Registered best model: {model_name} v{result.version}")
                print(f"   Run ID: {best_run_id}")
                print(f"   {metric_name}: {best_run[metric_name]}")

                return result.version

            except Exception:
                # Model might not have been logged as 'model', try other paths
                artifacts = client.list_artifacts(best_run_id)
                for artifact in artifacts:
                    if 'model' in artifact.path.lower():
                        try:
                            result = mlflow.register_model(
                                model_uri=f"runs:/{best_run_id}/{artifact.path}",
                                name=model_name
                            )
                            print(f"🏆 Registered best model: {model_name} v{result.version}")
                            return result.version
                        except:
                            continue

                print("Could not register model - no model artifacts found")
                return None

        except Exception as e:
            print(f"⚠️ Failed to register best model: {e}")
            return None

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, f"{new_key}_").items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _register_ensemble_model(
        self,
        ensemble_dir: Path,
        model_name: str
    ) -> str:
        """Register ensemble as a custom MLflow model"""
        # This is a simplified implementation
        # In practice, you'd create a custom MLflow model class
        # for proper ensemble inference

        try:
            # For now, just register the ensemble directory
            model_uri = f"runs:/{self.current_run_id}/ensemble"

            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            print(f"🏆 Registered ensemble model: {model_name} v{result.version}")
            return result.version

        except Exception as e:
            print(f"⚠️ Failed to register ensemble model: {e}")
            return ""


class MLflowModelRegistry:
    """
    MLflow Model Registry management utilities.
    """

    def __init__(self):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed")

        self.client = mlflow.tracking.MlflowClient()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        try:
            models = self.client.list_registered_models()

            model_list = []
            for model in models:
                latest_version = self.client.get_latest_versions(
                    model.name, stages=["Production", "Staging", "None"]
                )[0] if model.latest_versions else None

                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'latest_version': latest_version.version if latest_version else None,
                    'current_stage': latest_version.current_stage if latest_version else None,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp
                }
                model_list.append(model_info)

            return model_list

        except Exception as e:
            print(f"⚠️ Failed to list models: {e}")
            return []

    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production",
        archive_existing: bool = True
    ) -> bool:
        """
        Promote model to a stage.

        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage (Staging, Production)
            archive_existing: Whether to archive existing models in the stage

        Returns:
            Success status
        """
        try:
            # Archive existing models in the target stage
            if archive_existing:
                current_models = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                for model in current_models:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model.version,
                        stage="Archived"
                    )

            # Promote new model
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )

            print(f"🚀 Promoted {model_name} v{version} to {stage}")
            return True

        except Exception as e:
            print(f"⚠️ Failed to promote model: {e}")
            return False

    def load_production_model(self, model_name: str):
        """Load production model"""
        try:
            model_uri = f"models:/{model_name}/Production"

            # Try different model flavors
            try:
                return mlflow.tensorflow.load_model(model_uri)
            except:
                pass

            try:
                return mlflow.sklearn.load_model(model_uri)
            except:
                pass

            try:
                return mlflow.lightgbm.load_model(model_uri)
            except:
                pass

            # Generic model loading
            return mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            print(f"⚠️ Failed to load production model: {e}")
            return None


def setup_mlflow_tracking(
    tracking_uri: str = "file:./mlflow_tracking",
    experiment_name: str = "crypto_prediction"
) -> MLflowExperimentTracker:
    """
    Setup MLflow tracking with default configuration.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Name of the experiment

    Returns:
        Configured experiment tracker
    """
    return MLflowExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )