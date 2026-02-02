"""
Advanced MLflow Experiment Management System

Professional-grade experiment tracking with:
- Semantic versioning and automated changelog generation
- Comprehensive metrics logging with real-time streaming
- Artifact management with automated organization
- Experiment lineage and dependency tracking
- Custom visualizations and reporting
"""

from __future__ import annotations

import os
import json
import uuid
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import threading
import time
from contextlib import contextmanager
import warnings

try:
    import mlflow
    import mlflow.tracking
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Experiment, Run
    import mlflow.sklearn
    import mlflow.tensorflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version information with semantic versioning"""
    major: int = 1
    minor: int = 0
    patch: int = 0
    build: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f".{self.build}"
        return version

    def increment_patch(self) -> 'ModelVersion':
        """Increment patch version"""
        return ModelVersion(self.major, self.minor, self.patch + 1, self.build)

    def increment_minor(self) -> 'ModelVersion':
        """Increment minor version"""
        return ModelVersion(self.major, self.minor + 1, 0, self.build)

    def increment_major(self) -> 'ModelVersion':
        """Increment major version"""
        return ModelVersion(self.major + 1, 0, 0, self.build)


@dataclass
class ExperimentConfig:
    """Advanced experiment configuration"""
    experiment_name: str
    model_type: str
    version: ModelVersion
    description: str = ""
    tags: Dict[str, str] = None
    parent_run_id: Optional[str] = None
    tracking_uri: str = "file://./mlruns"
    auto_log: bool = True
    log_frequency: int = 10  # Log metrics every N steps
    artifact_location: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricsStreamer:
    """Real-time metrics streaming for live monitoring"""

    def __init__(self, run_id: str, client: MlflowClient):
        self.run_id = run_id
        self.client = client
        self.metrics_buffer = {}
        self.streaming = False
        self._lock = threading.Lock()
        self._thread = None

    def start_streaming(self, interval: float = 1.0):
        """Start real-time metrics streaming"""
        if self.streaming:
            return

        self.streaming = True
        self._thread = threading.Thread(
            target=self._stream_worker,
            args=(interval,),
            daemon=True
        )
        self._thread.start()
        logger.info(f"Started metrics streaming for run {self.run_id}")

    def stop_streaming(self):
        """Stop metrics streaming"""
        self.streaming = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Stopped metrics streaming")

    def add_metric(self, key: str, value: float, step: Optional[int] = None):
        """Add metric to streaming buffer"""
        with self._lock:
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []

            timestamp = int(time.time() * 1000)
            self.metrics_buffer[key].append({
                'value': value,
                'timestamp': timestamp,
                'step': step or len(self.metrics_buffer[key])
            })

    def _stream_worker(self, interval: float):
        """Worker thread for streaming metrics"""
        while self.streaming:
            try:
                self._flush_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in metrics streaming: {e}")

    def _flush_metrics(self):
        """Flush buffered metrics to MLflow"""
        with self._lock:
            for key, metrics in self.metrics_buffer.items():
                if metrics:
                    # Log the latest metric
                    latest = metrics[-1]
                    try:
                        self.client.log_metric(
                            self.run_id,
                            key,
                            latest['value'],
                            latest['timestamp'],
                            latest['step']
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log metric {key}: {e}")

            # Keep only recent metrics to prevent memory bloat
            for key in self.metrics_buffer:
                if len(self.metrics_buffer[key]) > 100:
                    self.metrics_buffer[key] = self.metrics_buffer[key][-50:]


class AdvancedExperimentManager:
    """
    Professional MLflow experiment manager with advanced features
    """

    def __init__(self, config: ExperimentConfig):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required but not installed")

        self.config = config
        self.client = MlflowClient(tracking_uri=config.tracking_uri)
        self.current_run = None
        self.metrics_streamer = None
        self.experiment_id = None

        # Setup MLflow tracking
        mlflow.set_tracking_uri(config.tracking_uri)

        # Create or get experiment
        self._setup_experiment()

        # Setup auto-logging if enabled
        if config.auto_log:
            self._setup_auto_logging()

    def _setup_experiment(self):
        """Setup or create MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self.experiment_id = self.client.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags={
                        "model_type": self.config.model_type,
                        "version": str(self.config.version),
                        "created_by": "AdvancedExperimentManager",
                        "creation_date": datetime.datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new experiment: {self.config.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.config.experiment_name}")

        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise

    def _setup_auto_logging(self):
        """Setup auto-logging for different frameworks"""
        try:
            mlflow.tensorflow.autolog(log_models=True, log_datasets=True)
            mlflow.sklearn.autolog(log_models=True, log_datasets=True)
            logger.info("Auto-logging enabled for TensorFlow and Scikit-learn")
        except Exception as e:
            logger.warning(f"Auto-logging setup failed: {e}")

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Context manager for MLflow runs with advanced tracking"""

        if run_name is None:
            run_name = f"{self.config.model_type}_v{self.config.version}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare run tags
        run_tags = {
            **self.config.tags,
            "model_type": self.config.model_type,
            "version": str(self.config.version),
            "run_id": str(uuid.uuid4()),
            "start_time": datetime.datetime.now().isoformat(),
            "framework": "AdvancedExperimentManager"
        }

        if self.config.parent_run_id:
            run_tags["parent_run_id"] = self.config.parent_run_id

        try:
            # Start MLflow run
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
                tags=run_tags
            ) as run:
                self.current_run = run

                # Setup metrics streaming
                self.metrics_streamer = MetricsStreamer(run.info.run_id, self.client)
                self.metrics_streamer.start_streaming()

                # Log experiment configuration
                self._log_experiment_config()

                logger.info(f"Started run: {run_name} (ID: {run.info.run_id})")

                yield self

        except Exception as e:
            logger.error(f"Error in MLflow run: {e}")
            raise

        finally:
            # Cleanup
            if self.metrics_streamer:
                self.metrics_streamer.stop_streaming()

            # Log run completion
            if self.current_run:
                self.log_metric("run_duration",
                               (datetime.datetime.now() -
                                datetime.datetime.fromisoformat(
                                    self.current_run.info.start_time
                                )).total_seconds())

            self.current_run = None
            logger.info(f"Completed run: {run_name}")

    def _log_experiment_config(self):
        """Log experiment configuration"""
        try:
            config_dict = asdict(self.config)
            # Convert non-serializable objects
            config_dict['version'] = str(self.config.version)

            mlflow.log_params(config_dict)

            # Save detailed config as artifact
            config_path = "experiment_config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            mlflow.log_artifact(config_path, "config")
            os.unlink(config_path)  # Cleanup

        except Exception as e:
            logger.warning(f"Failed to log experiment config: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log metric with streaming support"""
        try:
            mlflow.log_metric(key, value, step)

            # Also add to streaming buffer
            if self.metrics_streamer:
                self.metrics_streamer.add_metric(key, value, step)

        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")

    def log_metrics_batch(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics efficiently"""
        try:
            mlflow.log_metrics(metrics, step)

            # Add to streaming buffer
            if self.metrics_streamer:
                for key, value in metrics.items():
                    self.metrics_streamer.add_metric(key, value, step)

        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")

    def log_param(self, key: str, value: Any):
        """Log parameter"""
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log param {key}: {e}")

    def log_params_batch(self, params: Dict[str, Any]):
        """Log multiple parameters"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params batch: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact with organized structure"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def log_model_artifacts(self, model: Any, model_type: str,
                           model_signature=None, input_example=None):
        """Log model with comprehensive artifacts"""
        try:
            artifact_path = f"models/{model_type}_v{self.config.version}"

            if model_type.lower() in ['tensorflow', 'tf', 'keras']:
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=artifact_path,
                    signature=model_signature,
                    input_example=input_example
                )
            elif model_type.lower() in ['sklearn', 'scikit-learn']:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=model_signature,
                    input_example=input_example
                )
            else:
                # Generic model logging
                mlflow.log_artifact(str(model), artifact_path)

            logger.info(f"Logged {model_type} model artifacts")

        except Exception as e:
            logger.error(f"Failed to log model artifacts: {e}")

    def log_training_curves(self, history: Dict[str, List[float]],
                           title: str = "Training Curves"):
        """Log interactive training curves"""
        try:
            # Create plotly figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Loss', 'Accuracy', 'Learning Rate', 'Custom Metrics'],
                vertical_spacing=0.1
            )

            # Add training curves
            for metric, values in history.items():
                if 'loss' in metric.lower():
                    row, col = 1, 1
                elif 'acc' in metric.lower():
                    row, col = 1, 2
                elif 'lr' in metric.lower() or 'learning' in metric.lower():
                    row, col = 2, 1
                else:
                    row, col = 2, 2

                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode='lines',
                        name=metric,
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                title=title,
                height=800,
                showlegend=True
            )

            # Save and log
            html_file = "training_curves.html"
            fig.write_html(html_file)
            self.log_artifact(html_file, "visualizations")
            os.unlink(html_file)

            logger.info("Logged interactive training curves")

        except Exception as e:
            logger.warning(f"Failed to log training curves: {e}")

    def log_feature_importance(self, feature_names: List[str],
                              importance_values: List[float],
                              title: str = "Feature Importance"):
        """Log feature importance visualization"""
        try:
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=True).tail(20)

            # Create plotly figure
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=title,
                color='importance',
                color_continuous_scale='viridis'
            )

            fig.update_layout(
                height=600,
                xaxis_title="Importance Score",
                yaxis_title="Features"
            )

            # Save and log
            html_file = "feature_importance.html"
            fig.write_html(html_file)
            self.log_artifact(html_file, "visualizations")

            # Also log as CSV
            csv_file = "feature_importance.csv"
            importance_df.to_csv(csv_file, index=False)
            self.log_artifact(csv_file, "data")

            # Cleanup
            os.unlink(html_file)
            os.unlink(csv_file)

            logger.info("Logged feature importance visualization")

        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None):
        """Log confusion matrix heatmap"""
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm))
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save and log
            plot_file = "confusion_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.log_artifact(plot_file, "visualizations")
            plt.close()
            os.unlink(plot_file)

            logger.info("Logged confusion matrix")

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def log_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Prediction vs Actual Distribution"):
        """Log prediction distribution plots"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Scatter Plot', 'Residuals', 'Distribution Comparison']
            )

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    opacity=0.6
                ),
                row=1, col=1
            )

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=1
            )

            # Residuals
            residuals = y_true - y_pred
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    opacity=0.6
                ),
                row=1, col=2
            )

            # Zero line for residuals
            fig.add_trace(
                go.Scatter(
                    x=[y_pred.min(), y_pred.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=2
            )

            # Distribution comparison
            fig.add_trace(
                go.Histogram(
                    x=y_true,
                    name='Actual',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=3
            )

            fig.add_trace(
                go.Histogram(
                    x=y_pred,
                    name='Predicted',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=3
            )

            fig.update_layout(
                title=title,
                height=500,
                showlegend=True
            )

            # Save and log
            html_file = "prediction_analysis.html"
            fig.write_html(html_file)
            self.log_artifact(html_file, "visualizations")
            os.unlink(html_file)

            logger.info("Logged prediction distribution analysis")

        except Exception as e:
            logger.warning(f"Failed to log prediction distribution: {e}")

    def create_model_card(self, model_info: Dict[str, Any]) -> str:
        """Create comprehensive model card documentation"""
        try:
            timestamp = datetime.datetime.now().isoformat()

            model_card = f"""
# Model Card: {model_info.get('name', 'Unknown Model')}

## Model Information
- **Version**: {self.config.version}
- **Type**: {self.config.model_type}
- **Created**: {timestamp}
- **Run ID**: {self.current_run.info.run_id if self.current_run else 'N/A'}

## Model Description
{model_info.get('description', self.config.description)}

## Performance Metrics
"""

            # Add performance metrics
            if 'metrics' in model_info:
                for metric, value in model_info['metrics'].items():
                    model_card += f"- **{metric}**: {value:.4f}\n"

            model_card += f"""
## Training Configuration
- **Dataset**: {model_info.get('dataset', 'Unknown')}
- **Training Duration**: {model_info.get('training_time', 'Unknown')}
- **Hyperparameters**: See experiment artifacts

## Model Architecture
{model_info.get('architecture', 'Architecture details not provided')}

## Usage
```python
# Load model from MLflow
import mlflow
model = mlflow.pyfunc.load_model(f"runs:/{self.current_run.info.run_id if self.current_run else 'RUN_ID'}/model")

# Make predictions
predictions = model.predict(input_data)
```

## Limitations and Considerations
{model_info.get('limitations', 'No limitations documented')}

## Contact Information
- **Experiment**: {self.config.experiment_name}
- **Framework**: Advanced MLflow Integration
"""

            # Save model card
            card_file = "model_card.md"
            with open(card_file, 'w') as f:
                f.write(model_card)

            self.log_artifact(card_file, "documentation")
            os.unlink(card_file)

            logger.info("Created and logged model card")
            return model_card

        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
            return ""

    def get_experiment_lineage(self) -> Dict[str, Any]:
        """Get experiment lineage and dependencies"""
        try:
            if not self.current_run:
                return {}

            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=1000
            )

            lineage = {
                'current_run': self.current_run.info.run_id,
                'experiment_id': self.experiment_id,
                'parent_runs': [],
                'child_runs': [],
                'related_runs': []
            }

            current_tags = self.current_run.data.tags
            parent_id = current_tags.get('parent_run_id')

            for run in runs:
                if run.info.run_id == self.current_run.info.run_id:
                    continue

                run_tags = run.data.tags

                # Check for parent-child relationships
                if run.info.run_id == parent_id:
                    lineage['parent_runs'].append({
                        'run_id': run.info.run_id,
                        'name': run.data.tags.get('mlflow.runName', 'Unknown'),
                        'model_type': run_tags.get('model_type', 'Unknown')
                    })
                elif run_tags.get('parent_run_id') == self.current_run.info.run_id:
                    lineage['child_runs'].append({
                        'run_id': run.info.run_id,
                        'name': run.data.tags.get('mlflow.runName', 'Unknown'),
                        'model_type': run_tags.get('model_type', 'Unknown')
                    })
                elif run_tags.get('model_type') == current_tags.get('model_type'):
                    lineage['related_runs'].append({
                        'run_id': run.info.run_id,
                        'name': run.data.tags.get('mlflow.runName', 'Unknown'),
                        'version': run_tags.get('version', 'Unknown')
                    })

            return lineage

        except Exception as e:
            logger.warning(f"Failed to get experiment lineage: {e}")
            return {}

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs with comprehensive metrics"""
        try:
            comparison_data = []

            for run_id in run_ids:
                run = self.client.get_run(run_id)

                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'model_type': run.data.tags.get('model_type', 'Unknown'),
                    'version': run.data.tags.get('version', 'Unknown'),
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }

                # Add all metrics
                for key, value in run.data.metrics.items():
                    run_data[f"metric_{key}"] = value

                # Add key parameters
                for key, value in run.data.params.items():
                    run_data[f"param_{key}"] = value

                comparison_data.append(run_data)

            comparison_df = pd.DataFrame(comparison_data)

            # Save comparison
            csv_file = "run_comparison.csv"
            comparison_df.to_csv(csv_file, index=False)
            self.log_artifact(csv_file, "analysis")
            os.unlink(csv_file)

            logger.info(f"Compared {len(run_ids)} runs")
            return comparison_df

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()


def create_experiment_manager(
    experiment_name: str,
    model_type: str,
    version: Union[str, ModelVersion] = "1.0.0",
    description: str = "",
    **kwargs
) -> AdvancedExperimentManager:
    """
    Create an advanced experiment manager instance.

    Args:
        experiment_name: Name of the MLflow experiment
        model_type: Type of model being trained
        version: Model version (string or ModelVersion object)
        description: Experiment description
        **kwargs: Additional configuration options

    Returns:
        AdvancedExperimentManager instance
    """
    if isinstance(version, str):
        parts = version.split('.')
        version = ModelVersion(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            build=parts[3] if len(parts) > 3 else None
        )

    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_type=model_type,
        version=version,
        description=description,
        **kwargs
    )

    return AdvancedExperimentManager(config)


if __name__ == "__main__":
    # Example usage
    manager = create_experiment_manager(
        experiment_name="crypto_prediction_advanced",
        model_type="transformer",
        version="1.0.0",
        description="Advanced cryptocurrency prediction with enhanced tracking"
    )

    # Example training loop with comprehensive logging
    with manager.start_run("example_training"):
        # Log hyperparameters
        manager.log_params_batch({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "model_architecture": "transformer"
        })

        # Simulate training loop
        for epoch in range(5):
            # Simulate metrics
            train_loss = 1.0 / (epoch + 1)
            val_loss = 1.2 / (epoch + 1)

            manager.log_metrics_batch({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            }, step=epoch)

        # Log model artifacts and visualizations
        print("Advanced MLflow experiment tracking demo completed!")