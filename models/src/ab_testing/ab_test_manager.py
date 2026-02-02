"""
Advanced A/B Testing Framework for Model Comparison

Professional A/B testing system featuring:
- Traffic splitting with configurable allocation
- Bayesian A/B testing with early stopping
- Statistical significance testing
- Champion-Challenger model comparison
- Real-time performance monitoring
- Automated model promotion based on test results
"""

from __future__ import annotations

import os
import json
import time
import uuid
import random
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

try:
    from scipy import stats
    from scipy.stats import beta
    import bayesian_testing
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.panel import Panel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class ABTestStatus(Enum):
    """A/B test status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EARLY_STOPPED = "early_stopped"


class ModelVariant(Enum):
    """Model variant enumeration"""
    CONTROL = "control"
    TREATMENT = "treatment"
    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    description: str
    control_model: str
    treatment_model: str
    traffic_split: float = 0.5  # Percentage of traffic to treatment
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05  # 5% minimum improvement
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    early_stopping_confidence: float = 0.99
    success_metric: str = "accuracy"  # Primary metric for comparison
    guardrail_metrics: List[str] = None  # Metrics that should not degrade
    guardrail_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.guardrail_metrics is None:
            self.guardrail_metrics = ["response_time", "error_rate"]
        if self.guardrail_thresholds is None:
            self.guardrail_thresholds = {
                "response_time": 2.0,  # Max 2x degradation
                "error_rate": 1.5      # Max 1.5x degradation
            }


@dataclass
class ABTestMetrics:
    """A/B test metrics tracking"""
    control_samples: int = 0
    treatment_samples: int = 0
    control_successes: float = 0.0
    treatment_successes: float = 0.0
    control_mean: float = 0.0
    treatment_mean: float = 0.0
    control_std: float = 0.0
    treatment_std: float = 0.0
    statistical_power: float = 0.0
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    effect_size: float = 0.0
    is_significant: bool = False


@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    test_name: str
    winner: Optional[ModelVariant]
    confidence: float
    effect_size: float
    metrics: ABTestMetrics
    recommendation: str
    details: Dict[str, Any]


class BayesianABTesting:
    """Bayesian A/B testing implementation with early stopping"""

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def update_beliefs(self, successes: int, failures: int) -> Tuple[float, float]:
        """Update Bayesian beliefs based on observed data"""
        alpha_posterior = self.alpha_prior + successes
        beta_posterior = self.beta_prior + failures
        return alpha_posterior, beta_posterior

    def calculate_probability_of_superiority(
        self,
        control_successes: int,
        control_failures: int,
        treatment_successes: int,
        treatment_failures: int,
        n_samples: int = 100000
    ) -> float:
        """Calculate probability that treatment is better than control"""

        # Update posterior distributions
        control_alpha, control_beta = self.update_beliefs(control_successes, control_failures)
        treatment_alpha, treatment_beta = self.update_beliefs(treatment_successes, treatment_failures)

        # Sample from posterior distributions
        control_samples = np.random.beta(control_alpha, control_beta, n_samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)

        # Calculate probability of treatment being better
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        return prob_treatment_better

    def should_stop_early(
        self,
        control_successes: int,
        control_failures: int,
        treatment_successes: int,
        treatment_failures: int,
        confidence_threshold: float = 0.95
    ) -> Tuple[bool, str]:
        """Determine if test should be stopped early"""

        prob_treatment_better = self.calculate_probability_of_superiority(
            control_successes, control_failures,
            treatment_successes, treatment_failures
        )

        if prob_treatment_better >= confidence_threshold:
            return True, f"Treatment is significantly better (confidence: {prob_treatment_better:.3f})"
        elif prob_treatment_better <= (1 - confidence_threshold):
            return True, f"Control is significantly better (confidence: {1-prob_treatment_better:.3f})"
        else:
            return False, f"Insufficient evidence for early stopping (confidence: {max(prob_treatment_better, 1-prob_treatment_better):.3f})"


class StatisticalTesting:
    """Statistical significance testing utilities"""

    @staticmethod
    def two_sample_ttest(
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """Perform two-sample t-test"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for statistical testing")

        statistic, p_value = stats.ttest_ind(control_data, treatment_data)

        effect_size = (np.mean(treatment_data) - np.mean(control_data)) / np.sqrt(
            (np.var(control_data) + np.var(treatment_data)) / 2
        )

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': p_value < alpha,
            'effect_size': float(effect_size),
            'control_mean': float(np.mean(control_data)),
            'treatment_mean': float(np.mean(treatment_data)),
            'control_std': float(np.std(control_data)),
            'treatment_std': float(np.std(treatment_data))
        }

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for A/B test"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for sample size calculation")

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # Pooled standard deviation
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        pooled_p = (p1 + p2) / 2

        # Sample size calculation
        numerator = (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2

        denominator = (p2 - p1) ** 2

        sample_size = int(np.ceil(numerator / denominator))

        return sample_size


class TrafficSplitter:
    """Manages traffic allocation between model variants"""

    def __init__(self, split_ratio: float = 0.5, random_seed: Optional[int] = None):
        self.split_ratio = split_ratio
        self.random_generator = random.Random(random_seed)

    def assign_variant(self, user_id: str) -> ModelVariant:
        """Assign user to model variant based on consistent hashing"""
        # Use hash of user_id for consistent assignment
        hash_value = hash(user_id) % 10000
        threshold = int(self.split_ratio * 10000)

        if hash_value < threshold:
            return ModelVariant.TREATMENT
        else:
            return ModelVariant.CONTROL

    def update_split_ratio(self, new_ratio: float):
        """Update traffic split ratio"""
        if 0 <= new_ratio <= 1:
            self.split_ratio = new_ratio
            logger.info(f"Updated traffic split ratio to {new_ratio:.2%}")
        else:
            raise ValueError("Split ratio must be between 0 and 1")


class MetricsCollector:
    """Collects and aggregates A/B test metrics"""

    def __init__(self):
        self.metrics_data = {
            ModelVariant.CONTROL: [],
            ModelVariant.TREATMENT: []
        }
        self.performance_data = {
            ModelVariant.CONTROL: {},
            ModelVariant.TREATMENT: {}
        }
        self._lock = threading.Lock()

    def record_prediction(
        self,
        variant: ModelVariant,
        prediction: Any,
        actual: Any,
        response_time: float,
        user_id: str,
        additional_metrics: Dict[str, Any] = None
    ):
        """Record a prediction and its outcome"""
        with self._lock:
            record = {
                'timestamp': datetime.now(),
                'user_id': user_id,
                'prediction': prediction,
                'actual': actual,
                'response_time': response_time,
                'variant': variant,
                **(additional_metrics or {})
            }

            self.metrics_data[variant].append(record)

    def calculate_metrics(self, variant: ModelVariant) -> Dict[str, float]:
        """Calculate aggregated metrics for a variant"""
        with self._lock:
            data = self.metrics_data[variant]

            if not data:
                return {}

            df = pd.DataFrame(data)

            metrics = {
                'sample_count': len(df),
                'avg_response_time': df['response_time'].mean(),
                'p95_response_time': df['response_time'].quantile(0.95),
                'error_rate': (df['prediction'].isna().sum() / len(df)) if len(df) > 0 else 0.0
            }

            # Calculate accuracy if both prediction and actual are available
            valid_predictions = df.dropna(subset=['prediction', 'actual'])
            if len(valid_predictions) > 0:
                if isinstance(valid_predictions.iloc[0]['actual'], (int, float)):
                    # Regression metrics
                    mse = np.mean((valid_predictions['prediction'] - valid_predictions['actual']) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(valid_predictions['prediction'] - valid_predictions['actual']))

                    metrics.update({
                        'rmse': rmse,
                        'mae': mae,
                        'mse': mse
                    })
                else:
                    # Classification metrics
                    accuracy = (valid_predictions['prediction'] == valid_predictions['actual']).mean()
                    metrics['accuracy'] = accuracy

            return metrics

    def get_comparison_data(self) -> Dict[str, Dict[str, float]]:
        """Get comparison data for both variants"""
        return {
            'control': self.calculate_metrics(ModelVariant.CONTROL),
            'treatment': self.calculate_metrics(ModelVariant.TREATMENT)
        }

    def export_raw_data(self) -> pd.DataFrame:
        """Export raw data for analysis"""
        all_data = []
        for variant, data in self.metrics_data.items():
            for record in data:
                record_copy = record.copy()
                record_copy['variant'] = variant.value
                all_data.append(record_copy)

        return pd.DataFrame(all_data)


class ABTestManager:
    """Main A/B testing manager orchestrating all testing operations"""

    def __init__(self, mlflow_tracking_uri: str = "file://./mlruns"):
        self.tracking_uri = mlflow_tracking_uri
        self.client = MlflowClient(tracking_uri) if MLFLOW_AVAILABLE else None
        self.active_tests = {}
        self.test_history = {}
        self.bayesian_tester = BayesianABTesting()

    def create_ab_test(
        self,
        config: ABTestConfig,
        start_immediately: bool = False
    ) -> str:
        """Create a new A/B test"""
        test_id = f"abtest_{uuid.uuid4().hex[:8]}"

        # Calculate required sample size
        required_sample_size = StatisticalTesting.calculate_sample_size(
            baseline_rate=0.5,  # Assume 50% baseline for effect size calculation
            minimum_detectable_effect=config.minimum_effect_size,
            alpha=1 - config.confidence_level,
            power=0.8
        )

        ab_test = {
            'test_id': test_id,
            'config': config,
            'status': ABTestStatus.DRAFT,
            'created_at': datetime.now(),
            'started_at': None,
            'ended_at': None,
            'traffic_splitter': TrafficSplitter(config.traffic_split),
            'metrics_collector': MetricsCollector(),
            'required_sample_size': required_sample_size,
            'current_winner': None,
            'statistical_power': 0.0
        }

        self.active_tests[test_id] = ab_test

        logger.info(f"Created A/B test: {test_id} ({config.test_name})")
        logger.info(f"Required sample size: {required_sample_size}")

        if start_immediately:
            self.start_test(test_id)

        return test_id

    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        if test_id not in self.active_tests:
            logger.error(f"Test not found: {test_id}")
            return False

        ab_test = self.active_tests[test_id]

        if ab_test['status'] != ABTestStatus.DRAFT:
            logger.error(f"Test cannot be started in {ab_test['status']} state")
            return False

        ab_test['status'] = ABTestStatus.RUNNING
        ab_test['started_at'] = datetime.now()

        # Log to MLflow if available
        if self.client:
            self._log_test_to_mlflow(test_id, ab_test)

        logger.info(f"Started A/B test: {test_id}")
        return True

    def process_request(
        self,
        test_id: str,
        user_id: str,
        prediction: Any,
        actual: Optional[Any] = None,
        response_time: float = 0.0,
        additional_metrics: Dict[str, Any] = None
    ) -> ModelVariant:
        """Process a request through the A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        if ab_test['status'] != ABTestStatus.RUNNING:
            raise ValueError(f"Test is not running: {ab_test['status']}")

        # Assign variant
        variant = ab_test['traffic_splitter'].assign_variant(user_id)

        # Record metrics
        ab_test['metrics_collector'].record_prediction(
            variant=variant,
            prediction=prediction,
            actual=actual,
            response_time=response_time,
            user_id=user_id,
            additional_metrics=additional_metrics
        )

        # Check for early stopping
        if ab_test['config'].early_stopping_enabled:
            self._check_early_stopping(test_id)

        return variant

    def _check_early_stopping(self, test_id: str):
        """Check if test should be stopped early"""
        ab_test = self.active_tests[test_id]
        metrics_collector = ab_test['metrics_collector']

        # Get current metrics
        comparison_data = metrics_collector.get_comparison_data()

        if 'control' not in comparison_data or 'treatment' not in comparison_data:
            return

        control_metrics = comparison_data['control']
        treatment_metrics = comparison_data['treatment']

        # Check minimum sample size
        total_samples = control_metrics.get('sample_count', 0) + treatment_metrics.get('sample_count', 0)
        if total_samples < ab_test['config'].minimum_sample_size:
            return

        # Bayesian early stopping check
        success_metric = ab_test['config'].success_metric

        if success_metric in control_metrics and success_metric in treatment_metrics:
            # Convert metrics to successes/failures for Bayesian testing
            control_successes = int(control_metrics[success_metric] * control_metrics['sample_count'])
            control_failures = control_metrics['sample_count'] - control_successes

            treatment_successes = int(treatment_metrics[success_metric] * treatment_metrics['sample_count'])
            treatment_failures = treatment_metrics['sample_count'] - treatment_successes

            should_stop, reason = self.bayesian_tester.should_stop_early(
                control_successes, control_failures,
                treatment_successes, treatment_failures,
                ab_test['config'].early_stopping_confidence
            )

            if should_stop:
                logger.info(f"Early stopping triggered for test {test_id}: {reason}")
                self.stop_test(test_id, reason="early_stopping")

    def stop_test(self, test_id: str, reason: str = "manual") -> ABTestResult:
        """Stop an A/B test and generate results"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        if ab_test['status'] != ABTestStatus.RUNNING:
            raise ValueError(f"Test is not running: {ab_test['status']}")

        ab_test['ended_at'] = datetime.now()

        if reason == "early_stopping":
            ab_test['status'] = ABTestStatus.EARLY_STOPPED
        else:
            ab_test['status'] = ABTestStatus.COMPLETED

        # Generate final results
        results = self._generate_test_results(test_id)

        # Move to history
        self.test_history[test_id] = ab_test
        del self.active_tests[test_id]

        logger.info(f"Stopped A/B test: {test_id} (reason: {reason})")

        return results

    def _generate_test_results(self, test_id: str) -> ABTestResult:
        """Generate comprehensive test results"""
        ab_test = self.active_tests[test_id]
        config = ab_test['config']
        metrics_collector = ab_test['metrics_collector']

        # Get comparison data
        comparison_data = metrics_collector.get_comparison_data()
        control_metrics = comparison_data.get('control', {})
        treatment_metrics = comparison_data.get('treatment', {})

        # Perform statistical analysis
        success_metric = config.success_metric
        statistical_results = {}

        if success_metric in control_metrics and success_metric in treatment_metrics:
            # Get raw data for statistical testing
            raw_data = metrics_collector.export_raw_data()

            if len(raw_data) > 0:
                control_data = raw_data[raw_data['variant'] == 'control'][success_metric].dropna()
                treatment_data = raw_data[raw_data['variant'] == 'treatment'][success_metric].dropna()

                if len(control_data) > 0 and len(treatment_data) > 0:
                    statistical_results = StatisticalTesting.two_sample_ttest(
                        control_data.values, treatment_data.values
                    )

        # Create metrics summary
        metrics = ABTestMetrics(
            control_samples=control_metrics.get('sample_count', 0),
            treatment_samples=treatment_metrics.get('sample_count', 0),
            control_mean=control_metrics.get(success_metric, 0.0),
            treatment_mean=treatment_metrics.get(success_metric, 0.0),
            effect_size=statistical_results.get('effect_size', 0.0),
            p_value=statistical_results.get('p_value', 1.0),
            is_significant=statistical_results.get('is_significant', False)
        )

        # Determine winner
        winner = None
        confidence = 0.0

        if metrics.is_significant:
            if metrics.treatment_mean > metrics.control_mean:
                winner = ModelVariant.TREATMENT
                confidence = 1 - metrics.p_value
            else:
                winner = ModelVariant.CONTROL
                confidence = 1 - metrics.p_value

        # Generate recommendation
        recommendation = self._generate_recommendation(metrics, config, winner)

        result = ABTestResult(
            test_id=test_id,
            test_name=config.test_name,
            winner=winner,
            confidence=confidence,
            effect_size=metrics.effect_size,
            metrics=metrics,
            recommendation=recommendation,
            details={
                'control_metrics': control_metrics,
                'treatment_metrics': treatment_metrics,
                'statistical_results': statistical_results,
                'test_duration': (ab_test['ended_at'] - ab_test['started_at']).total_seconds() if ab_test['ended_at'] and ab_test['started_at'] else 0
            }
        )

        return result

    def _generate_recommendation(
        self,
        metrics: ABTestMetrics,
        config: ABTestConfig,
        winner: Optional[ModelVariant]
    ) -> str:
        """Generate actionable recommendation based on test results"""

        if not metrics.is_significant:
            return "No significant difference detected. Consider extending the test or investigating the effect size."

        if winner == ModelVariant.TREATMENT:
            if metrics.effect_size >= config.minimum_effect_size:
                return f"Deploy treatment model. Significant improvement of {metrics.effect_size:.2%} detected."
            else:
                return "Treatment shows statistical significance but effect size is below minimum threshold. Consider business impact."
        else:
            return "Keep control model. Treatment did not show improvement over control."

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current test status"""
        if test_id in self.active_tests:
            ab_test = self.active_tests[test_id]
            metrics_collector = ab_test['metrics_collector']
            comparison_data = metrics_collector.get_comparison_data()

            return {
                'test_id': test_id,
                'test_name': ab_test['config'].test_name,
                'status': ab_test['status'].value,
                'created_at': ab_test['created_at'].isoformat(),
                'started_at': ab_test['started_at'].isoformat() if ab_test['started_at'] else None,
                'required_sample_size': ab_test['required_sample_size'],
                'current_metrics': comparison_data,
                'progress': self._calculate_test_progress(ab_test)
            }
        elif test_id in self.test_history:
            ab_test = self.test_history[test_id]
            return {
                'test_id': test_id,
                'test_name': ab_test['config'].test_name,
                'status': ab_test['status'].value,
                'created_at': ab_test['created_at'].isoformat(),
                'started_at': ab_test['started_at'].isoformat() if ab_test['started_at'] else None,
                'ended_at': ab_test['ended_at'].isoformat() if ab_test['ended_at'] else None
            }
        else:
            return {'error': 'Test not found'}

    def _calculate_test_progress(self, ab_test: Dict[str, Any]) -> Dict[str, float]:
        """Calculate test progress metrics"""
        comparison_data = ab_test['metrics_collector'].get_comparison_data()

        total_samples = sum(
            metrics.get('sample_count', 0)
            for metrics in comparison_data.values()
        )

        sample_progress = min(1.0, total_samples / ab_test['required_sample_size'])

        time_progress = 0.0
        if ab_test['started_at']:
            elapsed_days = (datetime.now() - ab_test['started_at']).days
            time_progress = min(1.0, elapsed_days / ab_test['config'].max_duration_days)

        return {
            'sample_progress': sample_progress,
            'time_progress': time_progress,
            'overall_progress': max(sample_progress, time_progress)
        }

    def create_test_dashboard(self, test_id: str) -> str:
        """Create interactive dashboard for A/B test"""
        try:
            ab_test = self.active_tests.get(test_id) or self.test_history.get(test_id)
            if not ab_test:
                return ""

            # Get data for visualization
            raw_data = ab_test['metrics_collector'].export_raw_data()

            if raw_data.empty:
                return ""

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Sample Size Over Time',
                    'Success Metric Comparison',
                    'Response Time Distribution',
                    'Error Rate Comparison'
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )

            # Sample size over time
            sample_counts = raw_data.groupby(['variant', raw_data['timestamp'].dt.date]).size().unstack(level=0, fill_value=0).cumsum()

            for variant in ['control', 'treatment']:
                if variant in sample_counts.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=sample_counts.index,
                            y=sample_counts[variant],
                            mode='lines',
                            name=f'{variant.title()} Samples',
                            line=dict(width=3)
                        ),
                        row=1, col=1
                    )

            # Success metric comparison
            success_metric = ab_test['config'].success_metric
            if success_metric in raw_data.columns:
                for variant in ['control', 'treatment']:
                    variant_data = raw_data[raw_data['variant'] == variant][success_metric].dropna()
                    if len(variant_data) > 0:
                        fig.add_trace(
                            go.Box(
                                y=variant_data,
                                name=f'{variant.title()} {success_metric}',
                                boxpoints='outliers'
                            ),
                            row=1, col=2
                        )

            # Response time distribution
            for variant in ['control', 'treatment']:
                variant_data = raw_data[raw_data['variant'] == variant]['response_time'].dropna()
                if len(variant_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=variant_data,
                            name=f'{variant.title()} Response Time',
                            opacity=0.7,
                            nbinsx=30
                        ),
                        row=2, col=1
                    )

            # Error rate over time
            error_rates = raw_data.groupby(['variant', raw_data['timestamp'].dt.date]).apply(
                lambda x: x['prediction'].isna().mean()
            ).unstack(level=0, fill_value=0)

            for variant in ['control', 'treatment']:
                if variant in error_rates.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=error_rates.index,
                            y=error_rates[variant],
                            mode='lines+markers',
                            name=f'{variant.title()} Error Rate',
                            line=dict(width=3)
                        ),
                        row=2, col=2
                    )

            fig.update_layout(
                title=f"A/B Test Dashboard: {ab_test['config'].test_name}",
                height=800,
                showlegend=True
            )

            # Save dashboard
            dashboard_file = f"abtest_dashboard_{test_id}.html"
            fig.write_html(dashboard_file)

            logger.info(f"Created A/B test dashboard: {dashboard_file}")
            return dashboard_file

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return ""

    def _log_test_to_mlflow(self, test_id: str, ab_test: Dict[str, Any]):
        """Log A/B test to MLflow"""
        try:
            with mlflow.start_run(run_name=f"abtest_{test_id}"):
                config = ab_test['config']

                # Log test parameters
                mlflow.log_params({
                    'test_id': test_id,
                    'test_name': config.test_name,
                    'control_model': config.control_model,
                    'treatment_model': config.treatment_model,
                    'traffic_split': config.traffic_split,
                    'minimum_sample_size': config.minimum_sample_size,
                    'confidence_level': config.confidence_level,
                    'success_metric': config.success_metric
                })

                # Log test tags
                mlflow.set_tags({
                    'ab_test': 'true',
                    'test_name': config.test_name,
                    'control_model': config.control_model,
                    'treatment_model': config.treatment_model
                })

        except Exception as e:
            logger.warning(f"Failed to log A/B test to MLflow: {e}")


def create_ab_test_manager(mlflow_tracking_uri: str = "file://./mlruns") -> ABTestManager:
    """Create an A/B test manager instance"""
    return ABTestManager(mlflow_tracking_uri)


if __name__ == "__main__":
    # Example usage
    manager = create_ab_test_manager()

    # Create test configuration
    config = ABTestConfig(
        test_name="transformer_vs_lstm_accuracy",
        description="Compare accuracy between transformer and LSTM models",
        control_model="crypto_lstm_v1.0",
        treatment_model="crypto_transformer_v1.0",
        traffic_split=0.3,  # 30% to treatment
        minimum_sample_size=1000,
        success_metric="accuracy",
        early_stopping_enabled=True
    )

    # Create and start test
    test_id = manager.create_ab_test(config, start_immediately=True)

    # Simulate some requests
    for i in range(100):
        user_id = f"user_{i}"
        prediction = random.random()
        actual = random.random()
        response_time = random.uniform(0.1, 2.0)

        variant = manager.process_request(
            test_id=test_id,
            user_id=user_id,
            prediction=prediction,
            actual=actual,
            response_time=response_time
        )

        print(f"User {user_id} assigned to {variant.value}")

    # Check test status
    status = manager.get_test_status(test_id)
    print(f"Test status: {status}")

    # Create dashboard
    dashboard_file = manager.create_test_dashboard(test_id)
    print(f"Dashboard created: {dashboard_file}")