"""
Advanced Performance Monitoring and Alerting System

Professional monitoring system featuring:
- Real-time performance tracking
- Model drift detection and alerting
- System health monitoring
- Automated alert notifications (Slack, email)
- Performance degradation detection
- SLA monitoring and reporting
- Interactive monitoring dashboards
"""

from __future__ import annotations

import os
import json
import time
import uuid
import threading
import requests
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Optional email imports
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
import psutil
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from scipy import stats
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
from rich.panel import Panel
from rich.progress import Progress

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringMetric(Enum):
    """Monitoring metrics enumeration"""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    DRIFT_SCORE = "drift_score"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    MODEL_LATENCY = "model_latency"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: MonitoringMetric
    threshold: float
    comparison: str  # '>', '<', '>=', '<=', '=='
    severity: AlertSeverity
    window_minutes: int = 5
    min_samples: int = 10
    enabled: bool = True
    notification_channels: List[str] = None

    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["console"]


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    monitoring_interval: int = 60  # seconds
    retention_days: int = 30
    alert_cooldown_minutes: int = 15
    enable_system_monitoring: bool = True
    enable_model_monitoring: bool = True
    enable_drift_detection: bool = True
    metrics_storage_path: str = "monitoring/metrics"
    alerts_storage_path: str = "monitoring/alerts"
    dashboard_update_interval: int = 30  # seconds


@dataclass
class PerformanceThresholds:
    """Performance thresholds for SLA monitoring"""
    max_response_time_ms: float = 1000.0
    max_error_rate_percent: float = 1.0
    min_throughput_rps: float = 10.0
    min_accuracy_percent: float = 85.0
    max_drift_score: float = 0.1
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 85.0


class MetricsCollector:
    """Collects system and model performance metrics"""

    def __init__(self):
        self.metrics_buffer = {}
        self._lock = threading.Lock()

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network metrics (if available)
            network = psutil.net_io_counters()

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    def collect_model_metrics(
        self,
        model_name: str,
        response_time: float,
        prediction: Any,
        actual: Optional[Any] = None,
        error: Optional[str] = None
    ):
        """Collect model performance metrics"""
        with self._lock:
            if model_name not in self.metrics_buffer:
                self.metrics_buffer[model_name] = []

            metric_record = {
                'timestamp': datetime.now(),
                'response_time': response_time,
                'prediction': prediction,
                'actual': actual,
                'error': error,
                'has_error': error is not None
            }

            self.metrics_buffer[model_name].append(metric_record)

            # Keep only recent metrics (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics_buffer[model_name] = [
                m for m in self.metrics_buffer[model_name]
                if m['timestamp'] > cutoff_time
            ]

    def get_model_performance_summary(self, model_name: str) -> Dict[str, float]:
        """Get model performance summary"""
        with self._lock:
            if model_name not in self.metrics_buffer:
                return {}

            metrics = self.metrics_buffer[model_name]
            if not metrics:
                return {}

            # Calculate performance metrics
            response_times = [m['response_time'] for m in metrics]
            errors = [m for m in metrics if m['has_error']]

            # Calculate accuracy if we have actual values
            accurate_predictions = [
                m for m in metrics
                if m['actual'] is not None and m['prediction'] is not None
            ]

            summary = {
                'total_requests': len(metrics),
                'error_count': len(errors),
                'error_rate': len(errors) / len(metrics) if metrics else 0,
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
                'throughput_rps': len(metrics) / 3600 if metrics else 0  # requests per second over last hour
            }

            # Calculate accuracy for regression or classification
            if accurate_predictions:
                if isinstance(accurate_predictions[0]['actual'], (int, float)):
                    # Regression - calculate RMSE and MAE
                    actuals = [m['actual'] for m in accurate_predictions]
                    predictions = [m['prediction'] for m in accurate_predictions]

                    rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))
                    mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))

                    summary.update({
                        'rmse': rmse,
                        'mae': mae,
                        'accuracy_samples': len(accurate_predictions)
                    })
                else:
                    # Classification - calculate accuracy
                    correct = sum(
                        1 for m in accurate_predictions
                        if m['prediction'] == m['actual']
                    )
                    accuracy = correct / len(accurate_predictions)
                    summary['accuracy'] = accuracy

            return summary


class DriftMonitor:
    """Monitor model drift"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.reference_data = None
        self.recent_data = []

    def set_reference_data(self, data: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data = data
        logger.info(f"Set reference data with {len(data)} samples")

    def add_recent_data(self, data_point: np.ndarray):
        """Add recent data point for drift monitoring"""
        self.recent_data.append(data_point)

        # Keep only recent data within window
        if len(self.recent_data) > self.window_size:
            self.recent_data = self.recent_data[-self.window_size:]

    def calculate_drift_score(self) -> Dict[str, float]:
        """Calculate drift score using statistical tests"""
        if self.reference_data is None or len(self.recent_data) < 30:
            return {'drift_score': 0.0, 'p_value': 1.0}

        try:
            # Convert recent data to array
            recent_array = np.array(self.recent_data)

            # Calculate drift for each feature
            n_features = min(self.reference_data.shape[1], recent_array.shape[1])
            drift_scores = []
            p_values = []

            for i in range(n_features):
                if SCIPY_AVAILABLE:
                    # Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(
                        self.reference_data[:, i],
                        recent_array[:, i]
                    )
                    drift_scores.append(statistic)
                    p_values.append(p_value)
                else:
                    # Simple statistical comparison
                    ref_mean = np.mean(self.reference_data[:, i])
                    ref_std = np.std(self.reference_data[:, i])
                    recent_mean = np.mean(recent_array[:, i])

                    # Normalized difference
                    drift_score = abs(recent_mean - ref_mean) / (ref_std + 1e-8)
                    drift_scores.append(drift_score)
                    p_values.append(0.5)  # Placeholder

            overall_drift = np.mean(drift_scores)
            min_p_value = np.min(p_values) if p_values else 1.0

            return {
                'drift_score': float(overall_drift),
                'p_value': float(min_p_value),
                'feature_drift_scores': [float(s) for s in drift_scores]
            }

        except Exception as e:
            logger.error(f"Drift calculation failed: {e}")
            return {'drift_score': 0.0, 'p_value': 1.0}


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules = []
        self.alert_history = []
        self.alert_cooldowns = {}

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all alert rules against current metrics"""
        alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check if rule is in cooldown
            if self._is_in_cooldown(rule.name):
                continue

            # Get metric value
            metric_value = self._get_metric_value(metrics, rule.metric)
            if metric_value is None:
                continue

            # Check threshold
            if self._evaluate_threshold(metric_value, rule.threshold, rule.comparison):
                alert = self._create_alert(rule, metric_value, metrics)
                alerts.append(alert)

                # Set cooldown
                self.alert_cooldowns[rule.name] = datetime.now()

        # Store alerts in history
        self.alert_history.extend(alerts)

        # Cleanup old alerts
        self._cleanup_alert_history()

        return alerts

    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is in cooldown period"""
        if rule_name not in self.alert_cooldowns:
            return False

        cooldown_end = self.alert_cooldowns[rule_name] + timedelta(
            minutes=self.config.alert_cooldown_minutes
        )

        return datetime.now() < cooldown_end

    def _get_metric_value(self, metrics: Dict[str, Any], metric_type: MonitoringMetric) -> Optional[float]:
        """Extract metric value from metrics dictionary"""
        metric_key = metric_type.value

        # Direct lookup
        if metric_key in metrics:
            return float(metrics[metric_key])

        # Look in nested structures
        for key, value in metrics.items():
            if isinstance(value, dict) and metric_key in value:
                return float(value[metric_key])

        return None

    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate threshold comparison"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '>=':
            return value >= threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '==':
            return abs(value - threshold) < 1e-6
        else:
            return False

    def _create_alert(self, rule: AlertRule, value: float, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert dictionary"""
        return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'rule_name': rule.name,
            'metric': rule.metric.value,
            'value': value,
            'threshold': rule.threshold,
            'comparison': rule.comparison,
            'severity': rule.severity.value,
            'message': f"{rule.name}: {rule.metric.value} ({value:.3f}) {rule.comparison} {rule.threshold}",
            'notification_channels': rule.notification_channels,
            'full_metrics': metrics
        }

    def _cleanup_alert_history(self):
        """Remove old alerts from history"""
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]


class NotificationManager:
    """Manages alert notifications"""

    def __init__(self):
        self.notification_handlers = {
            'console': self._send_console_notification,
            'slack': self._send_slack_notification,
            'email': self._send_email_notification,
            'webhook': self._send_webhook_notification
        }

    def send_notifications(self, alerts: List[Dict[str, Any]]):
        """Send notifications for alerts"""
        for alert in alerts:
            for channel in alert['notification_channels']:
                self._send_notification(alert, channel)

    def _send_notification(self, alert: Dict[str, Any], channel: str):
        """Send notification to specific channel"""
        try:
            if channel in self.notification_handlers:
                self.notification_handlers[channel](alert)
            else:
                logger.warning(f"Unknown notification channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to send {channel} notification: {e}")

    def _send_console_notification(self, alert: Dict[str, Any]):
        """Send console notification"""
        severity_colors = {
            'info': 'blue',
            'warning': 'yellow',
            'critical': 'red',
            'emergency': 'magenta'
        }

        color = severity_colors.get(alert['severity'], 'white')

        console.print(
            Panel(
                f"[bold]{alert['message']}[/bold]\n"
                f"Severity: {alert['severity'].upper()}\n"
                f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Value: {alert['value']:.3f}",
                title=f"🚨 {alert['rule_name']}",
                border_style=color
            )
        )

    def _send_slack_notification(self, alert: Dict[str, Any]):
        """Send Slack notification"""
        # This would require Slack webhook URL configuration
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')

        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        severity_emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨',
            'emergency': '🔥'
        }

        emoji = severity_emojis.get(alert['severity'], '📊')

        message = {
            'text': f"{emoji} Model Monitoring Alert",
            'blocks': [
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f"*{alert['rule_name']}*\n{alert['message']}"
                    }
                },
                {
                    'type': 'section',
                    'fields': [
                        {
                            'type': 'mrkdwn',
                            'text': f"*Severity:*\n{alert['severity'].upper()}"
                        },
                        {
                            'type': 'mrkdwn',
                            'text': f"*Value:*\n{alert['value']:.3f}"
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=message, timeout=30)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def _send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available")
            return

        # Email configuration from environment variables
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        from_email = os.getenv('ALERT_FROM_EMAIL')
        to_emails = os.getenv('ALERT_TO_EMAILS', '').split(',')

        if not all([smtp_server, smtp_username, smtp_password, from_email]):
            logger.warning("Email configuration incomplete")
            return

        try:
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"Model Monitoring Alert: {alert['rule_name']}"

            body = f"""
Model Monitoring Alert

Rule: {alert['rule_name']}
Message: {alert['message']}
Severity: {alert['severity'].upper()}
Timestamp: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Value: {alert['value']:.3f}
Threshold: {alert['threshold']}

Please investigate immediately if this is a critical alert.
            """

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info("Email notification sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _send_webhook_notification(self, alert: Dict[str, Any]):
        """Send webhook notification"""
        webhook_url = os.getenv('MONITORING_WEBHOOK_URL')

        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return

        payload = {
            'alert_id': alert['id'],
            'rule_name': alert['rule_name'],
            'metric': alert['metric'],
            'value': alert['value'],
            'threshold': alert['threshold'],
            'severity': alert['severity'],
            'message': alert['message'],
            'timestamp': alert['timestamp'].isoformat()
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=30)
            if response.status_code == 200:
                logger.info("Webhook notification sent successfully")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")


class PerformanceMonitor:
    """Main performance monitoring system"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.drift_monitor = DriftMonitor()
        self.alert_manager = AlertManager(config)
        self.notification_manager = NotificationManager()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None

        # Storage
        self.metrics_history = []
        self.storage_path = Path(config.metrics_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Setup default alert rules
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High error rate
        self.alert_manager.add_alert_rule(AlertRule(
            name="High Error Rate",
            metric=MonitoringMetric.ERROR_RATE,
            threshold=0.05,  # 5%
            comparison='>',
            severity=AlertSeverity.WARNING,
            notification_channels=['console', 'slack']
        ))

        # High response time
        self.alert_manager.add_alert_rule(AlertRule(
            name="High Response Time",
            metric=MonitoringMetric.RESPONSE_TIME,
            threshold=2000,  # 2 seconds
            comparison='>',
            severity=AlertSeverity.WARNING,
            notification_channels=['console']
        ))

        # Model drift
        self.alert_manager.add_alert_rule(AlertRule(
            name="Model Drift Detected",
            metric=MonitoringMetric.DRIFT_SCORE,
            threshold=0.1,
            comparison='>',
            severity=AlertSeverity.CRITICAL,
            notification_channels=['console', 'slack', 'email']
        ))

        # System resource alerts
        self.alert_manager.add_alert_rule(AlertRule(
            name="High CPU Usage",
            metric=MonitoringMetric.CPU_USAGE,
            threshold=90.0,
            comparison='>',
            severity=AlertSeverity.WARNING,
            notification_channels=['console']
        ))

        self.alert_manager.add_alert_rule(AlertRule(
            name="High Memory Usage",
            metric=MonitoringMetric.MEMORY_USAGE,
            threshold=90.0,
            comparison='>',
            severity=AlertSeverity.CRITICAL,
            notification_channels=['console', 'slack']
        ))

    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                current_metrics = self._collect_all_metrics()

                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': current_metrics
                })

                # Check alerts
                alerts = self.alert_manager.check_alerts(current_metrics)

                # Send notifications
                if alerts:
                    self.notification_manager.send_notifications(alerts)

                # Cleanup old metrics
                self._cleanup_metrics_history()

                # Save metrics to disk periodically
                if len(self.metrics_history) % 10 == 0:
                    self._save_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.config.monitoring_interval)

    def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics"""
        metrics = {}

        # System metrics
        if self.config.enable_system_monitoring:
            system_metrics = self.metrics_collector.collect_system_metrics()
            metrics.update(system_metrics)

        # Model metrics
        if self.config.enable_model_monitoring:
            # This would be populated by model prediction calls
            pass

        # Drift metrics
        if self.config.enable_drift_detection:
            drift_metrics = self.drift_monitor.calculate_drift_score()
            metrics.update(drift_metrics)

        return metrics

    def record_prediction(
        self,
        model_name: str,
        prediction: Any,
        response_time: float,
        actual: Optional[Any] = None,
        error: Optional[str] = None,
        input_features: Optional[np.ndarray] = None
    ):
        """Record a model prediction for monitoring"""
        # Collect model metrics
        self.metrics_collector.collect_model_metrics(
            model_name, response_time, prediction, actual, error
        )

        # Update drift monitoring
        if input_features is not None and self.config.enable_drift_detection:
            self.drift_monitor.add_recent_data(input_features)

    def get_performance_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}

        # System performance
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]['metrics']
            summary['system'] = {
                'cpu_usage': latest_metrics.get('cpu_usage', 0),
                'memory_usage': latest_metrics.get('memory_usage', 0),
                'disk_usage': latest_metrics.get('disk_usage', 0),
                'timestamp': self.metrics_history[-1]['timestamp'].isoformat()
            }

        # Model performance
        if model_name:
            model_summary = self.metrics_collector.get_model_performance_summary(model_name)
            summary['model'] = model_summary

        # Recent alerts
        recent_alerts = [
            alert for alert in self.alert_manager.alert_history
            if alert['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        summary['recent_alerts'] = len(recent_alerts)

        # Drift status
        drift_metrics = self.drift_monitor.calculate_drift_score()
        summary['drift'] = drift_metrics

        return summary

    def create_monitoring_dashboard(self) -> str:
        """Create monitoring dashboard"""
        try:
            if not self.metrics_history:
                return ""

            # Prepare data
            timestamps = [m['timestamp'] for m in self.metrics_history]
            cpu_usage = [m['metrics'].get('cpu_usage', 0) for m in self.metrics_history]
            memory_usage = [m['metrics'].get('memory_usage', 0) for m in self.metrics_history]
            drift_scores = [m['metrics'].get('drift_score', 0) for m in self.metrics_history]

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'CPU Usage (%)',
                    'Memory Usage (%)',
                    'Model Drift Score',
                    'Alert History'
                ],
                vertical_spacing=0.1
            )

            # CPU usage
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_usage,
                    mode='lines',
                    name='CPU Usage',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # Memory usage
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode='lines',
                    name='Memory Usage',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )

            # Drift score
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=drift_scores,
                    mode='lines',
                    name='Drift Score',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )

            # Alert history
            alert_times = [alert['timestamp'] for alert in self.alert_manager.alert_history]
            alert_severities = [alert['severity'] for alert in self.alert_manager.alert_history]

            severity_colors = {'info': 'blue', 'warning': 'orange', 'critical': 'red', 'emergency': 'purple'}
            for severity in ['info', 'warning', 'critical', 'emergency']:
                severity_alerts = [t for t, s in zip(alert_times, alert_severities) if s == severity]
                if severity_alerts:
                    fig.add_trace(
                        go.Scatter(
                            x=severity_alerts,
                            y=[severity] * len(severity_alerts),
                            mode='markers',
                            name=f'{severity.title()} Alerts',
                            marker=dict(color=severity_colors[severity], size=10)
                        ),
                        row=2, col=2
                    )

            fig.update_layout(
                title="Performance Monitoring Dashboard",
                height=800,
                showlegend=True
            )

            # Save dashboard
            dashboard_file = f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(dashboard_file)

            logger.info(f"Created monitoring dashboard: {dashboard_file}")
            return dashboard_file

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return ""

    def _cleanup_metrics_history(self):
        """Remove old metrics from history"""
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        self.metrics_history = [
            m for m in self.metrics_history
            if m['timestamp'] > cutoff_time
        ]

    def _save_metrics(self):
        """Save metrics to disk"""
        try:
            metrics_file = self.storage_path / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"

            # Convert to serializable format
            serializable_metrics = []
            for m in self.metrics_history:
                serializable_m = m.copy()
                serializable_m['timestamp'] = m['timestamp'].isoformat()
                serializable_metrics.append(serializable_m)

            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def generate_sla_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate SLA compliance report"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.metrics_history
                if m['timestamp'] > cutoff_time
            ]

            if not recent_metrics:
                return {}

            thresholds = PerformanceThresholds()

            # Calculate SLA metrics
            cpu_violations = sum(
                1 for m in recent_metrics
                if m['metrics'].get('cpu_usage', 0) > thresholds.max_cpu_usage_percent
            )

            memory_violations = sum(
                1 for m in recent_metrics
                if m['metrics'].get('memory_usage', 0) > thresholds.max_memory_usage_percent
            )

            drift_violations = sum(
                1 for m in recent_metrics
                if m['metrics'].get('drift_score', 0) > thresholds.max_drift_score
            )

            total_samples = len(recent_metrics)

            sla_report = {
                'report_period_days': days,
                'total_samples': total_samples,
                'cpu_sla_compliance': 1 - (cpu_violations / total_samples) if total_samples > 0 else 1.0,
                'memory_sla_compliance': 1 - (memory_violations / total_samples) if total_samples > 0 else 1.0,
                'drift_sla_compliance': 1 - (drift_violations / total_samples) if total_samples > 0 else 1.0,
                'overall_sla_compliance': 1 - ((cpu_violations + memory_violations + drift_violations) / (3 * total_samples)) if total_samples > 0 else 1.0,
                'generated_at': datetime.now().isoformat()
            }

            # Save report
            report_file = self.storage_path / f"sla_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(sla_report, f, indent=2)

            return sla_report

        except Exception as e:
            logger.error(f"Failed to generate SLA report: {e}")
            return {}


def create_performance_monitor(
    monitoring_interval: int = 60,
    enable_slack_alerts: bool = False,
    enable_email_alerts: bool = False
) -> PerformanceMonitor:
    """Create a performance monitor instance"""
    config = MonitoringConfig(
        monitoring_interval=monitoring_interval,
        enable_system_monitoring=True,
        enable_model_monitoring=True,
        enable_drift_detection=True
    )

    return PerformanceMonitor(config)


if __name__ == "__main__":
    # Example usage
    monitor = create_performance_monitor(monitoring_interval=30)

    # Start monitoring
    monitor.start_monitoring()

    # Simulate some predictions
    import random
    for i in range(100):
        monitor.record_prediction(
            model_name="crypto_transformer",
            prediction=random.random(),
            response_time=random.uniform(0.1, 3.0),
            actual=random.random(),
            input_features=np.random.randn(50)
        )
        time.sleep(0.1)

    # Get performance summary
    summary = monitor.get_performance_summary("crypto_transformer")
    print("Performance Summary:", summary)

    # Generate SLA report
    sla_report = monitor.generate_sla_report(days=1)
    print("SLA Report:", sla_report)

    # Create dashboard
    dashboard_file = monitor.create_monitoring_dashboard()
    print(f"Dashboard: {dashboard_file}")

    # Stop monitoring
    monitor.stop_monitoring()