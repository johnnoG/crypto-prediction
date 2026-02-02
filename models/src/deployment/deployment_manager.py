"""
Advanced Model Deployment Pipeline

Professional deployment system featuring:
- Multi-stage deployment (Dev → Staging → Production)
- Blue-Green deployments with zero downtime
- Automated rollback capabilities
- Health checks and validation
- Deployment approval gates
- Performance monitoring integration
"""

from __future__ import annotations

import os
import json
import time
import uuid
import shutil
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
import subprocess
import requests
import psutil

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class DeploymentStage(Enum):
    """Deployment stage enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    TESTING = "testing"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ARCHIVED = "archived"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_name: str
    model_version: str
    stage: DeploymentStage
    replicas: int = 1
    cpu_limit: str = "500m"
    memory_limit: str = "1Gi"
    health_check_path: str = "/health"
    health_check_timeout: int = 30
    rollback_on_failure: bool = True
    auto_promote: bool = False
    approval_required: bool = True
    monitoring_enabled: bool = True
    traffic_percentage: int = 100  # For blue-green deployments


@dataclass
class DeploymentInfo:
    """Deployment information"""
    deployment_id: str
    model_name: str
    model_version: str
    stage: DeploymentStage
    status: DeploymentStatus
    config: DeploymentConfig
    created_at: datetime
    updated_at: datetime
    endpoint_url: Optional[str] = None
    health_status: Optional[str] = None
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class HealthChecker:
    """Health checking for deployed models"""

    def __init__(self, deployment_info: DeploymentInfo):
        self.deployment_info = deployment_info
        self.is_healthy = False
        self.last_check = None
        self.error_count = 0
        self.max_errors = 5

    def check_health(self) -> bool:
        """Perform health check"""
        try:
            if not self.deployment_info.endpoint_url:
                return False

            health_url = f"{self.deployment_info.endpoint_url}{self.deployment_info.config.health_check_path}"

            response = requests.get(
                health_url,
                timeout=self.deployment_info.config.health_check_timeout
            )

            if response.status_code == 200:
                self.is_healthy = True
                self.error_count = 0
                self.last_check = datetime.now()
                return True
            else:
                self.error_count += 1
                logger.warning(f"Health check failed: {response.status_code}")
                return False

        except Exception as e:
            self.error_count += 1
            logger.warning(f"Health check error: {e}")
            return False

    def is_deployment_healthy(self) -> bool:
        """Check if deployment is healthy"""
        return self.is_healthy and self.error_count < self.max_errors

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary"""
        return {
            'is_healthy': self.is_healthy,
            'error_count': self.error_count,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'max_errors': self.max_errors
        }


class DeploymentValidator:
    """Validates model deployments"""

    def __init__(self, test_data_path: Optional[str] = None):
        self.test_data_path = test_data_path
        self.validation_results = {}

    def validate_model_predictions(
        self,
        deployment_info: DeploymentInfo,
        sample_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Validate model predictions"""
        try:
            if not deployment_info.endpoint_url:
                return {'status': 'failed', 'error': 'No endpoint URL'}

            # Use sample data or load test data
            if sample_data is None and self.test_data_path:
                sample_data = self._load_test_data()

            if sample_data is None:
                # Generate dummy data for basic validation
                sample_data = np.random.randn(10, 50)

            # Make prediction request
            predict_url = f"{deployment_info.endpoint_url}/predict"
            response = requests.post(
                predict_url,
                json={'data': sample_data.tolist()},
                timeout=30
            )

            if response.status_code == 200:
                predictions = response.json()

                # Basic validation
                validation_result = {
                    'status': 'passed',
                    'response_time': response.elapsed.total_seconds(),
                    'prediction_shape': np.array(predictions).shape if predictions else None,
                    'sample_predictions': predictions[:5] if predictions else None
                }

                # Performance validation
                if response.elapsed.total_seconds() > 5.0:
                    validation_result['warnings'] = ['High response time detected']

                return validation_result

            else:
                return {
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}: {response.text}'
                }

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _load_test_data(self) -> Optional[np.ndarray]:
        """Load test data for validation"""
        try:
            if self.test_data_path and Path(self.test_data_path).exists():
                # Load based on file extension
                if self.test_data_path.endswith('.npy'):
                    return np.load(self.test_data_path)
                elif self.test_data_path.endswith('.csv'):
                    df = pd.read_csv(self.test_data_path)
                    return df.values
                else:
                    logger.warning(f"Unsupported test data format: {self.test_data_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load test data: {e}")
            return None

    def validate_performance(
        self,
        deployment_info: DeploymentInfo,
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate deployment performance against baseline"""
        try:
            # Get current performance metrics
            current_metrics = self._collect_performance_metrics(deployment_info)

            validation_result = {
                'status': 'passed',
                'current_metrics': current_metrics,
                'baseline_metrics': baseline_metrics,
                'performance_degradation': {}
            }

            # Check for performance degradation
            for metric, baseline_value in baseline_metrics.items():
                if metric in current_metrics:
                    current_value = current_metrics[metric]
                    degradation = (current_value - baseline_value) / baseline_value

                    if degradation > 0.1:  # 10% degradation threshold
                        validation_result['performance_degradation'][metric] = degradation
                        validation_result['status'] = 'warning'

                    if degradation > 0.25:  # 25% degradation threshold
                        validation_result['status'] = 'failed'

            return validation_result

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _collect_performance_metrics(self, deployment_info: DeploymentInfo) -> Dict[str, float]:
        """Collect current performance metrics"""
        try:
            # This would typically integrate with monitoring systems
            # For now, return mock metrics
            return {
                'response_time_p95': 0.5,
                'error_rate': 0.01,
                'throughput_rps': 100.0,
                'cpu_usage': 0.3,
                'memory_usage': 0.4
            }
        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")
            return {}


class BlueGreenDeploymentManager:
    """Manages blue-green deployments for zero downtime"""

    def __init__(self):
        self.active_deployments = {}  # deployment_id -> DeploymentInfo
        self.traffic_router = TrafficRouter()

    def deploy_blue_green(
        self,
        model_name: str,
        new_version: str,
        config: DeploymentConfig
    ) -> str:
        """Deploy new version using blue-green strategy"""
        deployment_id = f"{model_name}_{new_version}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting blue-green deployment: {deployment_id}")

        try:
            # Create new deployment info (Green)
            green_deployment = DeploymentInfo(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=new_version,
                stage=config.stage,
                status=DeploymentStatus.DEPLOYING,
                config=config,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Deploy green version
            self._deploy_green_version(green_deployment)

            # Health check green version
            if not self._wait_for_healthy(green_deployment):
                raise Exception("Green deployment failed health check")

            # Validate green version
            if not self._validate_green_version(green_deployment):
                raise Exception("Green deployment failed validation")

            # Switch traffic (Blue -> Green)
            self._switch_traffic(model_name, green_deployment)

            # Archive old blue version
            self._archive_blue_version(model_name, green_deployment)

            green_deployment.status = DeploymentStatus.ACTIVE
            green_deployment.updated_at = datetime.now()

            self.active_deployments[deployment_id] = green_deployment

            logger.info(f"Blue-green deployment completed successfully: {deployment_id}")
            return deployment_id

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")

            # Cleanup failed green deployment
            self._cleanup_failed_deployment(deployment_id)

            # Keep blue deployment active
            raise

    def _deploy_green_version(self, deployment_info: DeploymentInfo):
        """Deploy the green version"""
        logger.info(f"Deploying green version: {deployment_info.model_version}")

        # Simulate deployment process
        # In production, this would:
        # 1. Pull model from MLflow registry
        # 2. Create container/service
        # 3. Configure load balancer
        # 4. Set up monitoring

        time.sleep(2)  # Simulate deployment time

        # Set endpoint URL (would be actual URL in production)
        deployment_info.endpoint_url = f"http://green.{deployment_info.model_name}.local:8000"

        logger.info(f"Green deployment ready: {deployment_info.endpoint_url}")

    def _wait_for_healthy(self, deployment_info: DeploymentInfo) -> bool:
        """Wait for deployment to become healthy"""
        logger.info("Waiting for green deployment to become healthy...")

        health_checker = HealthChecker(deployment_info)
        max_attempts = 30

        for attempt in range(max_attempts):
            if health_checker.check_health():
                logger.info("Green deployment is healthy")
                return True

            time.sleep(10)
            logger.info(f"Health check attempt {attempt + 1}/{max_attempts}")

        logger.error("Green deployment failed to become healthy")
        return False

    def _validate_green_version(self, deployment_info: DeploymentInfo) -> bool:
        """Validate green version before traffic switch"""
        logger.info("Validating green deployment...")

        validator = DeploymentValidator()

        # Prediction validation
        pred_result = validator.validate_model_predictions(deployment_info)
        if pred_result['status'] != 'passed':
            logger.error(f"Prediction validation failed: {pred_result}")
            return False

        # Performance validation (if baseline exists)
        # perf_result = validator.validate_performance(deployment_info, baseline_metrics)

        logger.info("Green deployment validation passed")
        return True

    def _switch_traffic(self, model_name: str, green_deployment: DeploymentInfo):
        """Switch traffic from blue to green"""
        logger.info("Switching traffic from blue to green...")

        # Gradual traffic switching (canary-style)
        traffic_percentages = [10, 25, 50, 100]

        for percentage in traffic_percentages:
            self.traffic_router.set_traffic_split(
                model_name,
                green_deployment.deployment_id,
                percentage
            )

            logger.info(f"Routing {percentage}% traffic to green deployment")

            # Monitor for a period before increasing traffic
            time.sleep(30)

            # Check health during traffic switch
            health_checker = HealthChecker(green_deployment)
            if not health_checker.check_health():
                raise Exception(f"Health check failed during traffic switch at {percentage}%")

        logger.info("Traffic switch completed successfully")

    def _archive_blue_version(self, model_name: str, green_deployment: DeploymentInfo):
        """Archive the old blue version"""
        logger.info("Archiving blue deployment...")

        # Find and archive old deployments
        for deployment_id, deployment in self.active_deployments.items():
            if (deployment.model_name == model_name and
                deployment.deployment_id != green_deployment.deployment_id and
                deployment.status == DeploymentStatus.ACTIVE):

                deployment.status = DeploymentStatus.ARCHIVED
                deployment.updated_at = datetime.now()
                logger.info(f"Archived deployment: {deployment_id}")

    def _cleanup_failed_deployment(self, deployment_id: str):
        """Cleanup failed deployment"""
        logger.info(f"Cleaning up failed deployment: {deployment_id}")

        # Remove from active deployments
        if deployment_id in self.active_deployments:
            del self.active_deployments[deployment_id]

        # Cleanup resources (containers, services, etc.)
        # This would be implementation-specific

    def rollback_deployment(self, model_name: str, target_version: Optional[str] = None) -> bool:
        """Rollback to previous or specific version"""
        logger.info(f"Starting rollback for {model_name}")

        try:
            # Find current active deployment
            current_deployment = None
            for deployment in self.active_deployments.values():
                if (deployment.model_name == model_name and
                    deployment.status == DeploymentStatus.ACTIVE):
                    current_deployment = deployment
                    break

            if not current_deployment:
                logger.error(f"No active deployment found for {model_name}")
                return False

            # Find rollback target
            rollback_target = None
            if target_version:
                # Rollback to specific version
                for deployment in self.active_deployments.values():
                    if (deployment.model_name == model_name and
                        deployment.model_version == target_version and
                        deployment.status == DeploymentStatus.ARCHIVED):
                        rollback_target = deployment
                        break
            else:
                # Rollback to most recent archived version
                archived_deployments = [
                    d for d in self.active_deployments.values()
                    if (d.model_name == model_name and
                        d.status == DeploymentStatus.ARCHIVED)
                ]
                if archived_deployments:
                    rollback_target = max(archived_deployments, key=lambda x: x.created_at)

            if not rollback_target:
                logger.error(f"No rollback target found for {model_name}")
                return False

            # Perform rollback
            current_deployment.status = DeploymentStatus.ROLLING_BACK

            # Reactivate rollback target
            self._reactivate_deployment(rollback_target)

            # Switch traffic back
            self._switch_traffic(model_name, rollback_target)

            # Archive current deployment
            current_deployment.status = DeploymentStatus.ARCHIVED
            rollback_target.status = DeploymentStatus.ACTIVE

            logger.info(f"Rollback completed: {model_name} -> {rollback_target.model_version}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def _reactivate_deployment(self, deployment: DeploymentInfo):
        """Reactivate an archived deployment"""
        logger.info(f"Reactivating deployment: {deployment.deployment_id}")

        # This would restart containers/services for the archived deployment
        deployment.updated_at = datetime.now()


class TrafficRouter:
    """Manages traffic routing for blue-green deployments"""

    def __init__(self):
        self.traffic_rules = {}

    def set_traffic_split(self, model_name: str, deployment_id: str, percentage: int):
        """Set traffic split percentage for a deployment"""
        if model_name not in self.traffic_rules:
            self.traffic_rules[model_name] = {}

        self.traffic_rules[model_name][deployment_id] = percentage
        logger.info(f"Set traffic split: {model_name} -> {deployment_id}: {percentage}%")

    def get_traffic_rules(self, model_name: str) -> Dict[str, int]:
        """Get current traffic rules for a model"""
        return self.traffic_rules.get(model_name, {})


class DeploymentManager:
    """Main deployment manager orchestrating all deployment operations"""

    def __init__(self, mlflow_tracking_uri: str = "file://./mlruns"):
        self.tracking_uri = mlflow_tracking_uri
        self.client = MlflowClient(tracking_uri) if MLFLOW_AVAILABLE else None
        self.blue_green_manager = BlueGreenDeploymentManager()
        self.deployment_registry = {}
        self.approval_queue = {}

    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        stage: DeploymentStage,
        config: Optional[DeploymentConfig] = None,
        strategy: str = "blue_green"
    ) -> str:
        """Deploy model with specified strategy"""

        if config is None:
            config = DeploymentConfig(
                model_name=model_name,
                model_version=model_version,
                stage=stage
            )

        logger.info(f"Deploying {model_name} v{model_version} to {stage.value}")

        try:
            # Approval check for production deployments
            if stage == DeploymentStage.PRODUCTION and config.approval_required:
                if not self._check_deployment_approval(model_name, model_version):
                    approval_id = self._request_deployment_approval(model_name, model_version)
                    logger.info(f"Deployment pending approval: {approval_id}")
                    return approval_id

            # Deploy based on strategy
            if strategy == "blue_green":
                deployment_id = self.blue_green_manager.deploy_blue_green(
                    model_name, model_version, config
                )
            else:
                raise ValueError(f"Unsupported deployment strategy: {strategy}")

            # Register deployment
            self.deployment_registry[deployment_id] = {
                'model_name': model_name,
                'model_version': model_version,
                'stage': stage,
                'config': config,
                'created_at': datetime.now()
            }

            # Log to MLflow if available
            if self.client:
                self._log_deployment_to_mlflow(deployment_id, model_name, model_version, stage)

            logger.info(f"Deployment completed successfully: {deployment_id}")
            return deployment_id

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    def _check_deployment_approval(self, model_name: str, model_version: str) -> bool:
        """Check if deployment has been approved"""
        approval_key = f"{model_name}_{model_version}"
        return self.approval_queue.get(approval_key, {}).get('approved', False)

    def _request_deployment_approval(self, model_name: str, model_version: str) -> str:
        """Request deployment approval"""
        approval_id = f"approval_{uuid.uuid4().hex[:8]}"
        approval_key = f"{model_name}_{model_version}"

        self.approval_queue[approval_key] = {
            'approval_id': approval_id,
            'model_name': model_name,
            'model_version': model_version,
            'requested_at': datetime.now(),
            'approved': False,
            'approved_by': None,
            'approved_at': None
        }

        # In production, this would send notifications to approvers
        console.print(f"[yellow]Deployment approval required for {model_name} v{model_version}[/yellow]")
        console.print(f"[yellow]Approval ID: {approval_id}[/yellow]")

        return approval_id

    def approve_deployment(self, approval_id: str, approver: str) -> bool:
        """Approve a pending deployment"""
        for key, approval in self.approval_queue.items():
            if approval['approval_id'] == approval_id:
                approval['approved'] = True
                approval['approved_by'] = approver
                approval['approved_at'] = datetime.now()

                logger.info(f"Deployment approved by {approver}: {approval_id}")

                # Trigger the actual deployment
                self.deploy_model(
                    approval['model_name'],
                    approval['model_version'],
                    DeploymentStage.PRODUCTION,
                    strategy="blue_green"
                )

                return True

        logger.warning(f"Approval ID not found: {approval_id}")
        return False

    def rollback_model(self, model_name: str, target_version: Optional[str] = None) -> bool:
        """Rollback model deployment"""
        return self.blue_green_manager.rollback_deployment(model_name, target_version)

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get detailed deployment status"""
        if deployment_id in self.blue_green_manager.active_deployments:
            deployment = self.blue_green_manager.active_deployments[deployment_id]

            # Get health status
            health_checker = HealthChecker(deployment)
            health_status = health_checker.get_health_summary()

            return {
                'deployment_id': deployment_id,
                'model_name': deployment.model_name,
                'model_version': deployment.model_version,
                'stage': deployment.stage.value,
                'status': deployment.status.value,
                'created_at': deployment.created_at.isoformat(),
                'updated_at': deployment.updated_at.isoformat(),
                'endpoint_url': deployment.endpoint_url,
                'health_status': health_status,
                'performance_metrics': deployment.performance_metrics
            }
        else:
            return {'error': 'Deployment not found'}

    def list_deployments(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments"""
        deployments = []

        for deployment_id, deployment in self.blue_green_manager.active_deployments.items():
            if model_name is None or deployment.model_name == model_name:
                deployments.append(self.get_deployment_status(deployment_id))

        return deployments

    def _log_deployment_to_mlflow(self, deployment_id: str, model_name: str,
                                 model_version: str, stage: DeploymentStage):
        """Log deployment information to MLflow"""
        try:
            # Create deployment tracking run
            with mlflow.start_run(run_name=f"deployment_{deployment_id}"):
                mlflow.log_params({
                    'deployment_id': deployment_id,
                    'model_name': model_name,
                    'model_version': model_version,
                    'stage': stage.value,
                    'deployment_time': datetime.now().isoformat()
                })

                mlflow.set_tags({
                    'deployment': 'true',
                    'stage': stage.value,
                    'model_name': model_name
                })

        except Exception as e:
            logger.warning(f"Failed to log deployment to MLflow: {e}")

    def create_deployment_report(self) -> str:
        """Create comprehensive deployment report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_deployments': len(self.blue_green_manager.active_deployments),
                'deployments_by_stage': {},
                'deployments_by_status': {},
                'recent_deployments': [],
                'pending_approvals': len(self.approval_queue)
            }

            # Aggregate statistics
            for deployment in self.blue_green_manager.active_deployments.values():
                stage = deployment.stage.value
                status = deployment.status.value

                report['deployments_by_stage'][stage] = report['deployments_by_stage'].get(stage, 0) + 1
                report['deployments_by_status'][status] = report['deployments_by_status'].get(status, 0) + 1

            # Recent deployments (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent = [
                {
                    'model_name': d.model_name,
                    'model_version': d.model_version,
                    'stage': d.stage.value,
                    'status': d.status.value,
                    'created_at': d.created_at.isoformat()
                }
                for d in self.blue_green_manager.active_deployments.values()
                if d.created_at > week_ago
            ]
            report['recent_deployments'] = sorted(recent, key=lambda x: x['created_at'], reverse=True)

            # Save report
            report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Deployment report created: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"Failed to create deployment report: {e}")
            return ""


def create_deployment_manager(mlflow_tracking_uri: str = "file://./mlruns") -> DeploymentManager:
    """Create a deployment manager instance"""
    return DeploymentManager(mlflow_tracking_uri)


if __name__ == "__main__":
    # Example usage
    manager = create_deployment_manager()

    # Deploy a model
    config = DeploymentConfig(
        model_name="crypto_transformer",
        model_version="1.2.0",
        stage=DeploymentStage.PRODUCTION,
        replicas=3,
        health_check_path="/health",
        rollback_on_failure=True
    )

    deployment_id = manager.deploy_model(
        model_name="crypto_transformer",
        model_version="1.2.0",
        stage=DeploymentStage.PRODUCTION,
        config=config,
        strategy="blue_green"
    )

    print(f"Deployment started: {deployment_id}")

    # Check deployment status
    status = manager.get_deployment_status(deployment_id)
    print(f"Deployment status: {status}")

    # Create report
    report_file = manager.create_deployment_report()
    print(f"Report created: {report_file}")