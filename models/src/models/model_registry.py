"""
Model Registry for Cryptocurrency Forecasting Models

Manages model versions, metadata, and deployment:
- Version tracking
- Model comparison
- A/B testing support
- Production model selection
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    model_id: str
    model_type: str  # 'lightgbm', 'lstm', 'ensemble'
    version: str
    crypto_id: str
    trained_at: datetime
    model_path: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    status: str  # 'development', 'staging', 'production', 'archived'
    tags: List[str]
    description: Optional[str] = None


class ModelRegistry:
    """
    Central registry for all trained models.
    
    Provides versioning, comparison, and production deployment management.
    """
    
    def __init__(self, registry_dir: str = "models/artifacts"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "model_registry.json"
        self.models: Dict[str, ModelMetadata] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                
                for model_id, model_data in data.items():
                    # Convert datetime string back to datetime
                    model_data['trained_at'] = datetime.fromisoformat(model_data['trained_at'])
                    self.models[model_id] = ModelMetadata(**model_data)
            
            print(f"Loaded {len(self.models)} models from registry")
        else:
            print("No existing registry found, starting fresh")
    
    def _save_registry(self) -> None:
        """Save registry to disk"""
        data = {}
        
        for model_id, metadata in self.models.items():
            # Convert to dict and handle datetime
            model_dict = asdict(metadata)
            model_dict['trained_at'] = metadata.trained_at.isoformat()
            data[model_id] = model_dict
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model_type: str,
        version: str,
        crypto_id: str,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        status: str = 'development'
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_type: Type of model
            version: Version string (e.g., '1.0.0')
            crypto_id: Cryptocurrency ID
            model_path: Path to saved model file
            metrics: Performance metrics
            config: Model configuration
            tags: Optional tags for categorization
            description: Optional description
            status: Model status
            
        Returns:
            Model ID
        """
        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{crypto_id}_{model_type}_v{version}_{timestamp}"
        
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            crypto_id=crypto_id,
            trained_at=datetime.now(),
            model_path=model_path,
            metrics=metrics,
            config=config,
            status=status,
            tags=tags or [],
            description=description
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        
        print(f"Model registered: {model_id}")
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        crypto_id: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            crypto_id: Filter by cryptocurrency
            model_type: Filter by model type
            status: Filter by status
            
        Returns:
            List of matching models
        """
        models = list(self.models.values())
        
        if crypto_id:
            models = [m for m in models if m.crypto_id == crypto_id]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by training date (newest first)
        models.sort(key=lambda m: m.trained_at, reverse=True)
        
        return models
    
    def get_production_model(
        self,
        crypto_id: str,
        model_type: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get the production model for a cryptocurrency.
        
        Args:
            crypto_id: Cryptocurrency ID
            model_type: Optional model type filter
            
        Returns:
            Production model metadata
        """
        models = self.list_models(
            crypto_id=crypto_id,
            model_type=model_type,
            status='production'
        )
        
        if len(models) == 0:
            return None
        
        # Return the most recent production model
        return models[0]
    
    def promote_to_production(
        self,
        model_id: str,
        demote_existing: bool = True
    ) -> None:
        """
        Promote a model to production status.
        
        Args:
            model_id: Model ID to promote
            demote_existing: Whether to demote existing production models
        """
        model = self.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model not found: {model_id}")
        
        # Demote existing production models for same crypto
        if demote_existing:
            existing_prod = self.get_production_model(
                model.crypto_id,
                model.model_type
            )
            
            if existing_prod:
                existing_prod.status = 'archived'
                print(f"Demoted {existing_prod.model_id} to archived")
        
        # Promote new model
        model.status = 'production'
        self._save_registry()
        
        print(f"Promoted {model_id} to production")
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str = 'mape'
    ) -> pd.DataFrame:
        """
        Compare multiple models by a metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare ('mape', 'rmse', 'r2_score', etc.)
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_id in model_ids:
            model = self.get_model(model_id)
            if model:
                comparison_data.append({
                    'model_id': model_id,
                    'type': model.model_type,
                    'version': model.version,
                    'crypto': model.crypto_id,
                    'status': model.status,
                    'trained_at': model.trained_at,
                    metric: model.metrics.get(metric, np.nan)
                })
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty and metric in df.columns:
            df = df.sort_values(metric)
        
        return df
    
    def get_best_model(
        self,
        crypto_id: str,
        metric: str = 'mape',
        minimize: bool = True,
        status_filter: Optional[List[str]] = None
    ) -> Optional[ModelMetadata]:
        """
        Get the best performing model for a cryptocurrency.
        
        Args:
            crypto_id: Cryptocurrency ID
            metric: Metric to optimize
            minimize: Whether lower is better (True for MAPE/RMSE, False for R²)
            status_filter: Optional list of acceptable statuses
            
        Returns:
            Best model metadata
        """
        models = self.list_models(crypto_id=crypto_id)
        
        if status_filter:
            models = [m for m in models if m.status in status_filter]
        
        if len(models) == 0:
            return None
        
        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.metrics]
        
        if len(models_with_metric) == 0:
            return None
        
        # Find best
        if minimize:
            best = min(models_with_metric, key=lambda m: m.metrics[metric])
        else:
            best = max(models_with_metric, key=lambda m: m.metrics[metric])
        
        return best
    
    def archive_old_models(
        self,
        keep_n_per_crypto: int = 3,
        keep_production: bool = True
    ) -> int:
        """
        Archive old models to save disk space.
        
        Args:
            keep_n_per_crypto: Keep N most recent models per crypto
            keep_production: Don't archive production models
            
        Returns:
            Number of models archived
        """
        archived_count = 0
        
        # Group by crypto_id
        crypto_groups: Dict[str, List[ModelMetadata]] = {}
        for model in self.models.values():
            if model.crypto_id not in crypto_groups:
                crypto_groups[model.crypto_id] = []
            crypto_groups[model.crypto_id].append(model)
        
        # Process each crypto
        for crypto_id, models in crypto_groups.items():
            # Sort by training date (newest first)
            models.sort(key=lambda m: m.trained_at, reverse=True)
            
            # Keep recent and production models
            for i, model in enumerate(models):
                should_keep = (
                    i < keep_n_per_crypto or
                    (keep_production and model.status == 'production')
                )
                
                if not should_keep and model.status != 'archived':
                    model.status = 'archived'
                    archived_count += 1
        
        self._save_registry()
        
        print(f"Archived {archived_count} models")
        
        return archived_count
    
    def generate_registry_report(self) -> Dict[str, Any]:
        """
        Generate summary report of registry.
        
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.models),
            'by_status': {},
            'by_type': {},
            'by_crypto': {}
        }
        
        # Count by status
        for model in self.models.values():
            report['by_status'][model.status] = report['by_status'].get(model.status, 0) + 1
            report['by_type'][model.model_type] = report['by_type'].get(model.model_type, 0) + 1
            report['by_crypto'][model.crypto_id] = report['by_crypto'].get(model.crypto_id, 0) + 1
        
        return report


# Global registry instance
model_registry = ModelRegistry()

