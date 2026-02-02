"""
Comprehensive ML System Documentation Generator
Creates detailed documentation for the entire cryptocurrency prediction ML system
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import inspect
import importlib.util

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSystemDocumentationGenerator:
    """Generates comprehensive documentation for the ML system"""

    def __init__(self,
                 project_root: str = ".",
                 output_dir: str = "docs/ml_system",
                 include_code: bool = True):
        """
        Initialize documentation generator

        Args:
            project_root: Root directory of the project
            output_dir: Output directory for documentation
            include_code: Whether to include code snippets in documentation
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_code = include_code

        # Documentation structure
        self.documentation = {
            'system_overview': {},
            'data_pipeline': {},
            'models': {},
            'training': {},
            'visualization': {},
            'deployment': {},
            'api_reference': {},
            'usage_examples': {}
        }

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure and components"""
        structure = {
            'project_name': 'Cryptocurrency Prediction ML System',
            'creation_date': datetime.now().isoformat(),
            'components': {},
            'dependencies': {},
            'file_tree': {}
        }

        # Analyze main components
        models_dir = self.project_root / "models"
        if models_dir.exists():
            structure['components']['models'] = self._analyze_models_directory(models_dir)

        data_dir = self.project_root / "data"
        if data_dir.exists():
            structure['components']['data'] = self._analyze_data_directory(data_dir)

        # Analyze requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            structure['dependencies'] = self._analyze_dependencies(requirements_file)

        return structure

    def _analyze_models_directory(self, models_dir: Path) -> Dict[str, Any]:
        """Analyze models directory structure"""
        models_info = {
            'directory': str(models_dir),
            'subdirectories': {},
            'model_files': [],
            'implementation_files': {}
        }

        for item in models_dir.rglob("*"):
            if item.is_file():
                if item.suffix in ['.py']:
                    # Analyze Python files for model implementations
                    rel_path = item.relative_to(models_dir)
                    models_info['implementation_files'][str(rel_path)] = self._analyze_python_file(item)
                elif item.suffix in ['.h5', '.keras', '.pt', '.pth']:
                    # Found model artifacts
                    models_info['model_files'].append(str(item.relative_to(models_dir)))

        return models_info

    def _analyze_data_directory(self, data_dir: Path) -> Dict[str, Any]:
        """Analyze data directory structure"""
        data_info = {
            'directory': str(data_dir),
            'datasets': [],
            'total_size': 0,
            'file_types': {}
        }

        for item in data_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(data_dir)
                file_size = item.stat().st_size
                file_ext = item.suffix

                data_info['datasets'].append({
                    'path': str(rel_path),
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'type': file_ext
                })

                data_info['total_size'] += file_size
                data_info['file_types'][file_ext] = data_info['file_types'].get(file_ext, 0) + 1

        data_info['total_size_mb'] = round(data_info['total_size'] / (1024 * 1024), 2)
        return data_info

    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Python file for classes and functions"""
        info = {
            'classes': [],
            'functions': [],
            'imports': [],
            'docstring': None,
            'line_count': 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            info['line_count'] = len(content.split('\n'))

            # Load module and inspect
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Extract classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module.__name__:  # Only classes defined in this module
                        class_info = {
                            'name': name,
                            'docstring': inspect.getdoc(obj),
                            'methods': []
                        }

                        # Extract methods
                        for method_name, method_obj in inspect.getmembers(obj, inspect.ismethod):
                            if not method_name.startswith('_'):  # Skip private methods
                                method_info = {
                                    'name': method_name,
                                    'docstring': inspect.getdoc(method_obj),
                                    'signature': str(inspect.signature(method_obj))
                                }
                                class_info['methods'].append(method_info)

                        info['classes'].append(class_info)

                # Extract functions
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if obj.__module__ == module.__name__:  # Only functions defined in this module
                        func_info = {
                            'name': name,
                            'docstring': inspect.getdoc(obj),
                            'signature': str(inspect.signature(obj))
                        }
                        info['functions'].append(func_info)

                # Module docstring
                info['docstring'] = inspect.getdoc(module)

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")

        return info

    def _analyze_dependencies(self, requirements_file: Path) -> Dict[str, Any]:
        """Analyze project dependencies"""
        deps_info = {
            'total_packages': 0,
            'categories': {},
            'packages': []
        }

        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '>=' in line:
                        package, version = line.split('>=')
                        operator = '>='
                    elif '==' in line:
                        package, version = line.split('==')
                        operator = '=='
                    else:
                        package = line
                        version = 'latest'
                        operator = ''

                    package = package.strip()
                    version = version.strip() if version != 'latest' else version

                    # Categorize packages
                    category = self._categorize_package(package)

                    deps_info['packages'].append({
                        'name': package,
                        'version': version,
                        'operator': operator,
                        'category': category
                    })

                    deps_info['categories'][category] = deps_info['categories'].get(category, 0) + 1
                    deps_info['total_packages'] += 1

        except Exception as e:
            logger.warning(f"Failed to analyze dependencies: {e}")

        return deps_info

    def _categorize_package(self, package_name: str) -> str:
        """Categorize package by its purpose"""
        ml_packages = ['tensorflow', 'keras', 'torch', 'scikit-learn', 'lightgbm', 'xgboost', 'optuna', 'mlflow']
        data_packages = ['pandas', 'numpy', 'polars', 'pyarrow']
        viz_packages = ['matplotlib', 'plotly', 'seaborn', 'dash']
        web_packages = ['fastapi', 'uvicorn', 'httpx', 'requests', 'flask', 'django']
        db_packages = ['sqlalchemy', 'redis', 'psycopg2', 'alembic']
        stats_packages = ['statsmodels', 'scipy']
        finance_packages = ['ta', 'ta-lib', 'yfinance']

        package_lower = package_name.lower()

        if any(pkg in package_lower for pkg in ml_packages):
            return 'Machine Learning'
        elif any(pkg in package_lower for pkg in data_packages):
            return 'Data Processing'
        elif any(pkg in package_lower for pkg in viz_packages):
            return 'Visualization'
        elif any(pkg in package_lower for pkg in web_packages):
            return 'Web Framework'
        elif any(pkg in package_lower for pkg in db_packages):
            return 'Database'
        elif any(pkg in package_lower for pkg in stats_packages):
            return 'Statistics'
        elif any(pkg in package_lower for pkg in finance_packages):
            return 'Financial Analysis'
        else:
            return 'Utilities'

    def generate_system_overview_doc(self) -> str:
        """Generate system overview documentation"""
        structure = self.analyze_project_structure()

        doc = f"""
# 🚀 Cryptocurrency Prediction ML System Overview

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📋 System Description

This is a comprehensive machine learning system designed for cryptocurrency price prediction using advanced deep learning models, ensemble methods, and real-time analytics.

## 🏗️ System Architecture

The system consists of several integrated components:

### 📊 Data Pipeline
- **Data Sources**: Kaggle cryptocurrency datasets, real-time market data
- **Feature Engineering**: 50+ technical indicators, market regime detection
- **Data Storage**: PostgreSQL for structured data, Redis for caching

### 🤖 Machine Learning Models
- **Transformer Model**: Multi-head self-attention for temporal patterns
- **Enhanced LSTM**: Bidirectional LSTM with attention mechanisms
- **Advanced Ensemble**: Dynamic model combination with market regime awareness
- **Traditional Models**: LightGBM, XGBoost for baseline comparisons

### 🔬 Experiment Management
- **MLflow Integration**: Experiment tracking and model versioning
- **Optuna Optimization**: Bayesian hyperparameter tuning
- **Production Pipeline**: Automated training and deployment

### 📈 Visualization & Monitoring
- **Training Dashboard**: Real-time training progress monitoring
- **Hyperparameter Dashboard**: Interactive optimization analysis
- **Model Inspector**: Architecture and weight visualization
- **Analytics Dashboard**: Performance metrics and predictions

## 📦 Dependencies Overview

**Total Packages**: {structure['dependencies'].get('total_packages', 0)}

### Package Categories:
"""

        # Add dependency categories
        categories = structure['dependencies'].get('categories', {})
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            doc += f"- **{category}**: {count} packages\n"

        doc += f"""

## 📁 Project Structure

```
crypto-prediction/
├── data/                    # Dataset storage
├── models/                  # ML model implementations
│   ├── src/
│   │   ├── models/         # Model architectures
│   │   ├── training/       # Training pipelines
│   │   ├── visualization/  # Dashboards and plots
│   │   └── utils/          # Utility functions
│   └── artifacts/          # Trained model files
├── notebooks/              # Jupyter notebooks for analysis
├── docs/                   # Documentation
└── requirements.txt        # Dependencies
```

## 🎯 Key Features

### Advanced ML Models
- **Multi-head Attention**: Transformer architecture for long-term dependencies
- **Uncertainty Quantification**: Monte Carlo dropout for confidence intervals
- **Market Regime Detection**: Dynamic adaptation to market conditions
- **Multi-step Forecasting**: Predictions for multiple time horizons

### Production-Ready Infrastructure
- **Automated Training**: Scheduled retraining with performance monitoring
- **Model Versioning**: MLflow model registry with deployment tracking
- **Real-time Monitoring**: Live training progress and system metrics
- **Interactive Dashboards**: Web-based analysis and visualization tools

### Comprehensive Analytics
- **Training Visualization**: Real-time loss curves and metrics
- **Hyperparameter Analysis**: Interactive optimization results
- **Model Inspection**: Architecture graphs and weight distributions
- **Performance Monitoring**: Model accuracy and prediction confidence

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Training Pipeline**:
   ```python
   from models.src.training.production_pipeline import ProductionPipeline
   pipeline = ProductionPipeline()
   pipeline.run_complete_pipeline()
   ```

3. **Launch Monitoring Dashboards**:
   ```python
   # Training Monitor
   from models.src.visualization.training_monitor import create_monitoring_dashboard
   create_monitoring_dashboard(port=8050)

   # Hyperparameter Dashboard
   from models.src.visualization.hyperopt_dashboard import HyperparameterOptimizationDashboard
   dashboard = HyperparameterOptimizationDashboard()
   dashboard.run_dashboard(port=8051)

   # Model Inspector
   from models.src.visualization.model_inspector import ModelInspectorDashboard
   inspector = ModelInspectorDashboard()
   inspector.run_dashboard(port=8052)
   ```

## 📊 Model Performance

The system achieves state-of-the-art performance through:
- **Ensemble Learning**: Combining multiple model architectures
- **Adaptive Strategies**: Dynamic model selection based on market conditions
- **Uncertainty Quantification**: Reliable confidence intervals
- **Continuous Learning**: Automated retraining on new data

## 🔮 Future Enhancements

- **Real-time Data Streaming**: Live market data integration
- **Advanced Ensemble Methods**: Meta-learning and stacking
- **Explainable AI**: SHAP values and attention visualization
- **Cloud Deployment**: Kubernetes orchestration and auto-scaling
"""

        return doc

    def generate_api_reference(self) -> str:
        """Generate API reference documentation"""
        structure = self.analyze_project_structure()

        doc = """
# 📚 API Reference

*Generated API documentation for the ML system components*

## 🔧 Core Components

"""

        # Document models
        models_info = structure['components'].get('models', {})
        implementation_files = models_info.get('implementation_files', {})

        for file_path, file_info in implementation_files.items():
            if file_info.get('classes'):
                doc += f"\n### 📁 {file_path}\n\n"

                if file_info.get('docstring'):
                    doc += f"{file_info['docstring']}\n\n"

                for class_info in file_info['classes']:
                    doc += f"#### 🏷️ Class: `{class_info['name']}`\n\n"

                    if class_info.get('docstring'):
                        doc += f"{class_info['docstring']}\n\n"

                    if class_info.get('methods'):
                        doc += "**Methods:**\n\n"
                        for method in class_info['methods']:
                            doc += f"- `{method['name']}{method['signature']}`\n"
                            if method.get('docstring'):
                                doc += f"  - {method['docstring']}\n"
                        doc += "\n"

        return doc

    def generate_usage_examples(self) -> str:
        """Generate usage examples documentation"""
        doc = """
# 💡 Usage Examples

*Practical examples for using the ML system components*

## 🏃‍♂️ Quick Start Examples

### 1. Training a Transformer Model

```python
from models.src.models.transformer_model import TransformerForecaster
import numpy as np

# Initialize model
config = {
    'model_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'dropout_rate': 0.1,
    'learning_rate': 0.001
}

transformer = TransformerForecaster(config=config)

# Prepare data (example)
X_train = np.random.randn(1000, 60, 10)  # (samples, timesteps, features)
y_train = np.random.randn(1000, 3)       # (samples, prediction_horizons)

# Train model
transformer.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Make predictions
predictions = transformer.predict(X_train[:10])
print(f"Predictions shape: {predictions.shape}")
```

### 2. Running Hyperparameter Optimization

```python
from models.src.training.hyperopt_pipeline import HyperparameterOptimizer
import optuna

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    model_type='transformer',
    n_trials=100,
    study_name='crypto_transformer_opt'
)

# Run optimization
best_params = optimizer.optimize()
print(f"Best parameters: {best_params}")

# Load study for analysis
study = optuna.load_study(
    study_name='crypto_transformer_opt',
    storage=optimizer.storage_url
)
print(f"Best value: {study.best_value}")
```

### 3. Using the Advanced Ensemble

```python
from models.src.models.advanced_ensemble import AdvancedEnsemble
import pandas as pd

# Initialize ensemble
ensemble = AdvancedEnsemble()

# Prepare data
data = pd.read_parquet("data/processed/crypto_features.parquet")
X, y = ensemble.prepare_data(data)

# Train ensemble
ensemble.fit(X, y, validation_split=0.2)

# Get predictions with uncertainty
predictions, uncertainties = ensemble.predict_with_uncertainty(X[:10])
print(f"Predictions: {predictions}")
print(f"Uncertainties: {uncertainties}")

# Analyze market regime
regime = ensemble.detect_regime(data['close'].values[-100:])
print(f"Current market regime: {regime}")
```

### 4. Production Training Pipeline

```python
from models.src.training.production_pipeline import ProductionPipeline

# Initialize pipeline
pipeline = ProductionPipeline(
    data_path="data/processed/crypto_features.parquet",
    models_to_train=['transformer', 'enhanced_lstm', 'ensemble'],
    enable_mlflow=True,
    enable_monitoring=True
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()
print(f"Training completed. Results: {results}")

# Get best model
best_model = pipeline.get_best_model()
print(f"Best model: {best_model}")
```

### 5. Starting Monitoring Dashboards

```python
# Training Progress Monitor
from models.src.visualization.training_monitor import TrainingMonitor, create_monitoring_dashboard

monitor = TrainingMonitor()
monitor.start_monitoring()

# Register a training session
session_id = monitor.register_training_session(
    session_id="transformer_training_001",
    model_type="Transformer",
    config={"learning_rate": 0.001, "batch_size": 32}
)

# Start dashboard
create_monitoring_dashboard(monitor, port=8050)
```

```python
# Hyperparameter Optimization Dashboard
from models.src.visualization.hyperopt_dashboard import HyperparameterOptimizationDashboard

dashboard = HyperparameterOptimizationDashboard(
    studies_dir="models/optuna_studies"
)
dashboard.run_dashboard(port=8051)
```

```python
# Model Architecture Inspector
from models.src.visualization.model_inspector import ModelInspectorDashboard

inspector = ModelInspectorDashboard(
    models_dir="models/artifacts"
)
inspector.run_dashboard(port=8052)
```

## 🔧 Advanced Usage

### Custom Model Configuration

```python
# Custom Transformer configuration
transformer_config = {
    'model_dim': 512,           # Model dimension
    'num_heads': 16,            # Number of attention heads
    'num_layers': 8,            # Number of transformer layers
    'ff_dim': 2048,             # Feedforward dimension
    'dropout_rate': 0.15,       # Dropout rate
    'learning_rate': 0.0005,    # Learning rate
    'warmup_steps': 1000,       # Learning rate warmup
    'max_position_encoding': 1000,  # Maximum sequence length
    'prediction_horizons': [1, 7, 30]  # Prediction horizons
}

# Custom LSTM configuration
lstm_config = {
    'lstm_units': [256, 128, 64],    # LSTM layer sizes
    'use_bidirectional': True,        # Bidirectional LSTM
    'use_attention': True,            # Attention mechanism
    'dropout_rates': [0.2, 0.3, 0.1], # Dropout per layer
    'learning_rate': 0.001,           # Learning rate
    'prediction_horizons': [1, 7, 30] # Prediction horizons
}

# Custom ensemble configuration
ensemble_config = {
    'models': ['transformer', 'enhanced_lstm', 'lightgbm'],
    'ensemble_method': 'meta_learning',
    'meta_model_type': 'lightgbm',
    'dynamic_weighting': True,
    'market_regime_detection': True,
    'uncertainty_quantification': True
}
```

### MLflow Integration

```python
from models.src.training.mlflow_integration import MLflowExperimentTracker

# Initialize MLflow tracker
tracker = MLflowExperimentTracker(
    experiment_name="crypto_prediction_experiments",
    tracking_uri="http://localhost:5000"
)

# Start experiment
with tracker.start_run(run_name="transformer_experiment"):
    # Log parameters
    tracker.log_params(transformer_config)

    # Train model
    model = TransformerForecaster(config=transformer_config)
    model.fit(X_train, y_train)

    # Log metrics
    metrics = model.evaluate(X_test, y_test)
    tracker.log_metrics(metrics)

    # Log model
    tracker.log_model_tensorflow(model.model, "transformer")

    # Log artifacts
    tracker.log_artifact("config.json")
```

### Data Pipeline Integration

```python
# Load and prepare data
from models.src.utils.data_loader import CryptoDataLoader
from models.src.utils.feature_engineering import AdvancedFeatureEngineer

# Initialize data loader
loader = CryptoDataLoader(data_path="data/raw/crypto_data.parquet")
data = loader.load_data()

# Feature engineering
feature_engineer = AdvancedFeatureEngineer()
features = feature_engineer.create_features(data)

# Prepare for training
X, y = feature_engineer.prepare_training_data(
    features,
    target_column='close',
    sequence_length=60,
    prediction_horizons=[1, 7, 30]
)

print(f"Training data shape: {X.shape}")
print(f"Target data shape: {y.shape}")
```

## 🎯 Best Practices

### 1. Model Training
- Use validation splits for hyperparameter tuning
- Monitor training progress with MLflow
- Implement early stopping to prevent overfitting
- Use ensemble methods for better performance

### 2. Data Handling
- Ensure proper feature scaling and normalization
- Use time-aware cross-validation for time series
- Handle missing values appropriately
- Monitor data drift in production

### 3. Production Deployment
- Version your models with MLflow
- Implement proper logging and monitoring
- Use uncertainty quantification for risk assessment
- Regular model retraining on new data

### 4. Performance Monitoring
- Track model performance metrics over time
- Monitor prediction accuracy and calibration
- Set up alerts for model degradation
- Analyze model explanations regularly
"""

        return doc

    def generate_complete_documentation(self) -> Dict[str, str]:
        """Generate complete system documentation"""
        docs = {
            'system_overview': self.generate_system_overview_doc(),
            'api_reference': self.generate_api_reference(),
            'usage_examples': self.generate_usage_examples()
        }

        # Save documentation files
        for doc_type, content in docs.items():
            output_file = self.output_dir / f"{doc_type}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Generated {doc_type} documentation: {output_file}")

        # Generate index file
        index_content = self._generate_index_file()
        index_file = self.output_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)

        logger.info(f"Complete documentation generated in: {self.output_dir}")
        return docs

    def _generate_index_file(self) -> str:
        """Generate documentation index file"""
        return f"""
# 📚 ML System Documentation

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

Welcome to the comprehensive documentation for the Cryptocurrency Prediction ML System.

## 📋 Documentation Index

### 🚀 [System Overview](system_overview.md)
Complete overview of the system architecture, components, and capabilities.

### 📚 [API Reference](api_reference.md)
Detailed API documentation for all classes and methods.

### 💡 [Usage Examples](usage_examples.md)
Practical examples and best practices for using the system.

## 🎯 Quick Navigation

### Core Components
- **Models**: Transformer, Enhanced LSTM, Advanced Ensemble
- **Training**: Production Pipeline, Hyperparameter Optimization
- **Visualization**: Training Monitor, Hyperopt Dashboard, Model Inspector
- **Infrastructure**: MLflow Integration, Data Pipeline

### Getting Started
1. Review the [System Overview](system_overview.md) for architecture understanding
2. Check [Usage Examples](usage_examples.md) for practical implementation
3. Refer to [API Reference](api_reference.md) for detailed method documentation

### Dashboards Access
- **Training Monitor**: http://localhost:8050
- **Hyperparameter Dashboard**: http://localhost:8051
- **Model Inspector**: http://localhost:8052

## 🔗 Related Resources

- **Project Repository**: Main codebase and implementations
- **Data Documentation**: Dataset descriptions and feature engineering
- **Model Artifacts**: Trained models and experiment results
- **Notebooks**: Jupyter notebooks for analysis and exploration

---

*This documentation is automatically generated and regularly updated to reflect the latest system state.*
"""

if __name__ == "__main__":
    # Generate documentation
    generator = MLSystemDocumentationGenerator(
        project_root="/Users/yonatanglanzman/src/crypto-prediction",
        output_dir="/Users/yonatanglanzman/src/crypto-prediction/docs/ml_system"
    )

    docs = generator.generate_complete_documentation()
    print("📚 Documentation generation completed!")
    print(f"📁 Documentation available at: {generator.output_dir}")
    print("\n📋 Generated files:")
    for doc_type in docs.keys():
        print(f"  - {doc_type}.md")
    print("  - README.md (index)")