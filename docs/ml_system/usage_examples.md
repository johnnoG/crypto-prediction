
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
