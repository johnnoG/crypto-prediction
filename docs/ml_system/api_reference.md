
# 📚 API Reference

*Generated API documentation for the ML system components*

## 🔧 Core Components


### 📁 src/visualization/documentation_generator.py

Comprehensive ML System Documentation Generator
Creates detailed documentation for the entire cryptocurrency prediction ML system

#### 🏷️ Class: `MLSystemDocumentationGenerator`

Generates comprehensive documentation for the ML system


### 📁 src/visualization/training_dashboard.py

Advanced Training Analytics Dashboard

Comprehensive visualization system for cryptocurrency prediction models:
- Real-time training progress monitoring
- Interactive hyperparameter optimization visualization
- Model architecture and weight analysis
- Performance comparison dashboards
- Training diagnostics and debugging tools
- Export capabilities for presentations and reports

#### 🏷️ Class: `HyperparameterVisualizationDashboard`

Advanced visualization dashboard for hyperparameter optimization results.

#### 🏷️ Class: `ModelWeightVisualizer`

Advanced visualization for model weights and architecture analysis.

#### 🏷️ Class: `TrainingProgressVisualizer`

Real-time training progress visualization and monitoring.


### 📁 src/visualization/launch_dashboards.py

Unified Dashboard Launcher
Provides a single entry point to launch all visualization and monitoring dashboards

#### 🏷️ Class: `DashboardManager`

Manages multiple visualization dashboards


### 📁 src/visualization/training_monitor.py

Real-time Training Progress Monitoring System
Provides live monitoring and visualization of model training progress

#### 🏷️ Class: `TrainingLogger`

Enhanced training logger with structured logging

#### 🏷️ Class: `TrainingMonitor`

Real-time training progress monitor with live updates


### 📁 src/visualization/hyperopt_dashboard.py

Comprehensive Hyperparameter Optimization Visualization Dashboard
Provides interactive visualization and analysis of Optuna optimization studies

#### 🏷️ Class: `HyperparameterOptimizationDashboard`

Interactive dashboard for hyperparameter optimization analysis


### 📁 src/visualization/model_inspector.py

Comprehensive Model Architecture and Weight Visualization System
Provides detailed analysis and visualization of model internals, weights, and architecture

#### 🏷️ Class: `ModelArchitectureVisualizer`

Visualizes model architecture and layer details

#### 🏷️ Class: `ModelInspectorDashboard`

Comprehensive model inspection dashboard

#### 🏷️ Class: `WeightAnalyzer`

Analyzes and visualizes model weights


### 📁 src/training/mlflow_integration.py

MLflow Integration for Cryptocurrency Prediction Models

Comprehensive experiment tracking and model management:
- Experiment tracking for all model types
- Automated hyperparameter logging
- Model registry with versioning
- Performance comparison dashboards
- Artifact management (models, plots, datasets)
- Model deployment utilities

#### 🏷️ Class: `MLflowExperimentTracker`

Comprehensive MLflow experiment tracking for cryptocurrency models.

Features:
- Automatic experiment setup
- Model-specific parameter logging
- Performance metrics tracking
- Artifact management
- Model registry integration

#### 🏷️ Class: `MLflowModelRegistry`

MLflow Model Registry management utilities.


### 📁 src/models/transformer_model.py

Transformer Model for Cryptocurrency Price Forecasting

Modern attention-based architecture featuring:
- Multi-head self-attention for temporal pattern recognition
- Positional encoding for time series data
- Causal masking for proper temporal modeling
- Multi-step forecasting capabilities
- Uncertainty quantification through ensemble methods

#### 🏷️ Class: `MultiHeadSelfAttention`

Multi-head self-attention layer with causal masking for time series.

**Methods:**

- `from_config(config)`
  - Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the
same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype
config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

Args:
    config: A Python dictionary, typically the output of `get_config`.

Returns:
    An operation instance.

#### 🏷️ Class: `PositionalEncoding`

Positional encoding layer for time series Transformer.

Adds learnable position embeddings to capture temporal structure.

**Methods:**

- `from_config(config)`
  - Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the
same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype
config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

Args:
    config: A Python dictionary, typically the output of `get_config`.

Returns:
    An operation instance.

#### 🏷️ Class: `TransformerBlock`

Transformer encoder block with self-attention and feed-forward network.

**Methods:**

- `from_config(config)`
  - Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the
same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype
config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

Args:
    config: A Python dictionary, typically the output of `get_config`.

Returns:
    An operation instance.

#### 🏷️ Class: `TransformerForecaster`

Transformer-based cryptocurrency price forecaster.

Uses attention mechanisms to capture long-term dependencies
and complex temporal patterns in price movements.


### 📁 src/models/enhanced_lstm.py

Enhanced LSTM Model for Cryptocurrency Price Forecasting

Advanced LSTM architecture featuring:
- Bidirectional LSTM with residual connections
- Enhanced attention mechanism with learnable position encoding
- Multi-step forecasting capabilities (1, 7, 30 days)
- Uncertainty quantification through Monte Carlo dropout
- Teacher forcing for training stability
- Gradient clipping and advanced regularization

#### 🏷️ Class: `AttentionLayer`

Enhanced attention mechanism for LSTM.

Computes attention weights over LSTM hidden states to focus
on the most relevant time steps for prediction.

**Methods:**

- `from_config(config)`
  - Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the
same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype
config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

Args:
    config: A Python dictionary, typically the output of `get_config`.

Returns:
    An operation instance.

#### 🏷️ Class: `EnhancedLSTMForecaster`

Enhanced LSTM-based cryptocurrency price forecaster.

Features multi-step prediction, advanced attention,
uncertainty quantification, and robust training.

#### 🏷️ Class: `ResidualLSTMCell`

LSTM cell with residual connections for better gradient flow.

**Methods:**

- `from_config(config)`
  - Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the
same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype
config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

Args:
    config: A Python dictionary, typically the output of `get_config`.

Returns:
    An operation instance.

