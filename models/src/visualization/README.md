# 📊 ML System Visualization & Monitoring Suite

A comprehensive visualization and monitoring system for the cryptocurrency prediction ML pipeline, providing real-time insights into training progress, hyperparameter optimization, and model performance.

## 🌟 Features

### 🔍 Real-Time Training Monitor (`training_monitor.py`)
- **Live Training Progress**: Real-time visualization of training metrics
- **Session Management**: Track multiple training sessions simultaneously
- **Interactive Dashboard**: Web-based monitoring with auto-refresh
- **MLflow Integration**: Automatic experiment logging
- **Performance Metrics**: Loss curves, accuracy, learning rate tracking

### 🎯 Hyperparameter Optimization Dashboard (`hyperopt_dashboard.py`)
- **Optuna Integration**: Visualize Bayesian optimization studies
- **Parameter Importance**: Interactive parameter analysis
- **Trial History**: Complete optimization timeline
- **Best Trials Analysis**: Top-performing configurations
- **Parallel Coordinates**: Multi-dimensional parameter relationships

### 🏗️ Model Architecture Inspector (`model_inspector.py`)
- **Architecture Visualization**: Interactive model graphs
- **Weight Analysis**: Layer weight distributions and heatmaps
- **Parameter Statistics**: Comprehensive model statistics
- **TensorFlow/PyTorch Support**: Multi-framework compatibility
- **Model Comparison**: Side-by-side architecture analysis

### 📈 Training Analytics Dashboard (`training_dashboard.py`)
- **Comprehensive Metrics**: Advanced training visualizations
- **Model Performance**: Multi-metric comparison charts
- **Training History**: Historical performance tracking
- **Interactive Plots**: Plotly-based interactive charts
- **Export Capabilities**: Save visualizations and reports

### 📚 Documentation Generator (`documentation_generator.py`)
- **Auto-Generated Docs**: Complete system documentation
- **API Reference**: Detailed class and method documentation
- **Usage Examples**: Practical implementation guides
- **Project Analysis**: Automated codebase analysis

### 🚀 Unified Dashboard Launcher (`launch_dashboards.py`)
- **One-Click Launch**: Start all dashboards simultaneously
- **Port Management**: Automatic port allocation
- **Dependency Checking**: Verify required packages
- **Status Monitoring**: Track dashboard health
- **Graceful Shutdown**: Clean termination of all services

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch All Dashboards

```bash
python models/src/visualization/launch_dashboards.py
```

### 3. Access Dashboards

- **Training Monitor**: http://localhost:8050
- **Hyperparameter Dashboard**: http://localhost:8051
- **Model Inspector**: http://localhost:8052
- **MLflow UI**: http://localhost:5000

## 📋 Individual Dashboard Usage

### Training Monitor

```python
from models.src.visualization.training_monitor import TrainingMonitor, create_monitoring_dashboard

# Initialize monitor
monitor = TrainingMonitor()
monitor.start_monitoring()

# Register training session
session_id = monitor.register_training_session(
    session_id="transformer_training_001",
    model_type="Transformer",
    config={"learning_rate": 0.001, "batch_size": 32}
)

# Update metrics during training
monitor.update_metrics(
    session_id=session_id,
    epoch=1,
    metrics={"loss": 0.5, "val_loss": 0.6, "accuracy": 0.85}
)

# Launch dashboard
create_monitoring_dashboard(monitor, port=8050)
```

### Hyperparameter Dashboard

```python
from models.src.visualization.hyperopt_dashboard import HyperparameterOptimizationDashboard

# Initialize dashboard
dashboard = HyperparameterOptimizationDashboard(
    studies_dir="models/optuna_studies"
)

# Run dashboard
dashboard.run_dashboard(port=8051)
```

### Model Inspector

```python
from models.src.visualization.model_inspector import ModelInspectorDashboard

# Initialize inspector
inspector = ModelInspectorDashboard(
    models_dir="models/artifacts"
)

# Run dashboard
inspector.run_dashboard(port=8052)
```

## 🔧 Configuration

### Dashboard Ports

Default port configuration:
- Training Monitor: 8050
- Hyperparameter Dashboard: 8051
- Model Inspector: 8052
- Training Dashboard: 8053
- MLflow UI: 5000

### Customization

```python
# Custom port configuration
python launch_dashboards.py --port 9000

# Skip MLflow UI
python launch_dashboards.py --no-mlflow

# Generate documentation only
python launch_dashboards.py --docs-only

# Check dependencies
python launch_dashboards.py --check-deps
```

## 📊 Dashboard Features

### Training Monitor Features
- ✅ Real-time metric updates
- ✅ Multiple session tracking
- ✅ Interactive metric plots
- ✅ Session health monitoring
- ✅ MLflow integration
- ✅ Auto-refresh capabilities

### Hyperparameter Dashboard Features
- ✅ Parameter importance ranking
- ✅ Optimization history
- ✅ Best trials analysis
- ✅ Parallel coordinate plots
- ✅ Study comparison
- ✅ Export reports

### Model Inspector Features
- ✅ Architecture visualization
- ✅ Weight distribution analysis
- ✅ Layer-wise statistics
- ✅ Model comparison
- ✅ Interactive heatmaps
- ✅ Parameter counting

## 🛠️ Technical Details

### Dependencies
- **Core**: `dash`, `plotly`, `pandas`, `numpy`
- **ML Frameworks**: `tensorflow`, `torch` (optional)
- **Optimization**: `optuna` (optional)
- **Experiment Tracking**: `mlflow` (optional)
- **Web Framework**: `dash-bootstrap-components`

### Architecture
```
visualization/
├── training_monitor.py      # Real-time training monitoring
├── hyperopt_dashboard.py    # Hyperparameter optimization analysis
├── model_inspector.py       # Model architecture visualization
├── training_dashboard.py    # Comprehensive training analytics
├── documentation_generator.py # Auto-documentation system
├── launch_dashboards.py     # Unified dashboard launcher
└── README.md               # This file
```

### Data Flow
1. **Training Process** → **Monitoring System** → **Real-time Dashboard**
2. **Optuna Studies** → **Analysis Engine** → **Interactive Visualizations**
3. **Model Artifacts** → **Inspector** → **Architecture Graphs**
4. **System Components** → **Documentation Generator** → **Markdown Docs**

## 📈 Advanced Features

### Real-Time Monitoring
- Live metric streaming
- WebSocket connections
- Auto-refresh intervals
- Session state persistence

### Interactive Visualizations
- Plotly-based charts
- Hover information
- Zoom and pan capabilities
- Export functionality

### Multi-Model Support
- TensorFlow/Keras models
- PyTorch models
- Ensemble architectures
- Custom model formats

### Production Ready
- Error handling and recovery
- Logging and debugging
- Performance monitoring
- Graceful degradation

## 🔍 Monitoring Capabilities

### Training Metrics
- Loss curves (training/validation)
- Accuracy progression
- Learning rate schedules
- Custom metrics tracking

### System Health
- Memory usage monitoring
- GPU utilization tracking
- Training speed analysis
- Error rate monitoring

### Model Performance
- Prediction accuracy
- Uncertainty quantification
- Feature importance
- Model interpretability

## 📚 Documentation

The system automatically generates comprehensive documentation including:

- **System Overview**: Architecture and component descriptions
- **API Reference**: Detailed class and method documentation
- **Usage Examples**: Practical implementation guides
- **Best Practices**: Recommended usage patterns

Access documentation at: `docs/ml_system/README.md`

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   python launch_dashboards.py --port 9000
   ```

2. **Missing Dependencies**
   ```bash
   pip install dash plotly optuna mlflow
   ```

3. **Model Loading Errors**
   - Ensure model files are in `models/artifacts/`
   - Check TensorFlow/PyTorch versions

4. **Dashboard Not Loading**
   - Check terminal for error messages
   - Verify all dependencies installed
   - Try different port numbers

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH="${PYTHONPATH}:."
python -m models.src.visualization.launch_dashboards --check-deps
```

## 🤝 Contributing

To extend the visualization system:

1. **Add New Dashboard**: Create new dashboard class inheriting from base patterns
2. **Extend Metrics**: Add custom metric tracking in `training_monitor.py`
3. **Custom Visualizations**: Implement new chart types in respective modules
4. **Documentation**: Update auto-generation templates

## 📄 License

Part of the Cryptocurrency Prediction ML System. See project root for license details.

---

*For complete system documentation, see: `docs/ml_system/README.md`*