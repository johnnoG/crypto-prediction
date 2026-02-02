"""
Advanced Training Analytics Dashboard

Comprehensive visualization system for cryptocurrency prediction models:
- Real-time training progress monitoring
- Interactive hyperparameter optimization visualization
- Model architecture and weight analysis
- Performance comparison dashboards
- Training diagnostics and debugging tools
- Export capabilities for presentations and reports
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set style
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")


class TrainingProgressVisualizer:
    """
    Real-time training progress visualization and monitoring.
    """

    def __init__(self, output_dir: str = "training_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_data = {}
        self.live_plots = {}

    def create_training_dashboard(
        self,
        training_history: Dict[str, List[float]],
        model_type: str,
        model_config: Dict[str, Any],
        validation_scores: Optional[List[float]] = None
    ) -> str:
        """
        Create comprehensive training dashboard.

        Args:
            training_history: Training metrics history
            model_type: Type of model being trained
            model_config: Model configuration parameters
            validation_scores: Cross-validation scores

        Returns:
            Path to saved dashboard
        """
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_dashboard(
                training_history, model_type, model_config, validation_scores
            )

        print(f"🎨 Creating training dashboard for {model_type}")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Training & Validation Loss', 'Learning Rate Schedule', 'Metrics Evolution',
                'Loss Distribution', 'Gradient Norms', 'Model Configuration',
                'Cross-Validation Scores', 'Training Velocity', 'Performance Summary'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}, {"secondary_y": True}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "table"}],
                [{"type": "box"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )

        # 1. Training & Validation Loss
        self._add_loss_curves(fig, training_history, row=1, col=1)

        # 2. Learning Rate Schedule
        self._add_learning_rate_plot(fig, training_history, row=1, col=2)

        # 3. Metrics Evolution
        self._add_metrics_evolution(fig, training_history, row=1, col=3)

        # 4. Loss Distribution
        self._add_loss_distribution(fig, training_history, row=2, col=1)

        # 5. Gradient Norms (if available)
        self._add_gradient_norms(fig, training_history, row=2, col=2)

        # 6. Model Configuration Table
        self._add_config_table(fig, model_config, row=2, col=3)

        # 7. Cross-Validation Scores
        if validation_scores:
            self._add_cv_scores(fig, validation_scores, row=3, col=1)

        # 8. Training Velocity
        self._add_training_velocity(fig, training_history, row=3, col=2)

        # 9. Performance Summary
        self._add_performance_summary(fig, training_history, model_type, row=3, col=3)

        # Update layout
        fig.update_layout(
            title=f"{model_type.upper()} Training Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            height=1200,
            showlegend=True,
            template="plotly_white",
            font=dict(size=10)
        )

        # Save dashboard
        dashboard_path = self.output_dir / f"{model_type}_training_dashboard.html"
        fig.write_html(str(dashboard_path))

        print(f"✅ Training dashboard saved: {dashboard_path}")
        return str(dashboard_path)

    def _add_loss_curves(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add training and validation loss curves"""
        # Training loss
        if 'loss' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['loss']))),
                    y=training_history['loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=col
            )

        # Validation loss
        if 'val_loss' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['val_loss']))),
                    y=training_history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red', width=2)
                ),
                row=row, col=col
            )

        # Add early stopping point if available
        if 'val_loss' in training_history:
            best_epoch = np.argmin(training_history['val_loss'])
            fig.add_vline(
                x=best_epoch,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Best Epoch: {best_epoch}",
                row=row, col=col
            )

    def _add_learning_rate_plot(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add learning rate schedule"""
        if 'lr' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['lr']))),
                    y=training_history['lr'],
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='orange', width=2)
                ),
                row=row, col=col
            )

            # Use log scale for y-axis
            fig.update_yaxes(type="log", row=row, col=col)

    def _add_metrics_evolution(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add evolution of training metrics"""
        metric_colors = ['purple', 'green', 'brown', 'pink']
        color_idx = 0

        for metric, values in training_history.items():
            if metric not in ['loss', 'val_loss', 'lr'] and not metric.startswith('val_'):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        mode='lines',
                        name=metric,
                        line=dict(color=metric_colors[color_idx % len(metric_colors)], width=2)
                    ),
                    row=row, col=col
                )
                color_idx += 1

    def _add_loss_distribution(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add loss distribution histogram"""
        if 'loss' in training_history:
            fig.add_trace(
                go.Histogram(
                    x=training_history['loss'],
                    name='Training Loss Distribution',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=row, col=col
            )

    def _add_gradient_norms(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add gradient norms if available"""
        if 'gradient_norm' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['gradient_norm']))),
                    y=training_history['gradient_norm'],
                    mode='lines+markers',
                    name='Gradient Norm',
                    line=dict(color='red', width=2)
                ),
                row=row, col=col
            )
        else:
            # Placeholder text
            fig.add_annotation(
                text="Gradient norms not available",
                x=0.5, y=0.5,
                xref="x domain", yref="y domain",
                showarrow=False,
                row=row, col=col
            )

    def _add_config_table(self, fig, model_config: Dict[str, Any], row: int, col: int):
        """Add model configuration table"""
        # Prepare config data for table
        config_data = []
        for key, value in model_config.items():
            if isinstance(value, (int, float, str, bool)):
                config_data.append([key, str(value)])
            elif isinstance(value, list) and len(value) <= 5:
                config_data.append([key, str(value)])

        if config_data:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Parameter', 'Value'], fill_color='lightblue'),
                    cells=dict(
                        values=[[item[0] for item in config_data],
                               [item[1] for item in config_data]],
                        fill_color='white'
                    )
                ),
                row=row, col=col
            )

    def _add_cv_scores(self, fig, validation_scores: List[float], row: int, col: int):
        """Add cross-validation scores box plot"""
        fig.add_trace(
            go.Box(
                y=validation_scores,
                name='CV Scores',
                boxpoints='all'
            ),
            row=row, col=col
        )

    def _add_training_velocity(self, fig, training_history: Dict[str, List[float]], row: int, col: int):
        """Add training velocity (rate of improvement)"""
        if 'val_loss' in training_history:
            val_loss = training_history['val_loss']
            # Calculate velocity as negative gradient (improvement rate)
            velocity = [-np.gradient(val_loss)[i] for i in range(len(val_loss))]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(velocity))),
                    y=velocity,
                    mode='lines',
                    name='Training Velocity',
                    line=dict(color='darkgreen', width=2)
                ),
                row=row, col=col
            )

    def _add_performance_summary(self, fig, training_history: Dict[str, List[float]], model_type: str, row: int, col: int):
        """Add performance summary indicator"""
        # Calculate final performance metrics
        if 'val_loss' in training_history:
            final_val_loss = training_history['val_loss'][-1]
            best_val_loss = min(training_history['val_loss'])
            improvement = ((final_val_loss - best_val_loss) / best_val_loss) * 100

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=final_val_loss,
                    delta={'reference': best_val_loss},
                    title={'text': "Final Val Loss"},
                    gauge={
                        'axis': {'range': [None, max(training_history['val_loss'])]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, best_val_loss], 'color': "lightgray"},
                            {'range': [best_val_loss, final_val_loss], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': best_val_loss
                        }
                    }
                ),
                row=row, col=col
            )

    def _create_matplotlib_dashboard(
        self,
        training_history: Dict[str, List[float]],
        model_type: str,
        model_config: Dict[str, Any],
        validation_scores: Optional[List[float]] = None
    ) -> str:
        """Create dashboard using matplotlib as fallback"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_type.upper()} Training Dashboard', fontsize=16, fontweight='bold')

        # Training curves
        if 'loss' in training_history:
            axes[0, 0].plot(training_history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in training_history:
            axes[0, 0].plot(training_history['val_loss'], label='Validation Loss', linewidth=2)
            best_epoch = np.argmin(training_history['val_loss'])
            axes[0, 0].axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch: {best_epoch}')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        if 'lr' in training_history:
            axes[0, 1].plot(training_history['lr'], linewidth=2, color='orange')
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True, alpha=0.3)

        # Metrics evolution
        metric_count = 0
        for metric, values in training_history.items():
            if metric not in ['loss', 'val_loss', 'lr'] and not metric.startswith('val_'):
                axes[0, 2].plot(values, label=metric, linewidth=2)
                metric_count += 1
                if metric_count >= 4:  # Limit to avoid overcrowding
                    break
        axes[0, 2].set_title('Training Metrics Evolution')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Metric Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Loss distribution
        if 'loss' in training_history:
            axes[1, 0].hist(training_history['loss'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Training Loss Distribution')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # Cross-validation scores
        if validation_scores:
            axes[1, 1].boxplot(validation_scores)
            axes[1, 1].set_title('Cross-Validation Scores')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)

        # Training velocity
        if 'val_loss' in training_history:
            val_loss = training_history['val_loss']
            velocity = -np.gradient(val_loss)
            axes[1, 2].plot(velocity, linewidth=2, color='darkgreen')
            axes[1, 2].set_title('Training Velocity (Improvement Rate)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Velocity')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save dashboard
        dashboard_path = self.output_dir / f"{model_type}_training_dashboard_matplotlib.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(dashboard_path)

    def create_real_time_monitor(
        self,
        update_interval: int = 1000,
        max_epochs: int = 100
    ) -> None:
        """
        Create real-time training monitor (for Jupyter notebooks).

        Args:
            update_interval: Update interval in milliseconds
            max_epochs: Maximum number of epochs to display
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for real-time monitoring")
            return

        # Initialize plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Training data containers
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        def animate(frame):
            # Clear axes
            ax1.clear()
            ax2.clear()

            # Plot training curves
            if self.train_losses:
                ax1.plot(self.epochs, self.train_losses, label='Training Loss', color='blue', linewidth=2)
            if self.val_losses:
                ax1.plot(self.epochs, self.val_losses, label='Validation Loss', color='red', linewidth=2)

            ax1.set_title('Training Progress')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot learning rate
            if self.learning_rates:
                ax2.plot(self.epochs, self.learning_rates, color='orange', linewidth=2)
                ax2.set_yscale('log')

            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

        # Create animation
        anim = FuncAnimation(fig, animate, interval=update_interval, cache_frame_data=False)
        plt.show()

        print("Real-time monitor started. Use update_monitor() to add data points.")

    def update_monitor(self, epoch: int, train_loss: float, val_loss: float = None, lr: float = None):
        """Update real-time monitor with new data point"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)

        if lr is not None:
            self.learning_rates.append(lr)


class HyperparameterVisualizationDashboard:
    """
    Advanced visualization dashboard for hyperparameter optimization results.
    """

    def __init__(self, output_dir: str = "hyperopt_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_optuna_dashboard(
        self,
        study: 'optuna.Study',
        model_type: str
    ) -> str:
        """
        Create comprehensive Optuna optimization dashboard.

        Args:
            study: Optuna study object
            model_type: Type of model being optimized

        Returns:
            Path to saved dashboard
        """
        if not PLOTLY_AVAILABLE or not OPTUNA_AVAILABLE:
            return self._create_matplotlib_hyperopt_dashboard(study, model_type)

        print(f"🎨 Creating hyperparameter optimization dashboard for {model_type}")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Optimization History', 'Parameter Importance', 'Parallel Coordinate Plot',
                'Parameter Distribution', 'Convergence Analysis', 'Trial Timeline',
                'Best Trial Analysis', 'Parameter Correlation', 'Search Space Coverage'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}, {"type": "parcoords"}],
                [{"type": "histogram"}, {"secondary_y": True}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "heatmap"}, {"type": "scatter"}]
            ]
        )

        # 1. Optimization History
        self._add_optimization_history(fig, study, row=1, col=1)

        # 2. Parameter Importance
        self._add_parameter_importance(fig, study, row=1, col=2)

        # 3. Parallel Coordinate Plot
        self._add_parallel_coordinates(fig, study, row=1, col=3)

        # 4. Parameter Distributions
        self._add_parameter_distributions(fig, study, row=2, col=1)

        # 5. Convergence Analysis
        self._add_convergence_analysis(fig, study, row=2, col=2)

        # 6. Trial Timeline
        self._add_trial_timeline(fig, study, row=2, col=3)

        # 7. Best Trial Analysis
        self._add_best_trial_table(fig, study, row=3, col=1)

        # 8. Parameter Correlation
        self._add_parameter_correlation(fig, study, row=3, col=2)

        # 9. Search Space Coverage
        self._add_search_space_coverage(fig, study, row=3, col=3)

        # Update layout
        fig.update_layout(
            title=f"{model_type.upper()} Hyperparameter Optimization Dashboard",
            height=1400,
            showlegend=True,
            template="plotly_white"
        )

        # Save dashboard
        dashboard_path = self.output_dir / f"{model_type}_hyperopt_dashboard.html"
        fig.write_html(str(dashboard_path))

        print(f"✅ Hyperopt dashboard saved: {dashboard_path}")
        return str(dashboard_path)

    def _add_optimization_history(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add optimization history plot"""
        trials = study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]

        if not values:
            return

        # Objective values
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=values,
                mode='lines+markers',
                name='Objective Value',
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )

        # Best value so far
        best_values = []
        best_so_far = float('inf')
        for value in values:
            if value < best_so_far:
                best_so_far = value
            best_values.append(best_so_far)

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=best_values,
                mode='lines',
                name='Best So Far',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row, col=col
        )

    def _add_parameter_importance(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add parameter importance plot"""
        try:
            if hasattr(study, 'get_trials') and len(study.get_trials()) > 10:
                # Calculate parameter importance
                importance = optuna.importance.get_param_importances(study)

                params = list(importance.keys())
                importances = list(importance.values())

                fig.add_trace(
                    go.Bar(
                        x=importances,
                        y=params,
                        orientation='h',
                        name='Parameter Importance'
                    ),
                    row=row, col=col
                )
        except Exception as e:
            # Add placeholder if importance calculation fails
            fig.add_annotation(
                text=f"Parameter importance calculation failed: {str(e)}",
                x=0.5, y=0.5,
                xref="x domain", yref="y domain",
                showarrow=False,
                row=row, col=col
            )

    def _add_parallel_coordinates(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add parallel coordinates plot"""
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 2:
            return

        # Prepare data for parallel coordinates
        param_names = list(trials[0].params.keys())[:6]  # Limit to first 6 params

        dimensions = []
        for param in param_names:
            values = [t.params.get(param, 0) for t in trials]
            dimensions.append(
                dict(
                    label=param,
                    values=values
                )
            )

        # Add objective values
        objective_values = [t.value for t in trials]
        dimensions.append(
            dict(
                label='Objective',
                values=objective_values
            )
        )

        if dimensions:
            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color=objective_values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    dimensions=dimensions
                ),
                row=row, col=col
            )

    def _add_parameter_distributions(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add parameter distribution histograms"""
        trials = [t for t in study.trials if t.value is not None]
        if not trials:
            return

        # Get the most important parameter or first parameter
        param_names = list(trials[0].params.keys())
        if param_names:
            param_name = param_names[0]  # Take first parameter
            values = [t.params.get(param_name, 0) for t in trials]

            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=f'{param_name} Distribution',
                    opacity=0.7
                ),
                row=row, col=col
            )

    def _add_convergence_analysis(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add convergence analysis"""
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 5:
            return

        # Calculate running average and std
        values = [t.value for t in trials]
        window_size = max(5, len(values) // 10)

        running_mean = []
        running_std = []

        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            running_mean.append(np.mean(window))
            running_std.append(np.std(window))

        x_values = list(range(window_size, len(values)))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=running_mean,
                mode='lines',
                name='Running Mean',
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )

        # Add confidence band
        upper_bound = [m + s for m, s in zip(running_mean, running_std)]
        lower_bound = [m - s for m, s in zip(running_mean, running_std)]

        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std',
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_trial_timeline(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add trial timeline"""
        trials = [t for t in study.trials if t.value is not None and t.datetime_start]
        if not trials:
            return

        start_times = [t.datetime_start for t in trials]
        values = [t.value for t in trials]

        fig.add_trace(
            go.Scatter(
                x=start_times,
                y=values,
                mode='markers',
                name='Trial Timeline',
                marker=dict(size=8, color=values, colorscale='Viridis', showscale=True)
            ),
            row=row, col=col
        )

    def _add_best_trial_table(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add best trial information table"""
        if not study.best_trial:
            return

        best_trial = study.best_trial

        # Prepare table data
        data = [
            ['Best Value', f"{best_trial.value:.6f}"],
            ['Trial Number', str(best_trial.number)],
            ['Trial State', str(best_trial.state)]
        ]

        # Add best parameters
        for param, value in best_trial.params.items():
            data.append([f"Best {param}", str(value)])

        fig.add_trace(
            go.Table(
                header=dict(values=['Parameter', 'Value'], fill_color='lightblue'),
                cells=dict(
                    values=[[item[0] for item in data],
                           [item[1] for item in data]],
                    fill_color='white'
                )
            ),
            row=row, col=col
        )

    def _add_parameter_correlation(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add parameter correlation heatmap"""
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 5:
            return

        # Create DataFrame of parameters
        param_data = {}
        for trial in trials:
            for param, value in trial.params.items():
                if param not in param_data:
                    param_data[param] = []
                param_data[param].append(value)

        # Ensure all parameters have the same length
        max_len = max(len(values) for values in param_data.values())
        for param in param_data:
            while len(param_data[param]) < max_len:
                param_data[param].append(param_data[param][-1])  # Forward fill

        if len(param_data) >= 2:
            df = pd.DataFrame(param_data)
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()

                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0
                    ),
                    row=row, col=col
                )

    def _add_search_space_coverage(self, fig, study: 'optuna.Study', row: int, col: int):
        """Add search space coverage analysis"""
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 2:
            return

        # Take first two parameters for 2D visualization
        param_names = list(trials[0].params.keys())[:2]
        if len(param_names) >= 2:
            x_values = [t.params.get(param_names[0], 0) for t in trials]
            y_values = [t.params.get(param_names[1], 0) for t in trials]
            objective_values = [t.value for t in trials]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name='Search Space Coverage',
                    marker=dict(
                        size=8,
                        color=objective_values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Trial {t.number}" for t in trials],
                    hovertemplate="%{text}<br>%{x}<br>%{y}<extra></extra>"
                ),
                row=row, col=col
            )

    def _create_matplotlib_hyperopt_dashboard(
        self,
        study: 'optuna.Study',
        model_type: str
    ) -> str:
        """Create hyperopt dashboard using matplotlib"""
        if not MATPLOTLIB_AVAILABLE or not OPTUNA_AVAILABLE:
            return ""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_type.upper()} Hyperparameter Optimization Dashboard', fontsize=16, fontweight='bold')

        trials = [t for t in study.trials if t.value is not None]

        if not trials:
            return ""

        # 1. Optimization history
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]

        axes[0, 0].plot(trial_numbers, values, 'bo-', linewidth=2, markersize=4)
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Best value convergence
        best_values = []
        best_so_far = float('inf')
        for value in values:
            if value < best_so_far:
                best_so_far = value
            best_values.append(best_so_far)

        axes[0, 1].plot(trial_numbers, best_values, 'r-', linewidth=2)
        axes[0, 1].set_title('Best Value Convergence')
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Best Objective Value')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Parameter importance (if available)
        try:
            if len(trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:10]  # Top 10
                importances = [importance[p] for p in params]

                axes[0, 2].barh(params, importances)
                axes[0, 2].set_title('Parameter Importance')
                axes[0, 2].set_xlabel('Importance')
        except:
            axes[0, 2].text(0.5, 0.5, 'Parameter importance\nnot available',
                           ha='center', va='center', transform=axes[0, 2].transAxes)

        # 4. Parameter distribution
        if trials and trials[0].params:
            param_name = list(trials[0].params.keys())[0]
            param_values = [t.params.get(param_name, 0) for t in trials]

            axes[1, 0].hist(param_values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(f'{param_name} Distribution')
            axes[1, 0].set_xlabel(param_name)
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Trial durations (if available)
        durations = []
        for trial in trials:
            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                durations.append(duration)

        if durations:
            axes[1, 1].plot(range(len(durations)), durations, 'go-', linewidth=2, markersize=4)
            axes[1, 1].set_title('Trial Durations')
            axes[1, 1].set_xlabel('Trial Number')
            axes[1, 1].set_ylabel('Duration (seconds)')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Best trial summary
        if study.best_trial:
            best_trial = study.best_trial
            summary_text = f"Best Trial: {best_trial.number}\n"
            summary_text += f"Best Value: {best_trial.value:.6f}\n\n"
            summary_text += "Best Parameters:\n"
            for param, value in best_trial.params.items():
                summary_text += f"{param}: {value}\n"

            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Best Trial Summary')
            axes[1, 2].axis('off')

        plt.tight_layout()

        # Save dashboard
        dashboard_path = self.output_dir / f"{model_type}_hyperopt_dashboard_matplotlib.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(dashboard_path)


class ModelWeightVisualizer:
    """
    Advanced visualization for model weights and architecture analysis.
    """

    def __init__(self, output_dir: str = "model_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_model_architecture(
        self,
        model,
        model_type: str,
        save_architecture: bool = True
    ) -> str:
        """
        Create comprehensive model architecture visualization.

        Args:
            model: Trained model
            model_type: Type of model
            save_architecture: Whether to save architecture plot

        Returns:
            Path to saved visualization
        """
        if model_type in ['transformer', 'lstm'] and TENSORFLOW_AVAILABLE:
            return self._visualize_tensorflow_model(model, model_type, save_architecture)
        elif model_type == 'lightgbm':
            return self._visualize_lightgbm_model(model, model_type)
        elif model_type == 'ensemble':
            return self._visualize_ensemble_model(model, model_type)
        else:
            print(f"Visualization not supported for model type: {model_type}")
            return ""

    def _visualize_tensorflow_model(
        self,
        model,
        model_type: str,
        save_architecture: bool
    ) -> str:
        """Visualize TensorFlow model architecture and weights"""
        if not TENSORFLOW_AVAILABLE:
            return ""

        # Get the actual Keras model
        keras_model = model.model if hasattr(model, 'model') else model

        if keras_model is None:
            return ""

        visualizations = []

        # 1. Model architecture plot
        if save_architecture:
            try:
                tf.keras.utils.plot_model(
                    keras_model,
                    to_file=str(self.output_dir / f"{model_type}_architecture.png"),
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB',
                    expand_nested=True,
                    dpi=150
                )
                visualizations.append("architecture")
            except Exception as e:
                print(f"Failed to create architecture plot: {e}")

        # 2. Weight distributions
        weight_viz_path = self._visualize_weight_distributions(keras_model, model_type)
        if weight_viz_path:
            visualizations.append("weights")

        # 3. Layer activation analysis (if available)
        if hasattr(model, 'analyze_attention_weights'):
            try:
                # Create sample input for attention analysis
                input_shape = keras_model.input_shape
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]

                sample_input = np.random.randn(1, *input_shape[1:])
                attention_weights = model.analyze_attention_weights(sample_input)

                if attention_weights is not None:
                    self._visualize_attention_weights(attention_weights, model_type)
                    visualizations.append("attention")
            except Exception as e:
                print(f"Failed to analyze attention weights: {e}")

        # Create summary dashboard
        dashboard_path = self._create_model_summary_dashboard(keras_model, model_type, visualizations)

        return dashboard_path

    def _visualize_weight_distributions(self, model, model_type: str) -> str:
        """Visualize weight distributions across layers"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        weights_data = []
        layer_names = []

        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:
                for j, weight_tensor in enumerate(weights):
                    weights_data.append(weight_tensor.flatten())
                    layer_names.append(f"{layer.name}_{j}")

        if not weights_data:
            return ""

        # Create weight distribution plots
        n_layers = len(weights_data)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, (weights, layer_name) in enumerate(zip(weights_data, layer_names)):
            if i < len(axes):
                axes[i].hist(weights, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(layer_name)
                axes[i].set_xlabel('Weight Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                mean_weight = np.mean(weights)
                std_weight = np.std(weights)
                axes[i].axvline(mean_weight, color='red', linestyle='--',
                              label=f'μ={mean_weight:.4f}')
                axes[i].axvline(mean_weight + std_weight, color='orange', linestyle='--', alpha=0.7)
                axes[i].axvline(mean_weight - std_weight, color='orange', linestyle='--', alpha=0.7)
                axes[i].legend()

        # Remove empty subplots
        for i in range(len(weights_data), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'{model_type.upper()} Weight Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        weight_path = self.output_dir / f"{model_type}_weight_distributions.png"
        plt.savefig(weight_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(weight_path)

    def _visualize_attention_weights(self, attention_weights: np.ndarray, model_type: str) -> str:
        """Visualize attention weights heatmap"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(attention_weights.reshape(1, -1),
                   cmap='Blues',
                   cbar=True,
                   xticklabels=False,
                   yticklabels=False)

        plt.title(f'{model_type.upper()} Attention Weights')
        plt.xlabel('Time Steps')
        plt.ylabel('Attention')

        # Save plot
        attention_path = self.output_dir / f"{model_type}_attention_weights.png"
        plt.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(attention_path)

    def _visualize_lightgbm_model(self, model, model_type: str) -> str:
        """Visualize LightGBM model features and importance"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        # Get LightGBM model
        lgb_model = model.model if hasattr(model, 'model') else model

        # Feature importance
        if hasattr(lgb_model, 'feature_importance'):
            importance = lgb_model.feature_importance()
            feature_names = lgb_model.feature_name() if hasattr(lgb_model, 'feature_name') else [f"feature_{i}" for i in range(len(importance))]

            # Create feature importance plot
            plt.figure(figsize=(12, 8))

            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1][:20]  # Top 20 features
            sorted_importance = importance[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]

            plt.barh(range(len(sorted_importance)), sorted_importance)
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Feature Importance')
            plt.title('LightGBM Feature Importance (Top 20)')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            importance_path = self.output_dir / f"{model_type}_feature_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(importance_path)

        return ""

    def _visualize_ensemble_model(self, ensemble_model, model_type: str) -> str:
        """Visualize ensemble model weights and contributions"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        # Get model weights
        model_weights = ensemble_model.model_weights if hasattr(ensemble_model, 'model_weights') else {}

        if not model_weights:
            return ""

        # Create ensemble weight visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Weight distribution pie chart
        labels = list(model_weights.keys())
        sizes = list(model_weights.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        ax1.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90)
        ax1.set_title('Ensemble Model Weights')

        # Weight evolution (if performance history available)
        if hasattr(ensemble_model, 'performance_history'):
            performance_history = ensemble_model.performance_history

            for model_name in labels:
                if f"{model_name}_rmse" in performance_history:
                    history = performance_history[f"{model_name}_rmse"]
                    ax2.plot(history, label=model_name, linewidth=2)

            ax2.set_title('Model Performance History')
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('RMSE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Ensemble Model Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        ensemble_path = self.output_dir / f"{model_type}_ensemble_analysis.png"
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(ensemble_path)

    def _create_model_summary_dashboard(
        self,
        model,
        model_type: str,
        visualizations: List[str]
    ) -> str:
        """Create comprehensive model summary dashboard"""
        if not PLOTLY_AVAILABLE:
            return ""

        # Create summary info
        total_params = model.count_params() if hasattr(model, 'count_params') else 0
        trainable_params = total_params  # Simplified

        # Model architecture summary
        layer_info = []
        for layer in model.layers[:10]:  # First 10 layers
            layer_info.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params(),
                'output_shape': str(layer.output_shape)
            })

        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Model Summary', 'Layer Information', 'Parameter Distribution', 'Available Visualizations'],
            specs=[
                [{"type": "indicator"}, {"type": "table"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )

        # Model summary indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_params,
                title={'text': "Total Parameters"},
            ),
            row=1, col=1
        )

        # Layer information table
        if layer_info:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Layer Name', 'Type', 'Parameters', 'Output Shape']),
                    cells=dict(
                        values=[
                            [info['name'] for info in layer_info],
                            [info['type'] for info in layer_info],
                            [info['params'] for info in layer_info],
                            [info['output_shape'] for info in layer_info]
                        ]
                    )
                ),
                row=1, col=2
            )

        # Parameter distribution by layer type
        layer_types = {}
        for info in layer_info:
            layer_type = info['type']
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += info['params']

        if layer_types:
            fig.add_trace(
                go.Bar(
                    x=list(layer_types.keys()),
                    y=list(layer_types.values()),
                    name='Parameters by Layer Type'
                ),
                row=2, col=1
            )

        # Available visualizations
        viz_data = [['Visualization', 'Status']]
        for viz in ['architecture', 'weights', 'attention']:
            status = 'Available' if viz in visualizations else 'Not Available'
            viz_data.append([viz.capitalize(), status])

        fig.add_trace(
            go.Table(
                header=dict(values=['Visualization', 'Status']),
                cells=dict(
                    values=[
                        [item[0] for item in viz_data[1:]],
                        [item[1] for item in viz_data[1:]]
                    ]
                )
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f"{model_type.upper()} Model Summary Dashboard",
            height=800,
            template="plotly_white"
        )

        # Save dashboard
        dashboard_path = self.output_dir / f"{model_type}_model_summary.html"
        fig.write_html(str(dashboard_path))

        return str(dashboard_path)


def create_comprehensive_training_report(
    model_results: Dict[str, Any],
    output_dir: str = "training_reports"
) -> str:
    """
    Create comprehensive training report with all visualizations.

    Args:
        model_results: Dictionary containing all model training results
        output_dir: Output directory for reports

    Returns:
        Path to comprehensive report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize visualizers
    training_viz = TrainingProgressVisualizer(str(output_path / "training_progress"))
    hyperopt_viz = HyperparameterVisualizationDashboard(str(output_path / "hyperopt"))
    weight_viz = ModelWeightVisualizer(str(output_path / "model_weights"))

    report_links = {}

    for model_type, results in model_results.items():
        print(f"📊 Creating comprehensive report for {model_type}")

        # Training progress dashboard
        if 'training_history' in results:
            training_dashboard = training_viz.create_training_dashboard(
                results['training_history'],
                model_type,
                results.get('config', {}),
                results.get('cv_scores', None)
            )
            report_links[f"{model_type}_training"] = training_dashboard

        # Hyperparameter optimization dashboard
        if 'optuna_study' in results:
            hyperopt_dashboard = hyperopt_viz.create_optuna_dashboard(
                results['optuna_study'],
                model_type
            )
            report_links[f"{model_type}_hyperopt"] = hyperopt_dashboard

        # Model weights and architecture
        if 'model' in results:
            weight_dashboard = weight_viz.visualize_model_architecture(
                results['model'],
                model_type
            )
            if weight_dashboard:
                report_links[f"{model_type}_weights"] = weight_dashboard

    # Create master index HTML
    index_html = _create_master_index(report_links, model_results)
    index_path = output_path / "index.html"

    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"🎉 Comprehensive training report created: {index_path}")
    return str(index_path)


def _create_master_index(report_links: Dict[str, str], model_results: Dict[str, Any]) -> str:
    """Create master HTML index for all reports"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cryptocurrency Prediction Model Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .model-section { background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .link-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
            .dashboard-link {
                background-color: #3498db;
                color: white;
                padding: 15px;
                text-decoration: none;
                border-radius: 5px;
                text-align: center;
                display: block;
                transition: background-color 0.3s;
            }
            .dashboard-link:hover { background-color: #2980b9; }
            .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .summary-table th, .summary-table td { border: 1px solid #bdc3c7; padding: 12px; text-align: left; }
            .summary-table th { background-color: #34495e; color: white; }
            .performance-metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🚀 Cryptocurrency Prediction Model Training Report</h1>
        <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>

        <h2>📊 Executive Summary</h2>
        <div class="model-section">
            <h3>Training Overview</h3>
            <table class="summary-table">
                <tr><th>Model Type</th><th>Status</th><th>Key Metrics</th><th>Dashboards Available</th></tr>
    """

    # Add model summaries
    for model_type, results in model_results.items():
        status = "✅ Completed" if 'model' in results else "❌ Failed"

        # Extract key metrics
        metrics = ""
        if 'performance' in results:
            perf = results['performance']
            if 'val_rmse' in perf:
                metrics += f"Val RMSE: {perf['val_rmse']:.4f}"
            if 'test_rmse' in perf:
                metrics += f", Test RMSE: {perf['test_rmse']:.4f}"

        # Count dashboards
        dashboards = [k for k in report_links.keys() if k.startswith(model_type)]
        dashboard_count = len(dashboards)

        html_content += f"""
                <tr>
                    <td><strong>{model_type.upper()}</strong></td>
                    <td>{status}</td>
                    <td>{metrics}</td>
                    <td>{dashboard_count} dashboards</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    """

    # Add individual model sections
    for model_type, results in model_results.items():
        html_content += f"""
        <h2>🤖 {model_type.upper()} Model Analysis</h2>
        <div class="model-section">
        """

        # Performance metrics
        if 'performance' in results:
            html_content += "<h3>📈 Performance Metrics</h3>"
            for metric, value in results['performance'].items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="performance-metric"><strong>{metric}:</strong> {value:.4f}</div>'

        # Dashboard links
        model_links = [k for k in report_links.keys() if k.startswith(model_type)]
        if model_links:
            html_content += "<h3>📊 Available Dashboards</h3>"
            html_content += '<div class="link-grid">'

            for link_key in model_links:
                link_path = report_links[link_key]
                dashboard_name = link_key.replace(f"{model_type}_", "").replace("_", " ").title()
                # Convert to relative path
                relative_path = Path(link_path).relative_to(Path(link_path).parent.parent)
                html_content += f'<a href="{relative_path}" class="dashboard-link">{dashboard_name} Dashboard</a>'

            html_content += '</div>'

        html_content += '</div>'

    html_content += """
        <footer style="margin-top: 50px; text-align: center; color: #7f8c8d;">
            <p>Generated by Cryptocurrency Prediction Model Training Pipeline</p>
            <p>🎯 Advanced ML Models for Financial Forecasting</p>
        </footer>
    </body>
    </html>
    """

    return html_content


# Example usage functions
def monitor_training_progress(model_type: str, output_dir: str = "training_monitoring"):
    """
    Setup real-time training monitoring.

    Args:
        model_type: Type of model being trained
        output_dir: Output directory for monitoring files
    """
    visualizer = TrainingProgressVisualizer(output_dir)

    print(f"🔍 Setting up real-time monitoring for {model_type}")
    print("Use visualizer.update_monitor(epoch, train_loss, val_loss, lr) to update progress")

    # Create real-time monitor
    visualizer.create_real_time_monitor(update_interval=1000, max_epochs=100)

    return visualizer