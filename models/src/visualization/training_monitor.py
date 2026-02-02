"""
Real-time Training Progress Monitoring System
Provides live monitoring and visualization of model training progress
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.graph_objects import Figure
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Real-time training progress monitor with live updates"""

    def __init__(self,
                 experiment_name: str = "crypto_prediction",
                 monitor_dir: str = "models/monitoring",
                 refresh_interval: int = 5):
        """
        Initialize training monitor

        Args:
            experiment_name: MLflow experiment name to monitor
            monitor_dir: Directory to store monitoring files
            refresh_interval: Refresh interval in seconds
        """
        self.experiment_name = experiment_name
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_interval = refresh_interval

        # Training state tracking
        self.training_sessions = {}
        self.active_runs = {}
        self.historical_data = []

        # Initialize monitoring files
        self._init_monitoring_files()

        # Start background monitoring
        self.monitoring_active = False
        self.monitor_thread = None

    def _init_monitoring_files(self):
        """Initialize monitoring files and directories"""
        # Create monitoring subdirectories
        (self.monitor_dir / "sessions").mkdir(exist_ok=True)
        (self.monitor_dir / "metrics").mkdir(exist_ok=True)
        (self.monitor_dir / "logs").mkdir(exist_ok=True)

        # Initialize session tracking file
        self.session_file = self.monitor_dir / "active_sessions.json"
        if not self.session_file.exists():
            with open(self.session_file, 'w') as f:
                json.dump({}, f)

    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Training monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Training monitoring stopped")

    def register_training_session(self,
                                session_id: str,
                                model_type: str,
                                config: Dict[str, Any],
                                callbacks: Optional[List[Callable]] = None) -> str:
        """
        Register a new training session for monitoring

        Args:
            session_id: Unique session identifier
            model_type: Type of model being trained
            config: Training configuration
            callbacks: Optional callback functions

        Returns:
            Session ID for tracking
        """
        session_info = {
            'session_id': session_id,
            'model_type': model_type,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'metrics': {},
            'callbacks': callbacks or []
        }

        self.training_sessions[session_id] = session_info

        # Save session info
        session_file = self.monitor_dir / "sessions" / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_info, f, indent=2, default=str)

        logger.info(f"Registered training session: {session_id}")
        return session_id

    def update_metrics(self,
                      session_id: str,
                      epoch: int,
                      metrics: Dict[str, float],
                      learning_rate: Optional[float] = None):
        """Update training metrics for a session"""
        if session_id not in self.training_sessions:
            logger.warning(f"Session {session_id} not found")
            return

        timestamp = datetime.now().isoformat()

        metric_update = {
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': metrics,
            'learning_rate': learning_rate
        }

        # Update session metrics
        if 'metrics_history' not in self.training_sessions[session_id]:
            self.training_sessions[session_id]['metrics_history'] = []

        self.training_sessions[session_id]['metrics_history'].append(metric_update)

        # Save metrics to file
        metrics_file = self.monitor_dir / "metrics" / f"{session_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.training_sessions[session_id]['metrics_history'], f, indent=2)

        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"monitor_{session_id}"):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value, step=epoch)
                    if learning_rate:
                        mlflow.log_metric("learning_rate", learning_rate, step=epoch)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_monitoring_data()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.refresh_interval)

    def _update_monitoring_data(self):
        """Update monitoring data from various sources"""
        # Update from MLflow if available
        if MLFLOW_AVAILABLE:
            self._update_from_mlflow()

        # Update active sessions
        self._update_active_sessions()

    def _update_from_mlflow(self):
        """Update data from MLflow tracking server"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="status = 'RUNNING'"
                )

                for _, run in runs.iterrows():
                    run_id = run['run_id']
                    if run_id not in self.active_runs:
                        self.active_runs[run_id] = {
                            'start_time': run['start_time'],
                            'status': run['status'],
                            'metrics': {}
                        }
        except Exception as e:
            logger.warning(f"Failed to update from MLflow: {e}")

    def _update_active_sessions(self):
        """Update active session status"""
        current_time = datetime.now()

        for session_id, session in self.training_sessions.items():
            # Check if session is stale (no updates in 10 minutes)
            if session['status'] == 'active':
                last_update = datetime.fromisoformat(session.get('last_update', session['start_time']))
                if current_time - last_update > timedelta(minutes=10):
                    session['status'] = 'stale'
                    logger.warning(f"Session {session_id} marked as stale")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all training sessions"""
        summary = {
            'total_sessions': len(self.training_sessions),
            'active_sessions': len([s for s in self.training_sessions.values() if s['status'] == 'active']),
            'completed_sessions': len([s for s in self.training_sessions.values() if s['status'] == 'completed']),
            'failed_sessions': len([s for s in self.training_sessions.values() if s['status'] == 'failed']),
            'sessions': list(self.training_sessions.keys())
        }
        return summary

    def create_live_dashboard(self, port: int = 8050) -> dash.Dash:
        """Create live monitoring dashboard"""

        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔍 Training Progress Monitor", className="text-center mb-4"),
                    html.Hr(),
                ], width=12)
            ]),

            # Session Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Session Overview", className="card-title"),
                            html.Div(id="session-overview")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Live Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Live Training Metrics", className="card-title"),
                            dcc.Graph(id="live-metrics-plot")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚙️ Session Controls", className="card-title"),
                            dcc.Dropdown(
                                id="session-selector",
                                placeholder="Select training session...",
                                style={"margin-bottom": "10px"}
                            ),
                            dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", className="me-2"),
                            dbc.Button("⏸️ Pause", id="pause-btn", color="warning", className="me-2"),
                            dbc.Button("🛑 Stop", id="stop-btn", color="danger")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            # Detailed Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📋 Detailed Metrics", className="card-title"),
                            html.Div(id="detailed-metrics")
                        ])
                    ])
                ], width=12)
            ]),

            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.refresh_interval * 1000,  # in milliseconds
                n_intervals=0
            )

        ], fluid=True)

        # Callbacks
        @app.callback(
            [Output('session-overview', 'children'),
             Output('session-selector', 'options')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_overview(n):
            summary = self.get_session_summary()

            overview_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{summary['total_sessions']}", className="card-title"),
                            html.P("Total Sessions", className="card-text")
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{summary['active_sessions']}", className="card-title"),
                            html.P("Active Sessions", className="card-text")
                        ])
                    ], color="success", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{summary['completed_sessions']}", className="card-title"),
                            html.P("Completed Sessions", className="card-text")
                        ])
                    ], color="info", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{summary['failed_sessions']}", className="card-title"),
                            html.P("Failed Sessions", className="card-text")
                        ])
                    ], color="danger", outline=True)
                ], width=3)
            ])

            session_options = [
                {'label': f"{sid} ({self.training_sessions[sid]['model_type']})", 'value': sid}
                for sid in summary['sessions']
            ]

            return overview_cards, session_options

        @app.callback(
            Output('live-metrics-plot', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('session-selector', 'value')]
        )
        def update_live_metrics(n, selected_session):
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Validation Loss', 'Learning Rate', 'Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            if selected_session and selected_session in self.training_sessions:
                session = self.training_sessions[selected_session]

                if 'metrics_history' in session:
                    history = session['metrics_history']
                    epochs = [h['epoch'] for h in history]

                    # Training metrics
                    if epochs:
                        # Loss curves
                        train_loss = [h['metrics'].get('loss', 0) for h in history]
                        val_loss = [h['metrics'].get('val_loss', 0) for h in history]

                        fig.add_trace(
                            go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='blue')),
                            row=1, col=1
                        )

                        fig.add_trace(
                            go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='red')),
                            row=1, col=2
                        )

                        # Learning rate
                        lr_values = [h.get('learning_rate', 0) for h in history if h.get('learning_rate')]
                        if lr_values:
                            fig.add_trace(
                                go.Scatter(x=epochs[:len(lr_values)], y=lr_values, name='Learning Rate', line=dict(color='green')),
                                row=2, col=1
                            )

                        # Accuracy
                        accuracy = [h['metrics'].get('accuracy', 0) for h in history]
                        fig.add_trace(
                            go.Scatter(x=epochs, y=accuracy, name='Accuracy', line=dict(color='purple')),
                            row=2, col=2
                        )

            fig.update_layout(
                title_text=f"Live Training Metrics - {selected_session or 'Select a session'}",
                showlegend=True,
                height=600
            )

            return fig

        return app

class TrainingLogger:
    """Enhanced training logger with structured logging"""

    def __init__(self, log_dir: str = "models/logs", session_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}"

        # Set up structured logging
        self.logger = logging.getLogger(f"training.{self.session_id}")
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = self.log_dir / f"{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log_training_start(self, model_type: str, config: Dict[str, Any]):
        """Log training start"""
        self.logger.info(f"Training started - Model: {model_type}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float], duration: float):
        """Log epoch completion"""
        self.logger.info(
            f"Epoch {epoch} completed in {duration:.2f}s - "
            f"Metrics: {json.dumps(metrics, indent=2)}"
        )

    def log_training_complete(self, final_metrics: Dict[str, float], total_duration: float):
        """Log training completion"""
        self.logger.info(f"Training completed in {total_duration:.2f}s")
        self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

    def log_error(self, error: Exception, context: str = ""):
        """Log training errors"""
        self.logger.error(f"Training error {context}: {str(error)}")

def create_monitoring_dashboard(monitor: TrainingMonitor, port: int = 8050):
    """Create and run monitoring dashboard"""
    app = monitor.create_live_dashboard(port=port)

    print(f"🚀 Starting training monitor dashboard on http://localhost:{port}")
    print("📊 Monitor your training sessions in real-time!")
    print("🔄 Dashboard auto-refreshes every 5 seconds")

    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == "__main__":
    # Example usage
    monitor = TrainingMonitor()
    monitor.start_monitoring()

    # Register a sample training session
    session_id = monitor.register_training_session(
        session_id="transformer_training_001",
        model_type="Transformer",
        config={
            "model_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    )

    # Simulate some training updates
    for epoch in range(1, 11):
        metrics = {
            "loss": np.random.exponential(0.5) + 0.1,
            "val_loss": np.random.exponential(0.6) + 0.12,
            "accuracy": 0.7 + 0.3 * (1 - np.exp(-epoch/5)) + np.random.normal(0, 0.05)
        }
        monitor.update_metrics(session_id, epoch, metrics, learning_rate=0.001 * 0.95**epoch)
        time.sleep(1)

    # Start dashboard
    create_monitoring_dashboard(monitor, port=8050)