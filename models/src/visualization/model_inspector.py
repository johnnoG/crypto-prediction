"""
Comprehensive Model Architecture and Weight Visualization System
Provides detailed analysis and visualization of model internals, weights, and architecture
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objects import Figure
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.utils as keras_utils
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelArchitectureVisualizer:
    """Visualizes model architecture and layer details"""

    def __init__(self):
        self.model_info = {}
        self.layer_details = {}

    def analyze_tensorflow_model(self, model, model_name: str = "tf_model") -> Dict[str, Any]:
        """Analyze TensorFlow/Keras model architecture"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available")
            return {}

        analysis = {
            'model_name': model_name,
            'model_type': 'tensorflow',
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': [],
            'layer_graph': {},
            'model_summary': []
        }

        # Analyze each layer
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'class_name': layer.__class__.__name__,
                'input_shape': getattr(layer, 'input_shape', None),
                'output_shape': getattr(layer, 'output_shape', None),
                'params': layer.count_params(),
                'trainable': layer.trainable,
                'config': layer.get_config()
            }

            # Extract weights if available
            if layer.weights:
                layer_info['weights'] = []
                for weight in layer.weights:
                    weight_info = {
                        'name': weight.name,
                        'shape': weight.shape.as_list(),
                        'dtype': str(weight.dtype),
                        'mean': float(tf.reduce_mean(weight).numpy()),
                        'std': float(tf.math.reduce_std(weight).numpy()),
                        'min': float(tf.reduce_min(weight).numpy()),
                        'max': float(tf.reduce_max(weight).numpy())
                    }
                    layer_info['weights'].append(weight_info)

            analysis['layers'].append(layer_info)

        # Get model summary
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        analysis['model_summary'] = string_list

        self.model_info[model_name] = analysis
        return analysis

    def analyze_pytorch_model(self, model, model_name: str = "torch_model", input_shape: Tuple = None) -> Dict[str, Any]:
        """Analyze PyTorch model architecture"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            return {}

        analysis = {
            'model_name': model_name,
            'model_type': 'pytorch',
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layers': [],
            'layer_graph': {},
            'model_summary': []
        }

        # Analyze each module
        for name, module in model.named_modules():
            if name:  # Skip the root module
                layer_info = {
                    'name': name,
                    'class_name': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'trainable': any(p.requires_grad for p in module.parameters())
                }

                # Extract weights if available
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.detach().cpu().numpy()
                    layer_info['weight_stats'] = {
                        'shape': list(weight.shape),
                        'mean': float(np.mean(weight)),
                        'std': float(np.std(weight)),
                        'min': float(np.min(weight)),
                        'max': float(np.max(weight))
                    }

                analysis['layers'].append(layer_info)

        # Generate model summary string
        analysis['model_summary'] = [str(model)]

        self.model_info[model_name] = analysis
        return analysis

    def create_architecture_graph(self, model_name: str) -> Figure:
        """Create interactive architecture graph"""
        if model_name not in self.model_info:
            return go.Figure()

        analysis = self.model_info[model_name]
        layers = analysis['layers']

        if not layers:
            return go.Figure()

        # Create network graph
        fig = go.Figure()

        # Calculate positions for layers
        n_layers = len(layers)
        positions = {}
        for i, layer in enumerate(layers):
            x = i / max(n_layers - 1, 1)
            y = 0.5
            positions[layer['name']] = (x, y)

        # Add nodes (layers)
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []

        for layer in layers:
            pos = positions[layer['name']]
            node_x.append(pos[0])
            node_y.append(pos[1])

            # Create hover text
            hover_text = f"<b>{layer['name']}</b><br>"
            hover_text += f"Type: {layer['class_name']}<br>"
            if 'params' in layer:
                hover_text += f"Parameters: {layer['params']:,}<br>"
            if 'input_shape' in layer and layer['input_shape']:
                hover_text += f"Input: {layer['input_shape']}<br>"
            if 'output_shape' in layer and layer['output_shape']:
                hover_text += f"Output: {layer['output_shape']}"

            node_text.append(hover_text)

            # Size based on parameters
            param_count = layer.get('params', 0)
            if param_count > 0:
                size = 20 + min(np.log10(param_count + 1) * 5, 40)
            else:
                size = 15
            node_sizes.append(size)

            # Color based on layer type
            layer_type = layer['class_name'].lower()
            if 'dense' in layer_type or 'linear' in layer_type:
                color = 'blue'
            elif 'conv' in layer_type:
                color = 'green'
            elif 'lstm' in layer_type or 'gru' in layer_type:
                color = 'red'
            elif 'attention' in layer_type or 'transformer' in layer_type:
                color = 'purple'
            elif 'dropout' in layer_type or 'batch' in layer_type:
                color = 'orange'
            else:
                color = 'gray'
            node_colors.append(color)

        # Add edges (connections)
        edge_x = []
        edge_y = []
        for i in range(len(layers) - 1):
            x0, y0 = positions[layers[i]['name']]
            x1, y1 = positions[layers[i + 1]['name']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Add edges to plot
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))

        # Add nodes to plot
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[layer['name'] for layer in layers],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))

        fig.update_layout(
            title=f"Model Architecture: {analysis['model_name']}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400
        )

        return fig

class WeightAnalyzer:
    """Analyzes and visualizes model weights"""

    def __init__(self):
        self.weight_data = {}

    def analyze_layer_weights(self, weights: np.ndarray, layer_name: str) -> Dict[str, Any]:
        """Analyze weight statistics for a layer"""
        analysis = {
            'layer_name': layer_name,
            'shape': list(weights.shape),
            'total_params': weights.size,
            'statistics': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'median': float(np.median(weights)),
                'q1': float(np.percentile(weights, 25)),
                'q3': float(np.percentile(weights, 75)),
                'zeros': int(np.sum(weights == 0)),
                'near_zeros': int(np.sum(np.abs(weights) < 1e-6))
            },
            'distribution': {
                'histogram': np.histogram(weights.flatten(), bins=50),
                'sparsity': float(np.sum(weights == 0) / weights.size)
            }
        }

        self.weight_data[layer_name] = analysis
        return analysis

    def create_weight_distribution_plot(self, layer_name: str) -> Figure:
        """Create weight distribution visualization"""
        if layer_name not in self.weight_data:
            return go.Figure()

        data = self.weight_data[layer_name]
        hist_data, bin_edges = data['distribution']['histogram']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weight Distribution', 'Weight Statistics',
                          'Sparsity Analysis', 'Weight Magnitude'),
            specs=[[{"type": "histogram"}, {"type": "table"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )

        # Weight distribution histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(go.Histogram(
            x=bin_centers,
            y=hist_data,
            name='Weight Distribution',
            marker_color='blue',
            opacity=0.7
        ), row=1, col=1)

        # Statistics table
        stats = data['statistics']
        fig.add_trace(go.Table(
            header=dict(values=['Statistic', 'Value']),
            cells=dict(values=[
                ['Mean', 'Std', 'Min', 'Max', 'Median', 'Sparsity'],
                [f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                 f"{stats['min']:.6f}", f"{stats['max']:.6f}",
                 f"{stats['median']:.6f}", f"{data['distribution']['sparsity']:.4f}"]
            ])
        ), row=1, col=2)

        # Sparsity analysis
        sparsity_data = [
            ('Non-zero', data['total_params'] - stats['zeros']),
            ('Zero', stats['zeros']),
            ('Near-zero', stats['near_zeros'])
        ]

        fig.add_trace(go.Bar(
            x=[item[0] for item in sparsity_data],
            y=[item[1] for item in sparsity_data],
            marker_color=['green', 'red', 'orange'],
            name='Sparsity Analysis'
        ), row=2, col=1)

        # Weight magnitude distribution
        fig.add_trace(go.Histogram(
            x=np.abs(bin_centers),
            y=hist_data,
            name='Magnitude Distribution',
            marker_color='purple',
            opacity=0.7
        ), row=2, col=2)

        fig.update_layout(
            title_text=f"Weight Analysis: {layer_name}",
            showlegend=False,
            height=600
        )

        return fig

    def create_weight_heatmap(self, weights: np.ndarray, layer_name: str) -> Figure:
        """Create weight heatmap visualization"""
        if len(weights.shape) > 2:
            # For higher dimensional weights, reshape or take a slice
            if len(weights.shape) == 4:  # Conv weights (height, width, in_channels, out_channels)
                weights = weights[:, :, 0, 0] if weights.shape[2] > 0 and weights.shape[3] > 0 else weights.reshape(weights.shape[0], -1)
            else:
                weights = weights.reshape(weights.shape[0], -1)

        # Limit size for visualization
        max_size = 100
        if weights.shape[0] > max_size or weights.shape[1] > max_size:
            step_x = max(1, weights.shape[1] // max_size)
            step_y = max(1, weights.shape[0] // max_size)
            weights = weights[::step_y, ::step_x]

        fig = go.Figure(data=go.Heatmap(
            z=weights,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Weight Value")
        ))

        fig.update_layout(
            title=f"Weight Heatmap: {layer_name}",
            xaxis_title="Output Dimension",
            yaxis_title="Input Dimension",
            height=500
        )

        return fig

class ModelInspectorDashboard:
    """Comprehensive model inspection dashboard"""

    def __init__(self, models_dir: str = "models/artifacts", port: int = 8052):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.port = port

        self.arch_visualizer = ModelArchitectureVisualizer()
        self.weight_analyzer = WeightAnalyzer()

        self.available_models = {}
        self._load_models()

        # Setup dashboard
        self.app = self._setup_dashboard()

    def _load_models(self):
        """Load available models from artifacts directory"""
        model_files = []

        # Look for TensorFlow models
        model_files.extend(list(self.models_dir.glob("*.h5")))
        model_files.extend(list(self.models_dir.glob("*.keras")))

        # Look for PyTorch models
        model_files.extend(list(self.models_dir.glob("*.pt")))
        model_files.extend(list(self.models_dir.glob("*.pth")))

        # Look for saved model directories
        for item in self.models_dir.iterdir():
            if item.is_dir():
                if (item / "saved_model.pb").exists():
                    model_files.append(item)

        for model_file in model_files:
            model_name = model_file.stem
            self.available_models[model_name] = {
                'path': str(model_file),
                'type': self._detect_model_type(model_file),
                'loaded': False
            }

        logger.info(f"Found {len(self.available_models)} model files")

    def _detect_model_type(self, model_path: Path) -> str:
        """Detect model type from file extension"""
        suffix = model_path.suffix.lower()
        if suffix in ['.h5', '.keras']:
            return 'tensorflow'
        elif suffix in ['.pt', '.pth']:
            return 'pytorch'
        elif model_path.is_dir() and (model_path / "saved_model.pb").exists():
            return 'tensorflow_saved_model'
        else:
            return 'unknown'

    def _setup_dashboard(self):
        """Setup Dash application"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🔍 Model Architecture Inspector", className="text-center mb-4"),
                    html.Hr(),
                ], width=12)
            ]),

            # Model Selection
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📁 Model Selection", className="card-title"),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[
                                    {'label': f"{name} ({info['type']})", 'value': name}
                                    for name, info in self.available_models.items()
                                ],
                                placeholder="Select model to inspect...",
                                style={"margin-bottom": "15px"}
                            ),
                            html.Div(id="model-info"),
                            dbc.Button("🔄 Load Model", id="load-model-btn", color="primary")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🏗️ Architecture Overview", className="card-title"),
                            dcc.Graph(id="architecture-graph", style={"height": "300px"})
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),

            # Analysis Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            label="🏗️ Architecture Details",
                            tab_id="architecture",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("📋 Layer Summary"),
                                        html.Div(id="layer-table")
                                    ], width=6),
                                    dbc.Col([
                                        html.H5("📊 Parameter Distribution"),
                                        dcc.Graph(id="param-distribution-plot", style={"height": "400px"})
                                    ], width=6)
                                ], className="mt-3")
                            ]
                        ),
                        dbc.Tab(
                            label="⚖️ Weight Analysis",
                            tab_id="weights",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("🎚️ Layer Selection"),
                                        dcc.Dropdown(
                                            id="layer-selector",
                                            placeholder="Select layer for weight analysis...",
                                            style={"margin-bottom": "15px"}
                                        ),
                                    ], width=12)
                                ], className="mt-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="weight-distribution-plot", style={"height": "500px"})
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Graph(id="weight-heatmap-plot", style={"height": "500px"})
                                    ], width=6)
                                ], className="mt-3")
                            ]
                        ),
                        dbc.Tab(
                            label="📊 Model Summary",
                            tab_id="summary",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("📝 Model Summary"),
                                        html.Pre(id="model-summary", style={
                                            "background-color": "#f8f9fa",
                                            "padding": "15px",
                                            "border-radius": "5px",
                                            "max-height": "600px",
                                            "overflow-y": "auto"
                                        })
                                    ], width=12)
                                ], className="mt-3")
                            ]
                        )
                    ], id="inspector-tabs", active_tab="architecture")
                ], width=12)
            ])
        ], fluid=True)

        # Register callbacks
        self._register_callbacks(app)
        return app

    def _register_callbacks(self, app):
        """Register dashboard callbacks"""

        @app.callback(
            [Output('architecture-graph', 'figure'),
             Output('model-info', 'children'),
             Output('layer-table', 'children'),
             Output('param-distribution-plot', 'figure'),
             Output('model-summary', 'children'),
             Output('layer-selector', 'options')],
            [Input('load-model-btn', 'n_clicks')],
            [State('model-selector', 'value')]
        )
        def load_and_analyze_model(n_clicks, selected_model):
            if not n_clicks or not selected_model:
                return go.Figure(), "", "", go.Figure(), "", []

            if selected_model not in self.available_models:
                return go.Figure(), "Model not found", "", go.Figure(), "", []

            model_info = self.available_models[selected_model]

            try:
                # Load model based on type
                if model_info['type'] == 'tensorflow' and TF_AVAILABLE:
                    model = keras.models.load_model(model_info['path'])
                    analysis = self.arch_visualizer.analyze_tensorflow_model(model, selected_model)
                elif model_info['type'] == 'pytorch' and TORCH_AVAILABLE:
                    model = torch.load(model_info['path'], map_location='cpu')
                    analysis = self.arch_visualizer.analyze_pytorch_model(model, selected_model)
                else:
                    return go.Figure(), "Model type not supported or libraries not available", "", go.Figure(), "", []

                # Create architecture graph
                arch_fig = self.arch_visualizer.create_architecture_graph(selected_model)

                # Model info
                info_div = dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Total Parameters: "), f"{analysis['total_params']:,}"]),
                        html.P([html.Strong("Trainable Parameters: "), f"{analysis['trainable_params']:,}"]),
                        html.P([html.Strong("Model Type: "), analysis['model_type']]),
                        html.P([html.Strong("Number of Layers: "), str(len(analysis['layers']))])
                    ])
                ])

                # Layer table
                if analysis['layers']:
                    layer_data = []
                    for layer in analysis['layers']:
                        layer_data.append({
                            'Layer': layer['name'],
                            'Type': layer['class_name'],
                            'Parameters': f"{layer.get('params', 0):,}",
                            'Trainable': str(layer.get('trainable', 'N/A'))
                        })

                    layer_table = dash_table.DataTable(
                        data=layer_data,
                        columns=[{"name": col, "id": col} for col in layer_data[0].keys()],
                        style_cell={'textAlign': 'left', 'fontSize': 12},
                        style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
                        page_size=10,
                        style_table={'height': '400px', 'overflowY': 'auto'}
                    )
                else:
                    layer_table = html.P("No layer information available")

                # Parameter distribution plot
                if analysis['layers']:
                    param_counts = [layer.get('params', 0) for layer in analysis['layers'] if layer.get('params', 0) > 0]
                    layer_names = [layer['name'] for layer in analysis['layers'] if layer.get('params', 0) > 0]

                    param_fig = go.Figure(data=[
                        go.Bar(x=layer_names, y=param_counts, marker_color='steelblue')
                    ])
                    param_fig.update_layout(
                        title="Parameters per Layer",
                        xaxis_title="Layer",
                        yaxis_title="Parameter Count",
                        xaxis_tickangle=-45
                    )
                else:
                    param_fig = go.Figure()

                # Model summary
                summary_text = "\n".join(analysis['model_summary'])

                # Layer selector options
                layer_options = [
                    {'label': layer['name'], 'value': layer['name']}
                    for layer in analysis['layers']
                    if layer.get('params', 0) > 0
                ]

                return arch_fig, info_div, layer_table, param_fig, summary_text, layer_options

            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                logger.error(error_msg)
                return go.Figure(), error_msg, "", go.Figure(), error_msg, []

    def run_dashboard(self, debug: bool = False):
        """Run the model inspector dashboard"""
        print(f"🔍 Starting Model Inspector Dashboard on http://localhost:{self.port}")
        print("🏗️ Analyze model architectures and weights interactively!")
        print("⚖️ Explore layer details and parameter distributions")

        self.app.run(debug=debug, host='0.0.0.0', port=self.port)

if __name__ == "__main__":
    # Example usage
    inspector = ModelInspectorDashboard()
    inspector.run_dashboard(debug=False)