"""
Comprehensive Hyperparameter Optimization Visualization Dashboard
Provides interactive visualization and analysis of Optuna optimization studies
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objects import Figure
import dash
from dash import dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    from optuna.visualization import plot_parallel_coordinate, plot_slice
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizationDashboard:
    """Interactive dashboard for hyperparameter optimization analysis"""

    def __init__(self,
                 studies_dir: str = "models/optuna_studies",
                 dashboard_port: int = 8051):
        """
        Initialize hyperparameter optimization dashboard

        Args:
            studies_dir: Directory containing Optuna studies
            dashboard_port: Port for dashboard server
        """
        self.studies_dir = Path(studies_dir)
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_port = dashboard_port

        # Study management
        self.available_studies = {}
        self.current_study = None

        # Load existing studies
        self._load_studies()

        # Initialize dashboard
        self.app = None
        self._setup_dashboard()

    def _load_studies(self):
        """Load available Optuna studies"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available - hyperparameter dashboard will have limited functionality")
            return

        # Look for study databases
        study_files = list(self.studies_dir.glob("*.db"))
        study_files.extend(list(self.studies_dir.glob("*.json")))

        for study_file in study_files:
            try:
                if study_file.suffix == ".db":
                    # Load from SQLite database
                    storage_url = f"sqlite:///{study_file}"
                    study_summaries = optuna.get_study_summaries(storage=storage_url)

                    for summary in study_summaries:
                        study = optuna.load_study(
                            study_name=summary.study_name,
                            storage=storage_url
                        )
                        self.available_studies[summary.study_name] = {
                            'study': study,
                            'source': str(study_file),
                            'type': 'database'
                        }

                elif study_file.suffix == ".json":
                    # Load from JSON export
                    with open(study_file, 'r') as f:
                        study_data = json.load(f)

                    study_name = study_file.stem
                    self.available_studies[study_name] = {
                        'study_data': study_data,
                        'source': str(study_file),
                        'type': 'json'
                    }

            except Exception as e:
                logger.warning(f"Failed to load study from {study_file}: {e}")

        logger.info(f"Loaded {len(self.available_studies)} optimization studies")

    def _setup_dashboard(self):
        """Setup Dash application"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.app.layout = dbc.Container([
            dcc.Store(id='study-data-store'),

            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Hyperparameter Optimization Dashboard", className="text-center mb-4"),
                    html.Hr(),
                ], width=12)
            ]),

            # Study Selection and Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Study Selection", className="card-title"),
                            dcc.Dropdown(
                                id="study-selector",
                                options=[
                                    {'label': f"{name} ({info['type']})", 'value': name}
                                    for name, info in self.available_studies.items()
                                ],
                                placeholder="Select optimization study...",
                                style={"margin-bottom": "15px"}
                            ),
                            html.Div(id="study-overview")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Optimization Progress", className="card-title"),
                            dcc.Graph(id="optimization-history-plot", style={"height": "300px"})
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),

            # Detailed Analysis Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            label="🔍 Parameter Importance",
                            tab_id="param-importance",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="param-importance-plot", style={"height": "500px"})
                                    ], width=8),
                                    dbc.Col([
                                        html.H5("🎛️ Parameter Statistics"),
                                        html.Div(id="param-stats")
                                    ], width=4)
                                ], className="mt-3")
                            ]
                        ),
                        dbc.Tab(
                            label="📊 Parameter Relationships",
                            tab_id="param-relationships",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="parallel-coordinate-plot", style={"height": "500px"})
                                    ], width=12)
                                ], className="mt-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="param-correlation-plot", style={"height": "400px"})
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Graph(id="param-scatter-plot", style={"height": "400px"})
                                    ], width=6)
                                ], className="mt-3")
                            ]
                        ),
                        dbc.Tab(
                            label="🎯 Best Trials Analysis",
                            tab_id="best-trials",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("🏆 Top 10 Trials"),
                                        html.Div(id="best-trials-table")
                                    ], width=6),
                                    dbc.Col([
                                        html.H5("📊 Performance Distribution"),
                                        dcc.Graph(id="performance-distribution-plot", style={"height": "400px"})
                                    ], width=6)
                                ], className="mt-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("🔬 Hyperparameter Slice Analysis"),
                                        html.Div([
                                            dcc.Dropdown(
                                                id="slice-param-selector",
                                                placeholder="Select parameter for slice analysis...",
                                                style={"margin-bottom": "15px"}
                                            ),
                                            dcc.Graph(id="slice-plot", style={"height": "400px"})
                                        ])
                                    ], width=12)
                                ], className="mt-3")
                            ]
                        ),
                        dbc.Tab(
                            label="📈 Convergence Analysis",
                            tab_id="convergence",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="convergence-plot", style={"height": "400px"})
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Graph(id="trial-timeline-plot", style={"height": "400px"})
                                    ], width=6)
                                ], className="mt-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("📊 Optimization Statistics"),
                                        html.Div(id="optimization-stats")
                                    ], width=12)
                                ], className="mt-3")
                            ]
                        )
                    ], id="analysis-tabs", active_tab="param-importance")
                ], width=12)
            ])
        ], fluid=True)

        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register Dash callbacks"""

        @self.app.callback(
            [Output('study-data-store', 'data'),
             Output('study-overview', 'children')],
            [Input('study-selector', 'value')]
        )
        def update_study_data(selected_study):
            if not selected_study or selected_study not in self.available_studies:
                return {}, html.P("No study selected", className="text-muted")

            study_info = self.available_studies[selected_study]

            if study_info['type'] == 'database':
                study = study_info['study']
                trials_df = study.trials_dataframe()

                study_data = {
                    'study_name': study.study_name,
                    'direction': study.direction.name,
                    'n_trials': len(study.trials),
                    'best_value': study.best_value if study.trials else None,
                    'best_params': study.best_params if study.trials else {},
                    'trials': trials_df.to_dict('records') if not trials_df.empty else []
                }

                # Create overview
                overview = dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Study Name: "), study.study_name]),
                        html.P([html.Strong("Direction: "), study.direction.name]),
                        html.P([html.Strong("Total Trials: "), str(len(study.trials))]),
                        html.P([html.Strong("Best Value: "), f"{study.best_value:.6f}" if study.best_value else "N/A"]),
                    ])
                ])

            else:
                # JSON data
                study_data = study_info['study_data']
                overview = html.P("JSON study data loaded", className="text-info")

            return study_data, overview

        @self.app.callback(
            Output('optimization-history-plot', 'figure'),
            [Input('study-data-store', 'data')]
        )
        def update_optimization_history(study_data):
            if not study_data or 'trials' not in study_data or not study_data['trials']:
                return go.Figure()

            trials_df = pd.DataFrame(study_data['trials'])

            if 'value' not in trials_df.columns:
                return go.Figure()

            fig = go.Figure()

            # Best value progression
            best_values = []
            current_best = float('inf') if study_data.get('direction') == 'MINIMIZE' else float('-inf')

            for value in trials_df['value']:
                if pd.isna(value):
                    best_values.append(current_best)
                    continue

                if study_data.get('direction') == 'MINIMIZE':
                    current_best = min(current_best, value)
                else:
                    current_best = max(current_best, value)
                best_values.append(current_best)

            # Plot best value progression
            fig.add_trace(go.Scatter(
                x=list(range(1, len(best_values) + 1)),
                y=best_values,
                mode='lines+markers',
                name='Best Value',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))

            # Plot all trial values
            valid_trials = trials_df.dropna(subset=['value'])
            fig.add_trace(go.Scatter(
                x=valid_trials.index + 1,
                y=valid_trials['value'],
                mode='markers',
                name='Trial Values',
                marker=dict(color='lightblue', size=4, opacity=0.6)
            ))

            fig.update_layout(
                title="Optimization History",
                xaxis_title="Trial Number",
                yaxis_title="Objective Value",
                hovermode='closest',
                showlegend=True
            )

            return fig

        @self.app.callback(
            Output('param-importance-plot', 'figure'),
            [Input('study-data-store', 'data')]
        )
        def update_param_importance(study_data):
            if not study_data or 'trials' not in study_data or not study_data['trials']:
                return go.Figure()

            trials_df = pd.DataFrame(study_data['trials'])

            # Extract parameter columns
            param_cols = [col for col in trials_df.columns if col.startswith('params_')]
            if not param_cols:
                return go.Figure()

            # Calculate parameter importance (correlation with objective)
            importances = []
            param_names = []

            for col in param_cols:
                param_name = col.replace('params_', '')
                param_names.append(param_name)

                # Calculate correlation with objective
                if 'value' in trials_df.columns:
                    correlation = abs(trials_df[col].corr(trials_df['value']))
                    importances.append(correlation if not pd.isna(correlation) else 0)
                else:
                    importances.append(0)

            # Sort by importance
            sorted_data = sorted(zip(param_names, importances), key=lambda x: x[1], reverse=True)
            param_names, importances = zip(*sorted_data) if sorted_data else ([], [])

            fig = go.Figure(data=[
                go.Bar(x=list(param_names), y=list(importances),
                       marker_color='steelblue')
            ])

            fig.update_layout(
                title="Parameter Importance (Correlation with Objective)",
                xaxis_title="Parameters",
                yaxis_title="Importance Score",
                xaxis_tickangle=-45
            )

            return fig

        @self.app.callback(
            [Output('slice-param-selector', 'options'),
             Output('slice-param-selector', 'value')],
            [Input('study-data-store', 'data')]
        )
        def update_slice_selector(study_data):
            if not study_data or 'trials' not in study_data:
                return [], None

            trials_df = pd.DataFrame(study_data['trials'])
            param_cols = [col for col in trials_df.columns if col.startswith('params_')]
            param_names = [col.replace('params_', '') for col in param_cols]

            options = [{'label': name, 'value': name} for name in param_names]
            default_value = param_names[0] if param_names else None

            return options, default_value

        @self.app.callback(
            Output('parallel-coordinate-plot', 'figure'),
            [Input('study-data-store', 'data')]
        )
        def update_parallel_coordinate(study_data):
            if not study_data or 'trials' not in study_data or not study_data['trials']:
                return go.Figure()

            trials_df = pd.DataFrame(study_data['trials'])
            param_cols = [col for col in trials_df.columns if col.startswith('params_')]

            if len(param_cols) < 2 or 'value' not in trials_df.columns:
                return go.Figure()

            # Prepare data for parallel coordinates
            dimensions = []

            # Add objective value
            dimensions.append(dict(
                label='Objective Value',
                values=trials_df['value'].dropna(),
                range=[trials_df['value'].min(), trials_df['value'].max()]
            ))

            # Add parameters
            for col in param_cols[:8]:  # Limit to 8 parameters for readability
                param_name = col.replace('params_', '')
                param_values = trials_df[col].dropna()
                if len(param_values) > 0:
                    dimensions.append(dict(
                        label=param_name,
                        values=param_values,
                        range=[param_values.min(), param_values.max()]
                    ))

            fig = go.Figure(data=go.Parcoords(
                line=dict(color=trials_df['value'], colorscale='Viridis', showscale=True),
                dimensions=dimensions
            ))

            fig.update_layout(
                title="Parameter Relationships (Parallel Coordinates)",
                height=500
            )

            return fig

        @self.app.callback(
            Output('best-trials-table', 'children'),
            [Input('study-data-store', 'data')]
        )
        def update_best_trials_table(study_data):
            if not study_data or 'trials' not in study_data or not study_data['trials']:
                return html.P("No trials data available")

            trials_df = pd.DataFrame(study_data['trials'])

            if 'value' not in trials_df.columns:
                return html.P("No objective values available")

            # Sort by objective value
            ascending = study_data.get('direction') == 'MINIMIZE'
            best_trials = trials_df.dropna(subset=['value']).sort_values('value', ascending=ascending).head(10)

            # Prepare table data
            table_data = []
            param_cols = [col for col in best_trials.columns if col.startswith('params_')]

            for idx, (_, row) in enumerate(best_trials.iterrows()):
                trial_data = {
                    'Rank': idx + 1,
                    'Trial': int(row.get('number', 0)),
                    'Value': f"{row['value']:.6f}"
                }

                # Add parameters
                for col in param_cols:
                    param_name = col.replace('params_', '')
                    trial_data[param_name] = row[col]

                table_data.append(trial_data)

            if not table_data:
                return html.P("No completed trials available")

            columns = list(table_data[0].keys())

            return dash_table.DataTable(
                data=table_data,
                columns=[{"name": col, "id": col} for col in columns],
                style_cell={'textAlign': 'left', 'fontSize': 12},
                style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 0},
                        'backgroundColor': 'gold',
                        'color': 'black',
                    }
                ],
                page_size=10
            )

    def create_study_comparison_dashboard(self, study_names: List[str]) -> Figure:
        """Create comparison dashboard for multiple studies"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available")
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization History Comparison', 'Best Value Comparison',
                          'Trial Count Comparison', 'Parameter Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        colors = px.colors.qualitative.Set1

        for i, study_name in enumerate(study_names):
            if study_name not in self.available_studies:
                continue

            study_info = self.available_studies[study_name]
            color = colors[i % len(colors)]

            if study_info['type'] == 'database':
                study = study_info['study']
                trials_df = study.trials_dataframe()

                if not trials_df.empty and 'value' in trials_df.columns:
                    # Optimization history
                    best_values = []
                    current_best = float('inf') if study.direction.name == 'MINIMIZE' else float('-inf')

                    for value in trials_df['value']:
                        if pd.isna(value):
                            best_values.append(current_best)
                            continue

                        if study.direction.name == 'MINIMIZE':
                            current_best = min(current_best, value)
                        else:
                            current_best = max(current_best, value)
                        best_values.append(current_best)

                    fig.add_trace(
                        go.Scatter(x=list(range(1, len(best_values) + 1)), y=best_values,
                                 mode='lines', name=f'{study_name} - Best',
                                 line=dict(color=color)),
                        row=1, col=1
                    )

        fig.update_layout(title_text="Multi-Study Comparison Dashboard", showlegend=True)
        return fig

    def export_study_report(self, study_name: str, output_path: str = None) -> str:
        """Export comprehensive study analysis report"""
        if study_name not in self.available_studies:
            raise ValueError(f"Study {study_name} not found")

        study_info = self.available_studies[study_name]
        output_path = output_path or f"study_report_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        # Create comprehensive report
        report_html = f"""
        <html>
        <head>
            <title>Hyperparameter Optimization Report: {study_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Hyperparameter Optimization Report</h1>
                <h2>Study: {study_name}</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        if study_info['type'] == 'database':
            study = study_info['study']
            trials_df = study.trials_dataframe()

            report_html += f"""
            <div class="section">
                <h3>📊 Study Overview</h3>
                <div class="metric"><strong>Direction:</strong> {study.direction.name}</div>
                <div class="metric"><strong>Total Trials:</strong> {len(study.trials)}</div>
                <div class="metric"><strong>Completed Trials:</strong> {len([t for t in study.trials if t.state.name == 'COMPLETE'])}</div>
                <div class="metric"><strong>Best Value:</strong> {study.best_value:.6f if study.best_value else 'N/A'}</div>
            </div>

            <div class="section">
                <h3>🏆 Best Parameters</h3>
                <ul>
            """

            for param, value in study.best_params.items():
                report_html += f"<li><strong>{param}:</strong> {value}</li>"

            report_html += """
                </ul>
            </div>
            </body>
            </html>
            """

        with open(output_path, 'w') as f:
            f.write(report_html)

        logger.info(f"Study report exported to: {output_path}")
        return output_path

    def run_dashboard(self, debug: bool = False):
        """Run the hyperparameter optimization dashboard"""
        if not self.app:
            logger.error("Dashboard not initialized")
            return

        print(f"🎯 Starting Hyperparameter Optimization Dashboard on http://localhost:{self.dashboard_port}")
        print("📊 Analyze your optimization studies interactively!")
        print("🔍 Explore parameter importance, relationships, and best trials")

        self.app.run(debug=debug, host='0.0.0.0', port=self.dashboard_port)

if __name__ == "__main__":
    # Example usage
    dashboard = HyperparameterOptimizationDashboard()
    dashboard.run_dashboard(debug=False)