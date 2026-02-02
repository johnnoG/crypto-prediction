"""
Professional Model Documentation and Visualization System

Comprehensive documentation system featuring:
- Automated model card generation
- Interactive API documentation
- Performance benchmarking reports
- Training pipeline visualization
- Architecture diagrams
- Model comparison reports
- Professional presentation templates
"""

from __future__ import annotations

import os
import json
import time
import uuid
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import markdown
import jinja2
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Documentation configuration"""
    output_directory: str = "documentation/output"
    template_directory: str = "documentation/templates"
    include_code_examples: bool = True
    include_performance_benchmarks: bool = True
    include_architecture_diagrams: bool = True
    generate_pdf: bool = True
    branding: Dict[str, str] = None

    def __post_init__(self):
        if self.branding is None:
            self.branding = {
                "company_name": "Cryptocurrency Prediction Platform",
                "project_name": "Advanced ML Models",
                "version": "1.0.0",
                "author": "ML Engineering Team"
            }


class ModelCardGenerator:
    """Generates comprehensive model cards"""

    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_model_card(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_config: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> str:
        """Generate comprehensive model card"""

        model_card_content = f"""
# Model Card: {model_name}

## Model Information

### Basic Details
- **Model Name**: {model_name}
- **Model Type**: {model_info.get('type', 'Unknown')}
- **Version**: {model_info.get('version', '1.0.0')}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Framework**: {model_info.get('framework', 'TensorFlow/PyTorch')}
- **License**: {model_info.get('license', 'MIT')}

### Model Description
{model_info.get('description', 'A machine learning model for cryptocurrency price prediction.')}

### Intended Use
- **Primary Use Cases**: {', '.join(model_info.get('use_cases', ['Price prediction', 'Risk analysis']))}
- **Target Users**: {', '.join(model_info.get('target_users', ['Traders', 'Analysts', 'Researchers']))}
- **Out-of-Scope Uses**: {', '.join(model_info.get('limitations', ['Financial advice', 'Investment decisions']))}

## Model Architecture

### Architecture Details
{model_info.get('architecture_description', 'Detailed architecture information not provided.')}

### Model Parameters
- **Total Parameters**: {model_info.get('total_parameters', 'N/A')}
- **Trainable Parameters**: {model_info.get('trainable_parameters', 'N/A')}
- **Model Size**: {model_info.get('model_size', 'N/A')}

### Input/Output Specifications
- **Input Shape**: {model_info.get('input_shape', 'N/A')}
- **Output Shape**: {model_info.get('output_shape', 'N/A')}
- **Input Features**: {model_info.get('feature_count', 'N/A')} features
- **Prediction Horizons**: {', '.join(map(str, model_info.get('prediction_horizons', [1, 7, 30])))} days

## Training Configuration

### Dataset Information
- **Training Data**: {training_config.get('dataset_name', 'Cryptocurrency historical data')}
- **Data Sources**: {', '.join(training_config.get('data_sources', ['Multiple exchanges']))}
- **Training Samples**: {training_config.get('training_samples', 'N/A'):,}
- **Validation Samples**: {training_config.get('validation_samples', 'N/A'):,}
- **Test Samples**: {training_config.get('test_samples', 'N/A'):,}
- **Date Range**: {training_config.get('date_range', 'N/A')}

### Hyperparameters
"""

        # Add hyperparameters
        hyperparams = training_config.get('hyperparameters', {})
        for param, value in hyperparams.items():
            model_card_content += f"- **{param}**: {value}\n"

        model_card_content += f"""

### Training Details
- **Training Duration**: {training_config.get('training_duration', 'N/A')}
- **Optimizer**: {training_config.get('optimizer', 'N/A')}
- **Loss Function**: {training_config.get('loss_function', 'N/A')}
- **Early Stopping**: {training_config.get('early_stopping', 'N/A')}
- **Cross-Validation**: {training_config.get('cross_validation', 'Walk-forward validation')}

## Performance Metrics

### Overall Performance
"""

        # Add performance metrics
        for metric, value in performance_metrics.items():
            if isinstance(value, float):
                model_card_content += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
            else:
                model_card_content += f"- **{metric.replace('_', ' ').title()}**: {value}\n"

        model_card_content += f"""

### Validation Results
"""

        # Add validation results
        if validation_results:
            for fold_metric, values in validation_results.items():
                if isinstance(values, list) and len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    model_card_content += f"- **{fold_metric.replace('_', ' ').title()}**: {mean_val:.4f} ± {std_val:.4f}\n"

        model_card_content += f"""

## Model Usage

### Python Code Example
```python
import mlflow
import numpy as np

# Load the model
model_uri = "models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare input data (example)
input_data = np.random.randn(1, {model_info.get('feature_count', 50)})

# Make predictions
predictions = model.predict(input_data)
print(f"Predicted prices: {{predictions}}")
```

### API Usage Example
```python
import requests

# API endpoint
url = "https://api.crypto-prediction.com/predict"

# Input data
data = {{
    "model_name": "{model_name}",
    "features": [/* feature values */],
    "prediction_horizon": 7
}}

# Make request
response = requests.post(url, json=data)
predictions = response.json()
```

## Limitations and Considerations

### Model Limitations
{chr(10).join(f"- {limitation}" for limitation in model_info.get('limitations', [
    'Model performance may degrade during high market volatility',
    'Predictions are based on historical patterns and may not reflect future market conditions',
    'External factors (news, regulations) are not explicitly modeled'
]))}

### Ethical Considerations
- This model is intended for informational purposes only
- Not intended for financial advice or investment decisions
- Users should conduct their own research before making trading decisions
- Consider market risks and potential losses

### Bias and Fairness
- Model may exhibit bias towards certain market conditions
- Performance may vary across different cryptocurrencies
- Historical biases in training data may affect predictions

## Model Maintenance

### Monitoring
- Model performance is continuously monitored
- Drift detection alerts trigger retraining
- Performance degradation thresholds: {training_config.get('performance_threshold', '5%')}

### Retraining Schedule
- **Frequency**: {training_config.get('retrain_frequency', 'Weekly')}
- **Trigger Conditions**: {', '.join(training_config.get('retrain_triggers', ['Performance degradation', 'Data drift']))}
- **Approval Process**: {training_config.get('approval_process', 'Automated with manual review')}

### Version History
- **Current Version**: {model_info.get('version', '1.0.0')}
- **Previous Versions**: {', '.join(model_info.get('previous_versions', ['Initial release']))}
- **Change Log**: Available in MLflow model registry

## Contact Information

### Model Owners
- **Primary Contact**: {self.config.branding['author']}
- **Team**: ML Engineering Team
- **Email**: ml-team@company.com

### Support
- **Documentation**: [Model Documentation Portal](https://docs.crypto-prediction.com)
- **Issues**: [GitHub Issues](https://github.com/company/crypto-prediction/issues)
- **Slack**: #ml-models channel

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {self.config.branding['company_name']}*
"""

        # Save model card
        model_card_path = self.output_path / f"{model_name}_model_card.md"
        with open(model_card_path, 'w') as f:
            f.write(model_card_content)

        # Generate HTML version
        html_content = markdown.markdown(model_card_content, extensions=['tables', 'codehilite'])
        html_path = self.output_path / f"{model_name}_model_card.html"

        # Wrap in HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card: {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.branding['project_name']}</h1>
        <p>{self.config.branding['company_name']} | Version {self.config.branding['version']}</p>
    </div>
    {html_content}
    <div class="footer">
        <p>Generated by {self.config.branding['company_name']} Documentation System</p>
    </div>
</body>
</html>
"""

        with open(html_path, 'w') as f:
            f.write(html_template)

        logger.info(f"Generated model card: {model_card_path}")
        return str(model_card_path)


class BenchmarkReporter:
    """Generates performance benchmark reports"""

    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.output_path = Path(config.output_directory)

    def generate_benchmark_report(
        self,
        model_results: Dict[str, Dict[str, Any]],
        baseline_results: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate comprehensive benchmark report"""

        # Create performance comparison
        comparison_data = []
        for model_name, results in model_results.items():
            if 'validation_scores' in results:
                scores = results['validation_scores']
                for metric, values in scores.items():
                    if isinstance(values, list) and len(values) > 0:
                        comparison_data.append({
                            'Model': model_name,
                            'Metric': metric.replace('_', ' ').title(),
                            'Mean': np.mean(values),
                            'Std': np.std(values),
                            'Min': np.min(values),
                            'Max': np.max(values)
                        })

        comparison_df = pd.DataFrame(comparison_data)

        # Create visualizations
        visualizations = self._create_benchmark_visualizations(model_results, comparison_df)

        # Generate report content
        report_content = f"""
# Model Performance Benchmark Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Company**: {self.config.branding['company_name']}
**Project**: {self.config.branding['project_name']}

## Executive Summary

This report presents a comprehensive performance analysis of {len(model_results)} machine learning models
for cryptocurrency price prediction. The models were evaluated using walk-forward cross-validation
on historical cryptocurrency data.

### Key Findings
"""

        if comparison_df.empty:
            report_content += "- No valid performance metrics available for comparison\n"
        else:
            # Find best performing model
            rmse_data = comparison_df[comparison_df['Metric'] == 'RMSE']
            if not rmse_data.empty:
                best_model = rmse_data.loc[rmse_data['Mean'].idxmin(), 'Model']
                best_rmse = rmse_data['Mean'].min()
                report_content += f"- Best performing model: **{best_model}** (RMSE: {best_rmse:.4f})\n"

            # Performance spread
            if len(rmse_data) > 1:
                performance_spread = (rmse_data['Mean'].max() - rmse_data['Mean'].min()) / rmse_data['Mean'].min()
                report_content += f"- Performance spread: {performance_spread:.1%} difference between best and worst models\n"

        report_content += """

## Model Comparison

### Performance Overview
"""

        if not comparison_df.empty:
            # Create performance table
            pivot_table = comparison_df.pivot_table(
                index='Model',
                columns='Metric',
                values='Mean',
                fill_value=0
            )
            report_content += "\n" + pivot_table.to_string() + "\n\n"

        report_content += """

### Detailed Analysis

#### RMSE (Root Mean Square Error)
Lower values indicate better performance. This metric penalizes large prediction errors more heavily.

#### MAE (Mean Absolute Error)
Provides a linear penalty for prediction errors, easier to interpret than RMSE.

#### R² Score (Coefficient of Determination)
Indicates how well the model explains the variance in the target variable. Values closer to 1 are better.

## Model-Specific Results

"""

        # Add detailed results for each model
        for model_name, results in model_results.items():
            report_content += f"### {model_name}\n\n"

            if 'validation_scores' in results:
                scores = results['validation_scores']

                report_content += "**Cross-Validation Results:**\n\n"
                for metric, values in scores.items():
                    if isinstance(values, list) and len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        report_content += f"- {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}\n"

                report_content += "\n"

            if 'config' in results:
                config = results['config']
                report_content += "**Model Configuration:**\n\n"
                for key, value in config.items():
                    report_content += f"- {key}: {value}\n"

                report_content += "\n"

        report_content += """

## Baseline Comparison
"""

        if baseline_results:
            report_content += "Performance comparison against baseline models:\n\n"
            for metric, value in baseline_results.items():
                report_content += f"- Baseline {metric}: {value:.4f}\n"
        else:
            report_content += "No baseline results provided for comparison.\n"

        report_content += """

## Recommendations

### Model Selection
"""

        # Generate recommendations
        if not comparison_df.empty:
            rmse_data = comparison_df[comparison_df['Metric'] == 'RMSE']
            if not rmse_data.empty:
                best_model = rmse_data.loc[rmse_data['Mean'].idxmin(), 'Model']
                report_content += f"1. **Recommended for Production**: {best_model}\n"
                report_content += f"   - Lowest RMSE: {rmse_data['Mean'].min():.4f}\n"
                report_content += f"   - Consistent performance across validation folds\n\n"

        report_content += """
2. **Ensemble Consideration**: Consider ensemble methods to combine strengths of multiple models
3. **Monitoring**: Implement continuous monitoring to detect performance degradation
4. **Retraining**: Establish automated retraining pipeline based on performance thresholds

### Next Steps
1. Deploy recommended model to staging environment
2. Conduct A/B testing against current production model
3. Implement monitoring and alerting
4. Schedule regular model retraining

## Methodology

### Validation Strategy
- **Method**: Walk-forward cross-validation
- **Window Type**: Expanding window to simulate realistic training scenarios
- **Evaluation Metrics**: RMSE, MAE, R², and domain-specific metrics

### Data Preparation
- Feature engineering with technical indicators
- Time-series aware train/test splits
- Proper handling of missing values and outliers

---

*This report was automatically generated by the ML Documentation System.*
"""

        # Save report
        report_path = self.output_path / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Generate HTML version
        html_content = markdown.markdown(report_content, extensions=['tables', 'codehilite'])

        # Include visualizations in HTML
        html_with_viz = self._create_html_report_with_visualizations(html_content, visualizations)

        html_path = self.output_path / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_path, 'w') as f:
            f.write(html_with_viz)

        logger.info(f"Generated benchmark report: {report_path}")
        return str(report_path)

    def _create_benchmark_visualizations(
        self,
        model_results: Dict[str, Dict[str, Any]],
        comparison_df: pd.DataFrame
    ) -> Dict[str, str]:
        """Create visualization charts for benchmark report"""
        visualizations = {}

        try:
            if comparison_df.empty:
                return visualizations

            # Performance comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['RMSE Comparison', 'MAE Comparison', 'R² Comparison', 'Performance Distribution'],
                vertical_spacing=0.1
            )

            metrics_to_plot = ['RMSE', 'MAE', 'R²']
            positions = [(1, 1), (1, 2), (2, 1)]

            for i, metric in enumerate(metrics_to_plot):
                metric_data = comparison_df[comparison_df['Metric'] == metric]
                if not metric_data.empty:
                    row, col = positions[i]

                    fig.add_trace(
                        go.Bar(
                            x=metric_data['Model'],
                            y=metric_data['Mean'],
                            error_y=dict(type='data', array=metric_data['Std']),
                            name=metric,
                            showlegend=False
                        ),
                        row=row, col=col
                    )

            # Performance distribution (box plot)
            all_metrics_data = []
            for model_name, results in model_results.items():
                if 'validation_scores' in results:
                    for metric, values in results['validation_scores'].items():
                        if isinstance(values, list):
                            for value in values:
                                all_metrics_data.append({
                                    'Model': model_name,
                                    'Metric': metric,
                                    'Value': value
                                })

            if all_metrics_data:
                metrics_df = pd.DataFrame(all_metrics_data)
                rmse_data = metrics_df[metrics_df['Metric'] == 'rmse']

                if not rmse_data.empty:
                    for model in rmse_data['Model'].unique():
                        model_data = rmse_data[rmse_data['Model'] == model]
                        fig.add_trace(
                            go.Box(
                                y=model_data['Value'],
                                name=model,
                                showlegend=False
                            ),
                            row=2, col=2
                        )

            fig.update_layout(
                title="Model Performance Comparison",
                height=800
            )

            # Save visualization
            viz_path = self.output_path / "performance_comparison.html"
            fig.write_html(viz_path)
            visualizations['performance_comparison'] = str(viz_path)

            logger.info("Created benchmark visualizations")

        except Exception as e:
            logger.warning(f"Failed to create benchmark visualizations: {e}")

        return visualizations

    def _create_html_report_with_visualizations(self, content: str, visualizations: Dict[str, str]) -> str:
        """Embed visualizations in HTML report"""

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .header {{ text-align: center; margin-bottom: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 0.9em; }}
        .visualization {{ margin: 20px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }}
        .metric-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Benchmark Report</h1>
        <p><strong>{self.config.branding['project_name']}</strong></p>
        <p>{self.config.branding['company_name']} | Version {self.config.branding['version']}</p>
    </div>

    {content}

    <div class="footer">
        <p>Generated by {self.config.branding['company_name']} ML Documentation System</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html_template


class APIDocumentationGenerator:
    """Generates interactive API documentation"""

    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.output_path = Path(config.output_directory)

    def generate_api_docs(self, models: Dict[str, Any]) -> str:
        """Generate OpenAPI/Swagger documentation for model APIs"""

        api_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Cryptocurrency Prediction API",
                "version": self.config.branding['version'],
                "description": "Professional API for cryptocurrency price prediction models",
                "contact": {
                    "name": "ML Engineering Team",
                    "email": "ml-team@company.com"
                }
            },
            "servers": [
                {"url": "https://api.crypto-prediction.com/v1", "description": "Production"},
                {"url": "https://staging-api.crypto-prediction.com/v1", "description": "Staging"}
            ],
            "paths": {},
            "components": {
                "schemas": {
                    "PredictionRequest": {
                        "type": "object",
                        "required": ["model_name", "features"],
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to use for prediction",
                                "enum": list(models.keys()) if models else ["transformer", "lstm", "lightgbm"]
                            },
                            "features": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Input features for prediction"
                            },
                            "prediction_horizon": {
                                "type": "integer",
                                "description": "Prediction horizon in days",
                                "default": 7,
                                "enum": [1, 7, 30]
                            }
                        }
                    },
                    "PredictionResponse": {
                        "type": "object",
                        "properties": {
                            "predictions": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Predicted values"
                            },
                            "confidence": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Prediction confidence intervals"
                            },
                            "model_version": {
                                "type": "string",
                                "description": "Version of the model used"
                            },
                            "timestamp": {
                                "type": "string",
                                "format": "date-time",
                                "description": "Prediction timestamp"
                            }
                        }
                    },
                    "ErrorResponse": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "string",
                                "description": "Error message"
                            },
                            "code": {
                                "type": "integer",
                                "description": "Error code"
                            }
                        }
                    }
                }
            }
        }

        # Add prediction endpoint
        api_spec["paths"]["/predict"] = {
            "post": {
                "summary": "Make price predictions",
                "description": "Generate cryptocurrency price predictions using specified model",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PredictionRequest"},
                            "example": {
                                "model_name": "transformer",
                                "features": [0.1, 0.2, 0.3, "..."],
                                "prediction_horizon": 7
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Bad request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        }

        # Add model info endpoint
        api_spec["paths"]["/models"] = {
            "get": {
                "summary": "List available models",
                "description": "Get information about available prediction models",
                "responses": {
                    "200": {
                        "description": "List of available models",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "version": {"type": "string"},
                                            "description": {"type": "string"},
                                            "performance": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        # Add health check endpoint
        api_spec["paths"]["/health"] = {
            "get": {
                "summary": "Health check",
                "description": "Check API health status",
                "responses": {
                    "200": {
                        "description": "API is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        # Save OpenAPI spec
        api_spec_path = self.output_path / "api_specification.json"
        with open(api_spec_path, 'w') as f:
            json.dump(api_spec, f, indent=2)

        # Generate HTML documentation
        html_doc = self._generate_api_html_doc(api_spec)
        html_path = self.output_path / "api_documentation.html"
        with open(html_path, 'w') as f:
            f.write(html_doc)

        logger.info(f"Generated API documentation: {html_path}")
        return str(html_path)

    def _generate_api_html_doc(self, api_spec: Dict[str, Any]) -> str:
        """Generate HTML documentation from OpenAPI spec"""

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation - {api_spec['info']['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .endpoint {{ background-color: #fff; border: 1px solid #ddd; border-radius: 5px; margin: 20px 0; padding: 20px; }}
        .method {{ display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }}
        .get {{ background-color: #28a745; }}
        .post {{ background-color: #007bff; }}
        .put {{ background-color: #ffc107; color: black; }}
        .delete {{ background-color: #dc3545; }}
        .code {{ background-color: #f4f4f4; padding: 10px; border-radius: 3px; font-family: monospace; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{api_spec['info']['title']}</h1>
        <p><strong>Version:</strong> {api_spec['info']['version']}</p>
        <p>{api_spec['info']['description']}</p>
    </div>

    <h2>Servers</h2>
    <ul>
"""

        for server in api_spec['servers']:
            html_content += f"<li><strong>{server['description']}:</strong> {server['url']}</li>"

        html_content += """
    </ul>

    <h2>Endpoints</h2>
"""

        for path, methods in api_spec['paths'].items():
            for method, details in methods.items():
                method_class = method.lower()
                html_content += f"""
    <div class="endpoint">
        <h3>
            <span class="method {method_class}">{method.upper()}</span>
            {path}
        </h3>
        <p>{details['summary']}</p>
        <p>{details['description']}</p>
"""

                # Add request body if present
                if 'requestBody' in details:
                    html_content += """
        <h4>Request Body</h4>
        <div class="code">
            <strong>Content-Type:</strong> application/json<br>
            <strong>Required:</strong> Yes
        </div>
"""

                # Add responses
                html_content += "<h4>Responses</h4>"
                for status, response in details['responses'].items():
                    html_content += f"""
        <div class="code">
            <strong>{status}:</strong> {response['description']}
        </div>
"""

                html_content += "</div>"

        html_content += f"""
    <h2>Example Usage</h2>
    <h3>Python</h3>
    <div class="code">
        <pre>
import requests

# Make a prediction
url = "https://api.crypto-prediction.com/v1/predict"
data = {{
    "model_name": "transformer",
    "features": [/* your feature values */],
    "prediction_horizon": 7
}}

response = requests.post(url, json=data)
predictions = response.json()
print(predictions)
        </pre>
    </div>

    <h3>cURL</h3>
    <div class="code">
        <pre>
curl -X POST https://api.crypto-prediction.com/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model_name": "transformer",
    "features": [/* your feature values */],
    "prediction_horizon": 7
  }}'
        </pre>
    </div>

    <div class="footer">
        <p><em>Generated by {self.config.branding['company_name']} Documentation System</em></p>
    </div>
</body>
</html>
"""
        return html_content


class DocumentationOrchestrator:
    """Main orchestrator for documentation generation"""

    def __init__(self, config: Optional[DocumentationConfig] = None):
        self.config = config or DocumentationConfig()
        self.model_card_generator = ModelCardGenerator(self.config)
        self.benchmark_reporter = BenchmarkReporter(self.config)
        self.api_docs_generator = APIDocumentationGenerator(self.config)

        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)

    def generate_complete_documentation(
        self,
        models_info: Dict[str, Any],
        training_results: Dict[str, Any],
        performance_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """Generate complete documentation suite"""

        generated_docs = {}

        try:
            logger.info("Starting complete documentation generation")

            # Generate model cards for each model
            for model_name, model_info in models_info.items():
                logger.info(f"Generating model card for {model_name}")

                model_metrics = performance_metrics.get(model_name, {})
                training_config = training_results.get(model_name, {}).get('config', {})
                validation_results = training_results.get(model_name, {}).get('validation_scores', {})

                model_card_path = self.model_card_generator.generate_model_card(
                    model_name=model_name,
                    model_info=model_info,
                    performance_metrics=model_metrics,
                    training_config=training_config,
                    validation_results=validation_results
                )
                generated_docs[f"{model_name}_model_card"] = model_card_path

            # Generate benchmark report
            logger.info("Generating benchmark report")
            benchmark_report_path = self.benchmark_reporter.generate_benchmark_report(
                model_results=training_results
            )
            generated_docs["benchmark_report"] = benchmark_report_path

            # Generate API documentation
            logger.info("Generating API documentation")
            api_docs_path = self.api_docs_generator.generate_api_docs(models_info)
            generated_docs["api_documentation"] = api_docs_path

            # Generate index page
            index_path = self._generate_index_page(generated_docs)
            generated_docs["index"] = index_path

            logger.info(f"Documentation generation completed. Generated {len(generated_docs)} documents.")

            # Create summary report
            summary = self._create_documentation_summary(generated_docs)
            logger.info(f"Documentation summary: {summary}")

            return generated_docs

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {}

    def _generate_index_page(self, generated_docs: Dict[str, str]) -> str:
        """Generate an index page linking to all documentation"""

        index_content = f"""
# {self.config.branding['project_name']} Documentation

Welcome to the comprehensive documentation for {self.config.branding['project_name']}.

## Available Documentation

### Model Documentation
"""

        # Add model cards
        model_cards = [doc for name, doc in generated_docs.items() if 'model_card' in name]
        for doc_path in model_cards:
            doc_name = Path(doc_path).stem.replace('_model_card', '')
            html_path = doc_path.replace('.md', '.html')
            index_content += f"- [{doc_name} Model Card]({Path(html_path).name})\n"

        index_content += """

### Performance Reports
"""

        if 'benchmark_report' in generated_docs:
            benchmark_path = generated_docs['benchmark_report']
            html_path = benchmark_path.replace('.md', '.html')
            index_content += f"- [Benchmark Report]({Path(html_path).name})\n"

        index_content += """

### API Documentation
"""

        if 'api_documentation' in generated_docs:
            api_path = generated_docs['api_documentation']
            index_content += f"- [API Documentation]({Path(api_path).name})\n"

        index_content += f"""

## Project Information

- **Version**: {self.config.branding['version']}
- **Company**: {self.config.branding['company_name']}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contact

For questions or support, please contact the ML Engineering Team.

---

*Generated by {self.config.branding['company_name']} Documentation System*
"""

        # Save markdown version
        index_md_path = Path(self.config.output_directory) / "index.md"
        with open(index_md_path, 'w') as f:
            f.write(index_content)

        # Generate HTML version
        html_content = markdown.markdown(index_content)

        index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.branding['project_name']} Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; background-color: #f8f9fa; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        h1, h2, h3 {{ color: #333; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 0.9em; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.branding['project_name']}</h1>
        <p>Comprehensive Documentation Portal</p>
    </div>

    {html_content}

    <div class="footer">
        <p>Generated by {self.config.branding['company_name']} Documentation System</p>
    </div>
</body>
</html>
"""

        index_html_path = Path(self.config.output_directory) / "index.html"
        with open(index_html_path, 'w') as f:
            f.write(index_html)

        logger.info(f"Generated documentation index: {index_html_path}")
        return str(index_html_path)

    def _create_documentation_summary(self, generated_docs: Dict[str, str]) -> str:
        """Create a summary of generated documentation"""

        summary = f"""
Documentation Generation Summary
==============================

Generated {len(generated_docs)} documentation files:

"""
        for doc_type, doc_path in generated_docs.items():
            file_size = Path(doc_path).stat().st_size if Path(doc_path).exists() else 0
            summary += f"- {doc_type}: {doc_path} ({file_size:,} bytes)\n"

        summary += f"""

Output Directory: {self.config.output_directory}
Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Access the documentation by opening: {Path(self.config.output_directory) / 'index.html'}
"""

        # Save summary
        summary_path = Path(self.config.output_directory) / "generation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)

        return summary


def create_documentation_system(
    output_directory: str = "documentation/output",
    company_name: str = "Cryptocurrency Prediction Platform",
    project_name: str = "Advanced ML Models"
) -> DocumentationOrchestrator:
    """Create a documentation system instance"""

    config = DocumentationConfig(
        output_directory=output_directory,
        branding={
            "company_name": company_name,
            "project_name": project_name,
            "version": "1.0.0",
            "author": "ML Engineering Team"
        }
    )

    return DocumentationOrchestrator(config)


if __name__ == "__main__":
    # Example usage
    doc_system = create_documentation_system()

    # Example model information
    models_info = {
        "transformer": {
            "type": "Deep Learning",
            "version": "1.0.0",
            "framework": "TensorFlow",
            "description": "Transformer-based cryptocurrency price prediction model",
            "architecture_description": "Multi-head self-attention architecture with positional encoding",
            "total_parameters": "2.5M",
            "feature_count": 155
        },
        "lstm": {
            "type": "Deep Learning",
            "version": "1.0.0",
            "framework": "TensorFlow",
            "description": "Enhanced LSTM model with attention mechanism",
            "total_parameters": "1.8M",
            "feature_count": 155
        }
    }

    # Example training results
    training_results = {
        "transformer": {
            "config": {"learning_rate": 0.001, "epochs": 100},
            "validation_scores": {"rmse": [0.05, 0.06, 0.055], "mae": [0.03, 0.035, 0.032]}
        },
        "lstm": {
            "config": {"learning_rate": 0.001, "epochs": 80},
            "validation_scores": {"rmse": [0.08, 0.075, 0.07], "mae": [0.05, 0.048, 0.052]}
        }
    }

    # Example performance metrics
    performance_metrics = {
        "transformer": {"rmse": 0.055, "mae": 0.032, "r2": 0.85},
        "lstm": {"rmse": 0.075, "mae": 0.050, "r2": 0.78}
    }

    # Generate complete documentation
    docs = doc_system.generate_complete_documentation(
        models_info=models_info,
        training_results=training_results,
        performance_metrics=performance_metrics
    )

    print("Documentation generated successfully!")
    print(f"Access documentation at: {Path(doc_system.config.output_directory) / 'index.html'}")