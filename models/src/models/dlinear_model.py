"""
DLinear Model for Cryptocurrency Price Forecasting

Simple linear decomposition model that splits input into trend (moving average)
and seasonal (remainder) components, applies separate linear layers to each,
then sums. ~3000 parameters total — virtually impossible to overfit.

Reference: "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class DLinearForecaster:
    """
    DLinear-based cryptocurrency price forecaster.

    Decomposes input into trend + seasonal via moving average, applies
    separate linear projections, and sums for the final prediction.
    Very few parameters — strong baseline that resists overfitting.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/dlinear"
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed")

        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[keras.Model] = None
        self.training_history: Optional[keras.callbacks.History] = None
        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            'sequence_length': 60,
            'n_features': 1,
            'kernel_size': 25,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 150,
            'early_stopping_patience': 10,
            'multi_step': [1, 7, 30],
        }

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        seq_len, n_features = input_shape
        kernel = self.config['kernel_size']
        dropout = self.config['dropout_rate']

        inputs = layers.Input(shape=input_shape, name='input')

        # Trend: moving average along the time axis
        # AveragePooling1D with same padding approximates a centered MA
        trend = layers.AveragePooling1D(
            pool_size=kernel, strides=1, padding='same', name='trend_pool'
        )(inputs)

        # Seasonal: remainder after removing trend
        seasonal = layers.Subtract(name='seasonal_subtract')([inputs, trend])

        # Flatten both components
        trend_flat = layers.Flatten(name='trend_flatten')(trend)
        seasonal_flat = layers.Flatten(name='seasonal_flatten')(seasonal)

        trend_drop = layers.Dropout(dropout, name='trend_dropout')(trend_flat)
        seasonal_drop = layers.Dropout(dropout, name='seasonal_dropout')(seasonal_flat)

        # Multi-step output heads
        outputs = {}
        for horizon in self.config['multi_step']:
            t_pred = layers.Dense(1, name=f'trend_pred_{horizon}d')(trend_drop)
            s_pred = layers.Dense(1, name=f'seasonal_pred_{horizon}d')(seasonal_drop)
            out = layers.Add(name=f'output_{horizon}d')([t_pred, s_pred])
            outputs[f'output_{horizon}d'] = out

        model = keras.Model(inputs=inputs, outputs=outputs, name='DLinear')

        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
        )

        losses = {f'output_{h}d': 'huber' for h in self.config['multi_step']}
        loss_weights = {f'output_{h}d': float(1.0 / np.sqrt(h)) for h in self.config['multi_step']}

        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        multi_step_targets: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        seq_len = sequence_length or self.config['sequence_length']
        horizons = multi_step_targets or self.config['multi_step']
        max_horizon = max(horizons)

        X, y_dict = [], {f'{h}d': [] for h in horizons}

        for i in range(seq_len, len(data) - max_horizon):
            X.append(data[i - seq_len:i])
            for h in horizons:
                if i + h < len(data):
                    y_dict[f'{h}d'].append(data[i + h, 0])

        X = np.array(X)
        y = {f'output_{h}d': np.array(y_dict[f'{h}d']) for h in horizons}
        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train,
        X_val: Optional[np.ndarray] = None,
        y_val=None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        print("Training DLinear model...")
        print("=" * 50)

        if len(X_train.shape) == 3:
            self.config['sequence_length'] = X_train.shape[1]
            self.config['n_features'] = X_train.shape[2]

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self._build_model(input_shape)
        self.model.summary()

        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_dlinear.weights.h5'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                save_weights_only=True,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=verbose,
        )

        self.training_history = history

        # Compute metrics
        train_pred = self.model.predict(X_train, verbose=0)
        metrics = {}
        if isinstance(y_train, dict):
            output_names = list(y_train.keys())
            for i, output_name in enumerate(output_names):
                horizon = output_name.replace('output_', '').replace('d', '')
                pred_arr = train_pred[i] if isinstance(train_pred, list) else train_pred[output_name]
                train_rmse = np.sqrt(np.mean((y_train[output_name] - np.asarray(pred_arr).flatten()) ** 2))
                metrics[f'train_rmse_{horizon}d'] = float(train_rmse)
            metrics['epochs_trained'] = len(history.history['loss'])

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val, verbose=0)
            if isinstance(y_val, dict):
                for i, output_name in enumerate(list(y_val.keys())):
                    horizon = output_name.replace('output_', '').replace('d', '')
                    pred_arr = val_pred[i] if isinstance(val_pred, list) else val_pred[output_name]
                    val_rmse = np.sqrt(np.mean((y_val[output_name] - np.asarray(pred_arr).flatten()) ** 2))
                    metrics[f'val_rmse_{horizon}d'] = float(val_rmse)

        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val) if X_val is not None else 0,
            'input_shape': list(input_shape),
            'config': self.config,
            'metrics': metrics,
        }

        print("Training complete!")
        for key, value in metrics.items():
            if 'rmse' in key:
                print(f"{key}: {value:.4f}")

        return metrics

    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False,
        num_simulations: int = 50,
    ):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X, verbose=0)

        if isinstance(predictions, list):
            predictions = {f'horizon_{i+1}': np.asarray(pred).flatten()
                           for i, pred in enumerate(predictions)}

        if return_confidence:
            confidence_predictions = []
            for _ in range(num_simulations):
                pred = self.model(X, training=True)
                if isinstance(pred, dict):
                    pred = {k: v.numpy().flatten() for k, v in pred.items()}
                elif isinstance(pred, list):
                    pred = [p.numpy().flatten() for p in pred]
                else:
                    pred = pred.numpy().flatten()
                confidence_predictions.append(pred)

            if isinstance(confidence_predictions[0], (dict, list)):
                n_outputs = len(confidence_predictions[0]) if isinstance(confidence_predictions[0], list) else len(confidence_predictions[0])
                confidence_intervals = []
                for i in range(n_outputs):
                    if isinstance(confidence_predictions[0], dict):
                        key = list(confidence_predictions[0].keys())[i]
                        horizon_preds = np.array([cp[key] for cp in confidence_predictions])
                    else:
                        horizon_preds = np.array([cp[i] for cp in confidence_predictions])
                    lower = np.percentile(horizon_preds, 2.5, axis=0)
                    upper = np.percentile(horizon_preds, 97.5, axis=0)
                    confidence_intervals.append(np.column_stack([lower, upper]))
            else:
                arr = np.array(confidence_predictions)
                lower = np.percentile(arr, 2.5, axis=0)
                upper = np.percentile(arr, 97.5, axis=0)
                confidence_intervals = np.column_stack([lower, upper])

            return predictions, confidence_intervals

        return predictions

    def save_model(self, version: str = "1.0.0") -> Path:
        if self.model is None:
            raise ValueError("No model to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"dlinear_v{version}_{timestamp}.h5"
        metadata_path = self.model_dir / f"dlinear_v{version}_{timestamp}_metadata.json"

        self.model.save(model_path)

        metadata = {
            **self.metadata,
            'version': version,
            'model_path': str(model_path),
            'framework': 'tensorflow',
            'model_type': 'dlinear',
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"DLinear model saved to {model_path}")
        return model_path

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = keras.models.load_model(model_path)

        metadata_path = model_path.parent / model_path.name.replace('.h5', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.config = self.metadata.get('config', self._default_config())

        print(f"DLinear model loaded from {model_path}")
