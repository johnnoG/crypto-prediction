"""
Temporal Convolutional Network (TCN) for Cryptocurrency Price Forecasting

Dilated causal convolutions with residual connections. Parallelizable on
Metal GPU, fewer parameters than LSTM for the same receptive field.

Receptive field with 4 blocks, kernel_size=3:
  2^4 * (3-1) * 2 + 1 = 49 steps (covers most of 60-step window)
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


class TemporalBlock(layers.Layer):
    """Single TCN block: dilated causal conv → BN → ReLU → dropout, x2, + residual."""

    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int,
                 dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.conv1 = layers.Conv1D(
            n_filters, kernel_size, padding='causal',
            dilation_rate=dilation_rate, activation=None,
        )
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(
            n_filters, kernel_size, padding='causal',
            dilation_rate=dilation_rate, activation=None,
        )
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(dropout_rate)

        # 1x1 conv to match dimensions for residual if needed
        self.residual_conv = layers.Conv1D(n_filters, 1, padding='same')
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.drop2(x, training=training)

        # Residual connection
        if inputs.shape[-1] != self.n_filters:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs

        return self.activation(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
        })
        return config


class TCNForecaster:
    """
    TCN-based cryptocurrency price forecaster.

    Uses dilated causal convolutions to capture long-range dependencies
    with far fewer parameters than an LSTM of comparable receptive field.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/tcn",
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
            'n_filters': 32,
            'kernel_size': 3,
            'n_blocks': 4,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 150,
            'early_stopping_patience': 10,
            'multi_step': [1, 7, 30],
        }

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        seq_len, n_features = input_shape
        n_filters = self.config['n_filters']
        kernel_size = self.config['kernel_size']
        n_blocks = self.config['n_blocks']
        dropout = self.config['dropout_rate']

        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs

        # Stack of temporal blocks with exponentially increasing dilation
        for i in range(n_blocks):
            dilation = 2 ** i
            x = TemporalBlock(
                n_filters=n_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                dropout_rate=dropout,
                name=f'tcn_block_{i}',
            )(x)

        # Take last timestep (causal — only looks at past)
        x = layers.Lambda(lambda t: t[:, -1, :], name='last_step')(x)

        x = layers.Dropout(dropout, name='final_dropout')(x)

        # Multi-step output heads
        outputs = {}
        for horizon in self.config['multi_step']:
            out = layers.Dense(1, name=f'output_{horizon}d')(x)
            outputs[f'output_{horizon}d'] = out

        model = keras.Model(inputs=inputs, outputs=outputs, name='TCN')

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
        print("Training TCN model...")
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
                filepath=str(self.model_dir / 'best_tcn.weights.h5'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                save_weights_only=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
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
                n_outputs = len(confidence_predictions[0])
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
        model_path = self.model_dir / f"tcn_v{version}_{timestamp}.h5"
        metadata_path = self.model_dir / f"tcn_v{version}_{timestamp}_metadata.json"

        self.model.save(model_path)

        metadata = {
            **self.metadata,
            'version': version,
            'model_path': str(model_path),
            'framework': 'tensorflow',
            'model_type': 'tcn',
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"TCN model saved to {model_path}")
        return model_path

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        custom_objects = {'TemporalBlock': TemporalBlock}
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)

        metadata_path = model_path.parent / model_path.name.replace('.h5', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.config = self.metadata.get('config', self._default_config())

        print(f"TCN model loaded from {model_path}")
