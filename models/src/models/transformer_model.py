"""
Transformer Model for Cryptocurrency Price Forecasting

Modern attention-based architecture featuring:
- Multi-head self-attention for temporal pattern recognition
- Positional encoding for time series data
- Causal masking for proper temporal modeling
- Multi-step forecasting capabilities
- Uncertainty quantification through ensemble methods
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


class MCDropout(layers.Dropout):
    """Dropout that always applies, enabling Monte Carlo inference.

    Used in output heads so each MC forward pass gets a different mask,
    regardless of TF-Metal training flag propagation issues.
    """

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

    def get_config(self):
        return super().get_config()


class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer for time series Transformer.

    Adds learnable position embeddings to capture temporal structure.
    """

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Create learnable positional embeddings
        self.pos_embedding = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='random_normal',
            trainable=True,
            name='positional_encoding'
        )

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, d_model)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Broadcast positional embeddings to match batch size
        pos_encoding = self.pos_embedding[:seq_len, :]
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])

        return inputs + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention layer with causal masking for time series.
    """

    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, name='query')
        self.wk = layers.Dense(d_model, name='key')
        self.wv = layers.Dense(d_model, name='value')

        self.dense = layers.Dense(d_model, name='output')

    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, seq_len, self.d_model))

        output = self.dense(concat_attention)

        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate attention weights and apply to values"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale by sqrt(depth)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply causal mask (look only at past)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Transformer encoder block with self-attention and feed-forward network.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        # Self-attention block
        attn_output = self.att(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerForecaster:
    """
    Transformer-based cryptocurrency price forecaster.

    Uses attention mechanisms to capture long-term dependencies
    and complex temporal patterns in price movements.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/transformer"
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed")

        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[keras.Model] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_history: Optional[keras.callbacks.History] = None
        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default hyperparameters for Transformer"""
        return {
            'sequence_length': 60,
            'n_features': 1,  # Will be updated based on data
            'd_model': 512,  # Model dimension
            'num_heads': 8,  # Number of attention heads
            'ff_dim': 2048,  # Feed-forward dimension
            'num_layers': 6,  # Number of transformer blocks
            'dropout_rate': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'warmup_steps': 1000,  # Learning rate warmup
            'multi_step': [1, 7, 30],  # Multi-step forecasting horizons
            'use_causal_mask': True
        }

    def create_causal_mask(self, sequence_length: int) -> tf.Tensor:
        """Create causal mask for self-attention (look only at past)"""
        mask = tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
        return mask

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build Transformer model architecture.

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        seq_len, n_features = input_shape

        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')

        # Input projection to model dimension
        x = layers.Dense(self.config['d_model'], name='input_projection')(inputs)

        # Positional encoding
        x = PositionalEncoding(seq_len, self.config['d_model'])(x)
        x = layers.Dropout(self.config['dropout_rate'], name='pos_enc_dropout')(x)

        # Transformer blocks
        for i in range(self.config['num_layers']):
            x = TransformerBlock(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                ff_dim=self.config['ff_dim'],
                dropout_rate=self.config['dropout_rate'],
                name=f'transformer_block_{i}'
            )(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers for multi-step forecasting
        outputs = []
        for horizon in self.config['multi_step']:
            output = layers.Dense(
                128, activation='relu',
                name=f'dense_{horizon}d'
            )(x)
            output = MCDropout(self.config['dropout_rate'], name=f'mc_dropout_{horizon}d')(output)
            output = layers.Dense(
                1, name=f'output_{horizon}d'
            )(output)
            outputs.append(output)

        # Create model
        if len(outputs) == 1:
            model = keras.Model(inputs=inputs, outputs=outputs[0], name="CryptoTransformer")
        else:
            model = keras.Model(inputs=inputs, outputs=outputs, name="CryptoTransformerMultiStep")

        # Custom learning rate schedule with warmup
        learning_rate = self._get_learning_rate_schedule()

        # Compile model
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )

        if len(outputs) == 1:
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
        else:
            # Multi-output model
            losses = {f'output_{h}d': 'huber' for h in self.config['multi_step']}
            loss_weights = {f'output_{h}d': 1.0/h for h in self.config['multi_step']}  # Weight shorter horizons more

            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
            )

        return model

    def _get_learning_rate_schedule(self):
        """Learning rate schedule with warmup (Transformer-style)"""

        class TransformerSchedule(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps):
                super().__init__()
                self.d_model = tf.cast(d_model, tf.float32)
                self.warmup_steps = tf.cast(warmup_steps, tf.float32)

            def __call__(self, step):
                step = tf.cast(step + 1, tf.float32)  # avoid rsqrt(0)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)

            def get_config(self):
                return {"d_model": float(self.d_model.numpy()),
                        "warmup_steps": float(self.warmup_steps.numpy())}

        return TransformerSchedule(self.config['d_model'], self.config['warmup_steps'])

    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        multi_step_targets: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for Transformer training with multi-step targets.

        Args:
            data: Time series data (n_samples, n_features)
            sequence_length: Length of input sequences
            multi_step_targets: List of horizons for multi-step forecasting

        Returns:
            X (sequences), y (targets)
        """
        seq_len = sequence_length or self.config['sequence_length']
        horizons = multi_step_targets or self.config['multi_step']

        X, y_dict = [], {f'{h}d': [] for h in horizons}

        max_horizon = max(horizons)

        for i in range(seq_len, len(data) - max_horizon):
            X.append(data[i - seq_len:i])

            # Create targets for each horizon
            for h in horizons:
                if i + h < len(data):
                    y_dict[f'{h}d'].append(data[i + h, 0])  # Predict first feature (price)

        X = np.array(X)

        # Return single or multiple targets
        if len(horizons) == 1:
            y = np.array(y_dict[f'{horizons[0]}d'])
        else:
            y = {f'output_{h}d': np.array(y_dict[f'{h}d']) for h in horizons}

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train Transformer model.

        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (single or multi-step)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Verbosity level

        Returns:
            Training metrics
        """
        print("Training Transformer model...")
        print("=" * 50)

        # Update config based on data shape
        if len(X_train.shape) == 3:
            self.config['sequence_length'] = X_train.shape[1]
            self.config['n_features'] = X_train.shape[2]

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self._build_model(input_shape)

        print(f"Model architecture:")
        self.model.summary()

        # Callbacks
        # Note: ReduceLROnPlateau is NOT used here because the Transformer
        # optimizer uses a custom TransformerSchedule (LearningRateSchedule)
        # for warmup + decay. ReduceLROnPlateau conflicts with schedule objects.
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.weights.h5'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]

        # Per-batch progress logging
        class _BatchProgress(callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self._epoch = epoch
                self._t0 = __import__('time').time()
                self._steps = self.params.get('steps', '?')
            def on_batch_end(self, batch, logs=None):
                if batch > 0 and batch % 20 == 0:
                    loss = logs.get('loss', 0)
                    elapsed = __import__('time').time() - self._t0
                    print(f"  [Transformer] Epoch {self._epoch+1} | batch {batch}/{self._steps} | "
                          f"loss: {loss:.4f} | {elapsed:.0f}s", flush=True)
        callback_list.append(_BatchProgress())

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=verbose
        )

        self.training_history = history

        # Calculate final metrics
        train_pred = self.model.predict(X_train, verbose=0)

        if isinstance(y_train, dict):
            # Multi-step model - model.predict() returns a list, not a dict
            output_names = list(y_train.keys())
            metrics = {}
            for i, output_name in enumerate(output_names):
                pred_key = output_name.replace('output_', '').replace('d', '')
                pred_arr = train_pred[i] if isinstance(train_pred, list) else train_pred[output_name]
                train_rmse = np.sqrt(np.mean((y_train[output_name] - pred_arr.flatten()) ** 2))
                metrics[f'train_rmse_{pred_key}d'] = float(train_rmse)

            metrics['epochs_trained'] = len(history.history['loss'])
        else:
            # Single-step model
            train_rmse = np.sqrt(np.mean((y_train - train_pred.flatten()) ** 2))
            metrics = {
                'train_rmse': float(train_rmse),
                'train_loss': float(history.history['loss'][-1]),
                'epochs_trained': len(history.history['loss'])
            }

        if X_val is not None:
            val_pred = self.model.predict(X_val, verbose=0)
            if isinstance(y_val, dict):
                val_output_names = list(y_val.keys())
                for i, output_name in enumerate(val_output_names):
                    pred_key = output_name.replace('output_', '').replace('d', '')
                    pred_arr = val_pred[i] if isinstance(val_pred, list) else val_pred[output_name]
                    val_rmse = np.sqrt(np.mean((y_val[output_name] - pred_arr.flatten()) ** 2))
                    metrics[f'val_rmse_{pred_key}d'] = float(val_rmse)
            else:
                val_rmse = np.sqrt(np.mean((y_val - val_pred.flatten()) ** 2))
                metrics['val_rmse'] = float(val_rmse)

        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val) if X_val is not None else 0,
            'input_shape': list(input_shape),
            'config': self.config,
            'metrics': metrics
        }

        print(f"Training complete!")
        for key, value in metrics.items():
            if 'rmse' in key:
                print(f"{key}: {value:.4f}")

        return metrics

    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False,
        num_simulations: int = 50
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with optional Monte Carlo confidence intervals.

        Args:
            X: Input sequences
            return_confidence: Whether to return confidence intervals
            num_simulations: Number of MC simulations for confidence

        Returns:
            Predictions (and confidence intervals if requested)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X, verbose=0)

        # Handle multi-output models
        if isinstance(predictions, list):
            predictions = {f'horizon_{i+1}': pred.flatten()
                         for i, pred in enumerate(predictions)}
        elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        if return_confidence:
            # Monte Carlo dropout for uncertainty estimation
            # MCDropout layers always apply dropout, so training flag doesn't matter
            X_tensor = tf.constant(X, dtype=tf.float32)
            confidence_predictions = []
            for _ in range(num_simulations):
                pred = self.model(X_tensor, training=True)
                if isinstance(pred, list):
                    pred = [p.numpy().flatten() for p in pred]
                else:
                    pred = pred.numpy().flatten()
                confidence_predictions.append(pred)

            if isinstance(confidence_predictions[0], list):
                # Multi-output case
                confidence_intervals = {}
                for i in range(len(confidence_predictions[0])):
                    horizon_preds = np.array([cp[i] for cp in confidence_predictions])
                    lower_bound = np.percentile(horizon_preds, 2.5, axis=0)
                    upper_bound = np.percentile(horizon_preds, 97.5, axis=0)
                    confidence_intervals[f'horizon_{i+1}'] = np.column_stack([lower_bound, upper_bound])
            else:
                # Single output case
                confidence_predictions = np.array(confidence_predictions)
                lower_bound = np.percentile(confidence_predictions, 2.5, axis=0)
                upper_bound = np.percentile(confidence_predictions, 97.5, axis=0)
                confidence_intervals = np.column_stack([lower_bound, upper_bound])

            return predictions, confidence_intervals

        return predictions

    def save_model(self, version: str = "1.0.0") -> Path:
        """
        Save Transformer model and metadata.

        Args:
            version: Model version

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"transformer_v{version}_{timestamp}.h5"
        metadata_filename = f"transformer_v{version}_{timestamp}_metadata.json"

        model_path = self.model_dir / model_filename
        metadata_path = self.model_dir / metadata_filename

        # Save model
        self.model.save(model_path)

        # Save metadata
        metadata = {
            **self.metadata,
            'version': version,
            'model_path': str(model_path),
            'framework': 'tensorflow',
            'model_type': 'transformer'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Transformer model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")

        return model_path

    def load_model(self, model_path: str | Path) -> None:
        """
        Load Transformer model from disk.

        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Register custom layers
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'MCDropout': MCDropout,
        }

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Load metadata
        metadata_path = model_path.parent / model_path.name.replace('.h5', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.config = self.metadata.get('config', self._default_config())

        print(f"Transformer model loaded from {model_path}")

    def plot_training_history(self) -> None:
        """Plot training history"""
        if self.training_history is None:
            print("No training history available")
            return

        try:
            import matplotlib.pyplot as plt

            history = self.training_history.history

            # Determine number of subplots based on metrics
            metrics_keys = [k for k in history.keys() if not k.startswith('val_')]
            n_metrics = len([k for k in metrics_keys if 'loss' not in k])

            fig_width = 12
            fig_height = 4 if n_metrics <= 2 else 8

            plt.figure(figsize=(fig_width, fig_height))

            # Loss
            plt.subplot(2, 2, 1)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Learning rate (if available)
            plt.subplot(2, 2, 2)
            if 'lr' in history:
                plt.plot(history['lr'])
                plt.title('Learning Rate Schedule')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.model_dir / f"transformer_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {plot_path}")

            plt.close()

        except ImportError:
            print("matplotlib not installed, skipping plot")


def create_transformer_ensemble(
    n_models: int = 3,
    config: Optional[Dict[str, Any]] = None
) -> List[TransformerForecaster]:
    """
    Create ensemble of Transformer models with different configurations.

    Args:
        n_models: Number of models in ensemble
        config: Base configuration

    Returns:
        List of Transformer models
    """
    models = []
    base_config = config or TransformerForecaster._default_config()

    for i in range(n_models):
        model_config = base_config.copy()

        # Vary some hyperparameters for diversity
        if i == 1:
            model_config['num_heads'] = 4
            model_config['d_model'] = 256
            model_config['ff_dim'] = 1024
        elif i == 2:
            model_config['num_layers'] = 4
            model_config['dropout_rate'] = 0.15

        model = TransformerForecaster(config=model_config)
        models.append(model)

    return models