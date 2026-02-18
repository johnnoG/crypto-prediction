"""
Enhanced LSTM Model for Cryptocurrency Price Forecasting

Advanced LSTM architecture featuring:
- Bidirectional LSTM with residual connections
- Enhanced attention mechanism with learnable position encoding
- Multi-step forecasting capabilities (1, 7, 30 days)
- Uncertainty quantification through Monte Carlo dropout
- Teacher forcing for training stability
- Gradient clipping and advanced regularization
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


class AttentionLayer(layers.Layer):
    """
    Enhanced attention mechanism for LSTM.

    Computes attention weights over LSTM hidden states to focus
    on the most relevant time steps for prediction.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

        self.W_a = layers.Dense(units, use_bias=False)
        self.U_a = layers.Dense(units, use_bias=False)
        self.V_a = layers.Dense(1, use_bias=False)

    def call(self, hidden_states):
        # hidden_states shape: (batch_size, timesteps, lstm_units)

        # Compute attention scores
        score = self.V_a(tf.nn.tanh(self.W_a(hidden_states)))
        # score shape: (batch_size, timesteps, 1)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Apply attention weights to hidden states
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        # context_vector shape: (batch_size, lstm_units)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class ResidualLSTMCell(layers.Layer):
    """
    LSTM cell with residual connections for better gradient flow.
    """

    def __init__(self, units: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout

        self.lstm = layers.LSTM(units, return_sequences=True, dropout=dropout)
        self.projection = layers.Dense(units)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs):
        # LSTM output
        lstm_out = self.lstm(inputs)

        # Residual connection (if dimensions match)
        if inputs.shape[-1] == self.units:
            residual = inputs
        else:
            residual = self.projection(inputs)

        # Add residual and normalize
        output = self.layer_norm(lstm_out + residual)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout
        })
        return config


class EnhancedLSTMForecaster:
    """
    Enhanced LSTM-based cryptocurrency price forecaster.

    Features multi-step prediction, advanced attention,
    uncertainty quantification, and robust training.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/enhanced_lstm"
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
        """Default hyperparameters for Enhanced LSTM"""
        return {
            'sequence_length': 60,
            'n_features': 1,  # Will be updated based on data
            'lstm_units': [128, 64, 32],  # Multiple LSTM layers
            'dense_units': [64, 32],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 150,
            'early_stopping_patience': 20,
            'gradient_clip': 1.0,
            'use_bidirectional': True,
            'use_attention': True,
            'use_residual': True,
            'use_layer_norm': True,
            'multi_step': [1, 7, 30],  # Multi-step forecasting horizons
            'teacher_forcing_ratio': 0.5,  # For multi-step training
            'mc_samples': 100  # Monte Carlo samples for uncertainty
        }

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build enhanced LSTM model architecture.

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        from tensorflow.keras import regularizers

        seq_len, n_features = input_shape
        l2_reg = self.config.get('l2_reg', 0.0)
        kernel_reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')

        x = inputs

        # Input normalization
        if self.config['use_layer_norm']:
            x = layers.LayerNormalization(name='input_norm')(x)

        # Build LSTM layers
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = (i < len(self.config['lstm_units']) - 1) or self.config['use_attention']

            if self.config['use_residual'] and i > 0:
                # Use residual LSTM
                x = ResidualLSTMCell(
                    units=units,
                    dropout=self.config['dropout_rate'],
                    name=f'residual_lstm_{i}'
                )(x)
            else:
                # Standard LSTM
                if self.config['use_bidirectional']:
                    x = layers.Bidirectional(
                        layers.LSTM(
                            units,
                            return_sequences=return_sequences,
                            dropout=self.config['dropout_rate'],
                            recurrent_dropout=self.config['recurrent_dropout'],
                            kernel_regularizer=kernel_reg,
                        ),
                        name=f'bidirectional_lstm_{i}'
                    )(x)
                else:
                    x = layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config['dropout_rate'],
                        recurrent_dropout=self.config['recurrent_dropout'],
                        kernel_regularizer=kernel_reg,
                        name=f'lstm_{i}'
                    )(x)

            # Add dropout
            if i < len(self.config['lstm_units']) - 1:
                x = layers.Dropout(self.config['dropout_rate'], name=f'dropout_lstm_{i}')(x)

        # Attention mechanism
        if self.config['use_attention'] and len(x.shape) == 3:
            # x still has sequence dimension
            attention_units = self.config['lstm_units'][-1]
            if self.config['use_bidirectional']:
                attention_units *= 2

            context_vector, attention_weights = AttentionLayer(
                attention_units,
                name='attention'
            )(x)
            x = context_vector
        elif len(x.shape) == 3:
            # Global average pooling if no attention
            x = layers.GlobalAveragePooling1D(name='global_pooling')(x)

        # Dense layers
        for i, units in enumerate(self.config['dense_units']):
            x = layers.Dense(units, activation='relu',
                             kernel_regularizer=kernel_reg,
                             name=f'dense_{i}')(x)
            x = layers.Dropout(self.config['dropout_rate'], name=f'dropout_dense_{i}')(x)

            if self.config['use_layer_norm']:
                x = layers.LayerNormalization(name=f'dense_norm_{i}')(x)

        # Multi-step output heads
        outputs = []
        output_names = []

        for horizon in self.config['multi_step']:
            # Separate head for each prediction horizon
            output = layers.Dense(16, activation='relu', name=f'head_{horizon}d_dense')(x)
            output = layers.Dropout(self.config['dropout_rate']/2, name=f'head_{horizon}d_dropout')(output)
            output = layers.Dense(1, name=f'output_{horizon}d')(output)

            outputs.append(output)
            output_names.append(f'output_{horizon}d')

        # Create model
        if len(outputs) == 1:
            model = keras.Model(inputs=inputs, outputs=outputs[0], name="EnhancedLSTM")
        else:
            outputs_dict = {name: output for name, output in zip(output_names, outputs)}
            model = keras.Model(inputs=inputs, outputs=outputs_dict, name="EnhancedLSTMMultiStep")

        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config['gradient_clip']
        )

        if len(outputs) == 1:
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
        else:
            # Multi-output model with horizon-weighted losses
            losses = {f'output_{h}d': 'huber' for h in self.config['multi_step']}

            # Weight shorter horizons more heavily
            loss_weights = {f'output_{h}d': float(1.0 / np.sqrt(h)) for h in self.config['multi_step']}

            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
            )

        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        multi_step_targets: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with multi-step targets.

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
        Train Enhanced LSTM model.

        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (single or multi-step)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Verbosity level

        Returns:
            Training metrics
        """
        print("Training Enhanced LSTM model...")
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

        # Advanced callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_enhanced_lstm.weights.h5'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]

        # Add learning rate scheduler
        def lr_schedule(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 50:
                return lr * 0.95
            else:
                return lr * 0.90

        callback_list.append(callbacks.LearningRateScheduler(lr_schedule, verbose=0))

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
                    print(f"  [LSTM] Epoch {self._epoch+1} | batch {batch}/{self._steps} | "
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
                horizon = output_name.replace('output_', '').replace('d', '')
                pred_arr = train_pred[i] if isinstance(train_pred, list) else train_pred[output_name]
                train_rmse = np.sqrt(np.mean((y_train[output_name] - pred_arr.flatten()) ** 2))
                metrics[f'train_rmse_{horizon}d'] = float(train_rmse)

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
                    horizon = output_name.replace('output_', '').replace('d', '')
                    pred_arr = val_pred[i] if isinstance(val_pred, list) else val_pred[output_name]
                    val_rmse = np.sqrt(np.mean((y_val[output_name] - pred_arr.flatten()) ** 2))
                    metrics[f'val_rmse_{horizon}d'] = float(val_rmse)
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
        num_simulations: Optional[int] = None
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

        num_sims = num_simulations or self.config['mc_samples']

        predictions = self.model.predict(X, verbose=0)

        # Handle multi-output models
        if isinstance(predictions, dict):
            # Already a dictionary
            pass
        elif isinstance(predictions, list):
            # Convert list to dictionary
            predictions = {f'horizon_{i+1}': pred.flatten()
                         for i, pred in enumerate(predictions)}
        elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        if return_confidence:
            # Monte Carlo dropout for uncertainty estimation
            confidence_predictions = []

            for _ in range(num_sims):
                # Enable dropout during inference
                pred = self.model(X, training=True)

                if isinstance(pred, dict):
                    pred = {k: v.numpy().flatten() for k, v in pred.items()}
                elif isinstance(pred, list):
                    pred = [p.numpy().flatten() for p in pred]
                else:
                    pred = pred.numpy().flatten()

                confidence_predictions.append(pred)

            if isinstance(confidence_predictions[0], dict):
                # Multi-output case
                confidence_intervals = {}
                for key in confidence_predictions[0].keys():
                    horizon_preds = np.array([cp[key] for cp in confidence_predictions])
                    lower_bound = np.percentile(horizon_preds, 2.5, axis=0)
                    upper_bound = np.percentile(horizon_preds, 97.5, axis=0)
                    confidence_intervals[key] = np.column_stack([lower_bound, upper_bound])
            elif isinstance(confidence_predictions[0], list):
                # Multi-output list case
                confidence_intervals = []
                for i in range(len(confidence_predictions[0])):
                    horizon_preds = np.array([cp[i] for cp in confidence_predictions])
                    lower_bound = np.percentile(horizon_preds, 2.5, axis=0)
                    upper_bound = np.percentile(horizon_preds, 97.5, axis=0)
                    confidence_intervals.append(np.column_stack([lower_bound, upper_bound]))
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
        Save Enhanced LSTM model and metadata.

        Args:
            version: Model version

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"enhanced_lstm_v{version}_{timestamp}.h5"
        metadata_filename = f"enhanced_lstm_v{version}_{timestamp}_metadata.json"

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
            'model_type': 'enhanced_lstm'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Enhanced LSTM model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")

        return model_path

    def load_model(self, model_path: str | Path) -> None:
        """
        Load Enhanced LSTM model from disk.

        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Register custom layers
        custom_objects = {
            'AttentionLayer': AttentionLayer,
            'ResidualLSTMCell': ResidualLSTMCell
        }

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Load metadata
        metadata_path = model_path.parent / model_path.name.replace('.h5', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.config = self.metadata.get('config', self._default_config())

        print(f"Enhanced LSTM model loaded from {model_path}")

    def plot_training_history(self) -> None:
        """Plot training history with enhanced visualizations"""
        if self.training_history is None:
            print("No training history available")
            return

        try:
            import matplotlib.pyplot as plt

            history = self.training_history.history

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Loss
            axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Learning Rate
            if 'lr' in history:
                axes[0, 1].plot(history['lr'], linewidth=2, color='orange')
                axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)

            # MAE
            mae_keys = [k for k in history.keys() if 'mae' in k and not k.startswith('val')]
            if mae_keys:
                for key in mae_keys:
                    axes[1, 0].plot(history[key], label=key.replace('_', ' ').title(), linewidth=2)
                    if f'val_{key}' in history:
                        axes[1, 0].plot(history[f'val_{key}'],
                                      label=f'Val {key.replace("_", " ").title()}',
                                      linestyle='--', linewidth=2)
                axes[1, 0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # MSE
            mse_keys = [k for k in history.keys() if 'mse' in k and not k.startswith('val')]
            if mse_keys:
                for key in mse_keys:
                    axes[1, 1].plot(history[key], label=key.replace('_', ' ').title(), linewidth=2)
                    if f'val_{key}' in history:
                        axes[1, 1].plot(history[f'val_{key}'],
                                      label=f'Val {key.replace("_", " ").title()}',
                                      linestyle='--', linewidth=2)
                axes[1, 1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('MSE')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.model_dir / f"enhanced_lstm_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {plot_path}")

            plt.close()

        except ImportError:
            print("matplotlib not installed, skipping plot")

    def analyze_attention_weights(self, X_sample: np.ndarray) -> np.ndarray:
        """
        Analyze attention weights for a sample sequence.

        Args:
            X_sample: Sample input sequence (1, sequence_length, n_features)

        Returns:
            Attention weights
        """
        if self.model is None or not self.config['use_attention']:
            return None

        try:
            # Get attention layer
            attention_layer = None
            for layer in self.model.layers:
                if isinstance(layer, AttentionLayer):
                    attention_layer = layer
                    break

            if attention_layer is None:
                print("No attention layer found")
                return None

            # Create a model that outputs attention weights
            attention_model = keras.Model(
                inputs=self.model.input,
                outputs=[self.model.output, attention_layer.output[1]]  # attention_weights
            )

            predictions, attention_weights = attention_model.predict(X_sample, verbose=0)

            return attention_weights.squeeze()

        except Exception as e:
            print(f"Error analyzing attention weights: {e}")
            return None