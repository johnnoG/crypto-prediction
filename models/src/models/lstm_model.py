"""
LSTM Model for Cryptocurrency Price Forecasting

Deep learning time-series model:
- Bidirectional LSTM for capturing temporal patterns
- Attention mechanism for important time steps
- Dropout for regularization
- Multi-step ahead forecasting
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


class LSTMForecaster:
    """
    LSTM-based cryptocurrency price forecaster.
    
    Uses deep learning to capture complex temporal patterns
    in price movements. Slower than LightGBM but can model
    non-linear dependencies.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/artifacts/lstm"
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
        """Default hyperparameters for LSTM"""
        return {
            'sequence_length': 60,
            'n_features': 1,  # Will be updated based on data
            'lstm_units_1': 128,
            'lstm_units_2': 64,
            'dense_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'use_bidirectional': True,
            'use_attention': False
        }
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name="CryptoLSTM")
        
        # First LSTM layer
        if self.config['use_bidirectional']:
            model.add(layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units_1'],
                    return_sequences=True,
                    input_shape=input_shape
                ),
                name='bidirectional_lstm_1'
            ))
        else:
            model.add(layers.LSTM(
                self.config['lstm_units_1'],
                return_sequences=True,
                input_shape=input_shape,
                name='lstm_1'
            ))
        
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_1'))
        
        # Second LSTM layer
        if self.config['use_bidirectional']:
            model.add(layers.Bidirectional(
                layers.LSTM(self.config['lstm_units_2'], return_sequences=False),
                name='bidirectional_lstm_2'
            ))
        else:
            model.add(layers.LSTM(
                self.config['lstm_units_2'],
                return_sequences=False,
                name='lstm_2'
            ))
        
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_2'))
        
        # Dense layers
        model.add(layers.Dense(self.config['dense_units'], activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_3'))
        
        # Output layer
        model.add(layers.Dense(1, name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Time series data (n_samples, n_features)
            sequence_length: Length of input sequences
            
        Returns:
            X (sequences), y (targets)
        """
        seq_len = sequence_length or self.config['sequence_length']
        
        X, y = [], []
        
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i, 0])  # Predict first feature (price)
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Verbosity level
            
        Returns:
            Training metrics
        """
        print("Training LSTM model...")
        
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
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
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
        train_rmse = np.sqrt(np.mean((y_train - train_pred.flatten()) ** 2))
        
        metrics = {
            'train_rmse': float(train_rmse),
            'train_loss': float(history.history['loss'][-1]),
            'epochs_trained': len(history.history['loss'])
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val, verbose=0)
            val_rmse = np.sqrt(np.mean((y_val - val_pred.flatten()) ** 2))
            metrics['val_rmse'] = float(val_rmse)
            metrics['val_loss'] = float(history.history['val_loss'][-1])
        
        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val) if X_val is not None else 0,
            'input_shape': list(input_shape),
            'config': self.config,
            'metrics': metrics
        }
        
        print(f"Training complete. Train RMSE: {train_rmse:.4f}")
        if 'val_rmse' in metrics:
            print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False,
        num_simulations: int = 100
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
        
        predictions = self.model.predict(X, verbose=0).flatten()
        
        if return_confidence:
            # Monte Carlo dropout for uncertainty estimation
            confidence_predictions = []
            for _ in range(num_simulations):
                pred = self.model(X, training=True)  # Enable dropout
                confidence_predictions.append(pred.numpy().flatten())
            
            confidence_predictions = np.array(confidence_predictions)
            lower_bound = np.percentile(confidence_predictions, 2.5, axis=0)
            upper_bound = np.percentile(confidence_predictions, 97.5, axis=0)
            
            confidence_intervals = np.column_stack([lower_bound, upper_bound])
            return predictions, confidence_intervals
        
        return predictions
    
    def save_model(self, version: str = "1.0.0") -> Path:
        """
        Save LSTM model and metadata.
        
        Args:
            version: Model version
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"v{version}_{timestamp}.h5"
        metadata_filename = f"v{version}_{timestamp}_metadata.json"
        
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
            'model_type': 'lstm'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: str | Path) -> None:
        """
        Load LSTM model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.parent / model_path.name.replace('.h5', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.config = self.metadata.get('config', self._default_config())
        
        print(f"Model loaded from {model_path}")
    
    def plot_training_history(self) -> None:
        """Plot training history (loss curves)"""
        if self.training_history is None:
            print("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.training_history.history
            
            plt.figure(figsize=(12, 4))
            
            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # MAE
            plt.subplot(1, 2, 2)
            plt.plot(history['mae'], label='Train MAE')
            if 'val_mae' in history:
                plt.plot(history['val_mae'], label='Val MAE')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.model_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {plot_path}")
            
            plt.close()
            
        except ImportError:
            print("matplotlib not installed, skipping plot")


class AttentionLSTMForecaster(LSTMForecaster):
    """
    LSTM with Attention Mechanism for better long-term predictions.
    
    Attention helps the model focus on important time steps.
    """
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model with attention layer"""
        
        # Input
        inputs = layers.Input(shape=input_shape, name='input')
        
        # First LSTM layer
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units_1'],
                return_sequences=True
            ),
            name='bidirectional_lstm_1'
        )(inputs)
        
        lstm_out = layers.Dropout(self.config['dropout_rate'], name='dropout_1')(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh', name='attention_dense')(lstm_out)
        attention = layers.Flatten(name='attention_flatten')(attention)
        attention = layers.Activation('softmax', name='attention_softmax')(attention)
        attention = layers.RepeatVector(self.config['lstm_units_1'] * 2, name='attention_repeat')(attention)
        attention = layers.Permute([2, 1], name='attention_permute')(attention)
        
        # Apply attention
        attended = layers.multiply([lstm_out, attention], name='attention_multiply')
        
        # Second LSTM layer
        lstm_out_2 = layers.LSTM(
            self.config['lstm_units_2'],
            return_sequences=False,
            name='lstm_2'
        )(attended)
        
        lstm_out_2 = layers.Dropout(self.config['dropout_rate'], name='dropout_2')(lstm_out_2)
        
        # Dense layers
        dense = layers.Dense(self.config['dense_units'], activation='relu', name='dense_1')(lstm_out_2)
        dense = layers.Dropout(self.config['dropout_rate'], name='dropout_3')(dense)
        
        # Output
        output = layers.Dense(1, name='output')(dense)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=output, name='AttentionLSTM')
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model


def create_lstm_ensemble(
    n_models: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> List[LSTMForecaster]:
    """
    Create ensemble of LSTM models with different random seeds.
    
    Kaggle technique: Train multiple models and average predictions
    for better generalization.
    
    Args:
        n_models: Number of models in ensemble
        config: Model configuration
        
    Returns:
        List of trained models
    """
    models = []
    
    for i in range(n_models):
        model_config = config.copy() if config else {}
        model_config['random_state'] = i
        
        model = LSTMForecaster(config=model_config)
        models.append(model)
    
    return models


def predict_with_ensemble(
    models: List[LSTMForecaster],
    X: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Make predictions using ensemble of models.
    
    Args:
        models: List of trained LSTM models
        X: Input sequences
        method: 'mean' or 'median'
        
    Returns:
        Ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

