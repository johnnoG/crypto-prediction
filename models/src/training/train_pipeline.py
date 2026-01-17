"""
Training Pipeline for Cryptocurrency ML Models

Orchestrates the complete training workflow:
- Data loading and preprocessing
- Feature engineering  
- Model training with hyperparameter tuning
- Validation and evaluation
- Model saving and versioning
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from models.lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE
from models.lstm_model import LSTMForecaster, TENSORFLOW_AVAILABLE
from models.ensemble import HybridEnsemble
from evaluation.metrics import MetricsCalculator, ForecastMetrics

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Hyperparameter tuning unavailable.")


class TrainingPipeline:
    """
    Complete training pipeline for cryptocurrency forecasting models.
    
    Handles data loading, feature engineering, training, and evaluation.
    """
    
    def __init__(
        self,
        crypto_ids: List[str],
        days_history: int = 365,
        output_dir: str = "models/artifacts"
    ):
        self.crypto_ids = crypto_ids
        self.days_history = days_history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = CryptoDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.metrics_calculator = MetricsCalculator()
        
        self.training_results: Dict[str, Any] = {}
    
    async def prepare_data_for_crypto(
        self,
        crypto_id: str
    ) -> Dict[str, Any]:
        """
        Complete data preparation for one cryptocurrency.
        
        Args:
            crypto_id: CoinGecko ID
            
        Returns:
            Dictionary with train/val/test data
        """
        print(f"\n{'='*60}")
        print(f"Preparing data for {crypto_id}")
        print(f"{'='*60}")
        
        # 1. Load historical data
        df = await self.data_loader.load_crypto_data(crypto_id, self.days_history)
        print(f"Loaded {len(df)} days of data")
        
        # 2. Preprocess
        df = self.data_loader.preprocess_for_training(df, target_column='close')
        print(f"After preprocessing: {len(df)} samples")
        
        # 3. Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        print(f"Engineered {len(df_features.columns)} features")
        
        # 4. Prepare features and target
        X, y = self.feature_engineer.prepare_features_for_training(
            df_features,
            target_column='close'
        )
        
        # 5. Train/val/test split
        train_df, val_df, test_df = self.data_loader.train_val_test_split(df_features)
        
        X_train = train_df.drop(columns=['close']).values
        y_train = train_df['close'].values
        X_val = val_df.drop(columns=['close']).values
        y_val = val_df['close'].values
        X_test = test_df.drop(columns=['close']).values
        y_test = test_df['close'].values
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': self.feature_engineer.feature_names,
            'raw_data': df_features
        }
    
    def train_lightgbm(
        self,
        crypto_id: str,
        data: Dict[str, Any],
        tune_hyperparameters: bool = False
    ) -> LightGBMForecaster:
        """
        Train LightGBM model for a cryptocurrency.
        
        Args:
            crypto_id: CoinGecko ID
            data: Prepared data dictionary
            tune_hyperparameters: Whether to run hyperparameter tuning
            
        Returns:
            Trained model
        """
        print(f"\n[LightGBM] Training for {crypto_id}")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        # Hyperparameter tuning
        if tune_hyperparameters and OPTUNA_AVAILABLE:
            print("Running hyperparameter optimization...")
            best_params = self._tune_lightgbm_hyperparameters(data)
            model = LightGBMForecaster(config=best_params)
        else:
            model = LightGBMForecaster()
        
        # Train
        metrics = model.train(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            feature_names=data['feature_names']
        )
        
        # Evaluate on test set
        y_pred_test = model.predict(data['X_test'])
        test_metrics = self.metrics_calculator.calculate_all_metrics(
            data['y_test'],
            y_pred_test
        )
        
        print(f"Test MAPE: {test_metrics.mape:.2f}%")
        print(f"Test R²: {test_metrics.r2_score:.4f}")
        
        # Save model
        model.save_model(version="1.0.0")
        
        return model
    
    def _tune_lightgbm_hyperparameters(
        self,
        data: Dict[str, Any],
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Tune LightGBM hyperparameters using Optuna.
        
        Args:
            data: Training data
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'verbose': -1
            }
            
            model = LightGBMForecaster(config=params)
            metrics = model.train(
                data['X_train'],
                data['y_train'],
                data['X_val'],
                data['y_val']
            )
            
            return metrics.get('val_rmse', float('inf'))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return {**LightGBMForecaster._default_config(), **study.best_params}
    
    async def train_all_models(
        self,
        save_models: bool = True,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Any]:
        """
        Train models for all cryptocurrencies.
        
        Args:
            save_models: Whether to save trained models
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary with all training results
        """
        results = {}
        
        for crypto_id in self.crypto_ids:
            try:
                # Prepare data
                data = await self.prepare_data_for_crypto(crypto_id)
                
                # Train LightGBM
                if LIGHTGBM_AVAILABLE:
                    lgb_model = self.train_lightgbm(crypto_id, data, tune_hyperparameters)
                    results[crypto_id] = {
                        'lightgbm': lgb_model,
                        'status': 'success'
                    }
                
                # Note: LSTM and ensemble training would go here
                # Skipping for now to keep training fast
                
            except Exception as e:
                print(f"Error training {crypto_id}: {e}")
                results[crypto_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.training_results = results
        return results
    
    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_cryptos': len(self.crypto_ids),
            'cryptos': self.crypto_ids,
            'results': {}
        }
        
        for crypto_id, result in self.training_results.items():
            if result.get('status') == 'success':
                model = result.get('lightgbm')
                if model:
                    report['results'][crypto_id] = {
                        'status': 'success',
                        'metrics': model.metadata.get('metrics', {}),
                        'trained_at': model.metadata.get('trained_at')
                    }
            else:
                report['results'][crypto_id] = {
                    'status': 'error',
                    'error': result.get('error')
                }
        
        return report


# Standalone function for quick training
async def train_model_for_crypto(
    crypto_id: str,
    model_type: str = 'lightgbm',
    days: int = 365,
    save_model: bool = True
) -> Any:
    """
    Quick training function for a single cryptocurrency.
    
    Args:
        crypto_id: CoinGecko ID
        model_type: 'lightgbm', 'lstm', or 'ensemble'
        days: Historical data period
        save_model: Whether to save trained model
        
    Returns:
        Trained model
    """
    pipeline = TrainingPipeline([crypto_id], days_history=days)
    data = await pipeline.prepare_data_for_crypto(crypto_id)
    
    if model_type == 'lightgbm':
        model = pipeline.train_lightgbm(crypto_id, data, tune_hyperparameters=False)
    elif model_type == 'lstm':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        model = LSTMForecaster()
        # Prepare sequences for LSTM
        X_train_seq, y_train = pipeline.data_loader.create_sequences(
            data['X_train'], sequence_length=60
        )
        X_val_seq, y_val = pipeline.data_loader.create_sequences(
            data['X_val'], sequence_length=60
        )
        model.train(X_train_seq, y_train, X_val_seq, y_val)
    elif model_type == 'ensemble':
        model = HybridEnsemble()
        model.train(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            feature_names=data['feature_names']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if save_model:
        if hasattr(model, 'save_model'):
            model.save_model(version="1.0.0")
        elif hasattr(model, 'save_ensemble'):
            model.save_ensemble(version="1.0.0")
    
    return model

