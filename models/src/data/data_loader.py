"""
Data Loader for Cryptocurrency ML Models

Fetches historical OHLCV data from CoinGecko, caches it,
and prepares it for ML model training with proper train/val/test splits.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent.parent / "backend" / "app"
sys.path.insert(0, str(backend_path))

try:
    from clients.coingecko_client import CoinGeckoClient
    from cache import AsyncCache
except ImportError:
    print("Warning: Could not import backend modules. Some functionality may be limited.")


class CryptoDataLoader:
    """
    Load and prepare cryptocurrency data for ML training.
    
    Handles:
    - Fetching from CoinGecko with caching
    - Time-series train/val/test splits
    - Data preprocessing and cleaning
    """
    
    def __init__(
        self,
        cache_dir: str = "models/data/cache",
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    async def load_crypto_data(
        self,
        crypto_id: str,
        days: int = 365,
        vs_currency: str = 'usd',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for a cryptocurrency.
        
        Args:
            crypto_id: CoinGecko ID (e.g., 'bitcoin')
            days: Number of days of historical data
            vs_currency: Currency for pricing
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check cache first
        cache_file = self.cache_dir / f"{crypto_id}_{days}d.parquet"
        
        if use_cache and cache_file.exists():
            # Check if cache is recent (< 24 hours old)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                print(f"Loading {crypto_id} from cache ({cache_age.seconds // 3600}h old)")
                return pd.read_parquet(cache_file)
        
        # Fetch from CoinGecko
        print(f"Fetching {crypto_id} data from CoinGecko ({days} days)...")
        
        try:
            client = CoinGeckoClient(timeout_seconds=15.0)
            
            # Fetch OHLC data
            ohlc_data = await client.get_coin_ohlc_by_id(
                coin_id=crypto_id,
                vs_currency=vs_currency,
                days=days
            )
            
            await client.close()
            
            if not ohlc_data or len(ohlc_data) == 0:
                raise ValueError(f"No data received for {crypto_id}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            
            # Add volume if available (CoinGecko OHLC doesn't include volume, fetch separately)
            df['volume'] = 0  # Placeholder
            
            # Cache the data
            df.to_parquet(cache_file)
            print(f"Cached {crypto_id} data ({len(df)} records)")
            
            return df
            
        except Exception as e:
            print(f"Error fetching {crypto_id} data: {e}")
            # Try to load from old cache
            if cache_file.exists():
                print(f"Using stale cache for {crypto_id}")
                return pd.read_parquet(cache_file)
            raise
    
    async def load_multiple_cryptos(
        self,
        crypto_ids: List[str],
        days: int = 365,
        vs_currency: str = 'usd'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple cryptocurrencies in parallel.
        
        Returns:
            Dictionary mapping crypto_id to DataFrame
        """
        tasks = [
            self.load_crypto_data(crypto_id, days, vs_currency)
            for crypto_id in crypto_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for crypto_id, result in zip(crypto_ids, results):
            if isinstance(result, Exception):
                print(f"Failed to load {crypto_id}: {result}")
                continue
            data_dict[crypto_id] = result
        
        return data_dict
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int = 60,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/time-series models.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps ahead to predict
            
        Returns:
            X (sequences), y (targets)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data) - forecast_horizon + 1):
            X.append(data[i - sequence_length:i])
            y.append(data[i + forecast_horizon - 1, 0])  # Assume first column is price
        
        return np.array(X), np.array(y)
    
    def train_val_test_split(
        self,
        df: pd.DataFrame,
        target_column: str = 'close'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (time-series aware).
        
        Args:
            df: DataFrame with features
            target_column: Name of target variable
            
        Returns:
            train_df, val_df, test_df
        """
        n = len(df)
        
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def handle_missing_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing data in time series.
        
        Args:
            df: DataFrame with potential missing values
            method: 'ffill' (forward fill), 'interpolate', or 'drop'
            
        Returns:
            Cleaned DataFrame
        """
        if method == 'ffill':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate(method='time')
        elif method == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str = 'close',
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in price data.
        
        Args:
            df: DataFrame with price data
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def preprocess_for_training(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        remove_outliers: bool = True,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for ML training.
        
        Args:
            df: Raw DataFrame
            target_column: Target variable
            remove_outliers: Whether to remove outliers
            fill_method: Method for handling missing data
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        # 1. Handle missing data
        df = self.handle_missing_data(df, method=fill_method)
        
        # 2. Remove outliers (optional)
        if remove_outliers:
            outliers = self.detect_outliers(df, column=target_column, threshold=3.5)
            if outliers.sum() > 0:
                print(f"Removing {outliers.sum()} outliers ({outliers.sum()/len(df)*100:.2f}%)")
                df = df[~outliers]
        
        # 3. Ensure chronological order
        df = df.sort_index()
        
        # 4. Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    async def prepare_training_data(
        self,
        crypto_id: str,
        days: int = 365,
        sequence_length: int = 60,
        forecast_horizon: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Complete data preparation pipeline: fetch, preprocess, split, create sequences.
        
        Returns:
            Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Load data
        df = await self.load_crypto_data(crypto_id, days)
        
        # Preprocess
        df = self.preprocess_for_training(df, target_column='close')
        
        # Split into train/val/test
        train_df, val_df, test_df = self.train_val_test_split(df, target_column='close')
        
        # Create sequences for each split
        X_train, y_train = self.create_sequences(
            train_df[['close']].values,
            sequence_length,
            forecast_horizon
        )
        
        X_val, y_val = self.create_sequences(
            val_df[['close']].values,
            sequence_length,
            forecast_horizon
        )
        
        X_test, y_test = self.create_sequences(
            test_df[['close']].values,
            sequence_length,
            forecast_horizon
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }


# Global instance
data_loader = CryptoDataLoader()

