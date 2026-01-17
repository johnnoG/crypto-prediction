"""
Feature Engineering for Cryptocurrency Price Prediction

Kaggle-style comprehensive feature engineering with 50+ features:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic)
- Price-based features (returns, volatility, momentum)
- Volume features
- Time-based features
- Market features (BTC correlation, dominance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class FeatureEngineer:
    """
    Comprehensive feature engineering for crypto price prediction.
    Implements 50+ features used in successful Kaggle competitions.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.scaler_params: Dict[str, Tuple[float, float]] = {}
    
    def engineer_features(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        market_caps: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set from price data.
        
        Args:
            prices: DataFrame with columns [timestamp, price]
            volumes: Optional DataFrame with trading volumes
            market_caps: Optional DataFrame with market capitalizations
            sentiment_scores: Optional DataFrame with news sentiment
            
        Returns:
            DataFrame with engineered features
        """
        df = prices.copy()
        
        # Ensure we have a datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # 1. PRICE-BASED FEATURES
        df = self._add_price_features(df)
        
        # 2. TECHNICAL INDICATORS
        df = self._add_technical_indicators(df)
        
        # 3. VOLUME FEATURES
        if volumes is not None:
            df = self._add_volume_features(df, volumes)
        
        # 4. TIME-BASED FEATURES
        df = self._add_time_features(df)
        
        # 5. MARKET FEATURES
        if market_caps is not None:
            df = self._add_market_features(df, market_caps)
        
        # 6. SENTIMENT FEATURES
        if sentiment_scores is not None:
            df = self._add_sentiment_features(df, sentiment_scores)
        
        # 7. LAG FEATURES (Important for time series)
        df = self._add_lag_features(df)
        
        # 8. ROLLING STATISTICS
        df = self._add_rolling_statistics(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'price']
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Multiple period returns
        for period in [1, 3, 7, 14, 30]:
            df[f'return_{period}d'] = df['price'].pct_change(period)
        
        # Price momentum
        df['momentum_3d'] = df['price'] / df['price'].shift(3) - 1
        df['momentum_7d'] = df['price'] / df['price'].shift(7) - 1
        df['momentum_14d'] = df['price'] / df['price'].shift(14) - 1
        
        # Price acceleration (change in momentum)
        df['acceleration'] = df['returns'] - df['returns'].shift(1)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators (Kaggle approach)"""
        prices = df['price'].values
        
        # RSI (Relative Strength Index) - Multiple periods
        for period in [14, 21, 28]:
            df[f'rsi_{period}'] = self._calculate_rsi(prices, period)
        
        # MACD (Moving Average Convergence Divergence)
        macd, signal, histogram = self._calculate_macd(prices)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR (Average True Range) - Volatility measure
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(prices, 14, 3)
        
        # Moving Averages - Multiple periods
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['price'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['price'].ewm(span=period, adjust=False).mean()
        
        # Moving Average Crossovers (Important signals)
        df['sma_50_200_ratio'] = df['sma_50'] / (df['sma_200'] + 1e-10)
        df['ema_12_26_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-10)
        
        # Price distance from moving averages
        for period in [7, 14, 50, 200]:
            df[f'price_to_sma_{period}'] = df['price'] / (df[f'sma_{period}'] + 1e-10) - 1
        
        # Commodity Channel Index (CCI)
        df['cci_20'] = self._calculate_cci(df, 20)
        
        # Williams %R
        df['williams_r_14'] = self._calculate_williams_r(df, 14)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = df.join(volumes, rsuffix='_vol')
        
        if 'volume' in df.columns:
            # Volume momentum
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
            df['volume_ma_30'] = df['volume'].rolling(window=30).mean()
            
            # Volume ratio (current vs average)
            df['volume_ratio_7d'] = df['volume'] / (df['volume_ma_7'] + 1e-10)
            df['volume_ratio_30d'] = df['volume'] / (df['volume_ma_30'] + 1e-10)
            
            # Price-Volume correlation
            df['price_volume_corr_14d'] = df['price'].rolling(14).corr(df['volume'])
            
            # On-Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df)
            df['obv_ma_14'] = df['obv'].rolling(window=14).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding for time features (preserves ordinal nature)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        """Add market-related features"""
        df = df.join(market_caps, rsuffix='_mcap')
        
        if 'market_cap' in df.columns:
            df['mcap_change'] = df['market_cap'].pct_change()
            df['mcap_to_price_ratio'] = df['market_cap'] / (df['price'] + 1e-10)
            
            # Market cap momentum
            df['mcap_momentum_7d'] = df['market_cap'].pct_change(7)
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, sentiment_scores: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features from news"""
        df = df.join(sentiment_scores, rsuffix='_sent')
        
        if 'sentiment_score' in df.columns:
            df['sentiment_ma_3d'] = df['sentiment_score'].rolling(window=3).mean()
            df['sentiment_ma_7d'] = df['sentiment_score'].rolling(window=7).mean()
            df['sentiment_change'] = df['sentiment_score'].diff()
            df['sentiment_volatility'] = df['sentiment_score'].rolling(window=7).std()
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features (critical for time series)"""
        # Lag prices
        for lag in [1, 2, 3, 7, 14]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Lag technical indicators
        if 'rsi_14' in df.columns:
            for lag in [1, 3, 7]:
                df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics (Kaggle approach)"""
        # Rolling mean and std for multiple windows
        for window in [7, 14, 30]:
            df[f'price_rolling_mean_{window}'] = df['price'].rolling(window=window).mean()
            df[f'price_rolling_std_{window}'] = df['price'].rolling(window=window).std()
            df[f'price_rolling_min_{window}'] = df['price'].rolling(window=window).min()
            df[f'price_rolling_max_{window}'] = df['price'].rolling(window=window).max()
            
            # Coefficient of variation
            df[f'price_cv_{window}'] = df[f'price_rolling_std_{window}'] / (df[f'price_rolling_mean_{window}'] + 1e-10)
            
            # Z-score
            df[f'price_zscore_{window}'] = (df['price'] - df[f'price_rolling_mean_{window}']) / (df[f'price_rolling_std_{window}'] + 1e-10)
        
        # Volatility measures
        for window in [7, 14, 30]:
            df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(365)
        
        return df
    
    # ========== TECHNICAL INDICATOR CALCULATIONS ==========
    
    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))
        
        # Initialize first average
        if len(gains) >= period:
            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])
            
            # Calculate smoothed averages
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        # Calculate EMAs
        ema_fast = pd.Series(prices).ewm(span=fast_period, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow_period, adjust=False).mean().values
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = pd.Series(macd).ewm(span=signal_period, adjust=False).mean().values
        
        # Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def _calculate_bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean().values
        std = pd.Series(prices).rolling(window=period).std().values
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (volatility measure)"""
        # Approximate ATR using price ranges
        high_low = df['price'].rolling(window=2).max() - df['price'].rolling(window=2).min()
        atr = high_low.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_stochastic(
        prices: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        prices_series = pd.Series(prices)
        
        # %K line
        lowest_low = prices_series.rolling(window=k_period).min()
        highest_high = prices_series.rolling(window=k_period).max()
        
        stoch_k = 100 * (prices_series - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        # %D line (smoothed %K)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k.values, stoch_d.values
    
    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = df['price']  # Typical price (would use (high+low+close)/3 with OHLC data)
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci
    
    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['price'].rolling(window=period).max()
        lowest_low = df['price'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df['price']) / (highest_high - lowest_low + 1e-10)
        return williams_r
    
    @staticmethod
    def _calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        obv = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def prepare_features_for_training(
        self,
        df: pd.DataFrame,
        target_column: str = 'price',
        fill_method: str = 'ffill'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML model training.
        
        Args:
            df: DataFrame with engineered features
            target_column: Name of target variable
            fill_method: Method to fill NaN values
            
        Returns:
            X (features), y (target)
        """
        # Handle NaN values
        df = df.fillna(method=fill_method) if fill_method else df.dropna()
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove any remaining NaN
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, y
    
    def get_feature_importance_summary(self, feature_importances: np.ndarray) -> pd.DataFrame:
        """
        Generate feature importance summary.
        
        Args:
            feature_importances: Array of feature importance scores from model
            
        Returns:
            DataFrame with features ranked by importance
        """
        if len(feature_importances) != len(self.feature_names):
            raise ValueError("Feature importances length doesn't match feature names")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # Add cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        
        return importance_df
    
    def select_top_features(
        self,
        X: pd.DataFrame,
        feature_importances: np.ndarray,
        top_n: int = 30,
        cumulative_threshold: float = 0.95
    ) -> List[str]:
        """
        Select top features based on importance (Kaggle technique).
        
        Args:
            X: Feature DataFrame
            feature_importances: Importance scores
            top_n: Maximum number of features
            cumulative_threshold: Cumulative importance threshold
            
        Returns:
            List of selected feature names
        """
        importance_df = self.get_feature_importance_summary(feature_importances)
        
        # Select features that contribute to cumulative_threshold of importance
        selected_by_threshold = importance_df[
            importance_df['cumulative_importance'] <= cumulative_threshold
        ]['feature'].tolist()
        
        # Also ensure we don't exceed top_n
        selected = selected_by_threshold[:top_n]
        
        return selected


# Standalone functions for quick feature computation

def calculate_technical_features_quick(prices: List[float]) -> Dict[str, float]:
    """
    Quick technical feature calculation for real-time inference.
    Returns current values of key indicators.
    """
    if len(prices) < 30:
        return {}
    
    prices_array = np.array(prices)
    
    # RSI
    rsi_14 = FeatureEngineer._calculate_rsi(prices_array, 14)[-1]
    
    # MACD
    macd, signal, histogram = FeatureEngineer._calculate_macd(prices_array)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = FeatureEngineer._calculate_bollinger_bands(prices_array, 20, 2)
    bb_position = (prices_array[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-10)
    
    # Moving averages
    sma_50 = np.mean(prices_array[-50:]) if len(prices_array) >= 50 else prices_array[-1]
    sma_200 = np.mean(prices_array[-200:]) if len(prices_array) >= 200 else prices_array[-1]
    
    # Volatility
    returns = np.diff(prices_array) / prices_array[:-1]
    volatility_30d = np.std(returns[-30:]) * np.sqrt(365) if len(returns) >= 30 else 0
    
    return {
        'rsi_14': float(rsi_14),
        'macd': float(macd[-1]),
        'macd_signal': float(signal[-1]),
        'macd_histogram': float(histogram[-1]),
        'bb_upper': float(bb_upper[-1]),
        'bb_middle': float(bb_middle[-1]),
        'bb_lower': float(bb_lower[-1]),
        'bb_position': float(bb_position),
        'sma_50': float(sma_50),
        'sma_200': float(sma_200),
        'volatility_30d': float(volatility_30d),
        'momentum_7d': float((prices_array[-1] / prices_array[-7] - 1) if len(prices_array) >= 7 else 0),
        'momentum_14d': float((prices_array[-1] / prices_array[-14] - 1) if len(prices_array) >= 14 else 0),
    }

