from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

import polars as pl
import numpy as np


class TechnicalIndicators:
    """Technical analysis indicators for crypto data."""
    
    @staticmethod
    def sma(data: pl.Series, window: int) -> pl.Series:
        """Simple Moving Average."""
        return data.rolling_mean(window_size=window)
    
    @staticmethod
    def ema(data: pl.Series, window: int) -> pl.Series:
        """Exponential Moving Average."""
        return data.ewm_mean(alpha=2.0 / (window + 1))
    
    @staticmethod
    def rsi(data: pl.Series, window: int = 14) -> pl.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling_mean(window_size=window)
        avg_loss = loss.rolling_mean(window_size=window)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pl.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = data.ewm_mean(alpha=2.0 / (fast + 1))
        ema_slow = data.ewm_mean(alpha=2.0 / (slow + 1))
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(alpha=2.0 / (signal + 1))
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pl.Series, window: int = 20, std_dev: float = 2.0) -> Dict[str, pl.Series]:
        """Bollinger Bands."""
        sma = data.rolling_mean(window_size=window)
        std = data.rolling_std(window_size=window)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band
        }
    
    @staticmethod
    def stochastic(high: pl.Series, low: pl.Series, close: pl.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pl.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling_min(window_size=k_window)
        highest_high = high.rolling_max(window_size=k_window)
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling_mean(window_size=d_window)
        
        return {
            "k": k_percent,
            "d": d_percent
        }
    
    @staticmethod
    def atr(high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14) -> pl.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        atr = true_range.rolling_mean(window_size=window)
        
        return atr


class FeatureEngineer:
    """Feature engineering for crypto data."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def calculate_returns(self, prices: pl.Series) -> pl.Series:
        """Calculate simple returns."""
        return prices.pct_change()
    
    def calculate_log_returns(self, prices: pl.Series) -> pl.Series:
        """Calculate log returns."""
        return pl.log(prices / prices.shift(1))
    
    def calculate_volatility(self, returns: pl.Series, window: int = 20) -> pl.Series:
        """Calculate rolling volatility."""
        return returns.rolling_std(window_size=window) * np.sqrt(252)  # Annualized
    
    def calculate_vwap(self, high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> pl.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling_sum() / volume.rolling_sum()
    
    def get_feature_list(self) -> List[str]:
        """Get list of available features."""
        return [
            "returns", "log_returns", "volatility",
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_5", "ema_10", "ema_20", "ema_50",
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "stoch_k", "stoch_d", "atr", "vwap"
        ]


class FeaturePipeline:
    """Feature engineering pipeline for OHLCV data."""
    
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.indicators = TechnicalIndicators()
    
    def process_ohlcv_data(self, ohlcv_data: List[Dict[str, Any]], feature_set: str = "basic") -> pl.DataFrame:
        """Process OHLCV data and calculate features."""
        if not ohlcv_data:
            return pl.DataFrame()
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(ohlcv_data)
        
        # Ensure we have the required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return pl.DataFrame()
        
        # Convert timestamp to datetime if it's a string
        if df["timestamp"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S").alias("timestamp")
            )
        
        # Sort by timestamp
        df = df.sort("timestamp")
        
        # Calculate features based on feature set
        if feature_set in ["basic", "technical", "advanced", "all"]:
            df = self._add_basic_features(df)
        
        if feature_set in ["technical", "advanced", "all"]:
            df = self._add_technical_features(df)
        
        if feature_set in ["advanced", "all"]:
            df = self._add_advanced_features(df)
        
        return df
    
    def _add_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add basic features."""
        # Returns
        df = df.with_columns([
            self.engineer.calculate_returns(pl.col("close")).alias("returns"),
            self.engineer.calculate_log_returns(pl.col("close")).alias("log_returns"),
            self.engineer.calculate_volatility(pl.col("close").pct_change()).alias("volatility")
        ])
        
        # Simple Moving Averages
        df = df.with_columns([
            self.indicators.sma(pl.col("close"), 5).alias("sma_5"),
            self.indicators.sma(pl.col("close"), 10).alias("sma_10"),
            self.indicators.sma(pl.col("close"), 20).alias("sma_20"),
            self.indicators.sma(pl.col("close"), 50).alias("sma_50")
        ])
        
        # Exponential Moving Averages
        df = df.with_columns([
            self.indicators.ema(pl.col("close"), 5).alias("ema_5"),
            self.indicators.ema(pl.col("close"), 10).alias("ema_10"),
            self.indicators.ema(pl.col("close"), 20).alias("ema_20"),
            self.indicators.ema(pl.col("close"), 50).alias("ema_50")
        ])
        
        return df
    
    def _add_technical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add technical indicators."""
        # RSI
        df = df.with_columns([
            self.indicators.rsi(pl.col("close")).alias("rsi")
        ])
        
        # MACD
        macd_data = self.indicators.macd(pl.col("close"))
        df = df.with_columns([
            macd_data["macd"].alias("macd"),
            macd_data["signal"].alias("macd_signal"),
            macd_data["histogram"].alias("macd_histogram")
        ])
        
        # Bollinger Bands
        bb_data = self.indicators.bollinger_bands(pl.col("close"))
        df = df.with_columns([
            bb_data["upper"].alias("bb_upper"),
            bb_data["middle"].alias("bb_middle"),
            bb_data["lower"].alias("bb_lower"),
            (bb_data["upper"] - bb_data["lower"]).alias("bb_width")
        ])
        
        # Stochastic
        stoch_data = self.indicators.stochastic(pl.col("high"), pl.col("low"), pl.col("close"))
        df = df.with_columns([
            stoch_data["k"].alias("stoch_k"),
            stoch_data["d"].alias("stoch_d")
        ])
        
        return df
    
    def _add_advanced_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add advanced features."""
        # ATR
        df = df.with_columns([
            self.indicators.atr(pl.col("high"), pl.col("low"), pl.col("close")).alias("atr")
        ])
        
        # VWAP
        df = df.with_columns([
            self.engineer.calculate_vwap(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("vwap")
        ])
        
        # Price position relative to Bollinger Bands
        df = df.with_columns([
            ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower"))).alias("bb_position")
        ])
        
        # MACD divergence
        df = df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_divergence")
        ])
        
        return df
    
    def get_feature_sets(self) -> Dict[str, List[str]]:
        """Get available feature sets."""
        return {
            "basic": [
                "returns", "log_returns", "volatility",
                "sma_5", "sma_10", "sma_20", "sma_50",
                "ema_5", "ema_10", "ema_20", "ema_50"
            ],
            "technical": [
                "rsi", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_middle", "bb_lower", "bb_width",
                "stoch_k", "stoch_d"
            ],
            "advanced": [
                "atr", "vwap", "bb_position", "macd_divergence"
            ],
            "all": self.engineer.get_feature_list()
        }


# Global instance
_feature_pipeline_instance: Optional[FeaturePipeline] = None


def get_feature_pipeline() -> FeaturePipeline:
    """Get or create the global feature pipeline instance."""
    global _feature_pipeline_instance
    
    if _feature_pipeline_instance is None:
        _feature_pipeline_instance = FeaturePipeline()
    
    return _feature_pipeline_instance