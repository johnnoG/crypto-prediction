#!/usr/bin/env python3
"""
Simplified Feature Engineering for Cryptocurrency Data

Implements core technical indicators without complex dependencies:
- Basic Technical Analysis Indicators
- Price-based Features
- Volume-based Features
- Statistical Features
- Market Regime Detection
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

warnings.filterwarnings('ignore')

class SimpleFeatureEngineer:
    """
    Simplified feature engineering for cryptocurrency time series data
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path or "/Users/yonatanglanzman/src/crypto-prediction/data/processed"
        self.output_path = "/Users/yonatanglanzman/src/crypto-prediction/data/features"

        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.feature_descriptions = {}

    def engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer basic technical and statistical features
        """
        df = df.copy()

        # Clean and prepare data
        df = self._prepare_data(df)

        print("🔧 Engineering basic features...")

        # === 1. PRICE-BASED FEATURES ===
        self._add_price_features(df)

        # === 2. MOVING AVERAGES ===
        self._add_moving_averages(df)

        # === 3. MOMENTUM INDICATORS ===
        self._add_momentum_features(df)

        # === 4. VOLATILITY FEATURES ===
        self._add_volatility_features(df)

        # === 5. VOLUME FEATURES ===
        if 'volume' in df.columns and df['volume'].notna().any():
            self._add_volume_features(df)

        # === 6. STATISTICAL FEATURES ===
        self._add_statistical_features(df)

        # === 7. TIME-BASED FEATURES ===
        self._add_time_features(df)

        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for feature engineering"""
        # Remove duplicates in index
        if df.index.duplicated().any():
            print("⚠️ Removing duplicate dates...")
            df = df[~df.index.duplicated(keep='last')]

        # Sort by index
        df = df.sort_index()

        # Ensure numeric data types
        numeric_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            numeric_cols.append('volume')

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with all NaN values in OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')

        return df

    def _add_price_features(self, df: pd.DataFrame):
        """Add basic price-based features"""

        # Price returns
        for period in [1, 3, 5, 7, 14, 21, 30]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1

        # High-low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['oc_spread'] = (df['close'] - df['open']) / df['open']

        # True range (simplified)
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Price position within daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_position'] = df['price_position'].fillna(0.5)

        # Gap analysis
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_up'] = (df['gap_size'] > 0.01).astype(int)
        df['gap_down'] = (df['gap_size'] < -0.01).astype(int)

    def _add_moving_averages(self, df: pd.DataFrame):
        """Add moving average features"""

        # Simple moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # Price relative to moving averages
        for window in [20, 50, 200]:
            df[f'price_above_sma_{window}'] = (df['close'] > df[f'sma_{window}']).astype(int)
            df[f'price_distance_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

        # Moving average crossovers
        df['sma_20_above_sma_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma_50_above_sma_200'] = (df['sma_50'] > df['sma_200']).astype(int)

        # MACD (simplified)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

    def _add_momentum_features(self, df: pd.DataFrame):
        """Add momentum indicators"""

        # RSI (Relative Strength Index) - simplified calculation
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            df[f'rsi_{period}'] = rsi
            df[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)

        # Stochastic oscillator (simplified)
        window = 14
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_k'] = k_percent
        df['stoch_d'] = k_percent.rolling(window=3).mean()

        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * ((high_max - df['close']) / (high_max - low_min))

        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    def _add_volatility_features(self, df: pd.DataFrame):
        """Add volatility-based features"""

        # Bollinger Bands
        for window in [10, 20]:
            sma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()

            df[f'bb_upper_{window}'] = sma + (std * 2)
            df[f'bb_lower_{window}'] = sma - (std * 2)
            df[f'bb_middle_{window}'] = sma
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])

        # Average True Range (ATR)
        for window in [7, 14, 21]:
            df[f'atr_{window}'] = df['true_range'].rolling(window=window).mean()
            df[f'atr_ratio_{window}'] = df[f'atr_{window}'] / df['close']

        # Volatility measures
        for window in [10, 20, 30]:
            returns = df['close'].pct_change()
            df[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        # Keltner Channels (simplified)
        window = 20
        ema = df['close'].ewm(span=window).mean()
        atr = df['true_range'].rolling(window=window).mean()
        df['keltner_upper'] = ema + (2 * atr)
        df['keltner_lower'] = ema - (2 * atr)
        df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])

    def _add_volume_features(self, df: pd.DataFrame):
        """Add volume-based features"""

        # Volume moving averages
        for window in [10, 20, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']

        # On Balance Volume (OBV) - simplified
        price_change = np.where(df['close'] > df['close'].shift(1), 1,
                               np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv'] = (df['volume'] * price_change).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()

        # Volume Price Trend (VPT)
        price_change_pct = df['close'].pct_change()
        df['vpt'] = (price_change_pct * df['volume']).cumsum()

        # Volume oscillator
        df['volume_oscillator'] = (df['volume_sma_10'] / df['volume_sma_20'] - 1) * 100

        # Volume breakouts
        volume_mean = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        df['volume_breakout'] = (df['volume'] > volume_mean + 2 * volume_std).astype(int)

        # Price-Volume relationship
        returns = df['close'].pct_change()
        positive_volume = np.where(returns > 0, df['volume'], 0)
        negative_volume = np.where(returns < 0, df['volume'], 0)

        df['positive_volume_ratio'] = pd.Series(positive_volume).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['volume_sentiment'] = df['positive_volume_ratio'] - 0.5  # Center around 0

    def _add_statistical_features(self, df: pd.DataFrame):
        """Add statistical features"""

        returns = df['close'].pct_change()

        # Rolling statistical moments
        for window in [10, 20, 30]:
            df[f'skewness_{window}'] = returns.rolling(window).skew()
            df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            df[f'return_percentile_rank_{window}'] = returns.rolling(window).rank(pct=True)

        # Z-scores
        for window in [20, 50]:
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            df[f'return_zscore_{window}'] = (returns - rolling_mean) / rolling_std

        # Support and resistance levels
        for window in [20, 50, 100]:
            df[f'rolling_high_{window}'] = df['high'].rolling(window).max()
            df[f'rolling_low_{window}'] = df['low'].rolling(window).min()
            df[f'price_position_range_{window}'] = ((df['close'] - df[f'rolling_low_{window}']) /
                                                   (df[f'rolling_high_{window}'] - df[f'rolling_low_{window}']))

        # New highs and lows
        df['new_high_20'] = (df['high'] == df['high'].rolling(20).max()).astype(int)
        df['new_low_20'] = (df['low'] == df['low'].rolling(20).min()).astype(int)

        # Autocorrelations
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['close'].rolling(20).corr(df['close'].shift(lag))

    def _add_time_features(self, df: pd.DataFrame):
        """Add time-based features"""

        # Basic time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weekend and month effects
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)

    def engineer_cross_asset_features(self, all_crypto_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Engineer cross-asset correlation and market features
        """
        print("🔗 Engineering cross-asset correlation features...")

        # Create a combined price matrix for correlation analysis
        price_matrix = pd.DataFrame()
        for crypto, df in all_crypto_data.items():
            if len(df) > 100:  # Only include cryptos with substantial data
                clean_df = df.copy()
                if clean_df.index.duplicated().any():
                    clean_df = clean_df[~clean_df.index.duplicated(keep='last')]
                clean_df = clean_df.sort_index()

                # Only use 'close' column for correlation matrix
                if 'close' in clean_df.columns:
                    price_matrix[crypto] = clean_df['close']

        # Align all data to common index and remove cryptos with too little data
        price_matrix = price_matrix.dropna(axis=1, thresh=len(price_matrix) * 0.3)

        enhanced_data = {}

        for crypto, df in all_crypto_data.items():
            enhanced_df = df.copy()

            # Add cross-asset features if crypto is in price matrix
            if crypto in price_matrix.columns and len(price_matrix.columns) > 1:
                self._add_correlation_features(enhanced_df, price_matrix, crypto)
                self._add_market_regime_features(enhanced_df, price_matrix)

            enhanced_data[crypto] = enhanced_df

        return enhanced_data

    def _add_correlation_features(self, df: pd.DataFrame, price_matrix: pd.DataFrame, crypto: str):
        """Add correlation-based features"""

        # Rolling correlations with major cryptocurrencies
        major_cryptos = ['BTC', 'ETH', 'BNB']
        available_majors = [c for c in major_cryptos if c in price_matrix.columns and c != crypto]

        for major_crypto in available_majors:
            for window in [20, 60]:
                try:
                    # Align series for correlation calculation
                    crypto_prices = df['close'].reindex(price_matrix.index)
                    major_prices = price_matrix[major_crypto]

                    corr = crypto_prices.rolling(window).corr(major_prices)
                    df[f'corr_{major_crypto}_{window}d'] = corr.reindex(df.index)
                except:
                    continue

        # Beta calculation (vs BTC if available)
        if 'BTC' in price_matrix.columns and crypto != 'BTC':
            try:
                btc_returns = price_matrix['BTC'].pct_change()
                crypto_returns = df['close'].pct_change()

                # Align for calculation
                btc_returns_aligned = btc_returns.reindex(df.index)

                for window in [30, 90]:
                    covariance = crypto_returns.rolling(window).cov(btc_returns_aligned)
                    btc_variance = btc_returns_aligned.rolling(window).var()
                    df[f'beta_btc_{window}d'] = covariance / btc_variance
            except:
                pass

    def _add_market_regime_features(self, df: pd.DataFrame, price_matrix: pd.DataFrame):
        """Add market regime detection features"""

        try:
            # Calculate market-wide metrics
            market_returns = price_matrix.pct_change()
            market_avg_return = market_returns.mean(axis=1)
            market_volatility = market_returns.std(axis=1)

            # Align with current dataframe
            market_vol_aligned = market_volatility.reindex(df.index)
            market_ret_aligned = market_avg_return.reindex(df.index)

            # Market regime indicators
            df['market_volatility'] = market_vol_aligned
            df['market_average_return'] = market_ret_aligned

            # High volatility regime (above 80th percentile)
            df['high_volatility_regime'] = (
                market_vol_aligned > market_vol_aligned.rolling(252).quantile(0.8)
            ).astype(int)

            # Bull/Bear market regime
            market_trend_short = market_ret_aligned.rolling(50).mean()
            market_trend_long = market_ret_aligned.rolling(200).mean()
            df['bull_market_regime'] = (market_trend_short > market_trend_long).astype(int)

            # Market stress regime (high volatility + negative returns)
            df['market_stress_regime'] = (
                (market_vol_aligned > market_vol_aligned.rolling(252).quantile(0.8)) &
                (market_ret_aligned.rolling(5).mean() < 0)
            ).astype(int)

        except Exception as e:
            print(f"⚠️ Could not calculate market regime features: {e}")

    def process_all_cryptocurrencies(self, crypto_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all cryptocurrencies with feature engineering
        """
        print(f"🚀 Starting feature engineering for {len(crypto_data)} cryptocurrencies...")

        enhanced_data = {}

        # Step 1: Basic features for each crypto
        for i, (crypto, df) in enumerate(crypto_data.items(), 1):
            print(f"📊 Processing {crypto} ({i}/{len(crypto_data)})...")

            try:
                # Skip if not enough data
                if len(df) < 50:
                    print(f"⚠️ Skipping {crypto}: insufficient data ({len(df)} rows)")
                    continue

                # Basic feature engineering
                enhanced_df = self.engineer_basic_features(df)
                enhanced_data[crypto] = enhanced_df

            except Exception as e:
                print(f"❌ Error processing {crypto}: {e}")
                enhanced_data[crypto] = df  # Keep original if processing fails

        # Step 2: Cross-asset features
        if len(enhanced_data) > 1:
            enhanced_data = self.engineer_cross_asset_features(enhanced_data)

        print("✅ Feature engineering completed!")
        return enhanced_data

    def get_feature_summary(self, enhanced_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary of engineered features"""

        if not enhanced_data:
            return {}

        # Get sample dataframe to analyze features
        sample_crypto = list(enhanced_data.keys())[0]
        sample_df = enhanced_data[sample_crypto]

        # Original columns
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'ticker']
        original_count = len([col for col in original_cols if col in sample_df.columns])

        # New feature columns
        new_features = [col for col in sample_df.columns if col not in original_cols]

        # Categorize features
        feature_categories = {
            'price_features': [f for f in new_features if any(x in f for x in ['return', 'momentum', 'spread', 'gap', 'position'])],
            'moving_averages': [f for f in new_features if any(x in f for x in ['sma_', 'ema_', 'macd'])],
            'momentum_indicators': [f for f in new_features if any(x in f for x in ['rsi_', 'stoch_', 'williams_', 'roc_'])],
            'volatility_indicators': [f for f in new_features if any(x in f for x in ['bb_', 'atr_', 'volatility', 'keltner'])],
            'volume_indicators': [f for f in new_features if any(x in f for x in ['volume_', 'obv', 'vpt', 'sentiment'])],
            'statistical_features': [f for f in new_features if any(x in f for x in ['skewness', 'kurtosis', 'zscore', 'autocorr'])],
            'correlation_features': [f for f in new_features if any(x in f for x in ['corr_', 'beta_'])],
            'regime_features': [f for f in new_features if any(x in f for x in ['regime', 'market_'])],
            'time_features': [f for f in new_features if any(x in f for x in ['day_', 'month_', 'is_'])],
            'support_resistance': [f for f in new_features if any(x in f for x in ['rolling_', 'new_'])]
        }

        summary = {
            'total_cryptocurrencies': len(enhanced_data),
            'original_features': original_count,
            'engineered_features': len(new_features),
            'total_features': len(sample_df.columns),
            'feature_categories': {cat: len(features) for cat, features in feature_categories.items()},
            'sample_features_by_category': {cat: features[:5] for cat, features in feature_categories.items() if features}
        }

        return summary


def main():
    """
    Main execution function for feature engineering
    """
    print("🚀 Starting Simplified Feature Engineering for Cryptocurrency Data")
    print("=" * 70)

    # Load processed data
    data_path = "/Users/yonatanglanzman/src/crypto-prediction/data/processed"

    # Load all cryptocurrency data
    crypto_data = {}
    parquet_files = list(Path(data_path).glob("*_processed.parquet"))

    if not parquet_files:
        print("❌ No processed parquet files found. Looking for CSV files...")
        csv_files = list(Path(data_path).glob("*_processed.csv"))

        if not csv_files:
            print("❌ No processed data found. Please run data analysis first.")
            return

        # Load from CSV files
        for csv_file in csv_files:
            crypto_name = csv_file.stem.replace('_processed', '').upper()
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                crypto_data[crypto_name] = df
                print(f"✅ Loaded {crypto_name} from CSV")
            except Exception as e:
                print(f"❌ Error loading {crypto_name}: {e}")
    else:
        # Load from parquet files
        for parquet_file in parquet_files:
            crypto_name = parquet_file.stem.replace('_processed', '').upper()
            try:
                df = pd.read_parquet(parquet_file)
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                crypto_data[crypto_name] = df
                print(f"✅ Loaded {crypto_name} from Parquet")
            except Exception as e:
                print(f"❌ Error loading {crypto_name}: {e}")

    if not crypto_data:
        print("❌ No cryptocurrency data found.")
        return

    print(f"📊 Loaded {len(crypto_data)} cryptocurrencies for feature engineering")

    # Initialize feature engineer
    engineer = SimpleFeatureEngineer()

    # Process all cryptocurrencies
    enhanced_data = engineer.process_all_cryptocurrencies(crypto_data)

    if not enhanced_data:
        print("❌ No data was processed successfully.")
        return

    # Save enhanced data
    output_path = engineer.output_path
    for crypto, df in enhanced_data.items():
        output_file = f"{output_path}/{crypto}_features.parquet"
        df.to_parquet(output_file)
        print(f"💾 Saved enhanced features for {crypto}")

    # Generate and save feature summary
    summary = engineer.get_feature_summary(enhanced_data)
    with open(f"{output_path}/feature_engineering_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 FEATURE ENGINEERING SUMMARY:")
    print(f"📊 Processed {summary['total_cryptocurrencies']} cryptocurrencies")
    print(f"📈 Generated {summary['engineered_features']} new features")
    print(f"📋 Total features per cryptocurrency: {summary['total_features']}")
    print(f"\n📁 Enhanced data saved to: {output_path}")

    # Display feature categories
    print(f"\n📊 FEATURE CATEGORIES:")
    for category, count in summary['feature_categories'].items():
        if count > 0:
            print(f"  {category.replace('_', ' ').title()}: {count} features")

    print(f"\n✅ Feature engineering completed successfully!")


if __name__ == "__main__":
    main()