#!/usr/bin/env python3
"""
Comprehensive Cryptocurrency Data Analysis Script

Analyzes 103 cryptocurrency datasets from Kaggle (2010-2026)
- Data quality assessment
- Exploratory data analysis
- Feature engineering and technical indicators
- Data visualization and insights
- Preprocessing for ML models
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CryptoDataAnalyzer:
    """
    Comprehensive cryptocurrency data analyzer for 103 cryptocurrencies
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path or "crypto-prediction/data/kaggle-extracted"
        self.output_path = "crypto-prediction/data/processed"
        self.analysis_path = "crypto-prediction/data_analysis"

        # Create output directories
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_path).mkdir(parents=True, exist_ok=True)
        Path(f"{self.analysis_path}/plots").mkdir(parents=True, exist_ok=True)

        self.crypto_data = {}
        self.metadata = {}

    def load_all_crypto_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all cryptocurrency CSV files and perform initial analysis
        """
        print("🚀 Loading all cryptocurrency datasets...")

        csv_files = list(Path(self.data_path).glob("*.csv"))
        print(f"Found {len(csv_files)} cryptocurrency datasets")

        for csv_file in csv_files:
            crypto_name = csv_file.stem.upper()
            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df.set_index('date', inplace=True)

                # Basic data validation
                if len(df) > 0 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    self.crypto_data[crypto_name] = df
                    print(f"✅ Loaded {crypto_name}: {len(df)} records from {df.index.min()} to {df.index.max()}")
                else:
                    print(f"❌ Skipped {crypto_name}: Invalid data structure")

            except Exception as e:
                print(f"❌ Error loading {crypto_name}: {e}")

        print(f"\n📊 Successfully loaded {len(self.crypto_data)} cryptocurrencies")
        return self.crypto_data

    def analyze_data_quality(self) -> Dict:
        """
        Comprehensive data quality analysis
        """
        print("\n🔍 Analyzing data quality...")

        quality_report = {
            "summary": {},
            "missing_data": {},
            "date_ranges": {},
            "price_anomalies": {},
            "volume_analysis": {}
        }

        for crypto, df in self.crypto_data.items():
            # Basic statistics
            quality_report["summary"][crypto] = {
                "total_records": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "days_coverage": (df.index.max() - df.index.min()).days,
                "missing_days": self._count_missing_days(df),
                "columns": list(df.columns)
            }

            # Missing data analysis
            missing_data = df.isnull().sum()
            quality_report["missing_data"][crypto] = missing_data.to_dict()

            # Price anomalies
            price_stats = {
                "min_price": df['close'].min(),
                "max_price": df['close'].max(),
                "price_range_ratio": df['close'].max() / df['close'].min() if df['close'].min() > 0 else np.inf,
                "zero_prices": (df['close'] == 0).sum(),
                "negative_prices": (df['close'] < 0).sum()
            }
            quality_report["price_anomalies"][crypto] = price_stats

        # Generate quality summary
        self._save_quality_report(quality_report)
        return quality_report

    def _count_missing_days(self, df: pd.DataFrame) -> int:
        """Count missing days in the date series"""
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        return len(date_range) - len(df)

    def _save_quality_report(self, quality_report: Dict):
        """Save data quality report to JSON"""
        report_path = f"{self.analysis_path}/data_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        print(f"💾 Data quality report saved to {report_path}")

    def generate_market_overview(self):
        """
        Generate comprehensive market overview visualizations with professional styling
        """
        print("\n📈 Generating professional market overview visualizations...")

        # Set professional style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 18,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3
        })

        # Create separate high-quality plots
        self._create_price_evolution_plot()
        self._create_data_availability_plot()
        self._create_market_performance_plot()
        self._create_volatility_analysis_plot()

        print("✅ Professional market overview plots created")

    def _create_price_evolution_plot(self):
        """Create professional price evolution plot"""
        fig, ax = plt.subplots(figsize=(16, 10))

        # Get top 5 cryptocurrencies by market significance
        market_caps = {}
        for crypto, df in self.crypto_data.items():
            if len(df) > 100:  # Only consider cryptos with substantial data
                # Use latest price * data length as proxy for significance
                significance = df['close'].iloc[-1] * len(df)
                market_caps[crypto] = significance

        top_cryptos = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:5]

        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (crypto, _) in enumerate(top_cryptos):
            df = self.crypto_data[crypto]
            ax.plot(df.index, df['close'],
                   label=crypto, linewidth=2.5,
                   color=colors[i], alpha=0.9)

        ax.set_title('Price Evolution: Top 5 Cryptocurrencies by Market Significance',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # Log scale for better visibility

        # Improve legend
        ax.legend(loc='upper left', frameon=True, fancybox=True,
                 shadow=True, ncol=1, bbox_to_anchor=(0.02, 0.98))

        # Format x-axis
        years = pd.date_range(start='2010', end='2027', freq='2YE')
        ax.set_xticks(years)
        ax.set_xticklabels([str(year.year) for year in years], rotation=0)

        plt.tight_layout()
        plt.savefig(f"{self.analysis_path}/plots/price_evolution_top5.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _create_data_availability_plot(self):
        """Create professional data availability visualization"""
        fig, ax = plt.subplots(figsize=(16, 12))

        # Prepare data for heatmap
        years = list(range(2010, 2027))
        crypto_coverage = {}

        # Select top 30 cryptos by data coverage for better readability
        coverage_scores = {}
        for crypto, df in self.crypto_data.items():
            if len(df) > 0:
                coverage_scores[crypto] = len(df) * (df.index.max() - df.index.min()).days

        top_coverage = sorted(coverage_scores.items(), key=lambda x: x[1], reverse=True)[:30]

        # Create coverage matrix
        coverage_matrix = []
        crypto_names = []

        for crypto, _ in top_coverage:
            df = self.crypto_data[crypto]
            yearly_coverage = []

            for year in years:
                year_data = df[df.index.year == year]
                if year == 2026:
                    coverage = len(year_data) / 31.0  # Only January 2026
                else:
                    coverage = len(year_data) / 365.0
                yearly_coverage.append(min(coverage, 1.0))  # Cap at 1.0

            coverage_matrix.append(yearly_coverage)
            crypto_names.append(crypto)

        # Create heatmap
        im = ax.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')

        # Customize heatmap
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=0, fontsize=12)
        ax.set_yticks(range(len(crypto_names)))
        ax.set_yticklabels(crypto_names, fontsize=10)

        ax.set_title('Data Availability Heatmap: Top 30 Cryptocurrencies by Coverage',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cryptocurrency', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Data Coverage (0=No Data, 1=Complete)', rotation=270,
                      labelpad=20, fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.analysis_path}/plots/data_availability_heatmap.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _create_market_performance_plot(self):
        """Create professional market performance analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Calculate performance metrics
        performance_data = {
            'crypto': [],
            'total_return': [],
            'annualized_return': [],
            'volatility': [],
            'sharpe_ratio': []
        }

        for crypto, df in self.crypto_data.items():
            if len(df) > 252:  # At least 1 year of data
                # Calculate returns
                total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

                # Calculate annualized return
                years = (df.index.max() - df.index.min()).days / 365.25
                annualized_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (1/years) - 1) * 100

                # Calculate volatility
                daily_returns = df['close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100

                # Calculate Sharpe ratio (assuming 2% risk-free rate)
                sharpe = (annualized_return - 2) / volatility if volatility > 0 else 0

                performance_data['crypto'].append(crypto)
                performance_data['total_return'].append(total_return)
                performance_data['annualized_return'].append(annualized_return)
                performance_data['volatility'].append(volatility)
                performance_data['sharpe_ratio'].append(sharpe)

        # Plot 1: Risk vs Return
        scatter = ax1.scatter(performance_data['volatility'],
                            performance_data['annualized_return'],
                            s=80, alpha=0.7,
                            c=performance_data['sharpe_ratio'],
                            cmap='RdYlGn', edgecolors='black', linewidth=0.5)

        ax1.set_xlabel('Annualized Volatility (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Annualized Return (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Risk vs Return Analysis\n(Color = Sharpe Ratio)',
                     fontsize=16, fontweight='bold', pad=20)

        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Sharpe Ratio', rotation=270, labelpad=15, fontweight='bold')

        # Add diagonal line for reference
        max_vol = max(performance_data['volatility'])
        ax1.plot([0, max_vol], [0, max_vol], 'r--', alpha=0.5, linewidth=1)
        ax1.text(max_vol*0.7, max_vol*0.7, 'Return = Volatility',
                rotation=45, alpha=0.7, fontsize=10)

        # Plot 2: Top/Bottom performers
        # Get top 10 and bottom 5 by annualized return
        perf_sorted = list(zip(performance_data['crypto'], performance_data['annualized_return']))
        perf_sorted.sort(key=lambda x: x[1], reverse=True)

        top_performers = perf_sorted[:10]
        bottom_performers = perf_sorted[-5:]

        combined = top_performers + bottom_performers
        names = [x[0] for x in combined]
        returns = [x[1] for x in combined]
        colors = ['green' if r > 0 else 'red' for r in returns]

        bars = ax2.barh(range(len(names)), returns, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=10)
        ax2.set_xlabel('Annualized Return (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Top 10 & Bottom 5 Performers\n(Annualized Returns)',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Add value labels - all on the right side
        max_abs_return = max(abs(min(returns)), abs(max(returns)))
        for i, (bar, value) in enumerate(zip(bars, returns)):
            # Position all labels on the right side of the plot
            label_x = max_abs_return * 1.1
            ax2.text(label_x, i, f'{value:.1f}%',
                    va='center', ha='left',
                    fontweight='bold', fontsize=10)

        # Adjust x-axis to show labels properly
        ax2.set_xlim(min(returns) * 1.1, max_abs_return * 1.3)

        plt.tight_layout()
        plt.savefig(f"{self.analysis_path}/plots/market_performance_analysis.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _create_volatility_analysis_plot(self):
        """Create professional volatility analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Calculate rolling volatility for major cryptos
        major_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP']
        available_majors = [crypto for crypto in major_cryptos if crypto in self.crypto_data]

        colors = plt.cm.Set3(np.linspace(0, 1, len(available_majors)))

        # Plot 1: Rolling volatility over time
        for i, crypto in enumerate(available_majors):
            df = self.crypto_data[crypto]
            if len(df) > 100:
                returns = df['close'].pct_change().dropna()
                rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100

                ax1.plot(rolling_vol.index, rolling_vol,
                        label=crypto, linewidth=2, alpha=0.8, color=colors[i])

        ax1.set_title('30-Day Rolling Volatility: Major Cryptocurrencies',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Annualized Volatility (%)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        years = pd.date_range(start='2015', end='2027', freq='2YE')
        ax1.set_xticks(years)
        ax1.set_xticklabels([str(year.year) for year in years])

        # Plot 2: Current volatility distribution
        current_volatilities = []
        crypto_names = []

        for crypto, df in self.crypto_data.items():
            if len(df) > 100:
                recent_returns = df['close'].pct_change().dropna().tail(252)  # Last year
                if len(recent_returns) > 30:
                    vol = recent_returns.std() * np.sqrt(252) * 100
                    current_volatilities.append(vol)
                    crypto_names.append(crypto)

        # Create histogram
        ax2.hist(current_volatilities, bins=20, alpha=0.7, color='skyblue',
                edgecolor='black', linewidth=1)
        ax2.axvline(np.median(current_volatilities), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(current_volatilities):.1f}%')
        ax2.axvline(np.mean(current_volatilities), color='orange', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(current_volatilities):.1f}%')

        ax2.set_title('Distribution of Current Volatility (All Cryptocurrencies)',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Annualized Volatility (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.analysis_path}/plots/volatility_analysis.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _plot_data_availability_heatmap(self, ax):
        """Plot data availability heatmap"""
        # Create data availability matrix
        date_range = pd.date_range(start='2010-01-01', end='2026-12-31', freq='YE')
        availability_matrix = []
        crypto_names = []

        for crypto in sorted(list(self.crypto_data.keys()))[:20]:  # Top 20 for visibility
            df = self.crypto_data[crypto]
            yearly_data = []
            for year in date_range.year:
                year_data = df[df.index.year == year]
                coverage = len(year_data) / 365.0 if year != 2026 else len(year_data) / 31.0  # 2026 partial
                yearly_data.append(coverage)
            availability_matrix.append(yearly_data)
            crypto_names.append(crypto)

        sns.heatmap(availability_matrix,
                   xticklabels=[str(year) for year in date_range.year],
                   yticklabels=crypto_names,
                   cmap='RdYlGn',
                   ax=ax,
                   cbar_kws={'label': 'Data Coverage'})
        ax.set_title('Data Availability Heatmap (Top 20 Cryptos)')
        ax.set_xlabel('Year')

    def _plot_volume_analysis(self, ax):
        """Plot volume analysis for cryptocurrencies that have volume data"""
        volume_cryptos = []
        for crypto, df in self.crypto_data.items():
            if 'volume' in df.columns and df['volume'].notna().any():
                volume_cryptos.append((crypto, df['volume'].mean()))

        if volume_cryptos:
            volume_cryptos.sort(key=lambda x: x[1], reverse=True)
            top_volume = volume_cryptos[:10]

            cryptos = [item[0] for item in top_volume]
            volumes = [item[1] for item in top_volume]

            bars = ax.bar(range(len(cryptos)), volumes)
            ax.set_title('Average Trading Volume (Top 10)')
            ax.set_xlabel('Cryptocurrency')
            ax.set_ylabel('Average Volume')
            ax.set_xticks(range(len(cryptos)))
            ax.set_xticklabels(cryptos, rotation=45, ha='right')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Volume Data Available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14)
            ax.set_title('Volume Analysis - No Data')

    def _plot_market_statistics(self, ax, top_cryptos):
        """Plot market statistics"""
        stats_data = {
            'crypto': [],
            'total_return': [],
            'volatility': []
        }

        for crypto, _ in top_cryptos:
            df = self.crypto_data[crypto]
            if len(df) > 1:
                # Calculate total return
                total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

                # Calculate annualized volatility
                daily_returns = df['close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized

                stats_data['crypto'].append(crypto)
                stats_data['total_return'].append(total_return)
                stats_data['volatility'].append(volatility)

        # Create scatter plot
        ax.scatter(stats_data['volatility'], stats_data['total_return'],
                  s=100, alpha=0.7, c='darkblue')

        # Add labels for each point
        for i, crypto in enumerate(stats_data['crypto']):
            ax.annotate(crypto,
                       (stats_data['volatility'][i], stats_data['total_return'][i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Annualized Volatility (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk vs Return Analysis (Top 10)')
        ax.grid(True, alpha=0.3)

    def engineer_basic_features(self) -> Dict[str, pd.DataFrame]:
        """
        Engineer basic technical features for all cryptocurrencies
        """
        print("\n🔧 Engineering basic technical features...")

        enhanced_data = {}

        for crypto, df in self.crypto_data.items():
            try:
                enhanced_df = df.copy()

                # Price-based features
                enhanced_df['returns'] = enhanced_df['close'].pct_change()
                enhanced_df['log_returns'] = np.log(enhanced_df['close'] / enhanced_df['close'].shift(1))

                # Moving averages
                for window in [7, 14, 21, 50, 100, 200]:
                    enhanced_df[f'sma_{window}'] = enhanced_df['close'].rolling(window=window).mean()
                    enhanced_df[f'ema_{window}'] = enhanced_df['close'].ewm(span=window).mean()

                # Volatility measures
                enhanced_df['volatility_7d'] = enhanced_df['returns'].rolling(7).std()
                enhanced_df['volatility_30d'] = enhanced_df['returns'].rolling(30).std()

                # Price momentum
                enhanced_df['momentum_7d'] = enhanced_df['close'] / enhanced_df['close'].shift(7) - 1
                enhanced_df['momentum_14d'] = enhanced_df['close'] / enhanced_df['close'].shift(14) - 1
                enhanced_df['momentum_30d'] = enhanced_df['close'] / enhanced_df['close'].shift(30) - 1

                # High-low spread
                enhanced_df['hl_spread'] = (enhanced_df['high'] - enhanced_df['low']) / enhanced_df['close']
                enhanced_df['oc_spread'] = (enhanced_df['close'] - enhanced_df['open']) / enhanced_df['open']

                # Support and resistance levels (simplified)
                enhanced_df['high_52w'] = enhanced_df['high'].rolling(252).max()
                enhanced_df['low_52w'] = enhanced_df['low'].rolling(252).min()
                enhanced_df['price_position'] = (enhanced_df['close'] - enhanced_df['low_52w']) / (enhanced_df['high_52w'] - enhanced_df['low_52w'])

                enhanced_data[crypto] = enhanced_df

            except Exception as e:
                print(f"❌ Error processing {crypto}: {e}")
                enhanced_data[crypto] = df  # Keep original data

        print(f"✅ Enhanced {len(enhanced_data)} cryptocurrencies with technical features")
        return enhanced_data

    def save_processed_data(self, enhanced_data: Dict[str, pd.DataFrame]):
        """
        Save processed data in multiple formats for ML pipeline
        """
        print("\n💾 Saving processed data...")

        # Save individual crypto files
        for crypto, df in enhanced_data.items():
            # Save as CSV
            csv_path = f"{self.output_path}/{crypto}_processed.csv"
            df.to_csv(csv_path)

            # Save as Parquet (better for ML pipelines)
            parquet_path = f"{self.output_path}/{crypto}_processed.parquet"
            df.to_parquet(parquet_path)

        # Create combined dataset for cross-crypto analysis
        combined_data = []
        for crypto, df in enhanced_data.items():
            df_copy = df.copy()
            df_copy['crypto'] = crypto
            df_copy['date'] = df_copy.index
            combined_data.append(df_copy)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df.to_csv(f"{self.output_path}/all_cryptos_processed.csv", index=False)
            combined_df.to_parquet(f"{self.output_path}/all_cryptos_processed.parquet", index=False)

        print(f"✅ Processed data saved to {self.output_path}")

    def generate_analysis_summary(self):
        """
        Generate comprehensive analysis summary
        """
        print("\n📋 Generating analysis summary...")

        summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_cryptocurrencies": len(self.crypto_data),
            "data_overview": {},
            "market_insights": {},
            "data_quality": {}
        }

        # Data overview
        total_records = sum(len(df) for df in self.crypto_data.values())
        earliest_date = min(df.index.min() for df in self.crypto_data.values() if len(df) > 0)
        latest_date = max(df.index.max() for df in self.crypto_data.values() if len(df) > 0)

        summary["data_overview"] = {
            "total_records": total_records,
            "date_range": f"{earliest_date} to {latest_date}",
            "average_records_per_crypto": total_records / len(self.crypto_data) if self.crypto_data else 0,
            "cryptocurrencies_loaded": list(self.crypto_data.keys())
        }

        # Market insights
        current_prices = {}
        returns_analysis = {}

        for crypto, df in self.crypto_data.items():
            if len(df) > 1:
                current_prices[crypto] = df['close'].iloc[-1]
                total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                returns_analysis[crypto] = total_return

        # Top performers
        top_gainers = sorted(returns_analysis.items(), key=lambda x: x[1], reverse=True)[:10]
        top_losers = sorted(returns_analysis.items(), key=lambda x: x[1])[:10]

        summary["market_insights"] = {
            "top_gainers": dict(top_gainers),
            "top_losers": dict(top_losers),
            "highest_current_prices": dict(sorted(current_prices.items(), key=lambda x: x[1], reverse=True)[:10])
        }

        # Save summary
        summary_path = f"{self.analysis_path}/analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✅ Analysis summary saved to {summary_path}")

        # Print key insights
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"📊 Analyzed {len(self.crypto_data)} cryptocurrencies")
        print(f"📅 Data spans from {earliest_date} to {latest_date}")
        print(f"📈 Top gainer: {top_gainers[0][0]} (+{top_gainers[0][1]:.2f}%)")
        print(f"📉 Biggest loser: {top_losers[0][0]} ({top_losers[0][1]:.2f}%)")

        return summary

def main():
    """
    Main execution function
    """
    print("🚀 Starting Comprehensive Cryptocurrency Data Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CryptoDataAnalyzer()

    # Load all cryptocurrency data
    crypto_data = analyzer.load_all_crypto_data()

    if not crypto_data:
        print("❌ No cryptocurrency data loaded. Check data path.")
        return

    # Analyze data quality
    quality_report = analyzer.analyze_data_quality()

    # Generate market overview visualizations
    analyzer.generate_market_overview()

    # Engineer basic technical features
    enhanced_data = analyzer.engineer_basic_features()

    # Save processed data
    analyzer.save_processed_data(enhanced_data)

    # Generate analysis summary
    analyzer.generate_analysis_summary()

    print("\n🎉 Analysis complete! Check the following locations:")
    print(f"📊 Plots: {analyzer.analysis_path}/plots/")
    print(f"📋 Reports: {analyzer.analysis_path}/")
    print(f"💾 Processed data: {analyzer.output_path}/")

if __name__ == "__main__":
    main()