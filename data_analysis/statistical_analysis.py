#!/usr/bin/env python3
"""
Statistical Analysis and Correlation Matrices for Cryptocurrency Data

Generates comprehensive statistical summaries:
- Descriptive statistics for all features
- Correlation matrices across cryptocurrencies
- Feature importance rankings
- Market regime analysis
- Risk metrics and performance indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class CryptoStatisticalAnalyzer:
    """
    Comprehensive statistical analysis for cryptocurrency data
    """

    def __init__(self, features_path: str = None):
         # Get the project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
           
        self.features_path = features_path or str(project_root / "data" / "features")
        self.output_path = str(project_root / "analysis_results")
    

        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.crypto_data = {}
        self.statistical_summary = {}

    def load_feature_data(self, limit: int = None):
        """Load feature-engineered cryptocurrency data"""
        print("📊 Loading feature-engineered cryptocurrency data...")

        feature_files = list(Path(self.features_path).glob("*_features.parquet"))

        if limit:
            feature_files = feature_files[:limit]

        for file_path in feature_files:
            crypto_name = file_path.stem.replace('_features', '')
            try:
                df = pd.read_parquet(file_path)
                if len(df) > 100:  # Only include cryptos with substantial data
                    self.crypto_data[crypto_name] = df
                    print(f"✅ {crypto_name}: {len(df)} records, {len(df.columns)} features")
                else:
                    print(f"⚠️ Skipping {crypto_name}: insufficient data ({len(df)} records)")
            except Exception as e:
                print(f"❌ Error loading {crypto_name}: {e}")

        print(f"\n📈 Loaded {len(self.crypto_data)} cryptocurrencies")
        return len(self.crypto_data)

    def generate_descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("\n📊 Generating descriptive statistics...")

        all_stats = {}

        for crypto, df in self.crypto_data.items():
            print(f"   Analyzing {crypto}...")

            # Select numeric features only
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

            # Basic statistics
            basic_stats = df[numeric_features].describe()

            # Additional statistics
            additional_stats = pd.DataFrame({
                'skewness': df[numeric_features].skew(),
                'kurtosis': df[numeric_features].kurtosis(),
                'null_count': df[numeric_features].isnull().sum(),
                'null_percentage': (df[numeric_features].isnull().sum() / len(df)) * 100
            })

            # Combine statistics
            combined_stats = pd.concat([basic_stats.T, additional_stats], axis=1)
            all_stats[crypto] = combined_stats

        self.statistical_summary = all_stats

        # Save summary statistics
        summary_path = f"{self.output_path}/descriptive_statistics_summary.xlsx"
        with pd.ExcelWriter(summary_path) as writer:
            for crypto, stats_df in all_stats.items():
                stats_df.to_excel(writer, sheet_name=crypto[:31])  # Excel sheet name limit

        print(f"💾 Descriptive statistics saved to {summary_path}")
        return all_stats

    def create_correlation_matrices(self):
        """Create and analyze correlation matrices"""
        print("\n🔗 Creating correlation matrices...")

        correlation_results = {}

        # 1. Cross-asset price correlations
        print("   📈 Cross-asset price correlations...")
        price_matrix = self._create_price_matrix()

        if len(price_matrix.columns) > 1:
            price_correlations = price_matrix.corr()
            correlation_results['price_correlations'] = price_correlations

            # Visualize price correlations
            self._plot_correlation_heatmap(
                price_correlations,
                title='Cryptocurrency Price Correlations',
                save_name='price_correlations'
            )

        # 2. Feature correlations within each cryptocurrency
        print("   🔧 Feature correlations within cryptocurrencies...")
        sample_crypto = list(self.crypto_data.keys())[0]
        sample_df = self.crypto_data[sample_crypto]

        # Select key features for correlation analysis
        key_features = self._select_key_features(sample_df)

        if key_features:
            feature_corr = sample_df[key_features].corr()
            correlation_results['feature_correlations'] = feature_corr

            self._plot_correlation_heatmap(
                feature_corr,
                title=f'Feature Correlations - {sample_crypto}',
                save_name='feature_correlations'
            )

        # 3. Return correlations
        print("   📊 Return correlations...")
        returns_matrix = self._create_returns_matrix()

        if len(returns_matrix.columns) > 1:
            returns_correlations = returns_matrix.corr()
            correlation_results['returns_correlations'] = returns_correlations

            self._plot_correlation_heatmap(
                returns_correlations,
                title='Cryptocurrency Returns Correlations',
                save_name='returns_correlations'
            )

        # Save correlation matrices
        correlation_path = f"{self.output_path}/correlation_matrices.xlsx"
        with pd.ExcelWriter(correlation_path) as writer:
            for name, corr_matrix in correlation_results.items():
                corr_matrix.to_excel(writer, sheet_name=name)

        print(f"💾 Correlation matrices saved to {correlation_path}")
        return correlation_results

    def _create_price_matrix(self):
        """Create aligned price matrix for correlation analysis"""
        price_data = {}

        for crypto, df in self.crypto_data.items():
            if 'close' in df.columns:
                price_data[crypto] = df['close']

        if price_data:
            price_matrix = pd.DataFrame(price_data)
            # Use common time period
            price_matrix = price_matrix.dropna(axis=1, thresh=len(price_matrix) * 0.5)
            return price_matrix.dropna()

        return pd.DataFrame()

    def _create_returns_matrix(self):
        """Create aligned returns matrix"""
        returns_data = {}

        for crypto, df in self.crypto_data.items():
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                returns_data[crypto] = returns

        if returns_data:
            returns_matrix = pd.DataFrame(returns_data)
            return returns_matrix.dropna()

        return pd.DataFrame()

    def _select_key_features(self, df):
        """Select key features for correlation analysis"""
        # Priority features for analysis
        priority_features = [
            'return_1d', 'return_7d', 'return_30d',
            'rsi_14', 'rsi_21',
            'volatility_20d', 'volatility_30d',
            'bb_position_20', 'bb_width_20',
            'volume_ratio_20', 'volume_sentiment',
            'momentum_10d', 'momentum_20d',
            'sma_20', 'sma_50', 'ema_20',
            'macd', 'macd_signal',
            'atr_14', 'atr_21'
        ]

        available_features = [f for f in priority_features if f in df.columns]

        # If not enough priority features, add more numeric features
        if len(available_features) < 10:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            additional_features = [col for col in numeric_cols
                                 if col not in exclude_cols + available_features]
            available_features.extend(additional_features[:20])

        return available_features[:25]  # Limit to 25 features for readability

    def _plot_correlation_heatmap(self, corr_matrix, title, save_name):
        """Plot correlation heatmap"""
        plt.figure(figsize=(14, 12))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create heatmap
        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=True if len(corr_matrix) <= 15 else False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{save_name}_heatmap.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

    def perform_cluster_analysis(self):
        """Perform hierarchical clustering on cryptocurrencies"""
        print("\n🔍 Performing cluster analysis...")

        # Create returns matrix for clustering
        returns_matrix = self._create_returns_matrix()

        if len(returns_matrix.columns) < 3:
            print("⚠️ Insufficient data for cluster analysis")
            return None

        # Calculate correlation distance matrix
        correlation_matrix = returns_matrix.corr()
        distance_matrix = 1 - correlation_matrix

        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')

        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix,
                  labels=returns_matrix.columns,
                  orientation='top',
                  leaf_rotation=45)
        plt.title('Cryptocurrency Hierarchical Clustering\n(Based on Return Correlations)',
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/cluster_dendrogram.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

        # Create clusters
        n_clusters = min(5, len(returns_matrix.columns) // 2)
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        cluster_df = pd.DataFrame({
            'cryptocurrency': returns_matrix.columns,
            'cluster': clusters
        })

        print("🔍 Cluster Assignments:")
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cryptos = cluster_df[cluster_df['cluster'] == cluster_id]['cryptocurrency'].tolist()
            print(f"   Cluster {cluster_id}: {', '.join(cryptos)}")

        # Save cluster results
        cluster_df.to_csv(f"{self.output_path}/cluster_assignments.csv", index=False)

        return cluster_df

    def analyze_principal_components(self):
        """Perform PCA analysis on cryptocurrency returns"""
        print("\n🧮 Performing Principal Component Analysis...")

        returns_matrix = self._create_returns_matrix()

        if len(returns_matrix.columns) < 3:
            print("⚠️ Insufficient data for PCA analysis")
            return None

        # Standardize the data
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_matrix.fillna(0))

        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(returns_scaled)

        # Calculate explained variance
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)

        # Plot explained variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scree plot
        ax1.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio,
                'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot - Individual Explained Variance')
        ax1.grid(True, alpha=0.3)

        # Cumulative variance
        ax2.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio,
                'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='k', linestyle='--', alpha=0.7, label='80% Variance')
        ax2.axhline(y=0.9, color='k', linestyle='--', alpha=0.7, label='90% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Principal Component Analysis - Cryptocurrency Returns',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/pca_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

        # PCA loadings for first two components
        if len(returns_matrix.columns) > 2:
            loadings_df = pd.DataFrame({
                'cryptocurrency': returns_matrix.columns,
                'PC1': pca.components_[0],
                'PC2': pca.components_[1] if len(pca.components_) > 1 else 0
            })

            # Plot PCA loadings
            plt.figure(figsize=(12, 8))
            plt.scatter(loadings_df['PC1'], loadings_df['PC2'],
                       s=100, alpha=0.7, c='blue')

            for i, crypto in enumerate(loadings_df['cryptocurrency']):
                plt.annotate(crypto,
                           (loadings_df['PC1'].iloc[i], loadings_df['PC2'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)

            plt.xlabel(f'PC1 ({explained_var_ratio[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_var_ratio[1]:.2%} variance)')
            plt.title('PCA Loadings - First Two Principal Components')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{self.output_path}/pca_loadings.png",
                       dpi=300, bbox_inches='tight')
            plt.show()

            loadings_df.to_csv(f"{self.output_path}/pca_loadings.csv", index=False)

        # Save PCA results
        pca_summary = {
            'explained_variance_ratio': explained_var_ratio.tolist(),
            'cumulative_variance_ratio': cumulative_var_ratio.tolist(),
            'n_components_80_variance': int(np.argmax(cumulative_var_ratio >= 0.8) + 1),
            'n_components_90_variance': int(np.argmax(cumulative_var_ratio >= 0.9) + 1)
        }

        with open(f"{self.output_path}/pca_summary.json", 'w') as f:
            json.dump(pca_summary, f, indent=2)

        return pca_summary

    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        print("\n📊 Calculating risk metrics...")

        risk_metrics = []

        for crypto, df in self.crypto_data.items():
            if len(df) > 252 and 'close' in df.columns:  # At least 1 year of data
                try:
                    returns = df['close'].pct_change().dropna()

                    if len(returns) > 100:
                        metrics = self._calculate_crypto_risk_metrics(crypto, df, returns)
                        risk_metrics.append(metrics)
                except Exception as e:
                    print(f"⚠️ Error calculating risk metrics for {crypto}: {e}")

        if risk_metrics:
            risk_df = pd.DataFrame(risk_metrics)

            # Save risk metrics
            risk_df.to_csv(f"{self.output_path}/risk_metrics.csv", index=False)

            # Create risk metrics visualization
            self._visualize_risk_metrics(risk_df)

            return risk_df

        return None

    def _calculate_crypto_risk_metrics(self, crypto, df, returns):
        """Calculate risk metrics for a single cryptocurrency"""
        # Basic return metrics
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

        # Annualized metrics
        trading_days = 252
        years = len(df) / trading_days

        annualized_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (1/years) - 1) * 100
        annualized_volatility = returns.std() * np.sqrt(trading_days) * 100

        # Risk-adjusted returns
        risk_free_rate = 2  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Downside risk metrics
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days) * 100 if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Calmar ratio (ensure positive for visualization)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Value at Risk (VaR) - 5% and 1%
        var_5 = np.percentile(returns, 5) * 100
        var_1 = np.percentile(returns, 1) * 100

        # Expected Shortfall (Conditional VaR)
        es_5 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        es_1 = returns[returns <= np.percentile(returns, 1)].mean() * 100

        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        return {
            'cryptocurrency': crypto,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_5': var_5,
            'var_1': var_1,
            'expected_shortfall_5': es_5,
            'expected_shortfall_1': es_1,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'data_points': len(returns)
        }

    def _visualize_risk_metrics(self, risk_df):
        """Create comprehensive risk metrics visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cryptocurrency Risk Metrics Dashboard', fontsize=18, fontweight='bold')

        # 1. Risk vs Return scatter
        scatter = axes[0, 0].scatter(risk_df['annualized_volatility'],
                                    risk_df['annualized_return'],
                                    c=risk_df['sharpe_ratio'],
                                    s=100, alpha=0.7, cmap='RdYlGn')
        axes[0, 0].set_xlabel('Annualized Volatility (%)')
        axes[0, 0].set_ylabel('Annualized Return (%)')
        axes[0, 0].set_title('Risk vs Return (Color = Sharpe Ratio)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])

        # 2. Sharpe Ratio ranking
        top_sharpe = risk_df.nlargest(10, 'sharpe_ratio')
        axes[0, 1].barh(range(len(top_sharpe)), top_sharpe['sharpe_ratio'])
        axes[0, 1].set_yticks(range(len(top_sharpe)))
        axes[0, 1].set_yticklabels(top_sharpe['cryptocurrency'])
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_title('Top 10 Cryptocurrencies by Sharpe Ratio')

        # 3. Maximum Drawdown
        axes[0, 2].hist(risk_df['max_drawdown'], bins=15, alpha=0.7, color='red', edgecolor='black')
        axes[0, 2].axvline(risk_df['max_drawdown'].median(), color='black',
                          linestyle='--', linewidth=2, label=f'Median: {risk_df["max_drawdown"].median():.1f}%')
        axes[0, 2].set_xlabel('Maximum Drawdown (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Maximum Drawdowns')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. VaR Analysis
        axes[1, 0].scatter(risk_df['var_5'], risk_df['var_1'], alpha=0.7, s=100)
        axes[1, 0].plot([-10, 0], [-10, 0], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('VaR 5% (%)')
        axes[1, 0].set_ylabel('VaR 1% (%)')
        axes[1, 0].set_title('Value at Risk Comparison')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Skewness vs Kurtosis
        axes[1, 1].scatter(risk_df['skewness'], risk_df['kurtosis'], alpha=0.7, s=100)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Normal Kurtosis')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Normal Skewness')
        axes[1, 1].set_xlabel('Skewness')
        axes[1, 1].set_ylabel('Excess Kurtosis')
        axes[1, 1].set_title('Return Distribution Shape')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Sortino vs Sharpe
        axes[1, 2].scatter(risk_df['sharpe_ratio'], risk_df['sortino_ratio'], alpha=0.7, s=100)
        axes[1, 2].plot([risk_df['sharpe_ratio'].min(), risk_df['sharpe_ratio'].max()],
                       [risk_df['sharpe_ratio'].min(), risk_df['sharpe_ratio'].max()],
                       'r--', alpha=0.5, label='Equal Ratios')
        axes[1, 2].set_xlabel('Sharpe Ratio')
        axes[1, 2].set_ylabel('Sortino Ratio')
        axes[1, 2].set_title('Risk-Adjusted Returns Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/risk_metrics_dashboard.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self):
        """Generate final comprehensive analysis report"""
        print("\n📋 Generating comprehensive analysis report...")

        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_summary': {
                'total_cryptocurrencies': len(self.crypto_data),
                'total_features': len(list(self.crypto_data.values())[0].columns) if self.crypto_data else 0,
                'date_range': {
                    'start': min(df.index.min() for df in self.crypto_data.values()).isoformat(),
                    'end': max(df.index.max() for df in self.crypto_data.values()).isoformat()
                } if self.crypto_data else {}
            },
            'files_generated': [
                'descriptive_statistics_summary.xlsx',
                'correlation_matrices.xlsx',
                'risk_metrics.csv',
                'cluster_assignments.csv',
                'pca_loadings.csv',
                'pca_summary.json',
                'Various visualization PNG files'
            ]
        }

        # Save comprehensive report
        with open(f"{self.output_path}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {self.output_path}")
        print(f"📈 Generated {len(report['files_generated'])} analysis files")

        return report


def main():
    """
    Main execution function for statistical analysis
    """
    print("📊 Starting Comprehensive Statistical Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CryptoStatisticalAnalyzer()

    # Load data
    n_cryptos = analyzer.load_feature_data()

    if n_cryptos == 0:
        print("❌ No cryptocurrency data loaded")
        return

    # Perform analyses
    print("\n🔄 Executing analysis pipeline...")

    # 1. Descriptive statistics
    stats_summary = analyzer.generate_descriptive_statistics()

    # 2. Correlation analysis
    correlations = analyzer.create_correlation_matrices()

    # 3. Cluster analysis
    clusters = analyzer.perform_cluster_analysis()

    # 4. PCA analysis
    pca_results = analyzer.analyze_principal_components()

    # 5. Risk metrics
    risk_metrics = analyzer.calculate_risk_metrics()

    # 6. Generate final report
    report = analyzer.generate_comprehensive_report()

    print("\n✅ Statistical analysis completed successfully!")
    print("📊 All results saved to analysis_results directory")


if __name__ == "__main__":
    main()