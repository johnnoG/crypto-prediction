"""
Backtesting System for Cryptocurrency Forecasting Models

Implements walk-forward validation and comprehensive performance analysis:
- Time-series cross-validation
- Multiple evaluation metrics
- Strategy backtesting
- Performance visualization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import MetricsCalculator, ForecastMetrics


@dataclass
class BacktestResult:
    """Results from a single backtest fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_samples: int
    n_test_samples: int
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: ForecastMetrics


class ForecastBacktester:
    """
    Walk-forward backtesting for time series forecasting models.
    
    Simulates real-world deployment where model is periodically retrained.
    """
    
    def __init__(
        self,
        initial_train_size: int = 200,
        test_size: int = 30,
        step_size: int = 30,
        retrain_frequency: int = 30
    ):
        """
        Args:
            initial_train_size: Initial training set size (days)
            test_size: Test set size for each fold (days)
            step_size: How many days to move forward each iteration
            retrain_frequency: Retrain model every N days
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
        
        self.metrics_calculator = MetricsCalculator()
        self.results: List[BacktestResult] = []
    
    def run_walk_forward_validation(
        self,
        data: pd.DataFrame,
        model_trainer_fn: callable,
        target_column: str = 'close'
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.
        
        Args:
            data: Complete dataset with features
            model_trainer_fn: Function that trains model and returns it
                              Signature: fn(X_train, y_train) -> model
            target_column: Name of target variable
            
        Returns:
            List of backtest results for each fold
        """
        print("\nRunning Walk-Forward Validation")
        print("=" * 60)
        
        results = []
        n_samples = len(data)
        
        # Ensure data is sorted by time
        data = data.sort_index()
        
        fold_id = 0
        last_retrain_idx = 0
        trained_model = None
        
        for test_start_idx in range(
            self.initial_train_size,
            n_samples - self.test_size,
            self.step_size
        ):
            test_end_idx = min(test_start_idx + self.test_size, n_samples)
            train_end_idx = test_start_idx
            
            # Determine if we need to retrain
            should_retrain = (
                trained_model is None or
                (test_start_idx - last_retrain_idx) >= self.retrain_frequency
            )
            
            if should_retrain:
                # Train on all data up to test start
                train_data = data.iloc[:train_end_idx]
                
                X_train = train_data.drop(columns=[target_column]).values
                y_train = train_data[target_column].values
                
                print(f"\nFold {fold_id + 1}: Retraining model...")
                print(f"  Train: {len(train_data)} samples | Test: {test_end_idx - test_start_idx} samples")
                
                trained_model = model_trainer_fn(X_train, y_train)
                last_retrain_idx = test_start_idx
            else:
                print(f"\nFold {fold_id + 1}: Using existing model (no retrain)")
            
            # Test
            test_data = data.iloc[test_start_idx:test_end_idx]
            X_test = test_data.drop(columns=[target_column]).values
            y_test = test_data[target_column].values
            
            # Predict
            if hasattr(trained_model, 'predict'):
                y_pred = trained_model.predict(X_test)
            else:
                raise ValueError("Model does not have predict method")
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred)
            
            # Store result
            result = BacktestResult(
                fold_id=fold_id,
                train_start=data.index[0],
                train_end=data.index[train_end_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1],
                n_train_samples=train_end_idx,
                n_test_samples=len(X_test),
                predictions=y_pred,
                actuals=y_test,
                metrics=metrics
            )
            
            results.append(result)
            
            print(f"  MAPE: {metrics.mape:.2f}% | RMSE: {metrics.rmse:.4f} | R²: {metrics.r2_score:.4f}")
            
            fold_id += 1
        
        self.results = results
        
        print("\n" + "=" * 60)
        print("Walk-Forward Validation Complete")
        print(f"Total folds: {len(results)}")
        
        return results
    
    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all backtest folds.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if len(self.results) == 0:
            return {}
        
        # Collect metrics from all folds
        mapes = [r.metrics.mape for r in self.results]
        rmses = [r.metrics.rmse for r in self.results]
        maes = [r.metrics.mae for r in self.results]
        r2_scores = [r.metrics.r2_score for r in self.results]
        dir_accs = [r.metrics.directional_accuracy for r in self.results]
        
        aggregate = {
            'n_folds': len(self.results),
            'mape': {
                'mean': np.mean(mapes),
                'std': np.std(mapes),
                'min': np.min(mapes),
                'max': np.max(mapes),
                'median': np.median(mapes)
            },
            'rmse': {
                'mean': np.mean(rmses),
                'std': np.std(rmses),
                'min': np.min(rmses),
                'max': np.max(rmses),
                'median': np.median(rmses)
            },
            'r2_score': {
                'mean': np.mean(r2_scores),
                'std': np.std(r2_scores),
                'min': np.min(r2_scores),
                'max': np.max(r2_scores),
                'median': np.median(r2_scores)
            },
            'directional_accuracy': {
                'mean': np.mean(dir_accs),
                'std': np.std(dir_accs),
                'min': np.min(dir_accs),
                'max': np.max(dir_accs),
                'median': np.median(dir_accs)
            }
        }
        
        return aggregate
    
    def generate_backtest_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        aggregate_metrics = self.calculate_aggregate_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'initial_train_size': self.initial_train_size,
                'test_size': self.test_size,
                'step_size': self.step_size,
                'retrain_frequency': self.retrain_frequency
            },
            'aggregate_metrics': aggregate_metrics,
            'fold_details': []
        }
        
        # Add fold-level details
        for result in self.results:
            fold_detail = {
                'fold_id': result.fold_id,
                'test_period': f"{result.test_start.date()} to {result.test_end.date()}",
                'n_train': result.n_train_samples,
                'n_test': result.n_test_samples,
                'mape': result.metrics.mape,
                'rmse': result.metrics.rmse,
                'r2_score': result.metrics.r2_score,
                'directional_accuracy': result.metrics.directional_accuracy
            }
            report['fold_details'].append(fold_detail)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nBacktest report saved to {output_path}")
        
        return report
    
    def compare_with_baseline(
        self,
        baseline_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare model performance against naive baseline.
        
        Args:
            baseline_predictions: Predictions from naive persistence model
            
        Returns:
            Comparison statistics
        """
        if len(self.results) == 0:
            return {}
        
        # Collect all actuals and predictions
        all_actuals = np.concatenate([r.actuals for r in self.results])
        all_predictions = np.concatenate([r.predictions for r in self.results])
        
        # Calculate model metrics
        model_mape = self.metrics_calculator.calculate_mape(all_actuals, all_predictions)
        model_rmse = self.metrics_calculator.calculate_rmse(all_actuals, all_predictions)
        
        # Calculate baseline metrics
        baseline_mape = self.metrics_calculator.calculate_mape(all_actuals, baseline_predictions)
        baseline_rmse = self.metrics_calculator.calculate_rmse(all_actuals, baseline_predictions)
        
        # Calculate improvements
        mape_improvement = ((baseline_mape - model_mape) / baseline_mape * 100) if baseline_mape > 0 else 0
        rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0
        
        comparison = {
            'model_mape': model_mape,
            'baseline_mape': baseline_mape,
            'mape_improvement_pct': mape_improvement,
            'model_rmse': model_rmse,
            'baseline_rmse': baseline_rmse,
            'rmse_improvement_pct': rmse_improvement,
            'beats_baseline': model_mape < baseline_mape
        }
        
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)
        print(f"Model MAPE: {model_mape:.2f}% | Baseline MAPE: {baseline_mape:.2f}%")
        print(f"MAPE Improvement: {mape_improvement:+.2f}%")
        print(f"Beats Baseline: {'✓ YES' if comparison['beats_baseline'] else '✗ NO'}")
        
        return comparison


class TradingStrategyBacktester:
    """
    Backtest trading strategies based on forecast signals.
    
    Evaluates if forecasts can be profitably traded.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def backtest_forecast_strategy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.02  # 2% predicted move to trigger trade
    ) -> Dict[str, Any]:
        """
        Backtest a simple strategy: buy when forecast predicts >2% gain.
        
        Args:
            predictions: Forecasted prices
            actuals: Actual prices
            prices: Historical prices aligned with predictions
            threshold: Minimum predicted return to enter trade
            
        Returns:
            Strategy performance metrics
        """
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long
        trades = []
        equity = [capital]
        
        for i in range(len(predictions) - 1):
            predicted_return = (predictions[i] - prices[i]) / prices[i]
            
            # Trading logic
            if position == 0 and predicted_return > threshold:
                # Enter long position
                shares = capital / (prices[i] * (1 + self.slippage))
                cost = capital * (1 + self.commission)
                
                trades.append({
                    'type': 'BUY',
                    'price': prices[i],
                    'shares': shares,
                    'cost': cost,
                    'timestamp': i
                })
                
                position = 1
                capital -= cost
            
            elif position == 1 and (predicted_return < -threshold or i == len(predictions) - 2):
                # Exit position
                shares = trades[-1]['shares']
                revenue = shares * prices[i] * (1 - self.slippage) * (1 - self.commission)
                
                trades.append({
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': shares,
                    'revenue': revenue,
                    'timestamp': i
                })
                
                capital += revenue
                position = 0
            
            equity.append(capital + (shares * prices[i] if position == 1 else 0))
        
        # Calculate performance metrics
        final_equity = equity[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Calculate max drawdown
        equity_array = np.array(equity)
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax
        max_drawdown = np.min(drawdowns) * 100
        
        # Calculate Sharpe ratio
        equity_returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = self.metrics_calculator.calculate_sharpe_ratio(equity_returns) if len(equity_returns) > 0 else 0
        
        # Win rate
        profitable_trades = sum(1 for i in range(0, len(trades), 2) if i + 1 < len(trades) and trades[i+1]['revenue'] > trades[i]['cost'])
        total_trades = len(trades) // 2
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'n_trades': total_trades,
            'win_rate_pct': win_rate,
            'equity_curve': equity,
            'trades': trades
        }
        
        print(f"\nStrategy Performance:")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Win Rate: {win_rate:.2f}% ({profitable_trades}/{total_trades} trades)")
        
        return results
    
    def plot_results(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.results) == 0:
                print("No results to plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. MAPE over time
            mapes = [r.metrics.mape for r in self.results]
            axes[0, 0].plot(mapes, marker='o', linestyle='-', linewidth=2)
            axes[0, 0].axhline(y=10, color='r', linestyle='--', label='10% threshold')
            axes[0, 0].set_title('MAPE Over Time', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('MAPE (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. R² score over time
            r2_scores = [r.metrics.r2_score for r in self.results]
            axes[0, 1].plot(r2_scores, marker='s', linestyle='-', linewidth=2, color='green')
            axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='0.8 threshold')
            axes[0, 1].set_title('R² Score Over Time', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Directional Accuracy
            dir_accs = [r.metrics.directional_accuracy for r in self.results]
            axes[1, 0].plot(dir_accs, marker='^', linestyle='-', linewidth=2, color='purple')
            axes[1, 0].axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
            axes[1, 0].set_title('Directional Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Fold')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Predictions vs Actuals (last fold)
            last_result = self.results[-1]
            axes[1, 1].plot(last_result.actuals, label='Actual', linewidth=2)
            axes[1, 1].plot(last_result.predictions, label='Predicted', linewidth=2, linestyle='--')
            axes[1, 1].set_title('Last Fold: Predictions vs Actuals', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('Price')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib not installed, skipping plots")
    
    def save_backtest_results(
        self,
        output_path: str
    ) -> None:
        """
        Save backtest results to JSON.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'initial_train_size': self.initial_train_size,
                'test_size': self.test_size,
                'step_size': self.step_size,
                'retrain_frequency': self.retrain_frequency
            },
            'aggregate_metrics': self.calculate_aggregate_metrics(),
            'fold_results': [
                {
                    'fold_id': r.fold_id,
                    'test_period': f"{r.test_start.date()} to {r.test_end.date()}",
                    'mape': r.metrics.mape,
                    'rmse': r.metrics.rmse,
                    'mae': r.metrics.mae,
                    'r2_score': r.metrics.r2_score,
                    'directional_accuracy': r.metrics.directional_accuracy
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Backtest results saved to {output_path}")


# Convenience function
def run_quick_backtest(
    data: pd.DataFrame,
    model_type: str = 'lightgbm',
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run a quick backtest for a model.
    
    Args:
        data: DataFrame with features and target
        model_type: Type of model to backtest
        save_results: Whether to save results
        
    Returns:
        Backtest results
    """
    from models.lightgbm_model import LightGBMForecaster
    
    backtester = ForecastBacktester(
        initial_train_size=200,
        test_size=30,
        step_size=30,
        retrain_frequency=30
    )
    
    def train_fn(X, y):
        model = LightGBMForecaster()
        model.train(X, y)
        return model
    
    results = backtester.run_walk_forward_validation(data, train_fn, target_column='close')
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtester.save_backtest_results(f"models/artifacts/backtest_results_{timestamp}.json")
    
    return backtester.calculate_aggregate_metrics()

