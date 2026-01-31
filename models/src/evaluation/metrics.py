"""
Model Evaluation Metrics for Cryptocurrency Forecasting

Implements Kaggle-style evaluation metrics:
- MAPE, RMSE, MAE, R²
- Sharpe Ratio, Maximum Drawdown
- Directional Accuracy
- Confidence Calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics"""
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    r2_score: float  # R-squared
    directional_accuracy: float  # % of correct direction predictions
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    confidence_calibration: Optional[float] = None


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for forecasting models"""
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        
        Lower is better. MAPE < 10% is considered good for crypto.
        """
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R-squared (Coefficient of Determination)
        
        1.0 = perfect predictions
        0.0 = no better than mean
        < 0 = worse than mean
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prev: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate % of correct direction predictions.
        
        Important for trading: did we predict up/down correctly?
        """
        if y_prev is None:
            # Use shifted values
            y_prev = np.roll(y_true, 1)
            y_prev[0] = y_true[0]
        
        true_direction = np.sign(y_true - y_prev)
        pred_direction = np.sign(y_pred - y_prev)
        
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy * 100
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        Sharpe Ratio - Risk-adjusted returns
        
        Measures return per unit of risk.
        > 1.0 is good, > 2.0 is very good, > 3.0 is excellent
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """
        Maximum Drawdown - Largest peak-to-trough decline
        
        Measures worst-case loss from peak.
        Reported as negative percentage.
        """
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        return np.min(drawdown) * 100  # Convert to percentage
    
    @staticmethod
    def calculate_confidence_calibration(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_intervals: List[Tuple[float, float]],
        confidence_levels: List[float]
    ) -> float:
        """
        Confidence Calibration Score
        
        Measures how well predicted confidence matches actual accuracy.
        For 90% confidence interval, 90% of actual values should fall within it.
        
        Returns score from 0-100 (100 = perfect calibration)
        """
        if len(confidence_intervals) == 0:
            return 0.0
        
        calibration_errors = []
        
        for i, (conf_level, (lower, upper)) in enumerate(zip(confidence_levels, confidence_intervals)):
            # Check if true value falls within predicted interval
            within_interval = (y_true >= lower) & (y_true <= upper)
            actual_coverage = np.mean(within_interval)
            
            # Calculate calibration error
            expected_coverage = conf_level
            error = abs(actual_coverage - expected_coverage)
            calibration_errors.append(error)
        
        # Average calibration error
        avg_error = np.mean(calibration_errors)
        
        # Convert to score (lower error = higher score)
        calibration_score = max(0, 100 * (1 - avg_error))
        
        return calibration_score
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prev: Optional[np.ndarray] = None,
        prices_for_drawdown: Optional[np.ndarray] = None,
        confidence_intervals: Optional[List[Tuple[float, float]]] = None,
        confidence_levels: Optional[List[float]] = None
    ) -> ForecastMetrics:
        """
        Calculate all evaluation metrics at once.
        
        Returns:
            ForecastMetrics object with all computed metrics
        """
        mape = self.calculate_mape(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        mae = self.calculate_mae(y_true, y_pred)
        r2 = self.calculate_r2_score(y_true, y_pred)
        dir_acc = self.calculate_directional_accuracy(y_true, y_pred, y_prev)
        
        # Optional metrics
        sharpe = None
        if prices_for_drawdown is not None and len(prices_for_drawdown) > 1:
            returns = np.diff(prices_for_drawdown) / prices_for_drawdown[:-1]
            sharpe = self.calculate_sharpe_ratio(returns)
        
        max_dd = None
        if prices_for_drawdown is not None:
            max_dd = self.calculate_max_drawdown(prices_for_drawdown)
        
        conf_calib = None
        if confidence_intervals and confidence_levels:
            conf_calib = self.calculate_confidence_calibration(
                y_true, y_pred, confidence_intervals, confidence_levels
            )
        
        return ForecastMetrics(
            mape=mape,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            directional_accuracy=dir_acc,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            confidence_calibration=conf_calib
        )
    
    def compare_to_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare model performance against baseline (naive persistence).
        
        Returns:
            Dictionary with improvement metrics
        """
        model_mape = self.calculate_mape(y_true, y_pred)
        baseline_mape = self.calculate_mape(y_true, baseline_pred)
        
        model_rmse = self.calculate_rmse(y_true, y_pred)
        baseline_rmse = self.calculate_rmse(y_true, baseline_pred)
        
        return {
            'mape_improvement': ((baseline_mape - model_mape) / baseline_mape * 100) if baseline_mape > 0 else 0,
            'rmse_improvement': ((baseline_rmse - model_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0,
            'model_mape': model_mape,
            'baseline_mape': baseline_mape,
            'beats_baseline': model_mape < baseline_mape
        }
    
    @staticmethod
    def generate_performance_report(
        metrics: ForecastMetrics,
        model_name: str,
        crypto_id: str,
        baseline_comparison: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with formatted metrics and summary
        """
        report = {
            'model': model_name,
            'crypto': crypto_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'accuracy_metrics': {
                'MAPE': f"{metrics.mape:.2f}%",
                'RMSE': f"{metrics.rmse:.4f}",
                'MAE': f"{metrics.mae:.4f}",
                'R²': f"{metrics.r2_score:.4f}",
                'Directional_Accuracy': f"{metrics.directional_accuracy:.2f}%"
            },
            'risk_metrics': {}
        }
        
        if metrics.sharpe_ratio is not None:
            report['risk_metrics']['Sharpe_Ratio'] = f"{metrics.sharpe_ratio:.2f}"
        
        if metrics.max_drawdown is not None:
            report['risk_metrics']['Max_Drawdown'] = f"{metrics.max_drawdown:.2f}%"
        
        if metrics.confidence_calibration is not None:
            report['risk_metrics']['Confidence_Calibration'] = f"{metrics.confidence_calibration:.2f}%"
        
        if baseline_comparison:
            report['baseline_comparison'] = baseline_comparison
        
        # Add quality rating
        if metrics.mape < 5:
            rating = "Excellent"
        elif metrics.mape < 10:
            rating = "Good"
        elif metrics.mape < 15:
            rating = "Fair"
        else:
            rating = "Poor"
        
        report['quality_rating'] = rating
        
        return report


# Global instance
metrics_calculator = MetricsCalculator()

