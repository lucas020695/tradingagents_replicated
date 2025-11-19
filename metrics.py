"""
Performance metrics calculation utilities.

This module implements the four key metrics used in the TradingAgents paper:

1. Cumulative Return (CR): Total return over the entire backtest period
   CR = (Final Value - Initial Value) / Initial Value

2. Annualized Return (AR): Return normalized to annual basis
   AR = ((Final Value / Initial Value) ^ (1 / years)) - 1

3. Sharpe Ratio (SR): Risk-adjusted return
   SR = (Mean Daily Return - Risk Free Rate) / Std Dev of Daily Returns * sqrt(252)

4. Maximum Drawdown (MDD): Largest peak-to-trough decline
   MDD = min((Price - Max Prior Price) / Max Prior Price)

These metrics are SHARED across all three approaches (Perplexity API, weak LLM, quant-only),
ensuring fair and standardized comparison.

Reference: TradingAgents paper, Section 5.1 - Evaluation Metrics
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import date
import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Calculates and validates performance metrics for trading strategies.

    Usage:
        calc = MetricsCalculator(risk_free_rate=0.02)

        # Calculate all metrics at once
        metrics = calc.calculate_all(
            daily_values=daily_portfolio_values,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 29)
        )
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        # Or calculate individually
        cr = calc.cumulative_return(daily_values)
        sharpe = calc.sharpe_ratio(daily_values)
        mdd = calc.max_drawdown(daily_values)
    """

    def __init__(self, risk_free_rate: float = 0.0, trading_days_per_year: int = 252):
        """
        Initialize the metrics calculator.

        Args:
            risk_free_rate (float): Annual risk-free rate for Sharpe calculation (default 0%)
                Typically 0.02 for 2% annual rate
            trading_days_per_year (int): Number of trading days per year (default 252 for US markets)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    def cumulative_return(
            self,
            daily_values: List[float],
    ) -> float:
        """
        Calculate cumulative return over the entire period.

        Formula: CR = (Final Value - Initial Value) / Initial Value

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day

        Returns:
            float: Cumulative return as decimal (e.g., 0.15 = 15%)

        Example:
            daily_values = [100000, 101000, 100500, 102000]
            cr = calc.cumulative_return(daily_values)
            # Returns 0.02 (2% return)
        """
        if not daily_values or len(daily_values) < 2:
            return 0.0

        initial_value = daily_values[0]
        final_value = daily_values[-1]

        cr = (final_value - initial_value) / initial_value
        return float(cr)

    def annualized_return(
            self,
            daily_values: List[float],
            num_days: int,
    ) -> float:
        """
        Calculate annualized return.

        Formula: AR = ((Final Value / Initial Value) ^ (1 / years)) - 1

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day
            num_days (int): Total number of days in the backtest period

        Returns:
            float: Annualized return as decimal (e.g., 0.08 = 8% annual)

        Example:
            daily_values = [100000, ..., 115000]  # 15% gain over 3 months
            num_days = 90
            ar = calc.annualized_return(daily_values, num_days)
            # Returns ~0.80 (80% annualized, assuming linear scaling)
        """
        if not daily_values or len(daily_values) < 2 or num_days <= 0:
            return 0.0

        initial_value = daily_values[0]
        final_value = daily_values[-1]

        # Convert days to years (365 days = 1 year)
        years = num_days / 365.0

        if years <= 0:
            return 0.0

        # Compound annual growth rate (CAGR)
        ar = ((final_value / initial_value) ** (1 / years)) - 1
        return float(ar)

    def daily_returns(
            self,
            daily_values: List[float],
    ) -> np.ndarray:
        """
        Calculate daily returns (change from previous day).

        Formula: Daily Return_t = (Value_t - Value_{t-1}) / Value_{t-1}

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day

        Returns:
            np.ndarray: Array of daily returns (length = len(daily_values) - 1)

        Example:
            daily_values = [100000, 101000, 100500, 102000]
            returns = calc.daily_returns(daily_values)
            # Returns [0.01, -0.00495, 0.01493] (1%, -0.5%, 1.49%)
        """
        daily_values = np.array(daily_values, dtype=float)

        if len(daily_values) < 2:
            return np.array([])

        # Calculate day-to-day changes
        price_changes = np.diff(daily_values)

        # Divide by previous day's value
        previous_values = daily_values[:-1]

        # Avoid division by zero
        returns = np.divide(
            price_changes,
            previous_values,
            where=previous_values != 0,
            out=np.zeros_like(price_changes),
        )

        return returns

    def sharpe_ratio(
            self,
            daily_values: List[float],
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return).

        Formula: SR = (Mean Daily Return - Risk Free Rate / 252) / Std Dev of Daily Returns * sqrt(252)

        A Sharpe Ratio > 1.0 is considered good, > 2.0 is very good.

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day

        Returns:
            float: Sharpe ratio (higher is better)

        Example:
            daily_values = [100000, 101000, 100500, 102000, 103000]
            sr = calc.sharpe_ratio(daily_values)
            # Returns ~1.50 (good risk-adjusted return)
        """
        returns = self.daily_returns(daily_values)

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Use sample standard deviation

        if std_return == 0:
            return 0.0  # No volatility = undefined Sharpe

        # Daily risk-free rate
        daily_risk_free_rate = self.risk_free_rate / self.trading_days_per_year

        # Sharpe ratio annualized
        sharpe = (mean_return - daily_risk_free_rate) / std_return * np.sqrt(self.trading_days_per_year)

        return float(sharpe)

    def max_drawdown(
            self,
            daily_values: List[float],
    ) -> float:
        """
        Calculate maximum drawdown (worst peak-to-trough decline).

        Formula: MDD = min((Price_t - Max Prior Price) / Max Prior Price)

        Returns negative value (e.g., -0.15 = 15% drawdown).

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day

        Returns:
            float: Maximum drawdown as decimal (negative, e.g., -0.15 = -15%)

        Example:
            daily_values = [100000, 110000, 95000, 100000]
            mdd = calc.max_drawdown(daily_values)
            # Returns -0.1364 (peak was 110k, trough 95k, so -13.64%)
        """
        daily_values = np.array(daily_values, dtype=float)

        if len(daily_values) < 2:
            return 0.0

        # Running maximum up to each point
        cumulative_max = np.maximum.accumulate(daily_values)

        # Drawdown at each point
        drawdown = (daily_values - cumulative_max) / cumulative_max

        # Maximum drawdown (minimum value, which is most negative)
        mdd = np.min(drawdown)

        return float(mdd)

    def calculate_all(
            self,
            daily_values: List[float],
            start_date: date,
            end_date: date,
    ) -> Dict[str, float]:
        """
        Calculate all four metrics at once.

        Args:
            daily_values (List[float]): Portfolio value at end of each trading day
            start_date (date): Backtest start date
            end_date (date): Backtest end date

        Returns:
            Dict[str, float]: Dictionary with keys:
                - "cumulative_return": CR
                - "annualized_return": AR
                - "sharpe_ratio": Sharpe
                - "max_drawdown": MDD
        """
        num_days = (end_date - start_date).days

        metrics = {
            "cumulative_return": self.cumulative_return(daily_values),
            "annualized_return": self.annualized_return(daily_values, num_days),
            "sharpe_ratio": self.sharpe_ratio(daily_values),
            "max_drawdown": self.max_drawdown(daily_values),
            "num_days": num_days,
            "num_samples": len(daily_values),
        }

        return metrics

    def validate_metrics(
            self,
            metrics: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that calculated metrics are in reasonable ranges.

        Args:
            metrics (Dict[str, float]): Metrics dict from calculate_all()

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_warnings)

        Example:
            metrics = calc.calculate_all(daily_values, start, end)
            is_valid, warnings = calc.validate_metrics(metrics)
            for warning in warnings:
                print(f"⚠ {warning}")
        """
        warnings = []

        # Check Sharpe ratio range
        if metrics.get("sharpe_ratio", 0) > 10:
            warnings.append(
                f"Sharpe ratio {metrics['sharpe_ratio']:.2f} seems unrealistically high. "
                "Check for look-ahead bias or insufficient data."
            )

        # Check max drawdown
        if metrics.get("max_drawdown", 0) > 0:
            warnings.append("Max drawdown should be negative or zero.")

        if metrics.get("max_drawdown", 0) < -1.5:
            warnings.append(
                f"Max drawdown {metrics['max_drawdown']:.2%} exceeds -150%. "
                "Check for data quality or leverage issues."
            )

        # Check return consistency
        cr = metrics.get("cumulative_return", 0)
        ar = metrics.get("annualized_return", 0)
        if cr < 0 and ar > 0:
            warnings.append("Cumulative return is negative but annualized return is positive. Check data.")

        # Check number of samples
        if metrics.get("num_samples", 0) < 5:
            warnings.append(
                f"Only {metrics['num_samples']} samples. Results may not be statistically significant."
            )

        is_valid = len(warnings) == 0

        return is_valid, warnings

    def format_metrics(
            self,
            metrics: Dict[str, float],
    ) -> str:
        """
        Format metrics for human-readable display.

        Args:
            metrics (Dict[str, float]): Metrics dict from calculate_all()

        Returns:
            str: Formatted string for printing

        Example:
            print(calc.format_metrics(metrics))
            # Output:
            # Cumulative Return: 15.32%
            # Annualized Return: 18.45%
            # Sharpe Ratio: 1.85
            # Max Drawdown: -8.32%
        """
        output = []
        output.append("=" * 70)
        output.append("PERFORMANCE METRICS")
        output.append("=" * 70)
        output.append(f"Cumulative Return:    {metrics.get('cumulative_return', 0) * 100:>7.2f}%")
        output.append(f"Annualized Return:    {metrics.get('annualized_return', 0) * 100:>7.2f}%")
        output.append(f"Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>7.2f}")
        output.append(f"Max Drawdown:         {metrics.get('max_drawdown', 0) * 100:>7.2f}%")
        output.append(f"Trading Days:         {metrics.get('num_days', 0):>7.0f}")
        output.append(f"Daily Samples:        {metrics.get('num_samples', 0):>7.0f}")
        output.append("=" * 70)

        return "\n".join(output)


def compare_metrics(
        results: List[Tuple[str, Dict[str, float]]],
) -> str:
    """
    Compare metrics across multiple strategies.

    Args:
        results (List[Tuple[str, Dict]]): List of (strategy_name, metrics_dict)

    Returns:
        str: Formatted comparison table

    Example:
        results = [
            ("Buy & Hold", metrics_1),
            ("MACD", metrics_2),
            ("TradingAgents", metrics_3),
        ]
        print(compare_metrics(results))
    """
    df = pd.DataFrame([
        {
            "Strategy": name,
            "CR (%)": m.get("cumulative_return", 0) * 100,
            "AR (%)": m.get("annualized_return", 0) * 100,
            "Sharpe": m.get("sharpe_ratio", 0),
            "MDD (%)": m.get("max_drawdown", 0) * 100,
        }
        for name, m in results
    ])

    return df.to_string(index=False)