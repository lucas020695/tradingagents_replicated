"""
Baseline trading strategies for comparison with multi-agent approaches.

This module implements the five baseline strategies mentioned in the TradingAgents paper:
1. Buy and Hold (BH) - buy once, hold forever
2. MACD - Moving Average Convergence Divergence crossover
3. KDJ + RSI - Stochastic + Relative Strength Index combination
4. ZMR - Zero Mean Reversion (mean-reversion strategy)
5. SMA - Simple Moving Average crossover

All baselines follow the same interface:
    signal = strategy.generate_signal(context)
    # Returns: "buy", "sell", or "hold"

This ensures they can be plugged into the same backtesting engine as the multi-agent approaches.

Reference: TradingAgents paper, Section 5.1 - Baseline Descriptions
"""

from typing import Dict, Any, Optional, Tuple
from datetime import date
import pandas as pd
import numpy as np


class BaseStrategy:
    """Abstract base class for all strategies."""

    def __init__(self, symbol: str, lookback_window: int = 50):
        """
        Initialize strategy.

        Args:
            symbol (str): Ticker symbol
            lookback_window (int): Number of historical periods to consider
        """
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.position = 0  # Current position (shares)

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate trading signal based on context.

        Args:
            context (Dict): MarketContext-like object with technicals, prices, etc.

        Returns:
            str: "buy", "sell", or "hold"
        """
        raise NotImplementedError

    def reset(self):
        """Reset strategy state between backtests."""
        self.position = 0


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold: Purchase on first signal, hold until end.

    Logic:
        - Day 1: BUY (enter position)
        - All other days: HOLD (maintain position)

    Benchmark: Simple passive strategy, no active management.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, lookback_window=1)
        self.entered = False

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate buy-and-hold signal.

        Args:
            context (Dict): Must contain 'date' and 'close'

        Returns:
            str: "buy" on first day, "hold" thereafter
        """
        if not self.entered:
            self.entered = True
            return "buy"
        return "hold"

    def reset(self):
        super().reset()
        self.entered = False


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Logic:
        - BUY when MACD line crosses above signal line (bullish)
        - SELL when MACD line crosses below signal line (bearish)
        - HOLD otherwise

    Parameters:
        - MACD fast period: 12
        - MACD slow period: 26
        - Signal line period: 9

    Benchmark: Classic technical momentum indicator.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, lookback_window=26)
        self.prev_macd = None
        self.prev_signal = None

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate MACD crossover signal.

        Args:
            context (Dict): Must contain technicals['MACD*'] or similar

        Returns:
            str: "buy", "sell", or "hold"
        """
        technicals = context.get("technicals", {})

        # Extract MACD components (pandas_ta output format)
        macd_line = None
        signal_line = None

        # Try various naming conventions
        for key in technicals.keys():
            if "MACD" in key and "signal" not in key and "_" not in key:
                macd_line = technicals.get(key)
            if "MACDs" in key:  # Signal line
                signal_line = technicals.get(key)

        # Fallback: if not enough data, hold
        if macd_line is None or signal_line is None:
            return "hold"

        # Check for crossover
        if self.prev_macd is None or self.prev_signal is None:
            self.prev_macd = macd_line
            self.prev_signal = signal_line
            return "hold"

        # Bullish crossover: MACD crosses above signal
        if self.prev_macd <= self.prev_signal and macd_line > signal_line:
            self.prev_macd = macd_line
            self.prev_signal = signal_line
            return "buy"

        # Bearish crossover: MACD crosses below signal
        if self.prev_macd >= self.prev_signal and macd_line < signal_line:
            self.prev_macd = macd_line
            self.prev_signal = signal_line
            return "sell"

        self.prev_macd = macd_line
        self.prev_signal = signal_line
        return "hold"

    def reset(self):
        super().reset()
        self.prev_macd = None
        self.prev_signal = None


class KDJRSIStrategy(BaseStrategy):
    """
    KDJ + RSI Strategy: Combination of Stochastic and RSI indicators.

    Logic:
        - BUY when RSI < 30 (oversold) AND KDJ J line < 20 (strong oversold)
        - SELL when RSI > 70 (overbought) OR KDJ J line > 80 (overbought)
        - HOLD otherwise

    Parameters:
        - RSI period: 14
        - KDJ period: 9 (Stochastic K)

    Benchmark: Contrarian reversal strategy using dual oscillators.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, lookback_window=14)

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate KDJ + RSI signal.

        Args:
            context (Dict): Must contain technicals with RSI and KDJ components

        Returns:
            str: "buy", "sell", or "hold"
        """
        technicals = context.get("technicals", {})

        # Extract RSI
        rsi = technicals.get("RSI_14")

        # Extract KDJ components (simplified: using a placeholder)
        # In full implementation, would compute Stochastic K, D, J
        kdj_j = technicals.get("KDJ_J", 50)  # Placeholder, assume 50 if missing

        if rsi is None:
            return "hold"

        # BUY signal: oversold conditions
        if rsi < 30 and kdj_j < 20:
            return "buy"

        # SELL signal: overbought conditions
        if rsi > 70 or kdj_j > 80:
            return "sell"

        return "hold"

    def reset(self):
        super().reset()


class ZMRStrategy(BaseStrategy):
    """
    ZMR (Zero Mean Reversion) Strategy: Mean reversion on price deviations.

    Logic:
        - Calculate Z-score of price relative to moving average
        - BUY when Z-score < -1.5 (price far below mean, expect bounce)
        - SELL when Z-score > +1.5 (price far above mean, expect pullback)
        - HOLD when -1.5 <= Z-score <= +1.5 (price near mean)

    Formula:
        Z-score = (Price - SMA(50)) / StdDev(50)

    Benchmark: Statistical mean-reversion strategy.
    """

    def __init__(self, symbol: str, lookback: int = 50, z_threshold: float = 1.5):
        super().__init__(symbol, lookback_window=lookback)
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.price_history = []

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate ZMR signal based on Z-score.

        Args:
            context (Dict): Must contain 'close' price and technicals['SMA_50']

        Returns:
            str: "buy", "sell", or "hold"
        """
        close = context.get("close")
        technicals = context.get("technicals", {})
        sma = technicals.get("SMA_50")

        if close is None or sma is None:
            return "hold"

        # Track price history for volatility calculation
        self.price_history.append(close)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

        # Calculate Z-score
        if len(self.price_history) < self.lookback:
            return "hold"

        mean = np.mean(self.price_history)
        std = np.std(self.price_history)

        if std == 0:
            return "hold"

        z_score = (close - mean) / std

        # Trading logic based on Z-score
        if z_score < -self.z_threshold:
            return "buy"  # Oversold, expect mean reversion up
        elif z_score > self.z_threshold:
            return "sell"  # Overbought, expect mean reversion down
        else:
            return "hold"

    def reset(self):
        super().reset()
        self.price_history = []


class SMAStrategy(BaseStrategy):
    """
    SMA (Simple Moving Average) Crossover Strategy.

    Logic:
        - BUY when SMA(20) crosses above SMA(50) (bullish)
        - SELL when SMA(20) crosses below SMA(50) (bearish)
        - HOLD otherwise

    Parameters:
        - Fast MA: 20-day SMA
        - Slow MA: 50-day SMA

    Benchmark: Classic trend-following strategy using moving average crossover.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, lookback_window=50)
        self.prev_sma20 = None
        self.prev_sma50 = None

    def generate_signal(self, context: Dict[str, Any]) -> str:
        """
        Generate SMA crossover signal.

        Args:
            context (Dict): Must contain technicals['SMA_20'] and technicals['SMA_50']

        Returns:
            str: "buy", "sell", or "hold"
        """
        technicals = context.get("technicals", {})

        sma20 = technicals.get("SMA_20")
        sma50 = technicals.get("SMA_50")

        if sma20 is None or sma50 is None:
            return "hold"

        # First observation: just record values
        if self.prev_sma20 is None or self.prev_sma50 is None:
            self.prev_sma20 = sma20
            self.prev_sma50 = sma50
            return "hold"

        # Bullish crossover: SMA20 crosses above SMA50
        if self.prev_sma20 <= self.prev_sma50 and sma20 > sma50:
            self.prev_sma20 = sma20
            self.prev_sma50 = sma50
            return "buy"

        # Bearish crossover: SMA20 crosses below SMA50
        if self.prev_sma20 >= self.prev_sma50 and sma20 < sma50:
            self.prev_sma20 = sma20
            self.prev_sma50 = sma50
            return "sell"

        self.prev_sma20 = sma20
        self.prev_sma50 = sma50
        return "hold"

    def reset(self):
        super().reset()
        self.prev_sma20 = None
        self.prev_sma50 = None


# Factory function for easy instantiation
def get_baseline_strategy(strategy_name: str, symbol: str) -> BaseStrategy:
    """
    Factory function to instantiate any baseline strategy.

    Args:
        strategy_name (str): Name of strategy ("buy_hold", "macd", "kdj_rsi", "zmr", "sma")
        symbol (str): Ticker symbol

    Returns:
        BaseStrategy: Instantiated strategy object

    Example:
        strategy = get_baseline_strategy("macd", "AAPL")
        signal = strategy.generate_signal(context)
    """
    strategies = {
        "buy_hold": BuyAndHoldStrategy,
        "macd": MACDStrategy,
        "kdj_rsi": KDJRSIStrategy,
        "zmr": ZMRStrategy,
        "sma": SMAStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategies[strategy_name](symbol)


# All available strategies for iteration
ALL_BASELINE_STRATEGIES = [
    "buy_hold",
    "macd",
    "kdj_rsi",
    "zmr",
    "sma",
]