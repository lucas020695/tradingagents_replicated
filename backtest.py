"""
Backtesting engine for the TradingAgents replica framework.

This module implements the core simulation loop:
1. Load historical market data (OHLCV) for target date range
2. For each day and ticker:
   - Construct MarketContext with current portfolio state
   - Run OrchestrationEngine to get Decision
   - Execute Decision and update portfolio
   - Record metrics (PnL, returns, drawdown)
3. Compute aggregate performance metrics (CR, AR, Sharpe, MDD)

The backtester is APPROACH-AGNOSTIC: it calls abstract agents and doesn't care
whether they use LLM (Perplexity), weak LLM, or quant-only.

This ensures all approaches are tested on identical market data and conditions,
enabling fair comparison.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .types import (
    MarketContext,
    Decision,
    AgentState,
    BacktestResult,
)
from .agents_base import OrchestrationEngine


@dataclass
class PortfolioState:
    """
    Tracks the current portfolio holdings and cash balance.

    Attributes:
        cash (float): Available cash (uninvested capital)
        positions (Dict[str, float]): Holdings by ticker (shares, can be negative for short)
        entry_prices (Dict[str, float]): Average entry price per ticker (for debugging)
        trade_history (List[Dict]): Record of all executed trades
        daily_values (List[Tuple[date, float]]): Daily portfolio value (for metrics)
    """
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    entry_prices: Dict[str, float] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    daily_values: List[Tuple[date, float]] = field(default_factory=list)


@dataclass
class PositionMetrics:
    """
    Per-position tracking for detailed analysis.

    Attributes:
        symbol (str): Ticker
        entry_date (date): When position was opened
        exit_date (Optional[date]): When position was closed (None if still open)
        entry_price (float): Entry price
        exit_price (Optional[float]): Exit price (None if still open)
        quantity (float): Number of shares
        pnl (float): Profit/loss in dollars
        return_pct (float): Return as % (e.g., 0.05 = 5%)
        trade_days (int): Number of days held
    """
    symbol: str
    entry_date: date
    exit_date: Optional[date]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    return_pct: float
    trade_days: int


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the simulation loop.

    Usage:
        engine = BacktestEngine(
            orchestrator=orchestration_engine,
            initial_capital=100000,
            benchmark_symbol="SPY"
        )
        result = engine.run(
            market_data=price_df,
            symbols=["AAPL", "NVDA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 29)
        )
    """

    def __init__(
            self,
            orchestrator: OrchestrationEngine,
            initial_capital: float = 100000.0,
            benchmark_symbol: str = "SPY",
            transaction_cost_pct: float = 0.001,  # 0.1% per trade
            allow_short: bool = False,
    ):
        """
        Initialize the backtest engine.

        Args:
            orchestrator (OrchestrationEngine): Agent orchestrator for decision-making
            initial_capital (float): Starting portfolio value ($)
            benchmark_symbol (str): Ticker for benchmark comparison (e.g., "SPY")
            transaction_cost_pct (float): Transaction cost as % of trade value
            allow_short (bool): Whether to allow short selling
        """
        self.orchestrator = orchestrator
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol
        self.transaction_cost_pct = transaction_cost_pct
        self.allow_short = allow_short

        # Initialize portfolio state
        self.portfolio = PortfolioState(cash=initial_capital)
        self.position_metrics: List[PositionMetrics] = []

    def run(
            self,
            market_data: pd.DataFrame,
            symbols: List[str],
            start_date: date,
            end_date: date,
            strategy_name: str = "multi_agent",
            approach: str = "unknown",
    ) -> BacktestResult:
        """
        Run the complete backtest simulation.

        Args:
            market_data (pd.DataFrame): Historical OHLCV data indexed by (date, symbol)
                Expected columns: open, high, low, close, adj_close, volume
            symbols (List[str]): List of tickers to trade
            start_date (date): Backtest start date
            end_date (date): Backtest end date
            strategy_name (str): Human name for this strategy
            approach (str): Approach used ("llm_perplexity", "llm_weak", "quant_only", etc.)

        Returns:
            BacktestResult: Aggregated performance metrics
        """
        # Filter market data to date range
        mask = (market_data.index.get_level_values(0) >= start_date) & \
               (market_data.index.get_level_values(0) <= end_date)
        sim_data = market_data[mask]

        # Get unique trading dates in order
        trading_dates = sorted(sim_data.index.get_level_values(0).unique())

        print(f"\n{'=' * 70}")
        print(f"Backtesting: {strategy_name} ({approach})")
        print(f"Symbols: {symbols}")
        print(f"Period: {start_date} to {end_date} ({len(trading_dates)} trading days)")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"{'=' * 70}\n")

        # Main simulation loop: iterate through trading dates
        for trading_date in trading_dates:
            # For each ticker, get market context and run agent pipeline
            for symbol in symbols:
                try:
                    # Get OHLCV for this date and symbol
                    daily_row = market_data.loc[(trading_date, symbol)]
                    if daily_row.empty or daily_row.isnull().any():
                        continue  # Skip if data missing

                    # Construct MarketContext
                    context = self._build_market_context(
                        symbol=symbol,
                        as_of=trading_date,
                        daily_data=daily_row,
                        market_data=market_data,
                        trading_dates=trading_dates,
                    )

                    # Run orchestrator to get decision
                    agent_state = self.orchestrator.run_day(
                        symbol=symbol,
                        as_of=trading_date,
                        context=context,
                        portfolio_state=self._portfolio_dict(),
                    )

                    # Extract final decision
                    decision = agent_state.final_decision
                    if decision is None:
                        continue

                    # Execute decision
                    self._execute_decision(decision, daily_row, trading_date)

                except Exception as e:
                    print(f"[ERROR] {symbol} on {trading_date}: {e}")
                    continue

            # Record daily portfolio value
            daily_value = self._calculate_portfolio_value(
                trading_date, market_data, symbols
            )
            self.portfolio.daily_values.append((trading_date, daily_value))
            print(f"  {trading_date}: Portfolio value = ${daily_value:,.2f}")

        # Compute final metrics
        result = self._compute_metrics(
            start_date, end_date, strategy_name, approach, symbols
        )
        return result

    def _build_market_context(
            self,
            symbol: str,
            as_of: date,
            daily_data: pd.Series,
            market_data: pd.DataFrame,
            trading_dates: List[date],
    ) -> MarketContext:
        """
        Construct a MarketContext for a given ticker on a given day.

        Includes:
        - Daily OHLCV
        - Pre-computed technical indicators (if available)
        - Fundamental metrics (if available)
        - Sentiment scores (if approach supports)
        - Current portfolio state
        """
        # Extract basic OHLCV
        open_price = float(daily_data.get("open", 0.0))
        high_price = float(daily_data.get("high", 0.0))
        low_price = float(daily_data.get("low", 0.0))
        close_price = float(daily_data.get("close", 0.0))
        adj_close = float(daily_data.get("adj_close", close_price))
        volume = float(daily_data.get("volume", 0.0))

        # Compute technical indicators (simplified example)
        # In production, this would be pre-computed by data_loader.py
        technicals = self._compute_technicals(
            symbol, as_of, market_data, trading_dates
        )

        # Fundamental metrics (simplified; pulled from daily_data if available)
        fundamentals = {
            "pe_ratio": float(daily_data.get("pe_ratio", np.nan)),
            "pb_ratio": float(daily_data.get("pb_ratio", np.nan)),
            "roe": float(daily_data.get("roe", np.nan)),
        }
        # Remove NaN values
        fundamentals = {k: v for k, v in fundamentals.items() if not np.isnan(v)}

        # Sentiment scores (empty in quant-only; populated by LLM approaches)
        sentiment_scores = {}

        # Current portfolio state
        current_position = self.portfolio.positions.get(symbol, 0.0)
        portfolio_value = self._calculate_portfolio_value_now()

        context = MarketContext(
            symbol=symbol,
            as_of=as_of,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            adj_close=adj_close,
            volume=volume,
            technicals=technicals,
            fundamentals=fundamentals,
            sentiment_scores=sentiment_scores,
            current_position=current_position,
            cash=self.portfolio.cash,
            portfolio_value=portfolio_value,
            risk_limits={
                "max_position_pct": 0.2,
                "max_leverage": 1.0,
                "max_drawdown_pct": 0.1,
            },
        )
        return context

    def _compute_technicals(
            self,
            symbol: str,
            as_of: date,
            market_data: pd.DataFrame,
            trading_dates: List[date],
    ) -> Dict[str, float]:
        """
        Compute technical indicators for a symbol up to as_of date.

        In a real implementation, this would use pandas-ta or ta-lib.
        For now, returns a simplified set of indicators.

        Returns:
            Dict with keys like "MACD", "RSI", "SMA_20", "SMA_50", etc.
        """
        try:
            # Get historical data up to (and including) as_of date
            hist_data = market_data.loc[:as_of]
            hist_data = hist_data[hist_data.index.get_level_values(1) == symbol]

            if len(hist_data) < 50:
                return {}  # Not enough data

            closes = hist_data["close"].values
            last_close = closes[-1]

            # Simplified indicators (in production, use pandas-ta)
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.nan
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.nan

            # RSI (simplified)
            deltas = np.diff(closes[-14:])
            gains = np.mean([d for d in deltas if d > 0] or [0])
            losses = np.mean([-d for d in deltas if d < 0] or [0])
            rs = gains / losses if losses > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if losses > 0 else 50

            technicals = {
                "SMA_20": float(sma_20) if not np.isnan(sma_20) else 0.0,
                "SMA_50": float(sma_50) if not np.isnan(sma_50) else 0.0,
                "RSI": float(rsi),
                "last_close": float(last_close),
            }
            return {k: v for k, v in technicals.items() if not np.isnan(v)}

        except Exception as e:
            print(f"[WARN] Error computing technicals for {symbol}: {e}")
            return {}

    def _execute_decision(
            self,
            decision: Decision,
            daily_data: pd.Series,
            trading_date: date,
    ) -> None:
        """
        Execute a trading decision: update positions, cash, and trade history.

        Args:
            decision (Decision): Trading decision with action, size, etc.
            daily_data (pd.Series): OHLCV data for execution
            trading_date (date): Date of execution
        """
        symbol = decision.symbol
        action = decision.action
        trade_size = decision.trade_size
        reference_price = float(daily_data.get("close", decision.reference_price))

        if action == "hold" or trade_size == 0:
            return

        # Calculate transaction cost
        notional = abs(trade_size) * reference_price
        transaction_cost = notional * self.transaction_cost_pct

        # Check available cash
        if action == "buy" and trade_size > 0:
            cost = notional + transaction_cost
            if cost > self.portfolio.cash:
                print(
                    f"[WARN] Insufficient cash for {symbol} buy on {trading_date}. "
                    f"Need ${cost:,.2f}, have ${self.portfolio.cash:,.2f}"
                )
                return

        # Update position
        current_pos = self.portfolio.positions.get(symbol, 0.0)
        new_pos = current_pos + trade_size

        # Update cash
        self.portfolio.cash -= (notional + transaction_cost)

        # Record trade
        trade_record = {
            "date": trading_date,
            "symbol": symbol,
            "action": action,
            "size": trade_size,
            "price": reference_price,
            "cost": transaction_cost,
            "rationale": decision.rationale,
        }
        self.portfolio.trade_history.append(trade_record)

        # Update position tracking
        if new_pos != 0:
            if current_pos == 0:
                # Opening new position
                self.portfolio.positions[symbol] = new_pos
                self.portfolio.entry_prices[symbol] = reference_price
            else:
                # Scaling existing position
                avg_price = (
                                    current_pos * self.portfolio.entry_prices[symbol] +
                                    trade_size * reference_price
                            ) / new_pos
                self.portfolio.positions[symbol] = new_pos
                self.portfolio.entry_prices[symbol] = avg_price
        else:
            # Closing position
            if symbol in self.portfolio.positions:
                del self.portfolio.positions[symbol]
            if symbol in self.portfolio.entry_prices:
                del self.portfolio.entry_prices[symbol]

        print(
            f"    → {symbol} {action.upper()} {abs(trade_size):.0f} @ ${reference_price:.2f} "
            f"(cost ${transaction_cost:.2f})"
        )

    def _calculate_portfolio_value(
            self,
            as_of: date,
            market_data: pd.DataFrame,
            symbols: List[str],
    ) -> float:
        """Calculate total portfolio value (positions + cash) as of a given date."""
        value = self.portfolio.cash

        for symbol in symbols:
            qty = self.portfolio.positions.get(symbol, 0.0)
            if qty == 0:
                continue

            try:
                price_data = market_data.loc[(as_of, symbol)]
                price = float(price_data.get("close", 0.0))
                value += qty * price
            except Exception:
                pass

        return value

    def _calculate_portfolio_value_now(self) -> float:
        """Calculate current portfolio value (for MarketContext)."""
        # Simplified: just cash + position values at current prices
        # In production, would use latest market prices
        value = self.portfolio.cash
        for symbol, qty in self.portfolio.positions.items():
            entry_price = self.portfolio.entry_prices.get(symbol, 0.0)
            value += qty * entry_price
        return value

    def _portfolio_dict(self) -> Dict[str, Any]:
        """Return portfolio state as dict for agent context."""
        return {
            "cash": self.portfolio.cash,
            "positions": self.portfolio.positions.copy(),
            "entry_prices": self.portfolio.entry_prices.copy(),
        }

    def _compute_metrics(
            self,
            start_date: date,
            end_date: date,
            strategy_name: str,
            approach: str,
            symbols: List[str],
    ) -> BacktestResult:
        """
        Compute final performance metrics: CR, AR, Sharpe, MDD.

        Definitions (per TradingAgents paper):
        - CR: (Final Value - Initial Value) / Initial Value
        - AR: ((Final Value / Initial Value) ^ (1/n_years)) - 1
        - Sharpe: (Mean Daily Return - Risk-free Rate) / Std Dev of Returns
        - MDD: Maximum drawdown from peak to trough
        """
        daily_values = np.array([v for _, v in self.portfolio.daily_values])

        if len(daily_values) < 2:
            raise ValueError("Not enough data to compute metrics")

        final_value = daily_values[-1]
        initial_value = self.initial_capital

        # Cumulative Return
        cr = (final_value - initial_value) / initial_value

        # Annualized Return
        days = (end_date - start_date).days
        years = days / 252.0  # Trading days per year
        ar = ((final_value / initial_value) ** (1 / years)) - 1 if years > 0 else 0

        # Daily Returns
        daily_returns = np.diff(daily_values) / daily_values[:-1]

        # Sharpe Ratio (assuming 0% risk-free rate)
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Maximum Drawdown
        cumulative_max = np.maximum.accumulate(daily_values)
        drawdown = (daily_values - cumulative_max) / cumulative_max
        mdd = np.min(drawdown)

        # Trade statistics
        num_trades = len(self.portfolio.trade_history)
        winning_trades = sum(
            1 for t in self.portfolio.trade_history
            if t.get("pnl", 0) > 0
        )
        win_rate = winning_trades / num_trades if num_trades > 0 else 0

        print(f"\n{'=' * 70}")
        print(f"BACKTEST RESULTS: {strategy_name}")
        print(f"{'=' * 70}")
        print(f"Initial Capital:        ${initial_value:,.2f}")
        print(f"Final Value:            ${final_value:,.2f}")
        print(f"Cumulative Return:      {cr * 100:.2f}%")
        print(f"Annualized Return:      {ar * 100:.2f}%")
        print(f"Sharpe Ratio:           {sharpe:.2f}")
        print(f"Max Drawdown:           {mdd * 100:.2f}%")
        print(f"Total Trades:           {num_trades}")
        print(f"Win Rate:               {win_rate * 100:.2f}%")
        print(f"{'=' * 70}\n")

        return BacktestResult(
            symbol=",".join(symbols),
            start_date=start_date,
            end_date=end_date,
            cumulative_return=cr,
            annualized_return=ar,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            num_trades=num_trades,
            num_winning_trades=winning_trades,
            win_rate=win_rate,
            initial_capital=initial_value,
            final_value=final_value,
            approach=approach,
            strategy_name=strategy_name,
            metadata={
                "transaction_cost_pct": self.transaction_cost_pct,
                "allow_short": self.allow_short,
                "daily_values": daily_values.tolist(),
            },
        )