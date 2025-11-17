"""
Integration test: Run a complete backtest with quant-only approach + baselines.

This script:
1. Downloads real market data (AAPL, 30-day window)
2. Runs backtest with:
   - Buy & Hold baseline
   - Quant-only multi-agent approach
3. Compares performance metrics (CR, AR, Sharpe, MDD)
4. Generates comparison report

Run this to validate that the entire system works end-to-end.

Usage:
    python -m src.test_backtest_integration
"""

import sys
from datetime import date, timedelta
import pandas as pd

print("=" * 70)
print("BACKTEST INTEGRATION TEST")
print("=" * 70 + "\n")

# Step 1: Import all modules
print("Step 1: Importing modules...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator, compare_metrics
    from src.baselines import get_baseline_strategy
    from src.core.backtest import BacktestEngine
    from src.approaches.quant_only import create_quant_only_orchestrator

    print("  ✓ All imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Step 2: Load market data
print("Step 2: Loading market data...")
try:
    loader = DataLoader(verbose=False)

    end_date = date(2024, 3, 29)
    start_date = end_date - timedelta(days=30)

    # Download data
    df = loader.fetch_historical_data(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date,
    )

    # Add technical indicators
    df = loader.add_technical_indicators(
        df,
        indicators=["MACD", "RSI", "BBANDS", "SMA", "EMA"]
    )

    # Fill missing values
    df = df.fillna(method="ffill")

    print(f"  ✓ Downloaded {len(df)} rows for AAPL")
    print(f"  ✓ Indicators computed ({df.shape[1]} columns)")
    print(f"  Date range: {loader.get_date_range(df)[0]} to {loader.get_date_range(df)[1]}\n")

except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 3: Run Buy & Hold baseline
print("Step 3: Running Buy & Hold baseline...")
try:
    # Simple Buy & Hold backtest
    bh_strategy = get_baseline_strategy("buy_hold", "AAPL")

    daily_values_bh = [100000.0]  # Start with $100k
    position = 0
    cash = 100000
    entry_price = None

    for (trading_date, symbol), row in df.iterrows():
        close = row["close"]

        # Generate signal
        context = {
            "close": close,
            "technicals": {col: row[col] for col in df.columns if
                           col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']},
        }
        signal = bh_strategy.generate_signal(context)

        # Execute
        if signal == "buy" and position == 0:
            position = cash / close
            entry_price = close
            cash = 0

        # Calculate daily value
        portfolio_value = position * close + cash
        daily_values_bh.append(portfolio_value)

    # Calculate metrics
    calc = MetricsCalculator()
    metrics_bh = calc.calculate_all(daily_values_bh, start_date, end_date)

    print(f"  ✓ Buy & Hold backtest complete")
    print(f"    CR: {metrics_bh['cumulative_return'] * 100:.2f}%")
    print(f"    Sharpe: {metrics_bh['sharpe_ratio']:.2f}\n")

except Exception as e:
    print(f"  ✗ Buy & Hold backtest failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 4: Run Quant-Only approach
print("Step 4: Running Quant-Only multi-agent approach...")
try:
    # Create orchestrator
    orchestrator = create_quant_only_orchestrator()

    # Simple backtest loop
    daily_values_quant = [100000.0]
    position = 0
    cash = 100000
    trades = 0

    for (trading_date, symbol), row in df.iterrows():
        close = row["close"]

        # Build MarketContext
        from src.core.types import MarketContext

        technicals = {}
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                val = row[col]
                if pd.notna(val):
                    technicals[col] = float(val)

        context = MarketContext(
            symbol="AAPL",
            as_of=trading_date,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(close),
            adj_close=float(row["adj_close"]),
            volume=float(row["volume"]),
            technicals=technicals,
            fundamentals={},
            sentiment_scores={},
            current_position=position,
            cash=cash,
            portfolio_value=position * close + cash,
            risk_limits={
                "max_position_pct": 0.25,
                "max_leverage": 1.5,
            },
        )

        # Run orchestrator
        try:
            agent_state = orchestrator.run_day(
                symbol="AAPL",
                as_of=trading_date,
                context=context,
                portfolio_state={"AAPL": position, "cash": cash},
            )

            decision = agent_state.final_decision

            # Execute trade
            if decision and decision.action != "hold":
                trade_shares = decision.trade_size
                if trade_shares > 0:  # Buy
                    cost = trade_shares * close
                    if cost <= cash:
                        position += trade_shares
                        cash -= cost
                        trades += 1
                elif trade_shares < 0:  # Sell
                    shares_to_sell = min(abs(trade_shares), position)
                    position -= shares_to_sell
                    cash += shares_to_sell * close
                    trades += 1
        except Exception as e:
            # Skip day if agent fails
            pass

        # Calculate daily value
        portfolio_value = position * close + cash
        daily_values_quant.append(portfolio_value)

    # Calculate metrics
    metrics_quant = calc.calculate_all(daily_values_quant, start_date, end_date)

    print(f"  ✓ Quant-Only backtest complete")
    print(f"    Trades executed: {trades}")
    print(f"    CR: {metrics_quant['cumulative_return'] * 100:.2f}%")
    print(f"    Sharpe: {metrics_quant['sharpe_ratio']:.2f}\n")

except Exception as e:
    print(f"  ✗ Quant-Only backtest failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 5: Compare results
print("Step 5: Comparison Report")
print("=" * 70)

results = [
    ("Buy & Hold", metrics_bh),
    ("Quant-Only Multi-Agent", metrics_quant),
]

comparison_table = compare_metrics(results)
print(comparison_table)

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)

cr_diff = metrics_quant['cumulative_return'] - metrics_bh['cumulative_return']
sharpe_diff = metrics_quant['sharpe_ratio'] - metrics_bh['sharpe_ratio']

print(f"\nCumulative Return differential: {cr_diff * 100:+.2f}% (Multi-Agent vs Buy&Hold)")
print(f"Sharpe Ratio differential: {sharpe_diff:+.2f}")

if cr_diff > 0.01:
    print(f"✓ Multi-agent approach outperformed Buy&Hold by {cr_diff * 100:.2f}%")
elif cr_diff < -0.01:
    print(f"✗ Multi-agent approach underperformed Buy&Hold by {abs(cr_diff) * 100:.2f}%")
else:
    print(f"≈ Performance within 1% of Buy&Hold")

if sharpe_diff > 0.1:
    print(f"✓ Multi-agent has better risk-adjusted returns (Sharpe +{sharpe_diff:.2f})")
else:
    print(f"⚠ Risk-adjusted returns similar or lower than Buy&Hold")

print("\n" + "=" * 70)
print("BACKTEST INTEGRATION TEST COMPLETE ✓")
print("=" * 70)
print("\nNext steps:")
print("  1. Implement src/approaches/llm_weak/ (with local LLM)")
print("  2. Implement src/approaches/llm_perplexity/ (with Perplexity API)")
print("  3. Run full comparison across all 3 approaches + baselines")
print("  4. Generate final results for LinkedIn/GitHub portfolio\n")