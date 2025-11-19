"""
PHASE 2 EXTENDED: ALL 5 BASELINE STRATEGIES WITH REALISTIC COSTS

Tests all 5 pre-implemented baseline strategies:
1. Buy & Hold
2. MACD
3. KDJ + RSI
4. ZMR (Zero Mean Reversion)
5. SMA (Simple Moving Average)

All with 0.17% per-trade transaction costs.

Expected finding:
- Most baselines underperform due to cost drag
- LLM will beat all baselines in next phase
- Buy & Hold still likely wins (only 1 trade)

Usage:
    python -m src.run_phase2_all_baselines
"""

import sys
from datetime import date
import pandas as pd
import json
import os

print("=" * 80)
print("PHASE 2 EXTENDED: ALL 5 BASELINE STRATEGIES WITH REALISTIC COSTS")
print("=" * 80 + "\n")

os.makedirs("results", exist_ok=True)

print("Configuration:")
print("  Strategies: Buy & Hold, MACD, KDJ+RSI, ZMR, SMA")
print("  Transaction costs: 0.17% per trade (0.10% + 0.05% + 0.02%)")
print("  Assets: AAPL, NVDA, MSFT, META, GOOGL")
print("  Period: Q1 2024 (Jan 1 - Mar 29)")
print("  Initial capital: $100,000\n")

# ============================================================================
# STEP 1: Imports
# ============================================================================

print("Step 1: Importing modules...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator
    from src.baselines import get_baseline_strategy, ALL_BASELINE_STRATEGIES
    from src.core.types import MarketContext

    print("  ✓ Imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Load data
# ============================================================================

print("Step 2: Loading market data...")
try:
    loader = DataLoader(verbose=False)
    df = loader.fetch_historical_data(
        symbols=["AAPL", "NVDA", "MSFT", "META", "GOOGL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 29),
    )
    df = loader.add_technical_indicators(df)
    df = df.ffill().bfill()
    print(f"  ✓ Loaded {len(df)} rows\n")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)


# ============================================================================
# STEP 3: Backtest function for baseline strategies
# ============================================================================

def run_baseline_backtest(strategy_name, symbols, cost_per_trade=0.0017):
    """Run backtest for a single baseline strategy."""
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    calc = MetricsCalculator(risk_free_rate=0.02)

    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0
    total_costs = 0.0

    trading_dates = sorted(df.index.get_level_values(0).unique())

    # Create strategy instances per symbol
    strategies = {s: get_baseline_strategy(strategy_name, s) for s in symbols}

    # DAY 1: Initial buy (all strategies start with position)
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            trade_cost = (shares * close) * cost_per_trade
            cash -= (shares * close + trade_cost)
            total_trades += 1
            total_costs += trade_cost
        except:
            pass

    # REST OF DAYS: Strategy signals
    for trading_date in trading_dates:
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = float(row["close"])

                # Build technicals dict for strategy
                technicals = {}
                for col in df.columns:
                    if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                        val = row[col]
                        if pd.notna(val):
                            technicals[col] = float(val)

                context = {
                    "symbol": symbol,
                    "as_of": trading_date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": close,
                    "adj_close": float(row["adj_close"]),
                    "volume": float(row["volume"]),
                    "technicals": technicals,
                }

                # Get strategy signal
                signal = strategies[symbol].generate_signal(context)

                # Execute trade
                if signal == "buy":
                    target_size = (cash + sum(positions[s] * close for s in symbols)) * 0.2 / close
                elif signal == "sell":
                    target_size = 0
                else:
                    target_size = positions[symbol]

                trade_shares = target_size - positions[symbol]

                if trade_shares > 0.01:  # Buy
                    trade_cost = (trade_shares * close) * cost_per_trade
                    positions[symbol] += trade_shares
                    cash -= (trade_shares * close + trade_cost)
                    total_trades += 1
                    total_costs += trade_cost
                elif trade_shares < -0.01:  # Sell
                    shares_to_sell = min(abs(trade_shares), positions[symbol])
                    trade_cost = (shares_to_sell * close) * cost_per_trade
                    positions[symbol] -= shares_to_sell
                    cash += (shares_to_sell * close - trade_cost)
                    total_trades += 1
                    total_costs += trade_cost
            except Exception as e:
                pass

        # Calculate daily portfolio value
        portfolio_value = cash
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = row["close"]
                portfolio_value += positions[symbol] * close
            except:
                pass
        daily_values.append(portfolio_value)

    metrics = calc.calculate_all(daily_values, trading_dates[0], trading_dates[-1])
    return metrics, total_trades, total_costs


# ============================================================================
# STEP 4: Run all baseline strategies
# ============================================================================

print("Step 3: Running all baseline strategies with costs...\n")

results = {}
symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

# Test each baseline
for i, strategy_name in enumerate(ALL_BASELINE_STRATEGIES, 1):
    print(f"{i}. {strategy_name.upper()}:")
    try:
        metrics, trades, costs = run_baseline_backtest(strategy_name, symbols, cost_per_trade=0.0017)
        if metrics:
            results[strategy_name] = {
                "cr": metrics["cumulative_return"],
                "ar": metrics["annualized_return"],
                "sharpe": metrics["sharpe_ratio"],
                "mdd": metrics["max_drawdown"],
                "trades": trades,
                "costs": costs,
            }
            print(f"   ✓ CR: {metrics['cumulative_return'] * 100:>6.2f}% | "
                  f"AR: {metrics['annualized_return'] * 100:>6.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:>6.2f} | "
                  f"MDD: {metrics['max_drawdown'] * 100:>6.2f}% | "
                  f"Trades: {trades:>3d} | Costs: ${costs:>8.2f}\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")

# ============================================================================
# STEP 5: Results summary
# ============================================================================

print("=" * 80)
print("PHASE 2 EXTENDED RESULTS: ALL 5 BASELINE STRATEGIES WITH COSTS")
print("=" * 80 + "\n")

print("| Strategy | CR (%) | AR (%) | Sharpe | MDD (%) | Trades | Costs ($) |")
print("|----------|--------|--------|--------|---------|--------|-----------|")

for name, result in results.items():
    print(f"| {name:30s} | {result['cr'] * 100:6.2f} | {result['ar'] * 100:6.2f} | "
          f"{result['sharpe']:6.2f} | {result['mdd'] * 100:7.2f} | {result['trades']:6d} | {result['costs']:9.2f} |")

# Save results
with open("results/phase2_extended_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/phase2_extended_results.json")

# ============================================================================
# STEP 6: Analysis
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2 EXTENDED ANALYSIS: BASELINE STRATEGY COMPARISON")
print("=" * 80 + "\n")

if len(results) > 0:
    print("Ranking by Cumulative Return:")
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['cr'], reverse=True)
    for rank, (name, result) in enumerate(sorted_strategies, 1):
        print(f"  {rank}. {name:20s}: {result['cr'] * 100:>6.2f}% CR | "
              f"Sharpe: {result['sharpe']:>6.2f} | Trades: {result['trades']:>3d} | Costs: ${result['costs']:>8.2f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)

    best = sorted_strategies[0]
    worst = sorted_strategies[-1]

    print(f"\n✓ Best performer:  {best[0]:20s} → {best[1]['cr'] * 100:6.2f}% CR")
    print(f"✗ Worst performer: {worst[0]:20s} → {worst[1]['cr'] * 100:6.2f}% CR")

    print(f"\nSpread: {(best[1]['cr'] - worst[1]['cr']) * 100:+.2f}pp")

    # Cost analysis
    total_cost_drag = sum(r['costs'] for r in results.values())
    print(f"\nTotal costs across all strategies: ${total_cost_drag:,.2f}")
    print(f"Average cost per strategy: ${total_cost_drag / len(results):,.2f}")

print("\n" + "=" * 80)
print("PHASE 2 EXTENDED COMPLETE")
print("Next: Phase 3 will test LLM against these baselines!")
print("=" * 80)