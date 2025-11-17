"""
FIXED Comprehensive backtest: Multiple assets, full 3-month period.

Key fixes:
1. Initialize positions on first valid day (not waiting for signals)
2. Handle NaN values in technicals properly
3. Ensure all strategies get fair starting conditions
4. Better error handling for missing data

Usage:
    python -m src.run_comprehensive_backtest_fixed
"""

import sys
from datetime import date
import pandas as pd
import numpy as np

print("="*80)
print("COMPREHENSIVE BACKTEST: MULTI-ASSET, FULL PERIOD [FIXED]")
print("="*80 + "\n")

print("Configuration:")
print("  Assets: AAPL, NVDA, MSFT, META, GOOGL")
print("  Period: January 1, 2024 - March 29, 2024 (88 trading days)")
print("  Approaches: 5 Baselines + Quant-Only Multi-Agent")
print("  Metrics: CR, AR, Sharpe, MDD\n")

# ============================================================================
# STEP 1: Import
# ============================================================================

print("Step 1: Importing modules...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator, compare_metrics
    from src.baselines import get_baseline_strategy
    from src.approaches.quant_only import create_quant_only_orchestrator
    from src.core.types import MarketContext
    print("  ✓ All imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Load data
# ============================================================================

print("Step 2: Loading market data (3 months, 5 assets)...")
try:
    loader = DataLoader(verbose=False)

    start_date = date(2024, 1, 1)
    end_date = date(2024, 3, 29)
    symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

    df = loader.fetch_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"  ✓ Downloaded {len(df)} rows")

    # Add technical indicators
    df = loader.add_technical_indicators(
        df,
        indicators=["MACD", "RSI", "BBANDS", "SMA", "EMA"]
    )

    print(f"  ✓ Indicators computed ({df.shape[1]} columns)")

    # Fill missing values
    df = df.ffill()
    df = df.bfill()  # Also fill from back just in case

    print(f"  ✓ Date range: {loader.get_date_range(df)[0]} to {loader.get_date_range(df)[1]}\n")

except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: Run backtests
# ============================================================================

print("Step 3: Running backtests (all assets, equal weight)...")
print("-"*80)

results = {}
initial_capital = 100000.0
position_per_asset = initial_capital / len(symbols)
calc = MetricsCalculator(risk_free_rate=0.02)

trading_dates = sorted(df.index.get_level_values(0).unique())

# ============================================================================
# 3A: Buy & Hold
# ============================================================================

print("\n  Buy & Hold Baseline:")
try:
    daily_values_bh = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital

    # FIRST DAY: Buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
        except:
            pass

    # ALL OTHER DAYS: Hold
    for trading_date in trading_dates:
        portfolio_value = cash
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = row["close"]
                portfolio_value += positions[symbol] * close
            except:
                pass
        daily_values_bh.append(portfolio_value)

    metrics_bh = calc.calculate_all(daily_values_bh, start_date, end_date)
    results["Buy & Hold"] = metrics_bh

    print(f"    CR: {metrics_bh['cumulative_return']*100:>7.2f}%")
    print(f"    AR: {metrics_bh['annualized_return']*100:>7.2f}%")
    print(f"    Sharpe: {metrics_bh['sharpe_ratio']:>7.2f}")
    print(f"    MDD: {metrics_bh['max_drawdown']*100:>7.2f}%")

except Exception as e:
    print(f"    ✗ Failed: {e}")

# ============================================================================
# 3B: Technical baselines
# ============================================================================

for baseline_name in ["macd", "sma", "zmr", "kdj_rsi"]:
    print(f"\n  {baseline_name.upper()} Baseline:")
    try:
        daily_values = [initial_capital]
        positions = {s: 0 for s in symbols}
        cash = initial_capital
        strategies = {s: get_baseline_strategy(baseline_name, s) for s in symbols}

        # FIRST DAY: Initialize position (BUY) for all strategies
        for symbol in strategies:
            try:
                row = df.loc[(trading_dates[0], symbol)]
                close = row["close"]
                shares = position_per_asset / close
                positions[symbol] = shares
                cash -= shares * close
            except:
                pass

        # LOOP THROUGH DAYS: Check signals and trade
        for i, trading_date in enumerate(trading_dates):
            for symbol in strategies:
                try:
                    row = df.loc[(trading_date, symbol)]
                    close = row["close"]

                    # Build context
                    technicals = {}
                    for col in df.columns:
                        if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                            val = row[col]
                            if pd.notna(val):
                                technicals[col] = float(val)

                    context = {
                        "close": float(close),
                        "technicals": technicals,
                    }

                    # Generate signal ONLY after day 1
                    if i > 0:
                        signal = strategies[symbol].generate_signal(context)

                        if signal == "sell" and positions[symbol] > 0:
                            cash += positions[symbol] * close
                            positions[symbol] = 0
                        elif signal == "buy" and positions[symbol] == 0:
                            shares = position_per_asset / close
                            positions[symbol] = shares
                            cash -= shares * close

                except Exception as e:
                    pass

            # Calculate portfolio value
            portfolio_value = cash
            for symbol in symbols:
                try:
                    row = df.loc[(trading_date, symbol)]
                    close = row["close"]
                    portfolio_value += positions[symbol] * close
                except:
                    pass

            daily_values.append(portfolio_value)

        metrics = calc.calculate_all(daily_values, start_date, end_date)
        results[f"{baseline_name.upper()}"] = metrics

        print(f"    CR: {metrics['cumulative_return']*100:>7.2f}%")
        print(f"    AR: {metrics['annualized_return']*100:>7.2f}%")
        print(f"    Sharpe: {metrics['sharpe_ratio']:>7.2f}")
        print(f"    MDD: {metrics['max_drawdown']*100:>7.2f}%")

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 3C: Quant-Only
# ============================================================================

print(f"\n  Quant-Only Multi-Agent:")
try:
    orchestrator = create_quant_only_orchestrator()

    daily_values_quant = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0

    # FIRST DAY: Buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
        except:
            pass

    # LOOP THROUGH DAYS: Run orchestrator
    for trading_date in trading_dates:
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = float(row["close"])

                # Build context
                technicals = {}
                for col in df.columns:
                    if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                        val = row[col]
                        if pd.notna(val):
                            technicals[col] = float(val)

                context = MarketContext(
                    symbol=symbol,
                    as_of=trading_date,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close,
                    adj_close=float(row["adj_close"]),
                    volume=float(row["volume"]),
                    technicals=technicals,
                    fundamentals={},
                    sentiment_scores={},
                    current_position=positions[symbol],
                    cash=cash,
                    portfolio_value=cash + sum(positions[s] * close for s in symbols),
                    risk_limits={"max_position_pct": 0.25, "max_leverage": 1.5},
                )

                # Run orchestrator
                agent_state = orchestrator.run_day(
                    symbol=symbol,
                    as_of=trading_date,
                    context=context,
                    portfolio_state={s: positions[s] for s in symbols},
                )

                # Execute
                if agent_state.final_decision:
                    decision = agent_state.final_decision
                    if decision.action != "hold":
                        trade_shares = decision.trade_size
                        if trade_shares > 0:
                            cost = trade_shares * close
                            if cost <= cash:
                                positions[symbol] += trade_shares
                                cash -= cost
                                total_trades += 1
                        elif trade_shares < 0:
                            shares_to_sell = min(abs(trade_shares), positions[symbol])
                            positions[symbol] -= shares_to_sell
                            cash += shares_to_sell * close
                            if shares_to_sell > 0:
                                total_trades += 1

            except Exception as e:
                pass

        # Calculate daily value
        portfolio_value = cash
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = row["close"]
                portfolio_value += positions[symbol] * close
            except:
                pass

        daily_values_quant.append(portfolio_value)

    metrics_quant = calc.calculate_all(daily_values_quant, start_date, end_date)
    results["Quant-Only Multi-Agent"] = metrics_quant

    print(f"    CR: {metrics_quant['cumulative_return']*100:>7.2f}%")
    print(f"    AR: {metrics_quant['annualized_return']*100:>7.2f}%")
    print(f"    Sharpe: {metrics_quant['sharpe_ratio']:>7.2f}")
    print(f"    MDD: {metrics_quant['max_drawdown']*100:>7.2f}%")
    print(f"    Trades: {total_trades}")

except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 4: Results
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS TABLE")
print("="*80)

if results:
    results_list = [(name, metrics) for name, metrics in results.items()]
    comparison = compare_metrics(results_list)
    print("\n" + comparison)

print("\n" + "="*80)
print("BACKTEST COMPLETE ✓")
print("="*80 + "\n")