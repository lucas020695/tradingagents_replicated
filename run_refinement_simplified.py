"""
SIMPLER APPROACH: Don't refactor the entire orchestrator.
Just modify the Trader behavior based on config BEFORE creating orchestrator.

This is pragmatic and avoids abstract class issues.
"""

import sys
from datetime import date
import pandas as pd
import json
import os

print("=" * 80)
print("QUANT-ONLY REFINEMENT: SIMPLIFIED ABLATION STUDY")
print("=" * 80 + "\n")

os.makedirs("results", exist_ok=True)

print("Step 1: Imports...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator
    from src.core.types import MarketContext
    from src.approaches.quant_only import create_quant_only_orchestrator

    print("  ✓ Imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("Step 2: Loading data for Q1 & Q2...")
try:
    loader = DataLoader(verbose=False)

    df_q1 = loader.fetch_historical_data(
        symbols=["AAPL", "NVDA", "MSFT", "META", "GOOGL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 29),
    )
    df_q1 = loader.add_technical_indicators(df_q1)
    df_q1 = df_q1.ffill().bfill()

    df_q2 = loader.fetch_historical_data(
        symbols=["AAPL", "NVDA", "MSFT", "META", "GOOGL"],
        start_date=date(2024, 4, 1),
        end_date=date(2024, 6, 28),
    )
    df_q2 = loader.add_technical_indicators(df_q2)
    df_q2 = df_q2.ffill().bfill()

    print(f"  ✓ Q1: {len(df_q1)} rows")
    print(f"  ✓ Q2: {len(df_q2)} rows\n")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("Step 3: Loading fundamentals...")
try:
    fundamentals = loader.fetch_fundamentals(["AAPL", "NVDA", "MSFT", "META", "GOOGL"])
    print(f"  ✓ Fundamentals loaded\n")
except:
    fundamentals = {}

# Define configurations
CONFIGURATIONS = {
    "A_Baseline": {
        "description": "Original (0.2 position scale, no hysteresis)",
        "use_fundamentals": False,
    },
    "B_WithFundamentals": {
        "description": "Add real fundamentals",
        "use_fundamentals": True,
    },
}

print("Step 4: Running ablation study...\n")


def run_backtest(df, use_fundamentals, fundamentals_data, symbols):
    """Simple backtest: buy all day 1, hold or trade based on rules."""
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    calc = MetricsCalculator(risk_free_rate=0.02)

    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0

    trading_dates = sorted(df.index.get_level_values(0).unique())

    orchestrator = create_quant_only_orchestrator()

    # DAY 1: Buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
        except:
            pass

    # REST OF DAYS
    for trading_date in trading_dates:
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = float(row["close"])

                technicals = {}
                for col in df.columns:
                    if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                        val = row[col]
                        if pd.notna(val):
                            technicals[col] = float(val)

                # KEY CHANGE: Include fundamentals if requested
                fund_data = fundamentals_data.get(symbol, {}) if use_fundamentals else {}

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
                    fundamentals=fund_data,
                    sentiment_scores={},
                    current_position=positions[symbol],
                    cash=cash,
                    portfolio_value=cash + sum(positions[s] * close for s in symbols),
                    risk_limits={"max_position_pct": 0.25, "max_leverage": 1.5},
                )

                agent_state = orchestrator.run_day(
                    symbol=symbol,
                    as_of=trading_date,
                    context=context,
                    portfolio_state={s: positions[s] for s in symbols},
                )

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
                            if shares_to_sell > 0:
                                positions[symbol] -= shares_to_sell
                                cash += shares_to_sell * close
                                total_trades += 1
            except:
                pass

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
    return metrics, total_trades


results = {"metadata": {}, "configs": {}}
symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

for config_name, config in CONFIGURATIONS.items():
    print(f"{config_name}: {config['description']}")

    try:
        metrics_q1, trades_q1 = run_backtest(df_q1, config["use_fundamentals"], fundamentals, symbols)
        print(f"  Q1: CR={metrics_q1['cumulative_return'] * 100:>6.2f}% | "
              f"Sharpe={metrics_q1['sharpe_ratio']:>5.2f} | "
              f"Trades={trades_q1}")

        metrics_q2, trades_q2 = run_backtest(df_q2, config["use_fundamentals"], fundamentals, symbols)
        print(f"  Q2: CR={metrics_q2['cumulative_return'] * 100:>6.2f}% | "
              f"Sharpe={metrics_q2['sharpe_ratio']:>5.2f} | "
              f"Trades={trades_q2}")

        q1_q2_delta = abs(metrics_q1['cumulative_return'] - metrics_q2['cumulative_return'])
        if q1_q2_delta < 0.05:
            print(f"  ✅ Generalization OK: delta={q1_q2_delta * 100:.2f}%")
        else:
            print(f"  ⚠️ Possible regime change: delta={q1_q2_delta * 100:.2f}%")

        results["configs"][config_name] = {
            "config": config,
            "q1_cr": float(metrics_q1["cumulative_return"]),
            "q2_cr": float(metrics_q2["cumulative_return"]),
            "q1_sharpe": float(metrics_q1["sharpe_ratio"]),
            "q2_sharpe": float(metrics_q2["sharpe_ratio"]),
            "delta": float(q1_q2_delta),
            "trades_q1": trades_q1,
            "trades_q2": trades_q2,
        }
        print()

    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        import traceback

        traceback.print_exc()

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80 + "\n")

print("| Config | Q1 CR | Q2 CR | Delta | Q1 Sharpe | Q2 Sharpe | Trades Q1 | Trades Q2 |")
print("|--------|-------|-------|-------|-----------|-----------|-----------|-----------|")

for name, result in results["configs"].items():
    print(f"| {name:20s} | {result['q1_cr'] * 100:5.2f}% | {result['q2_cr'] * 100:5.2f}% | "
          f"{result['delta'] * 100:5.2f}% | {result['q1_sharpe']:9.2f} | {result['q2_sharpe']:9.2f} | "
          f"{result['trades_q1']:9d} | {result['trades_q2']:9d} |")

with open("results/refinement_results_simple.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/refinement_results_simple.json")
print("\nKEY INSIGHTS:")
print("- If B (with fundamentals) > A (baseline), fundamentals help")
print("- If Q1/Q2 deltas are similar, approach is stable")
print("- Trade count shows if we're active or passive")