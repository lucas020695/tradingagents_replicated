"""
FIXED Refinement: Ablation study with configs actually being applied.

Key fix: Use create_configurable_quant_only_orchestrator() instead of generic one.
"""

import sys
from datetime import date
import pandas as pd
import json
import os

print("="*80)
print("QUANT-ONLY REFINEMENT: ABLATION STUDY [FIXED]")
print("="*80 + "\n")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# ============================================================================
# STEP 1: Setup
# ============================================================================

print("Step 1: Setup and imports...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator
    from src.core.types import MarketContext
    from experiments.quant_only_configurable import create_configurable_quant_only_orchestrator
    print("  ✓ Imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: Load data
# ============================================================================

print("Step 2: Loading market data for Q1 & Q2 2024...")
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
    print(f"  ✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: Load fundamentals
# ============================================================================

print("Step 3: Loading fundamentals...")
try:
    fundamentals = loader.fetch_fundamentals(["AAPL", "NVDA", "MSFT", "META", "GOOGL"])
    print(f"  ✓ Fundamentals loaded\n")
except Exception as e:
    print(f"  ⚠ Fundamentals optional: {e}")
    fundamentals = {}

# ============================================================================
# STEP 4: Define configurations
# ============================================================================

CONFIGURATIONS = {
    "A_Baseline": {
        "use_fundamentals": False,
        "use_hysteresis": False,
        "position_scale": 0.2,
        "weight_by_confidence": False,
        "hysteresis_threshold": 0.0,
        "description": "Original baseline (no changes)"
    },
    "B_AddFundamentals": {
        "use_fundamentals": True,
        "use_hysteresis": False,
        "position_scale": 0.2,
        "weight_by_confidence": False,
        "hysteresis_threshold": 0.0,
        "description": "Baseline + Real fundamentals"
    },
    "C_AddHysteresis": {
        "use_fundamentals": False,
        "use_hysteresis": True,
        "position_scale": 0.2,
        "weight_by_confidence": False,
        "hysteresis_threshold": 0.2,
        "description": "Baseline + Hysteresis (reduce whipsaw)"
    },
    "D_HigherPositionScale": {
        "use_fundamentals": False,
        "use_hysteresis": False,
        "position_scale": 0.35,
        "weight_by_confidence": False,
        "hysteresis_threshold": 0.0,
        "description": "Baseline + position_scale 0.2→0.35"
    },
    "E_AllRefinements": {
        "use_fundamentals": True,
        "use_hysteresis": True,
        "position_scale": 0.35,
        "weight_by_confidence": True,
        "hysteresis_threshold": 0.2,
        "description": "All refinements combined"
    }
}

# ============================================================================
# STEP 5: Run backtest function
# ============================================================================

def run_backtest(df, config, fundamentals_data, symbols, period_name):
    """Run backtest with specific config."""
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    calc = MetricsCalculator(risk_free_rate=0.02)

    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0

    trading_dates = sorted(df.index.get_level_values(0).unique())

    # Create orchestrator with THIS config
    orchestrator = create_configurable_quant_only_orchestrator(config)

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

    # LOOP THROUGH DAYS
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

                # Add fundamentals if config says so
                fund_data = fundamentals_data.get(symbol, {}) if config["use_fundamentals"] else {}

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

                # Run orchestrator with CONFIG
                agent_state = orchestrator.run_day(
                    symbol=symbol,
                    as_of=trading_date,
                    context=context,
                    portfolio_state={s: positions[s] for s in symbols},
                )

                # Execute decision
                if agent_state.final_decision:
                    decision = agent_state.final_decision
                    if decision.action != "hold":
                        trade_shares = decision.trade_size
                        if trade_shares > 0:  # Buy
                            cost = trade_shares * close
                            if cost <= cash:
                                positions[symbol] += trade_shares
                                cash -= cost
                                total_trades += 1
                        elif trade_shares < 0:  # Sell
                            shares_to_sell = min(abs(trade_shares), positions[symbol])
                            if shares_to_sell > 0:
                                positions[symbol] -= shares_to_sell
                                cash += shares_to_sell * close
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

        daily_values.append(portfolio_value)

    metrics = calc.calculate_all(daily_values, trading_dates[0], trading_dates[-1])
    return metrics, total_trades

# ============================================================================
# STEP 6: Run all configs on Q1 & Q2
# ============================================================================

print("Step 4: Running ablation study with configs...\n")

results = {
    "metadata": {
        "train_period": "2024-01-01 to 2024-03-29",
        "test_period": "2024-04-01 to 2024-06-28"
    },
    "configs": {}
}

symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

for config_name in ["A_Baseline", "B_AddFundamentals", "C_AddHysteresis", "D_HigherPositionScale", "E_AllRefinements"]:
    config = CONFIGURATIONS[config_name]

    print(f"{config_name}: {config['description']}")

    try:
        metrics_q1, trades_q1 = run_backtest(df_q1, config, fundamentals, symbols, "Q1")
        print(f"  Q1: CR={metrics_q1['cumulative_return']*100:>6.2f}% | "
              f"Sharpe={metrics_q1['sharpe_ratio']:>5.2f} | "
              f"MDD={metrics_q1['max_drawdown']*100:>6.2f}% | "
              f"Trades={trades_q1}")

        metrics_q2, trades_q2 = run_backtest(df_q2, config, fundamentals, symbols, "Q2")
        print(f"  Q2: CR={metrics_q2['cumulative_return']*100:>6.2f}% | "
              f"Sharpe={metrics_q2['sharpe_ratio']:>5.2f} | "
              f"MDD={metrics_q2['max_drawdown']*100:>6.2f}% | "
              f"Trades={trades_q2}")

        # Check for overfitting
        q1_q2_delta = abs(metrics_q1['cumulative_return'] - metrics_q2['cumulative_return'])
        if q1_q2_delta < 0.05:
            print(f"  ✅ Good: Q1/Q2 delta={q1_q2_delta*100:.2f}% (< 5%)")
        else:
            print(f"  ⚠️ High: Q1/Q2 delta={q1_q2_delta*100:.2f}% (> 5%, possible regime change)")

        results["configs"][config_name] = {
            "config": config,
            "q1": {k: float(v) for k, v in metrics_q1.items()},
            "q2": {k: float(v) for k, v in metrics_q2.items()},
            "q1_q2_delta": float(q1_q2_delta),
            "trades_q1": trades_q1,
            "trades_q2": trades_q2,
        }

        print()

    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 7: Summary
# ============================================================================

print("="*80)
print("ABLATION STUDY SUMMARY")
print("="*80 + "\n")

print("| Config | Q1 CR | Q2 CR | Δ | Sharpe Q1 | Sharpe Q2 | Trades Q1 | Trades Q2 |")
print("|--------|-------|-------|---|-----------|-----------|-----------|-----------|")

best_q1 = None
best_q2 = None
best_q1_cr = -1
best_q2_cr = -1

for config_name in sorted(results["configs"].keys()):
    result = results["configs"][config_name]
    q1_cr = result["q1"]["cumulative_return"] * 100
    q2_cr = result["q2"]["cumulative_return"] * 100
    delta = result["q1_q2_delta"] * 100
    sharpe_q1 = result["q1"]["sharpe_ratio"]
    sharpe_q2 = result["q2"]["sharpe_ratio"]
    trades_q1 = result["trades_q1"]
    trades_q2 = result["trades_q2"]

    print(f"| {config_name:20s} | {q1_cr:5.2f}% | {q2_cr:5.2f}% | {delta:5.2f}% | {sharpe_q1:9.2f} | {sharpe_q2:9.2f} | {trades_q1:9d} | {trades_q2:9d} |")

    if q1_cr > best_q1_cr:
        best_q1_cr = q1_cr
        best_q1 = config_name
    if q2_cr > best_q2_cr:
        best_q2_cr = q2_cr
        best_q2 = config_name

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

print(f"\nBest on Q1 (TRAIN): {best_q1} with {best_q1_cr:.2f}% CR")
print(f"Best on Q2 (TEST):  {best_q2} with {best_q2_cr:.2f}% CR")

if best_q1 == best_q2:
    print(f"\n✅ STABLE: Same config wins on both Q1 & Q2")
    print(f"   Recommendation: Use {best_q1} for production")
else:
    print(f"\n⚠️ INSTABILITY: Different configs on Q1 vs Q2")
    print(f"   This suggests market regime change or overfitting")

# Save results
with open("results/refinement_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/refinement_ablation_results.json")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Analyze which refinements matter most (check config deltas)")
print("2. Select best config based on Q1 & Q2 consistency")
print("3. Document findings in REFINEMENT_REPORT.md")
print("4. Proceed to Weak LLM or LLM Perplexity implementation\n")