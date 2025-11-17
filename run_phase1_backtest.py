"""
PHASE 1 BACKTEST: Paper Replication (Cost-Agnostic)

Backtests:
1. Quant-Only (baseline, 0.73% expected)
2. LLM Weak (Ollama Mistral)
3. Buy & Hold + other baselines (for reference)

NO transaction costs (replicates paper assumptions).

Usage:
    python -m src.run_phase1_backtest
"""

import sys
from datetime import date
import pandas as pd
import json
import os

print("=" * 80)
print("PHASE 1 BACKTEST: PAPER REPLICATION (COST-AGNOSTIC)")
print("=" * 80 + "\n")

os.makedirs("results", exist_ok=True)

print("Configuration:")
print("  Transaction costs: $0 (cost-agnostic model)")
print("  Assets: AAPL, NVDA, MSFT, META, GOOGL")
print("  Period: Q1 2024 (Jan 1 - Mar 29)")
print("  Initial capital: $100,000")
print("  Approaches: Quant-Only, LLM Weak, Buy & Hold (baseline)\n")

# ============================================================================
# STEP 1: Imports
# ============================================================================

print("Step 1: Importing modules...")
try:
    from src.utils.data_loader import DataLoader
    from src.utils.metrics import MetricsCalculator
    from src.baselines import get_baseline_strategy
    from src.approaches.quant_only import create_quant_only_orchestrator
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
# STEP 3: Generic backtest function
# ============================================================================

def run_backtest(orchestrator_func, orchestrator_name, symbols):
    """
    Run backtest with given orchestrator.

    Args:
        orchestrator_func: Function that returns orchestrator instance
        orchestrator_name: Name of approach
        symbols: List of symbols

    Returns:
        (metrics, trades)
    """
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    calc = MetricsCalculator(risk_free_rate=0.02)

    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0

    trading_dates = sorted(df.index.get_level_values(0).unique())

    # Create orchestrator
    try:
        orchestrator = orchestrator_func()
    except Exception as e:
        print(f"    ✗ Failed to create orchestrator: {e}")
        return None, 0

    # DAY 1: Buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
            total_trades += 1
        except:
            pass

    # REST OF DAYS
    for trading_date in trading_dates:
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = float(row["close"])

                # Build technicals
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

                # Execute decision
                if agent_state.final_decision:
                    decision = agent_state.final_decision
                    if decision.action != "hold":
                        trade_shares = decision.trade_size
                        if trade_shares > 0:  # Buy
                            positions[symbol] += trade_shares
                            cash -= trade_shares * close
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
# STEP 4: Run backtests
# ============================================================================

print("Step 3: Running backtests...\n")

results = {}
symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

# Quant-Only (baseline)
print("1. Quant-Only (Rule-Based, Baseline):")
try:
    metrics, trades = run_backtest(
        create_quant_only_orchestrator,
        "Quant-Only",
        symbols
    )
    if metrics:
        results["Quant-Only"] = {
            "cr": metrics["cumulative_return"],
            "ar": metrics["annualized_return"],
            "sharpe": metrics["sharpe_ratio"],
            "mdd": metrics["max_drawdown"],
            "trades": trades,
        }
        print(f"   ✓ CR: {metrics['cumulative_return'] * 100:>6.2f}% | "
              f"AR: {metrics['annualized_return'] * 100:>6.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:>6.2f} | "
              f"MDD: {metrics['max_drawdown'] * 100:>6.2f}% | "
              f"Trades: {trades:>3d}\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

# LLM Weak (Ollama)
print("2. LLM Weak (Ollama Mistral):")
try:
    from src.approaches.llm_weak.llm_weak import create_llm_weak_orchestrator

    metrics, trades = run_backtest(
        create_llm_weak_orchestrator,
        "LLM Weak",
        symbols
    )
    if metrics:
        results["LLM Weak"] = {
            "cr": metrics["cumulative_return"],
            "ar": metrics["annualized_return"],
            "sharpe": metrics["sharpe_ratio"],
            "mdd": metrics["max_drawdown"],
            "trades": trades,
        }
        print(f"   ✓ CR: {metrics['cumulative_return'] * 100:>6.2f}% | "
              f"AR: {metrics['annualized_return'] * 100:>6.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:>6.2f} | "
              f"MDD: {metrics['max_drawdown'] * 100:>6.2f}% | "
              f"Trades: {trades:>3d}\n")
    else:
        print(f"   ⚠ Skipped (LLM not available)\n")
except ImportError:
    print(f"   ⚠ Skipped (LLM module not found)\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

# Buy & Hold (reference)
print("3. Buy & Hold (Reference Baseline):")
try:
    bh_strategy = get_baseline_strategy("buy_hold", "AAPL")

    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital

    trading_dates = sorted(df.index.get_level_values(0).unique())

    # Day 1: buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
        except:
            pass

    # Hold
    for trading_date in trading_dates:
        portfolio_value = cash
        for symbol in symbols:
            try:
                row = df.loc[(trading_date, symbol)]
                close = row["close"]
                portfolio_value += positions[symbol] * close
            except:
                pass
        daily_values.append(portfolio_value)

    calc = MetricsCalculator(risk_free_rate=0.02)
    metrics = calc.calculate_all(daily_values, trading_dates[0], trading_dates[-1])

    results["Buy & Hold"] = {
        "cr": metrics["cumulative_return"],
        "ar": metrics["annualized_return"],
        "sharpe": metrics["sharpe_ratio"],
        "mdd": metrics["max_drawdown"],
        "trades": 1,  # Buy day 1
    }

    print(f"   ✓ CR: {metrics['cumulative_return'] * 100:>6.2f}% | "
          f"AR: {metrics['annualized_return'] * 100:>6.2f}% | "
          f"Sharpe: {metrics['sharpe_ratio']:>6.2f} | "
          f"MDD: {metrics['max_drawdown'] * 100:>6.2f}% | "
          f"Trades: 1\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

# ============================================================================
# STEP 5: Results summary
# ============================================================================

print("=" * 80)
print("PHASE 1 RESULTS (COST-AGNOSTIC)")
print("=" * 80 + "\n")

print("| Approach | CR (%) | AR (%) | Sharpe | MDD (%) | Trades |")
print("|----------|--------|--------|--------|---------|--------|")

for name, result in results.items():
    print(f"| {name:20s} | {result['cr'] * 100:6.2f} | {result['ar'] * 100:6.2f} | "
          f"{result['sharpe']:6.2f} | {result['mdd'] * 100:7.2f} | {result['trades']:6d} |")

# Save results
with open("results/phase1_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/phase1_results.json")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80 + "\n")

if len(results) >= 2:
    if "Quant-Only" in results and "LLM Weak" in results:
        qo = results["Quant-Only"]
        llm = results["LLM Weak"]

        cr_delta = (llm["cr"] - qo["cr"]) * 100
        sharpe_delta = llm["sharpe"] - qo["sharpe"]
        trade_delta = llm["trades"] - qo["trades"]

        print(f"Quant-Only vs LLM Weak:")
        print(f"  CR delta: {cr_delta:+.2f}pp")
        print(f"  Sharpe delta: {sharpe_delta:+.2f}")
        print(f"  Trade delta: {trade_delta:+.0f}")

        if cr_delta > 0:
            print(f"  → LLM Weak outperformed Quant-Only by {cr_delta:.2f}pp")
        else:
            print(f"  → Quant-Only outperformed LLM Weak by {-cr_delta:.2f}pp")

print("\nKey insights:")
print("  - This is PHASE 1 (cost-agnostic, replicates paper)")
print("  - Differences show LLM quality impact isolated from costs")
print("  - Phase 2 will add realistic costs and cost-aware logic\n")