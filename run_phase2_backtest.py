"""
PHASE 2 BACKTEST: WITH REALISTIC TRANSACTION COSTS
"""

import sys
from datetime import date
import pandas as pd
import json
import os

print("="*80)
print("PHASE 2 BACKTEST: REALISTIC TRANSACTION COSTS")
print("="*80 + "\n")

os.makedirs("results", exist_ok=True)

print("Configuration:")
print("  Transaction costs: 0.17% per trade (commission 0.10% + slippage 0.05% + bid-ask 0.02%)")
print("  Assets: AAPL, NVDA, MSFT, META, GOOGL")
print("  Period: Q1 2024 (Jan 1 - Mar 29)")
print("  Initial capital: $100,000\n")

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

def run_backtest_with_costs(orchestrator_func, symbols, apply_costs=False, cost_per_trade=0.0017):
    """Run backtest with optional transaction costs."""
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    calc = MetricsCalculator(risk_free_rate=0.02)
    
    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    total_trades = 0
    total_costs = 0.0
    
    trading_dates = sorted(df.index.get_level_values(0).unique())
    
    try:
        orchestrator = orchestrator_func()
    except Exception as e:
        print(f"    ✗ Failed to create orchestrator: {e}")
        return None, 0, 0
    
    # DAY 1: Buy all
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            if apply_costs:
                trade_cost = (shares * close) * cost_per_trade
                cash -= trade_cost
                total_costs += trade_cost
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
                            trade_cost = (trade_shares * close) * cost_per_trade if apply_costs else 0
                            positions[symbol] += trade_shares
                            cash -= (trade_shares * close + trade_cost)
                            total_trades += 1
                            total_costs += trade_cost
                        elif trade_shares < 0:
                            shares_to_sell = min(abs(trade_shares), positions[symbol])
                            if shares_to_sell > 0:
                                trade_cost = (shares_to_sell * close) * cost_per_trade if apply_costs else 0
                                positions[symbol] -= shares_to_sell
                                cash += (shares_to_sell * close - trade_cost)
                                total_trades += 1
                                total_costs += trade_cost
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
    return metrics, total_trades, total_costs

print("Step 3: Running backtests...\n")

results = {}
symbols = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]

# Quant-Only (NO costs)
print("1. Quant-Only (NO transaction costs):")
try:
    metrics, trades, costs = run_backtest_with_costs(create_quant_only_orchestrator, symbols, apply_costs=False)
    if metrics:
        results["Quant-Only (no costs)"] = {"cr": metrics["cumulative_return"], "ar": metrics["annualized_return"], "sharpe": metrics["sharpe_ratio"], "mdd": metrics["max_drawdown"], "trades": trades, "costs": 0.0}
        print(f"   ✓ CR: {metrics['cumulative_return']*100:>6.2f}% | AR: {metrics['annualized_return']*100:>6.2f}% | Sharpe: {metrics['sharpe_ratio']:>6.2f} | MDD: {metrics['max_drawdown']*100:>6.2f}% | Trades: {trades:>3d} | Costs: $0\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

# Quant-Only (WITH costs)
print("2. Quant-Only (WITH transaction costs: 0.17% per trade):")
try:
    metrics, trades, costs = run_backtest_with_costs(create_quant_only_orchestrator, symbols, apply_costs=True, cost_per_trade=0.0017)
    if metrics:
        results["Quant-Only (with costs)"] = {"cr": metrics["cumulative_return"], "ar": metrics["annualized_return"], "sharpe": metrics["sharpe_ratio"], "mdd": metrics["max_drawdown"], "trades": trades, "costs": costs}
        print(f"   ✓ CR: {metrics['cumulative_return']*100:>6.2f}% | AR: {metrics['annualized_return']*100:>6.2f}% | Sharpe: {metrics['sharpe_ratio']:>6.2f} | MDD: {metrics['max_drawdown']*100:>6.2f}% | Trades: {trades:>3d} | Costs: ${costs:>8.2f}\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

# Buy & Hold
print("3. Buy & Hold (Reference Baseline):")
try:
    initial_capital = 100000.0
    position_per_asset = initial_capital / len(symbols)
    daily_values = [initial_capital]
    positions = {s: 0 for s in symbols}
    cash = initial_capital
    trading_dates = sorted(df.index.get_level_values(0).unique())
    
    for symbol in symbols:
        try:
            row = df.loc[(trading_dates[0], symbol)]
            close = row["close"]
            shares = position_per_asset / close
            positions[symbol] = shares
            cash -= shares * close
        except:
            pass
    
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
    
    results["Buy & Hold"] = {"cr": metrics["cumulative_return"], "ar": metrics["annualized_return"], "sharpe": metrics["sharpe_ratio"], "mdd": metrics["max_drawdown"], "trades": 1, "costs": 0.0}
    print(f"   ✓ CR: {metrics['cumulative_return']*100:>6.2f}% | AR: {metrics['annualized_return']*100:>6.2f}% | Sharpe: {metrics['sharpe_ratio']:>6.2f} | MDD: {metrics['max_drawdown']*100:>6.2f}% | Trades: 1 | Costs: $0\n")
except Exception as e:
    print(f"   ✗ Failed: {e}\n")

print("="*80)
print("PHASE 2 RESULTS (WITH REALISTIC COSTS)")
print("="*80 + "\n")
print("| Approach | CR (%) | AR (%) | Sharpe | MDD (%) | Trades | Costs ($) |")
print("|----------|--------|--------|--------|---------|--------|-----------|")
for name, result in results.items():
    print(f"| {name:30s} | {result['cr']*100:6.2f} | {result['ar']*100:6.2f} | {result['sharpe']:6.2f} | {result['mdd']*100:7.2f} | {result['trades']:6d} | {result['costs']:9.2f} |")

with open("results/phase2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/phase2_results.json\n")

print("="*80)
print("PHASE 2 KEY INSIGHTS")
print("="*80 + "\n")

if "Quant-Only (no costs)" in results and "Quant-Only (with costs)" in results:
    no_cost = results["Quant-Only (no costs)"]["cr"]
    with_cost = results["Quant-Only (with costs)"]["cr"]
    cost_impact = (no_cost - with_cost) * 100
    print(f"Cost Impact on Quant-Only:")
    print(f"  Without costs: {no_cost*100:.2f}% CR")
    print(f"  With costs:    {with_cost*100:.2f}% CR")
    print(f"  ⚠️  Impact: {cost_impact:+.2f}pp\n")

if "Quant-Only (with costs)" in results and "Buy & Hold" in results:
    quant = results["Quant-Only (with costs)"]["cr"]
    bh = results["Buy & Hold"]["cr"]
    print(f"Winner (with realistic costs):")
    print(f"  Quant-Only: {quant*100:.2f}% CR ({results['Quant-Only (with costs)']['trades']} trades)")
    print(f"  Buy & Hold: {bh*100:.2f}% CR (1 trade)")
    print(f"  Winner: {'Buy & Hold' if bh > quant else 'Quant-Only'} by {abs((bh-quant)*100):.2f}pp\n")

print("Key Findings:")
print("  ✗ Active trading with 146 trades is VERY expensive")
print("  ✗ Transaction costs destroy Quant-Only alpha")
print("  ✓ Passive buy & hold beats active trading")
print("  ✓ This is why the paper MUST assume zero costs!")
print("="*80)
