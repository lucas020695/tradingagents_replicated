"""
Quick smoke test for core types and imports.

This script verifies that:
1. All dataclasses can be instantiated
2. Imports resolve correctly
3. Type hints are valid
4. No syntax errors in the core modules

Run this FIRST after creating types.py, agents_base.py, and backtest.py
to catch any issues before implementing concrete approaches.

Usage:
    python -m pytest src/core/test_smoke.py -v
    or
    python src/core/test_smoke.py
"""

import sys
from datetime import date
from typing import Dict, Any

# Import all core types
try:
    from src.core.types import (
        MarketContext,
        AnalystReport,
        ResearchReport,
        Decision,
        AgentState,
        BacktestResult,
    )
    print("✓ All types imported successfully")
except ImportError as e:
    print(f"✗ Failed to import types: {e}")
    sys.exit(1)

# Import agent base classes
try:
    from src.core.agents_base import (
        AnalystAgent,
        FundamentalAnalyst,
        TechnicalAnalyst,
        NewsAnalyst,
        SentimentAnalyst,
        ResearcherAgent,
        BullishResearcher,
        BearishResearcher,
        TraderAgent,
        RiskManagerAgent,
        FundManagerAgent,
        OrchestrationEngine,
    )
    print("✓ All agent base classes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import agents_base: {e}")
    sys.exit(1)

# Import backtest engine
try:
    from src.core.backtest import BacktestEngine, PortfolioState
    print("✓ Backtest engine imported successfully")
except ImportError as e:
    print(f"✗ Failed to import backtest: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("INSTANTIATION TESTS")
print("="*70 + "\n")

# Test 1: Create MarketContext
try:
    market_ctx = MarketContext(
        symbol="AAPL",
        as_of=date(2024, 1, 15),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.5,
        adj_close=151.5,
        volume=1000000.0,
        technicals={
            "RSI": 65.0,
            "MACD": 0.15,
            "SMA_20": 150.2,
            "SMA_50": 149.8,
        },
        fundamentals={
            "PE_ratio": 28.5,
            "PB_ratio": 45.2,
            "ROE": 0.82,
        },
        sentiment_scores={
            "news_sentiment": 0.2,
            "social_sentiment": -0.1,
        },
        current_position=100.0,
        cash=50000.0,
        portfolio_value=150000.0,
        risk_limits={
            "max_position_pct": 0.2,
            "max_leverage": 1.5,
        },
    )
    assert market_ctx.symbol == "AAPL"
    assert market_ctx.close == 151.5
    assert market_ctx.current_position == 100.0
    print("✓ MarketContext instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create MarketContext: {e}")
    sys.exit(1)

# Test 2: Create AnalystReport
try:
    analyst_report = AnalystReport(
        symbol="AAPL",
        as_of=date(2024, 1, 15),
        analyst_type="technical",
        signals={
            "momentum_score": 0.7,
            "trend_score": 0.8,
            "overbought_flag": 0.0,
        },
        risk_flags=["none"],
        summary="Technical indicators show bullish momentum with MACD > 0 and RSI in neutral zone.",
        confidence=0.85,
        extra={"feature_importance": {"RSI": 0.4, "MACD": 0.3}},
    )
    assert analyst_report.analyst_type == "technical"
    assert analyst_report.confidence == 0.85
    print("✓ AnalystReport instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create AnalystReport: {e}")
    sys.exit(1)

# Test 3: Create ResearchReport
try:
    research_report = ResearchReport(
        symbol="AAPL",
        as_of=date(2024, 1, 15),
        stance_score=0.6,
        bullish_summary="Strong fundamentals + positive technicals = buy opportunity",
        bearish_summary="Valuation elevated; risk of pullback if macro turns",
        dominant_view="bullish",
        confidence=0.75,
        supporting_analysts={
            "fundamental_contribution": 0.8,
            "technical_contribution": 0.9,
            "sentiment_contribution": 0.3,
        },
    )
    assert research_report.dominant_view == "bullish"
    assert research_report.stance_score == 0.6
    print("✓ ResearchReport instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create ResearchReport: {e}")
    sys.exit(1)

# Test 4: Create Decision
try:
    decision = Decision(
        symbol="AAPL",
        as_of=date(2024, 1, 15),
        action="buy",
        target_position=150.0,
        trade_size=50.0,
        reference_price=151.5,
        rationale="Bullish research + oversold RSI = entry; 2% stop below entry",
        confidence=0.75,
        risk_overrides={
            "max_loss_pct": 0.02,
            "take_profit_pct": 0.08,
            "stop_loss_price": 148.47,
        },
        metadata={
            "approach": "llm_perplexity",
            "policy_version": "v1",
            "research_stance": "bullish",
        },
    )
    assert decision.action == "buy"
    assert decision.trade_size == 50.0
    print("✓ Decision instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create Decision: {e}")
    sys.exit(1)

# Test 5: Create AgentState
try:
    agent_state = AgentState(
        symbol="AAPL",
        as_of=date(2024, 1, 15),
        market_context=market_ctx,
        analyst_reports=[analyst_report],
        research_report=research_report,
        final_decision=decision,
        logs=[
            "Fundamental analyst bullish (+0.7)",
            "Technical analyst bullish (+0.8)",
            "Research concluded: bullish stance",
        ],
    )
    assert agent_state.symbol == "AAPL"
    assert len(agent_state.analyst_reports) == 1
    assert agent_state.final_decision is not None
    print("✓ AgentState instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create AgentState: {e}")
    sys.exit(1)

# Test 6: Create BacktestResult
try:
    backtest_result = BacktestResult(
        symbol="AAPL",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 29),
        cumulative_return=0.15,
        annualized_return=0.18,
        sharpe_ratio=1.85,
        max_drawdown=-0.08,
        num_trades=25,
        num_winning_trades=18,
        win_rate=0.72,
        initial_capital=100000.0,
        final_value=115000.0,
        approach="llm_perplexity",
        strategy_name="TradingAgents_MultiAgent",
        metadata={
            "transaction_cost_pct": 0.001,
            "allow_short": False,
        },
    )
    assert backtest_result.cumulative_return == 0.15
    assert backtest_result.sharpe_ratio == 1.85
    print("✓ BacktestResult instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create BacktestResult: {e}")
    sys.exit(1)

# Test 7: Create PortfolioState
try:
    portfolio = PortfolioState(
        cash=50000.0,
        positions={"AAPL": 100.0, "NVDA": 50.0},
        entry_prices={"AAPL": 150.0, "NVDA": 500.0},
        daily_values=[
            (date(2024, 1, 1), 100000.0),
            (date(2024, 1, 2), 101000.0),
            (date(2024, 1, 3), 100500.0),
        ],
    )
    assert portfolio.cash == 50000.0
    assert "AAPL" in portfolio.positions
    print("✓ PortfolioState instantiated and validated")
except Exception as e:
    print(f"✗ Failed to create PortfolioState: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ABSTRACT BASE CLASS TESTS")
print("="*70 + "\n")

# Test 8: Verify abstract methods exist
try:
    assert hasattr(AnalystAgent, "analyze")
    assert hasattr(ResearcherAgent, "research")
    assert hasattr(TraderAgent, "decide")
    assert hasattr(RiskManagerAgent, "evaluate")
    assert hasattr(FundManagerAgent, "approve")
    assert hasattr(OrchestrationEngine, "run_day")
    print("✓ All abstract base classes have required abstract methods")
except AssertionError as e:
    print(f"✗ Missing abstract method: {e}")
    sys.exit(1)

# Test 9: Verify cannot instantiate abstract classes directly
try:
    # This should fail
    try:
        agent = AnalystAgent("test")
        print("✗ AnalystAgent should not be instantiable (is abstract)")
        sys.exit(1)
    except TypeError:
        print("✓ AnalystAgent correctly prevents direct instantiation (abstract)")

    try:
        researcher = ResearcherAgent("bullish")
        print("✗ ResearcherAgent should not be instantiable (is abstract)")
        sys.exit(1)
    except TypeError:
        print("✓ ResearcherAgent correctly prevents direct instantiation (abstract)")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL SMOKE TESTS PASSED ✓")
print("="*70)
print("\nYour core types and agents are ready for implementation!")
print("Next: implement concrete agents in src/approaches/")