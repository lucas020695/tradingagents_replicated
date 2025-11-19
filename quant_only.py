"""
Quant-only approach: Multi-agent trading system without LLM.

This approach replaces LLM reasoning with pure quantitative logic and rule-based scoring.
All agents use the same interface as LLM-based approaches, but implement decisions through:
- Technical indicator thresholds
- Fundamental ratio comparisons
- Weighted scoring systems
- Simple decision rules

Architecture:
- AnalystAgents: Convert technicals/fundamentals into numerical scores
- ResearcherAgents: Aggregate analyst scores into bull/bear stance
- TraderAgent: Generate buy/sell/hold based on stance + portfolio state
- RiskManagerAgents: Apply position sizing and risk limits
- FundManagerAgent: Final approval and global constraints

This enables direct comparison with LLM approaches on identical data/metrics.

Reference: TradingAgents paper - we replicate the multi-agent flow without LLMs.
"""

from typing import Dict, List, Any, Optional
from datetime import date

from src.core.types import (
    MarketContext,
    AnalystReport,
    ResearchReport,
    Decision,
    AgentState,
)
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


# ============================================================================
# ANALYST IMPLEMENTATIONS (Quant-Only)
# ============================================================================

class QuantFundamentalAnalyst(FundamentalAnalyst):
    """
    Analyzes fundamentals using pure quantitative rules.

    Scoring logic:
    - value_score: based on P/E ratio vs historical average
    - quality_score: based on ROE, debt/equity, margins
    - growth_score: based on revenue growth rate
    """

    def analyze(self, context: MarketContext) -> AnalystReport:
        """
        Analyze fundamentals quantitatively.

        Returns analyst report with:
        - signals: {"value_score": 0-1, "quality_score": 0-1, "growth_score": 0-1}
        - risk_flags: ["high_valuation", "weak_roe", etc.]
        - summary: template-based narrative
        """
        fundamentals = context.fundamentals or {}

        # Extract metrics
        pe_ratio = fundamentals.get("pe_ratio", 0)
        pb_ratio = fundamentals.get("pb_ratio", 0)
        roe = fundamentals.get("roe", 0)
        debt_equity = fundamentals.get("debt_to_equity", 0)
        revenue_growth = fundamentals.get("revenue_growth", 0)

        # Value scoring: lower P/E is more attractive
        # Assume P/E of 20 is neutral (50% score), scale from 10 (100%) to 40 (0%)
        if pe_ratio > 0:
            value_score = max(0, min(1, 1 - (pe_ratio - 10) / 30))
        else:
            value_score = 0.5  # Unknown, neutral

        # Quality scoring: higher ROE and lower debt is better
        quality_score = 0.5
        if roe > 0:
            quality_score += min(0.3, roe / 0.3)  # Max +0.3 for ROE
        if debt_equity > 0:
            quality_score -= min(0.3, debt_equity / 2)  # Max -0.3 for debt
        quality_score = max(0, min(1, quality_score))

        # Growth scoring
        if revenue_growth > 0:
            growth_score = min(1, revenue_growth * 5)  # 20% growth = 100%
        else:
            growth_score = 0.5

        signals = {
            "value_score": float(value_score),
            "quality_score": float(quality_score),
            "growth_score": float(growth_score),
        }

        # Risk flags
        risk_flags = []
        if pe_ratio > 30:
            risk_flags.append("high_valuation")
        if roe < 0.05:
            risk_flags.append("low_roe")
        if debt_equity > 1.5:
            risk_flags.append("high_debt")

        # Template-based summary
        summary = f"Fundamental view: Value={value_score:.2f}, Quality={quality_score:.2f}, Growth={growth_score:.2f}. "
        if value_score > 0.7:
            summary += "Stock appears undervalued. "
        if quality_score > 0.7:
            summary += "Strong business quality indicators. "
        if growth_score > 0.6:
            summary += "Solid revenue growth. "

        confidence = 0.7 if len(fundamentals) > 3 else 0.4

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="fundamental",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
            extra={"fundamentals_used": len(fundamentals)},
        )


class QuantTechnicalAnalyst(TechnicalAnalyst):
    """
    Analyzes technicals using indicator thresholds and trend scoring.

    Signals:
    - momentum_score: from MACD, RSI, momentum indicators
    - trend_score: from moving averages
    - overbought_flag: from RSI extremes
    """

    def analyze(self, context: MarketContext) -> AnalystReport:
        """
        Analyze technicals quantitatively.
        """
        technicals = context.technicals or {}

        # RSI analysis (0-100 scale, 50 = neutral)
        rsi = technicals.get("RSI_14", 50)
        momentum_score = rsi / 100  # Normalize to 0-1

        overbought_flag = 0.0
        if rsi > 70:
            overbought_flag = 1.0  # Overbought
        elif rsi < 30:
            overbought_flag = -1.0  # Oversold

        # Trend analysis: SMA crossover
        sma20 = technicals.get("SMA_20", context.close)
        sma50 = technicals.get("SMA_50", context.close)
        current_price = context.close

        if current_price > sma20 > sma50:
            trend_score = 0.9  # Strong uptrend
        elif current_price > sma20:
            trend_score = 0.6  # Weak uptrend
        elif current_price < sma20 < sma50:
            trend_score = 0.1  # Strong downtrend
        else:
            trend_score = 0.4  # Weak downtrend

        signals = {
            "momentum_score": float(momentum_score),
            "trend_score": float(trend_score),
            "overbought_flag": float(overbought_flag),
        }

        risk_flags = []
        if rsi > 75:
            risk_flags.append("extremely_overbought")
        elif rsi < 25:
            risk_flags.append("extremely_oversold")

        # Template summary
        summary = f"Technical view: Momentum={momentum_score:.2f}, Trend={trend_score:.2f}. "
        if rsi > 70:
            summary += "RSI in overbought zone. "
        elif rsi < 30:
            summary += "RSI in oversold zone. "
        if trend_score > 0.7:
            summary += "Strong uptrend. "
        elif trend_score < 0.3:
            summary += "Strong downtrend. "

        confidence = 0.8 if len(technicals) > 3 else 0.5

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="technical",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
            extra={"indicators_available": len(technicals)},
        )


class QuantNewsAnalyst(NewsAnalyst):
    """
    Simplified news analyst (no LLM, uses aggregated sentiment).
    In quant-only, this returns neutral scores since we lack text data.
    """

    def analyze(self, context: MarketContext) -> AnalystReport:
        """
        Placeholder: would use news sentiment if data available.
        """
        signals = {
            "positive_events": 0,
            "negative_events": 0,
            "net_sentiment": 0.0,
        }
        risk_flags = []
        summary = "News analysis: No sentiment data available in quant-only approach."
        confidence = 0.3  # Low confidence without text

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="news",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
        )


class QuantSentimentAnalyst(SentimentAnalyst):
    """
    Placeholder sentiment analyst (no LLM, no social data access).
    Returns neutral scores.
    """

    def analyze(self, context: MarketContext) -> AnalystReport:
        """
        Placeholder: would use social sentiment if data available.
        """
        signals = {
            "social_sentiment": 0.0,
            "insider_score": 0.5,
            "options_flow": 0.0,
        }
        risk_flags = []
        summary = "Sentiment analysis: Limited data in quant-only approach."
        confidence = 0.2

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="sentiment",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
        )


# ============================================================================
# RESEARCHER IMPLEMENTATIONS (Quant-Only)
# ============================================================================

class QuantBullishResearcher(BullishResearcher):
    """
    Bullish researcher: aggregates analyst scores for bull case.
    """

    def research(
            self,
            context: MarketContext,
            analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        """
        Aggregate analyst signals into bullish stance.
        """
        # Extract signals from all analysts
        scores = {}
        for report in analyst_reports:
            scores[report.analyst_type] = report.signals

        # Compute weighted bull score
        bull_score = 0.0
        weights = {
            "fundamental": 0.3,
            "technical": 0.4,
            "news": 0.15,
            "sentiment": 0.15,
        }

        for analyst_type, weight in weights.items():
            if analyst_type in scores:
                analyst_signals = scores[analyst_type]
                # Average of available signals
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bull_score += weight * avg_signal

        # Bullish researcher emphasizes positive aspects
        bullish_summary = f"Bull case: Strong fundamentals ({scores.get('fundamental', {}).get('value_score', 0.5):.2f}), "
        bullish_summary += f"positive technicals ({scores.get('technical', {}).get('trend_score', 0.5):.2f}). "
        bullish_summary += "Recommend accumulating on dips."

        # Bearish case (counterpoint)
        bearish_summary = f"Bear case: Some valuation concerns, technicals show mixed signals. "
        bearish_summary += "Consider waiting for clearer entry."

        # Dominant view
        dominant_view = "bullish" if bull_score > 0.55 else "neutral"
        confidence = 0.7 if len(analyst_reports) > 2 else 0.5

        return ResearchReport(
            symbol=context.symbol,
            as_of=context.as_of,
            stance_score=2 * bull_score - 1,  # Convert 0-1 to -1 to +1
            bullish_summary=bullish_summary,
            bearish_summary=bearish_summary,
            dominant_view=dominant_view,
            confidence=confidence,
            supporting_analysts={
                f"analyst_{i}": report.confidence
                for i, report in enumerate(analyst_reports)
            },
        )


class QuantBearishResearcher(BearishResearcher):
    """
    Bearish researcher: focuses on downside risks.
    """

    def research(
            self,
            context: MarketContext,
            analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        """
        Aggregate analyst signals into bearish stance.
        """
        scores = {}
        for report in analyst_reports:
            scores[report.analyst_type] = report.signals

        # Compute weighted bear score (inverse of bull)
        bear_score = 0.0
        weights = {
            "fundamental": 0.3,
            "technical": 0.4,
            "news": 0.15,
            "sentiment": 0.15,
        }

        for analyst_type, weight in weights.items():
            if analyst_type in scores:
                analyst_signals = scores[analyst_type]
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bear_score += weight * (1 - avg_signal)  # Invert for bearish

        # Bearish narrative
        bearish_summary = f"Bear case: Valuation concerns ({1 - scores.get('fundamental', {}).get('value_score', 0.5):.2f}), "
        bearish_summary += f"weakening technicals ({1 - scores.get('technical', {}).get('trend_score', 0.5):.2f}). "
        bearish_summary += "Risk of pullback; recommend caution."

        bullish_summary = "Bull case: Longer-term fundamentals remain intact despite near-term weakness."

        dominant_view = "bearish" if bear_score > 0.55 else "neutral"
        confidence = 0.7

        return ResearchReport(
            symbol=context.symbol,
            as_of=context.as_of,
            stance_score=1 - 2 * bear_score,  # Convert to -1 to +1
            bullish_summary=bullish_summary,
            bearish_summary=bearish_summary,
            dominant_view=dominant_view,
            confidence=confidence,
            supporting_analysts={
                f"analyst_{i}": report.confidence
                for i, report in enumerate(analyst_reports)
            },
        )


# ============================================================================
# TRADER IMPLEMENTATION (Quant-Only)
# ============================================================================

class QuantTrader(TraderAgent):
    """
    Trader: converts research into buy/sell/hold decisions with position sizing.
    """

    def decide(
            self,
            context: MarketContext,
            research_report: ResearchReport,
            analyst_reports: List[AnalystReport],
    ) -> Decision:
        """
        Generate trading decision from research + context.
        """
        stance = research_report.stance_score  # -1 to +1
        confidence = research_report.confidence

        # Position sizing based on stance and confidence
        position_scale = self.policy.get("position_scale", 0.2)  # 20% of capital default
        max_position = self.policy.get("max_position_pct", 0.25)  # 25% max

        current_value = context.current_position * context.close
        available_capital = context.cash

        # Determine action and size
        if stance > 0.3:  # Bullish
            action = "buy"
            # Size proportional to stance and confidence
            target_size = (stance * confidence) * (available_capital / context.close)
            target_size = min(target_size, context.portfolio_value * max_position / context.close)
        elif stance < -0.3:  # Bearish
            action = "sell"
            # Reduce or exit position
            target_size = context.current_position * (1 - abs(stance) * confidence)
        else:  # Neutral
            action = "hold"
            target_size = context.current_position

        trade_size = target_size - context.current_position

        rationale = (
            f"Stance: {research_report.dominant_view.upper()}. "
            f"Analyst consensus supports {research_report.dominant_view} view. "
            f"Confidence: {confidence:.2%}. "
            f"Execution: {'BUY' if action == 'buy' else 'SELL' if action == 'sell' else 'HOLD'} "
            f"{abs(trade_size):.0f} shares @ {context.close:.2f}."
        )

        decision = Decision(
            symbol=context.symbol,
            as_of=context.as_of,
            action=action,
            target_position=target_size,
            trade_size=trade_size,
            reference_price=context.close,
            rationale=rationale,
            confidence=confidence,
            risk_overrides={
                "max_loss_pct": 0.03,
                "take_profit_pct": 0.08,
            },
            metadata={
                "approach": "quant_only",
                "stance": research_report.dominant_view,
            },
        )

        return decision


# ============================================================================
# RISK MANAGER & FUND MANAGER (Quant-Only)
# ============================================================================

class QuantRiskManager(RiskManagerAgent):
    """
    Risk manager: applies position limits and risk controls.
    """

    def evaluate(
            self,
            context: MarketContext,
            proposed_decision: Decision,
    ) -> Decision:
        """
        Modify decision to respect risk limits.
        """
        # Apply risk limits based on profile
        limits = self._get_limits_for_profile()

        # Cap position size
        max_position = context.portfolio_value * limits["max_position_pct"]
        proposed_decision.target_position = min(
            proposed_decision.target_position,
            max_position / context.close,
        )

        # Recalculate trade size
        proposed_decision.trade_size = (
                proposed_decision.target_position - context.current_position
        )

        # Add risk overrides
        entry_price = context.close
        proposed_decision.risk_overrides["stop_loss_price"] = (
                entry_price * (1 - limits["max_loss_pct"])
        )
        proposed_decision.risk_overrides["take_profit_price"] = (
                entry_price * (1 + limits["take_profit_pct"])
        )

        return proposed_decision

    def _get_limits_for_profile(self) -> Dict[str, float]:
        """Get risk limits based on risk profile."""
        limits = {
            "conservative": {
                "max_position_pct": 0.10,
                "max_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
            "neutral": {
                "max_position_pct": 0.20,
                "max_loss_pct": 0.03,
                "take_profit_pct": 0.08,
            },
            "aggressive": {
                "max_position_pct": 0.30,
                "max_loss_pct": 0.05,
                "take_profit_pct": 0.12,
            },
        }
        return limits.get(self.risk_profile, limits["neutral"])


class QuantFundManager(FundManagerAgent):
    """
    Fund manager: final approval with portfolio-level constraints.
    """

    def approve(
            self,
            context: MarketContext,
            risk_modified_decision: Decision,
            portfolio_state: Dict[str, Any],
    ) -> Decision:
        """
        Final approval (mostly pass-through in quant-only).
        """
        # Could add global portfolio constraints here
        return risk_modified_decision


# ============================================================================
# ORCHESTRATOR (Quant-Only)
# ============================================================================

class QuantOrchestrator(OrchestrationEngine):
    """
    Orchestrates the quant-only multi-agent pipeline for one day.
    """

    def run_day(
            self,
            symbol: str,
            as_of: date,
            context: MarketContext,
            portfolio_state: Dict[str, Any],
    ) -> AgentState:
        """
        Execute one day's decision cycle.
        """
        # Step 1: Run all analysts
        analyst_reports = []
        for analyst_name, analyst in self.analysts.items():
            report = analyst.analyze(context)
            analyst_reports.append(report)

        # Step 2: Run researchers
        research_report = None
        # Use bullish researcher (bearish could also be run, but we simplify here)
        bullish_res = self.researchers["bullish"]
        research_report = bullish_res.research(context, analyst_reports)

        # Step 3: Trader generates decision
        decision = self.trader.decide(context, research_report, analyst_reports)

        # Step 4: Risk manager reviews
        for rm in self.risk_managers.values():
            decision = rm.evaluate(context, decision)

        # Step 5: Fund manager approves
        decision = self.fund_manager.approve(context, decision, portfolio_state)

        # Build agent state for audit trail
        agent_state = AgentState(
            symbol=symbol,
            as_of=as_of,
            market_context=context,
            analyst_reports=analyst_reports,
            research_report=research_report,
            final_decision=decision,
            logs=[
                     f"Analyst: {r.analyst_type} → {list(r.signals.values())}"
                     for r in analyst_reports
                 ] + [
                     f"Research: {research_report.dominant_view.upper()} (confidence={research_report.confidence:.2f})",
                     f"Trader: {decision.action.upper()} {abs(decision.trade_size):.0f} shares",
                 ],
        )

        return agent_state


# ============================================================================
# FACTORY: Quick instantiation
# ============================================================================

def create_quant_only_orchestrator() -> QuantOrchestrator:
    """
    Factory function to create a fully configured quant-only orchestrator.

    Usage:
        orchestrator = create_quant_only_orchestrator()
        agent_state = orchestrator.run_day(symbol, date, context, portfolio)
    """
    # Create all analysts
    analysts = {
        "fundamental": QuantFundamentalAnalyst(),
        "technical": QuantTechnicalAnalyst(),
        "news": QuantNewsAnalyst(),
        "sentiment": QuantSentimentAnalyst(),
    }

    # Create researchers
    researchers = {
        "bullish": QuantBullishResearcher(),
        "bearish": QuantBearishResearcher(),
    }

    # Create trader
    trader = QuantTrader(
        policy={
            "position_scale": 0.2,
            "max_position_pct": 0.25,
        }
    )

    # Create risk managers (3 profiles)
    risk_managers = {
        "conservative": QuantRiskManager("conservative"),
        "neutral": QuantRiskManager("neutral"),
        "aggressive": QuantRiskManager("aggressive"),
    }

    # Create fund manager
    fund_manager = QuantFundManager()

    # Assemble orchestrator
    orchestrator = QuantOrchestrator(
        analysts=analysts,
        researchers=researchers,
        trader=trader,
        risk_managers=risk_managers,
        fund_manager=fund_manager,
    )

    return orchestrator