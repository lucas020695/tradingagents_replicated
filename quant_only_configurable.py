"""
Refactored Quant-Only approach: Config-driven, suitable for ablation study.

FIXED: Uses concrete QuantRiskManager and QuantFundManager from original module.
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
    FundamentalAnalyst,
    TechnicalAnalyst,
    NewsAnalyst,
    SentimentAnalyst,
    BullishResearcher,
    BearishResearcher,
    TraderAgent,
    RiskManagerAgent,
    FundManagerAgent,
    OrchestrationEngine,
)

import numpy as np


# ============================================================================
# CONFIGURABLE ANALYSTS
# ============================================================================

class ConfigurableFundamentalAnalyst(FundamentalAnalyst):
    """Fundamental analyst that uses real fundamentals if provided."""

    def analyze(self, context: MarketContext) -> AnalystReport:
        fundamentals = context.fundamentals or {}

        pe_ratio = fundamentals.get("pe_ratio", 20)
        pb_ratio = fundamentals.get("pb_ratio", 3)
        roe = fundamentals.get("roe", 0.15)
        debt_equity = fundamentals.get("debt_to_equity", 1.0)
        revenue_growth = fundamentals.get("revenue_growth", 0.05)

        value_score = max(0, min(1, 1 - (pe_ratio - 10) / 30))

        quality_score = 0.5
        if roe > 0:
            quality_score += min(0.3, roe / 0.3)
        if debt_equity > 0:
            quality_score -= min(0.3, debt_equity / 2)
        quality_score = max(0, min(1, quality_score))

        if revenue_growth > 0:
            growth_score = min(1, revenue_growth * 5)
        else:
            growth_score = 0.5

        signals = {
            "value_score": float(value_score),
            "quality_score": float(quality_score),
            "growth_score": float(growth_score),
        }

        risk_flags = []
        if pe_ratio > 30:
            risk_flags.append("high_valuation")
        if roe < 0.05:
            risk_flags.append("low_roe")
        if debt_equity > 1.5:
            risk_flags.append("high_debt")

        summary = f"Value={value_score:.2f}, Quality={quality_score:.2f}, Growth={growth_score:.2f}"
        confidence = 0.7 if len(fundamentals) > 2 else 0.4

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="fundamental",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
        )


class ConfigurableTechnicalAnalyst(TechnicalAnalyst):
    """Technical analyst."""

    def analyze(self, context: MarketContext) -> AnalystReport:
        technicals = context.technicals or {}

        rsi = technicals.get("RSI_14", 50)
        momentum_score = rsi / 100

        overbought_flag = 0.0
        if rsi > 70:
            overbought_flag = 1.0
        elif rsi < 30:
            overbought_flag = -1.0

        sma20 = technicals.get("SMA_20", context.close)
        sma50 = technicals.get("SMA_50", context.close)
        current_price = context.close

        if current_price > sma20 > sma50:
            trend_score = 0.9
        elif current_price > sma20:
            trend_score = 0.6
        elif current_price < sma20 < sma50:
            trend_score = 0.1
        else:
            trend_score = 0.4

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

        summary = f"Momentum={momentum_score:.2f}, Trend={trend_score:.2f}"
        confidence = 0.8 if len(technicals) > 3 else 0.5

        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="technical",
            signals=signals,
            risk_flags=risk_flags,
            summary=summary,
            confidence=confidence,
        )


class ConfigurableNewsAnalyst(NewsAnalyst):
    """News analyst placeholder."""

    def analyze(self, context: MarketContext) -> AnalystReport:
        signals = {"positive_events": 0, "negative_events": 0, "net_sentiment": 0.0}
        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="news",
            signals=signals,
            risk_flags=[],
            summary="News: no data",
            confidence=0.3,
        )


class ConfigurableSentimentAnalyst(SentimentAnalyst):
    """Sentiment analyst placeholder."""

    def analyze(self, context: MarketContext) -> AnalystReport:
        signals = {"social_sentiment": 0.0, "insider_score": 0.5, "options_flow": 0.0}
        return AnalystReport(
            symbol=context.symbol,
            as_of=context.as_of,
            analyst_type="sentiment",
            signals=signals,
            risk_flags=[],
            summary="Sentiment: limited data",
            confidence=0.2,
        )


# ============================================================================
# CONFIGURABLE RESEARCHERS
# ============================================================================

class ConfigurableBullishResearcher(BullishResearcher):
    """Bullish researcher with optional confidence weighting."""

    def __init__(self, weight_by_confidence: bool = False):
        super().__init__()
        self.weight_by_confidence = weight_by_confidence

    def research(
        self,
        context: MarketContext,
        analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        scores = {}
        confidences = {}
        for report in analyst_reports:
            scores[report.analyst_type] = report.signals
            confidences[report.analyst_type] = report.confidence

        bull_score = 0.0
        if self.weight_by_confidence:
            total_weight = 0.0
            for analyst_type in scores:
                weight = confidences.get(analyst_type, 0.5)
                analyst_signals = scores[analyst_type]
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bull_score += weight * avg_signal
                total_weight += weight
            bull_score = bull_score / total_weight if total_weight > 0 else 0.5
        else:
            for analyst_type in scores:
                analyst_signals = scores[analyst_type]
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bull_score += avg_signal * 0.25

        bullish_summary = f"Bull case: consensus={bull_score:.2f}"
        bearish_summary = f"Bear case: opposing view"
        dominant_view = "bullish" if bull_score > 0.55 else "neutral"

        return ResearchReport(
            symbol=context.symbol,
            as_of=context.as_of,
            stance_score=2 * bull_score - 1,
            bullish_summary=bullish_summary,
            bearish_summary=bearish_summary,
            dominant_view=dominant_view,
            confidence=0.7,
            supporting_analysts={f"analyst_{i}": r.confidence for i, r in enumerate(analyst_reports)},
        )


class ConfigurableBearishResearcher(BearishResearcher):
    """Bearish researcher with optional confidence weighting."""

    def __init__(self, weight_by_confidence: bool = False):
        super().__init__()
        self.weight_by_confidence = weight_by_confidence

    def research(
        self,
        context: MarketContext,
        analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        scores = {}
        confidences = {}
        for report in analyst_reports:
            scores[report.analyst_type] = report.signals
            confidences[report.analyst_type] = report.confidence

        bear_score = 0.0
        if self.weight_by_confidence:
            total_weight = 0.0
            for analyst_type in scores:
                weight = confidences.get(analyst_type, 0.5)
                analyst_signals = scores[analyst_type]
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bear_score += weight * (1 - avg_signal)
                total_weight += weight
            bear_score = bear_score / total_weight if total_weight > 0 else 0.5
        else:
            for analyst_type in scores:
                analyst_signals = scores[analyst_type]
                avg_signal = sum(analyst_signals.values()) / len(analyst_signals) if analyst_signals else 0.5
                bear_score += (1 - avg_signal) * 0.25

        bearish_summary = f"Bear case: consensus={bear_score:.2f}"
        bullish_summary = f"Bull case: opposing view"
        dominant_view = "bearish" if bear_score > 0.55 else "neutral"

        return ResearchReport(
            symbol=context.symbol,
            as_of=context.as_of,
            stance_score=1 - 2 * bear_score,
            bullish_summary=bullish_summary,
            bearish_summary=bearish_summary,
            dominant_view=dominant_view,
            confidence=0.7,
            supporting_analysts={f"analyst_{i}": r.confidence for i, r in enumerate(analyst_reports)},
        )


# ============================================================================
# CONFIGURABLE TRADER
# ============================================================================

class ConfigurableTrader(TraderAgent):
    """Trader with hysteresis support and configurable position sizing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.position_scale = config.get("position_scale", 0.2)
        self.use_hysteresis = config.get("use_hysteresis", False)
        self.hysteresis_threshold = config.get("hysteresis_threshold", 0.2)
        self.prev_stances = {}

    def decide(
        self,
        context: MarketContext,
        research_report: ResearchReport,
        analyst_reports: List[AnalystReport],
    ) -> Decision:
        stance = research_report.stance_score
        confidence = research_report.confidence

        # Apply hysteresis if enabled
        if self.use_hysteresis:
            prev_stance = self.prev_stances.get(context.symbol, stance)
            stance_delta = abs(stance - prev_stance)

            if stance_delta < self.hysteresis_threshold:
                decision = Decision(
                    symbol=context.symbol,
                    as_of=context.as_of,
                    action="hold",
                    target_position=context.current_position,
                    trade_size=0,
                    reference_price=context.close,
                    rationale=f"Hysteresis: delta={stance_delta:.2f} < {self.hysteresis_threshold}",
                    confidence=confidence,
                )
                return decision

            self.prev_stances[context.symbol] = stance

        # Position sizing
        max_position_pct = 0.25
        target_size = (stance * confidence) * (context.portfolio_value * self.position_scale / context.close)
        target_size = min(target_size, context.portfolio_value * max_position_pct / context.close)

        if stance > 0.3:
            action = "buy"
        elif stance < -0.3:
            action = "sell"
        else:
            action = "hold"

        trade_size = target_size - context.current_position
        rationale = f"Stance={stance:.2f}, Confidence={confidence:.2f}"

        decision = Decision(
            symbol=context.symbol,
            as_of=context.as_of,
            action=action,
            target_position=target_size,
            trade_size=trade_size,
            reference_price=context.close,
            rationale=rationale,
            confidence=confidence,
        )

        return decision


# ============================================================================
# CONCRETE IMPLEMENTATIONS (copied from quant_only.py)
# ============================================================================

class QuantRiskManager(RiskManagerAgent):
    """Risk manager implementation."""

    def __init__(self, risk_profile: str = "neutral"):
        self.risk_profile = risk_profile

    def evaluate(self, context: MarketContext, proposed_decision: Decision) -> Decision:
        limits = self._get_limits_for_profile()
        max_position = context.portfolio_value * limits["max_position_pct"]
        proposed_decision.target_position = min(
            proposed_decision.target_position,
            max_position / context.close,
        )
        proposed_decision.trade_size = (
            proposed_decision.target_position - context.current_position
        )
        entry_price = context.close
        proposed_decision.risk_overrides["stop_loss_price"] = (
            entry_price * (1 - limits["max_loss_pct"])
        )
        proposed_decision.risk_overrides["take_profit_price"] = (
            entry_price * (1 + limits["take_profit_pct"])
        )
        return proposed_decision

    def _get_limits_for_profile(self) -> Dict[str, float]:
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
    """Fund manager implementation."""

    def approve(
        self,
        context: MarketContext,
        risk_modified_decision: Decision,
        portfolio_state: Dict[str, Any],
    ) -> Decision:
        return risk_modified_decision


# ============================================================================
# FACTORY
# ============================================================================

def create_configurable_quant_only_orchestrator(config: Dict[str, Any]) -> OrchestrationEngine:
    """Create orchestrator with config support."""

    analysts = {
        "fundamental": ConfigurableFundamentalAnalyst(),
        "technical": ConfigurableTechnicalAnalyst(),
        "news": ConfigurableNewsAnalyst(),
        "sentiment": ConfigurableSentimentAnalyst(),
    }

    weight_by_conf = config.get("weight_by_confidence", False)
    researchers = {
        "bullish": ConfigurableBullishResearcher(weight_by_confidence=weight_by_conf),
        "bearish": ConfigurableBearishResearcher(weight_by_confidence=weight_by_conf),
    }

    trader = ConfigurableTrader(config)

    risk_managers = {
        "conservative": QuantRiskManager("conservative"),
        "neutral": QuantRiskManager("neutral"),
        "aggressive": QuantRiskManager("aggressive"),
    }

    fund_manager = QuantFundManager()

    orchestrator = OrchestrationEngine(
        analysts=analysts,
        researchers=researchers,
        trader=trader,
        risk_managers=risk_managers,
        fund_manager=fund_manager,
    )

    return orchestrator