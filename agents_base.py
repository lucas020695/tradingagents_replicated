"""
Base classes and interfaces for all agent types in the TradingAgents replica framework.

This module defines abstract base classes (ABCs) that specify the contract
each agent type must fulfill across all three approaches (Perplexity API, weak LLM, quant-only).

Key principle: The interface is IDENTICAL across approaches; only the implementation changes.
This ensures full comparability and allows plug-and-play swapping of approaches.

Reference: TradingAgents paper uses structured messaging between specialized agents.
We enforce this via abstract methods and type hints.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import date

from .types import (
    MarketContext,
    AnalystReport,
    ResearchReport,
    Decision,
    AgentState,
)


# ============================================================================
# ANALYST AGENTS (Fundamental, Technical, News, Sentiment)
# ============================================================================

class AnalystAgent(ABC):
    """
    Abstract base class for all specialist analysts.

    Each analyst receives a MarketContext and produces an AnalystReport
    with standardized numerical signals, risk flags, and a human-readable summary.

    The key design: all four analysts (fundamental, technical, news, sentiment)
    implement this interface, making it easy to swap implementations across approaches.
    """

    def __init__(self, analyst_type: str, llm_client: Optional[Any] = None):
        """
        Initialize an analyst agent.

        Args:
            analyst_type (str): "fundamental", "technical", "news", or "sentiment"
            llm_client (Optional[Any]): LLM client for text generation (None in quant-only).
                Allows all approaches to share the same AnalystAgent interface.
        """
        self.analyst_type = analyst_type
        self.llm_client = llm_client

    @abstractmethod
    def analyze(self, context: MarketContext) -> AnalystReport:
        """
        Analyze market context and produce a structured report.

        Implementations differ by approach:
        - LLM approaches: call LLM to generate summary, compute signals from context + LLM output
        - Quant-only: compute signals from context alone, generate summary from template

        Args:
            context (MarketContext): Market data for a single ticker on a single day

        Returns:
            AnalystReport: Structured analysis with signals, risk_flags, summary, confidence
        """
        pass

    def validate_report(self, report: AnalystReport) -> bool:
        """
        Validate that an AnalystReport has all required fields and valid ranges.

        Use this in __init__ or tests to catch bugs early.

        Returns:
            bool: True if valid, raises ValueError if not
        """
        assert report.analyst_type == self.analyst_type
        assert -1.0 <= report.confidence <= 1.0, "confidence must be in [0, 1]"
        assert len(report.signals) > 0, "signals cannot be empty"
        # Add more validations as needed
        return True


class FundamentalAnalyst(AnalystAgent):
    """
    Specialist in valuation, quality, and growth metrics.

    Inputs:
        - P/E, P/B, ROE, dividend yield, debt/equity, revenue growth, etc.
        - Available from MarketContext.fundamentals (if data exists)

    Outputs (AnalystReport.signals):
        - value_score: is the stock cheap relative to fundamentals? (0-1)
        - quality_score: how good is the business? (0-1)
        - growth_score: is the company growing? (0-1)

    Risk flags:
        - "high_valuation": P/E significantly above sector average
        - "weak_fundamentals": declining revenues or margins
        - "high_debt": debt/equity above sector norms
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(analyst_type="fundamental", llm_client=llm_client)

    @abstractmethod
    def analyze(self, context: MarketContext) -> AnalystReport:
        """Analyze fundamental metrics (implemented differently in each approach)"""
        pass


class TechnicalAnalyst(AnalystAgent):
    """
    Specialist in price action, momentum, and trend analysis.

    Inputs:
        - MACD, RSI, Bollinger Bands, SMA/EMA, ADX, etc.
        - Available from MarketContext.technicals (pre-computed by data_loader)

    Outputs (AnalystReport.signals):
        - momentum_score: how strong is the price trend? (0-1)
        - overbought_flag: is RSI in danger zone? (-1 to +1, -1 = overbought)
        - trend_score: is trend up or down? (0-1, higher = more bullish)

    Risk flags:
        - "overbought": RSI > 70
        - "oversold": RSI < 30
        - "trend_reversal": MACD crossing or divergence detected
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(analyst_type="technical", llm_client=llm_client)

    @abstractmethod
    def analyze(self, context: MarketContext) -> AnalystReport:
        """Analyze technical indicators (implemented differently in each approach)"""
        pass


class NewsAnalyst(AnalystAgent):
    """
    Specialist in news sentiment and corporate announcements.

    Inputs:
        - News articles, press releases, earnings announcements
        - Sentiment pre-computed from LLM or lexicon-based analysis

    Outputs (AnalystReport.signals):
        - positive_event_count: number of positive news items
        - negative_event_count: number of negative news items
        - net_sentiment: aggregate sentiment (-1 to +1)

    Risk flags:
        - "earnings_miss": recent earnings guidance miss
        - "sec_filing": new 8-K or regulatory filing
        - "negative_news_surge": unusual spike in negative articles

    Note: In quant-only approach, this may be simplified or skipped.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(analyst_type="news", llm_client=llm_client)

    @abstractmethod
    def analyze(self, context: MarketContext) -> AnalystReport:
        """Analyze news and events (implemented differently in each approach)"""
        pass


class SentimentAnalyst(AnalystAgent):
    """
    Specialist in social/alternative sentiment (Reddit, Twitter, insider transactions).

    Inputs:
        - Social media posts, insider trading activity, options flow, etc.
        - Available from MarketContext.sentiment_scores (if approach supports it)

    Outputs (AnalystReport.signals):
        - social_sentiment: aggregate social media sentiment (-1 to +1)
        - insider_score: insider buy/sell ratio (0-1, higher = more buying)
        - options_flow_score: unusual options activity (-1 to +1)

    Risk flags:
        - "insider_selling": insiders selling more than usual
        - "extreme_bullish_sentiment": social media frothy, potential bubble
        - "unusual_options_activity": large options positions or unusual IV

    Note: In quant-only approach, this is simplified or skipped.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(analyst_type="sentiment", llm_client=llm_client)

    @abstractmethod
    def analyze(self, context: MarketContext) -> AnalystReport:
        """Analyze sentiment signals (implemented differently in each approach)"""
        pass


# ============================================================================
# RESEARCHER AGENTS (Bullish & Bearish debate)
# ============================================================================

class ResearcherAgent(ABC):
    """
    Abstract base for Bull/Bear researchers who debate the case for/against a position.

    Each researcher receives all AnalystReports and produces a ResearchReport
    summarizing the bullish or bearish case.

    In LLM approaches, multiple rounds of debate happen via LLM calls.
    In quant-only, the debate is a weighted combination of analyst signals.
    """

    def __init__(self, perspective: str, llm_client: Optional[Any] = None):
        """
        Initialize a researcher agent.

        Args:
            perspective (str): "bullish" or "bearish"
            llm_client (Optional[Any]): LLM client for debate (None in quant-only)
        """
        assert perspective in ["bullish", "bearish"]
        self.perspective = perspective
        self.llm_client = llm_client

    @abstractmethod
    def research(
            self,
            context: MarketContext,
            analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        """
        Generate a research report from analyst inputs.

        Implementations differ:
        - LLM approaches: debate with opposing researcher via LLM
        - Quant-only: aggregate analyst signals via weighted formula

        Args:
            context (MarketContext): Market data for context
            analyst_reports (List[AnalystReport]): Reports from all 4 analysts

        Returns:
            ResearchReport: Aggregated bull/bear stance with supporting narrative
        """
        pass


class BullishResearcher(ResearcherAgent):
    """Advocates for buying or holding positions. Makes the case for upside."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(perspective="bullish", llm_client=llm_client)

    @abstractmethod
    def research(
            self,
            context: MarketContext,
            analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        pass


class BearishResearcher(ResearcherAgent):
    """Advocates for selling or avoiding positions. Makes the case for downside."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(perspective="bearish", llm_client=llm_client)

    @abstractmethod
    def research(
            self,
            context: MarketContext,
            analyst_reports: List[AnalystReport],
    ) -> ResearchReport:
        pass


# ============================================================================
# TRADER AGENT
# ============================================================================

class TraderAgent(ABC):
    """
    Converts research and analyst consensus into a concrete trading decision.

    Inputs:
        - ResearchReport (bull/bear stance + confidence)
        - All AnalystReports (for additional context if needed)
        - MarketContext (current position, cash, risk limits)

    Outputs:
        - Decision with action (buy/sell/hold), size, and rationale

    The Trader does NOT enforce risk limits; that's the RiskManager's job.
    The Trader focuses on the best trade SIZE given sentiment, not position limits.
    """

    def __init__(self, llm_client: Optional[Any] = None, policy: Optional[Dict[str, Any]] = None):
        """
        Initialize the Trader agent.

        Args:
            llm_client (Optional[Any]): LLM client (None in quant-only)
            policy (Optional[Dict]): Trading policy (position sizing rules, etc.)
                Example: {"max_position_size": 0.2, "position_scale": "log"}
        """
        self.llm_client = llm_client
        self.policy = policy or {}

    @abstractmethod
    def decide(
            self,
            context: MarketContext,
            research_report: ResearchReport,
            analyst_reports: List[AnalystReport],
    ) -> Decision:
        """
        Generate a trading decision.

        Args:
            context (MarketContext): Current market and portfolio state
            research_report (ResearchReport): Bull/bear aggregated view
            analyst_reports (List[AnalystReport]): Individual analyst inputs

        Returns:
            Decision: Action (buy/sell/hold), size, confidence, rationale
        """
        pass


# ============================================================================
# RISK MANAGER AGENT
# ============================================================================

class RiskManagerAgent(ABC):
    """
    Enforces risk limits and position constraints.

    Takes a Decision from the Trader and either:
    - Approves as-is
    - Modifies position size to respect limits
    - Rejects if risk is unacceptable

    Multiple risk profiles (risk-seeking, neutral, conservative) can coexist,
    each applying different risk constraints.
    """

    def __init__(self, risk_profile: str, llm_client: Optional[Any] = None):
        """
        Initialize a risk manager.

        Args:
            risk_profile (str): "conservative", "neutral", or "aggressive"
            llm_client (Optional[Any]): LLM client (None in quant-only)
        """
        assert risk_profile in ["conservative", "neutral", "aggressive"]
        self.risk_profile = risk_profile
        self.llm_client = llm_client

    @abstractmethod
    def evaluate(
            self,
            context: MarketContext,
            proposed_decision: Decision,
    ) -> Decision:
        """
        Evaluate and potentially modify a proposed decision.

        Implementations may:
        - Add stop-loss/take-profit levels
        - Reduce position size
        - Reject the trade entirely

        Args:
            context (MarketContext): Current portfolio and risk limits
            proposed_decision (Decision): Trader's proposed decision

        Returns:
            Decision: Modified decision with risk_overrides applied
        """
        pass


# ============================================================================
# FUND MANAGER AGENT (Final Approval)
# ============================================================================

class FundManagerAgent(ABC):
    """
    Final authority: reviews Risk Manager's modified decision and executes or rejects.

    Responsibilities:
    - Enforce global portfolio constraints (max leverage, max drawdown, etc.)
    - Aggregate decisions across multiple tickers
    - Log final decisions and audit trail
    - Approve, modify, or veto decisions from RiskManager
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the Fund Manager.

        Args:
            llm_client (Optional[Any]): LLM client (None in quant-only)
        """
        self.llm_client = llm_client

    @abstractmethod
    def approve(
            self,
            context: MarketContext,
            risk_modified_decision: Decision,
            portfolio_state: Dict[str, Any],
    ) -> Decision:
        """
        Final review and approval of a trading decision.

        Args:
            context (MarketContext): Current market and portfolio state
            risk_modified_decision (Decision): Decision after risk review
            portfolio_state (Dict): Aggregate portfolio across all tickers

        Returns:
            Decision: Final decision (approved, modified, or action="hold")
        """
        pass


# ============================================================================
# ORCHESTRATOR (Pipeline Coordinator)
# ============================================================================

class OrchestrationEngine(ABC):
    """
    Orchestrates the flow of agents through a single day's decision cycle.

    Sequence (per ticker, per day):
    1. Fetch MarketContext
    2. Run all 4 Analysts in parallel (or sequence)
    3. Run Bullish and Bearish Researchers (optionally in debate)
    4. Run Trader with consensus
    5. Run RiskManager(s) to review decision
    6. Run FundManager for final approval
    7. Record Decision and update portfolio

    This orchestrator is IDENTICAL across approaches; implementations only change
    the agent factories/configurations.
    """

    def __init__(
            self,
            analysts: Dict[str, AnalystAgent],
            researchers: Dict[str, ResearcherAgent],
            trader: TraderAgent,
            risk_managers: Dict[str, RiskManagerAgent],
            fund_manager: FundManagerAgent,
    ):
        """
        Initialize the orchestration engine with agent instances.

        Args:
            analysts: {"fundamental": ..., "technical": ..., "news": ..., "sentiment": ...}
            researchers: {"bullish": ..., "bearish": ...}
            trader: Single TraderAgent instance
            risk_managers: {"conservative": ..., "neutral": ..., "aggressive": ...}
            fund_manager: Single FundManagerAgent instance
        """
        self.analysts = analysts
        self.researchers = researchers
        self.trader = trader
        self.risk_managers = risk_managers
        self.fund_manager = fund_manager

    @abstractmethod
    def run_day(
            self,
            symbol: str,
            as_of: date,
            context: MarketContext,
            portfolio_state: Dict[str, Any],
    ) -> AgentState:
        """
        Run a complete decision cycle for one ticker on one day.

        Returns AgentState capturing all intermediate outputs (for debugging/audit).

        Args:
            symbol (str): Ticker being analyzed
            as_of (date): Decision date
            context (MarketContext): Market data for this date
            portfolio_state (Dict): Current portfolio holdings/cash

        Returns:
            AgentState: Complete state trace for this day
        """
        pass