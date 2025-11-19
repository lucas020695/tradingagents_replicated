"""
LLM Weak with Groq API - FAST VERSION

Uses Groq's free API (0.5s per call vs 60s Ollama).
Expected runtime: ~1 hour for Phase 1 backtest.

Setup:
1. Create free account: https://console.groq.com
2. Get API key
3. Set environment variable: GROQ_API_KEY=your_key_here
"""

from src.core.types import MarketContext, Decision, AgentState
from groq import Groq
from datetime import date
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set!")
    raise ValueError("Set GROQ_API_KEY environment variable")


class GroqLLMWeakOrchestrator:
    """Fast LLM using Groq API."""

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.daily_decisions = {}
        self.current_date = None
        self.call_count = 0

    def run_day(self, symbol: str, as_of: date, context: MarketContext, portfolio_state: dict) -> AgentState:
        """Run trading for one symbol on one day."""

        if as_of != self.current_date:
            self.daily_decisions = {}
            self.current_date = as_of
            self.call_count = 0

        if symbol not in self.daily_decisions:
            self.call_count += 1
            if self.call_count % 25 == 0:
                print(f"      [LLM calls: {self.call_count}...]")

            decision = self._call_groq(symbol, context)
            self.daily_decisions[symbol] = decision

        return AgentState(
            symbol=symbol,
            as_of=as_of,
            analyst_reports=[],
            research_reports=[],
            risk_decisions=[],
            final_decision=self.daily_decisions[symbol],
        )

    def _call_groq(self, symbol: str, context: MarketContext) -> Decision:
        """Call Groq API for trading decision."""

        prompt = f"""Trading decision for {symbol}.
Price: ${context.close:.2f}
Current position: {context.current_position:.0f} shares

Respond with ONLY: BUY, SELL, or HOLD"""

        try:
            message = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gemma-7b-it",
                temperature=0.3,
                max_tokens=10,
            )

            response = message.choices[0].message.content.upper().strip()

            if "BUY" in response:
                action = "buy"
            elif "SELL" in response:
                action = "sell"
            else:
                action = "hold"
        except Exception as e:
            print(f"      [Groq error: {str(e)[:50]}]")
            action = "hold"

        if action == "buy":
            target_size = (context.portfolio_value * 0.2) / context.close
        elif action == "sell":
            target_size = 0
        else:
            target_size = context.current_position

        trade_size = target_size - context.current_position

        return Decision(
            symbol=symbol,
            as_of=context.as_of,
            action=action,
            target_position=target_size,
            trade_size=trade_size,
            reference_price=context.close,
            rationale=f"Groq: {action.upper()}",
            confidence=0.7,
        )


def create_llm_weak_orchestrator():
    return GroqLLMWeakOrchestrator()
