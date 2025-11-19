"""
LLM Weak with Perplexity API (usando requests direto)

Chama Perplexity API diretamente via requests.
Runtime esperado: ~2-3 minutos para Phase 3.

Setup:
1. pip install requests
2. Set environment variable: PERPLEXITY_API_KEY=your_key_here
"""

from src.core.types import MarketContext, Decision, AgentState
from datetime import date
import os
import requests

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

class PerplexityLLMWeakOrchestrator:
    """Fast LLM using Perplexity API via requests."""
    
    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.daily_decisions = {}
        self.current_date = None
        self.call_count = 0
    
    def run_day(self, symbol: str, as_of: date, context: MarketContext, portfolio_state: dict) -> AgentState:
        if as_of != self.current_date:
            self.daily_decisions = {}
            self.current_date = as_of
            self.call_count = 0
        
        if symbol not in self.daily_decisions:
            self.call_count += 1
            if self.call_count % 25 == 0:
                print(f"      [LLM calls: {self.call_count}...]")
            decision = self._call_perplexity(symbol, context)
            self.daily_decisions[symbol] = decision
        
        decision = self.daily_decisions[symbol]
        return AgentState(symbol=symbol, as_of=as_of, analyst_reports=[], research_reports=[], risk_decisions=[], final_decision=decision)
    
    def _call_perplexity(self, symbol: str, context: MarketContext) -> Decision:
        prompt = f"""You are a trading AI for {symbol}.

Current market data:
- Price: ${context.close:.2f}
- Current position: {context.current_position:.0f} shares
- Portfolio value: ${context.portfolio_value:,.0f}

Make a QUICK trading decision: BUY, SELL, or HOLD?
Respond with ONLY one word: BUY, SELL, or HOLD"""
        
        action = "hold"
        
        try:
            if not self.api_key:
                action = "hold"
            else:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": "pplx-7b-online",
                    "messages": [
                        {"role": "system", "content": "You are a trading AI. Respond with only one word: BUY, SELL, or HOLD."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 5,
                }
                
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data["choices"][0]["message"]["content"].upper().strip()
                    if "BUY" in response_text:
                        action = "buy"
                    elif "SELL" in response_text:
                        action = "sell"
                    else:
                        action = "hold"
        except Exception as e:
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
            rationale=f"Perplexity LLM: {action.upper()}",
            confidence=0.7,
        )

def create_llm_weak_orchestrator():
    return PerplexityLLMWeakOrchestrator()
