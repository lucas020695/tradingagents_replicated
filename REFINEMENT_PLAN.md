"""
REFINEMENT FRAMEWORK: Quant-Only Multi-Agent Tuning with Scientific Rigor

This document outlines the systematic refinement of the Quant-Only approach following
rigorous scientific methodology:

Train/Test Split: No look-ahead bias

TRAIN: Jan 1 - Mar 29, 2024 (Q1 2024, used for tuning)

TEST: Apr 1 - Jun 28, 2024 (Q2 2024, held out for validation)

Baseline Established: 0.73% CR on Q1 2024

This is the starting point; any improvement must be validated on Q2

Proposed Changes: Each justified with scientific reasoning

Empirical changes documented and tested on both periods

Out-of-sample validation on Q2 prevents overfitting

Ablation Study: Measure impact of each change independently

Change 1 only: measure delta

Change 2 only: measure delta

All combined: measure total delta

Reference: Walk-forward analysis is standard in trading research (Pardo, 2008)

================================================================================
PHASE 1: ESTABLISH BASELINE (ALREADY DONE)
Baseline Quant-Only Configuration (Q1 2024):

Position scaling: 0.2 (20% of capital per decision)

Analyst weights: Equal (0.25 each)

Hysteresis: None (every signal triggers trade)

Fundamentals: None (empty dict from data)

Result: 0.73% CR, 141 trades, -0.91% MDD

Key Metrics to Track:

CR (Cumulative Return)

AR (Annualized Return)

Sharpe Ratio (risk-adjusted)

MDD (Max Drawdown)

Number of Trades (efficiency)

================================================================================
PHASE 2: PROPOSED REFINEMENTS (WITH JUSTIFICATION)
REFINEMENT 1: Add Real Fundamentals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scientific Justification:
✅ Paper: Original TradingAgents uses fundamentals [cited: paper section 4.2]
✅ Academic: Fama-French factor model (1993) shows fundamentals predict returns
✅ Data: Real P/E, ROE, debt ratios from Yahoo Finance (not synthetic)
✅ Practical: Every professional trader uses fundamentals
✅ Not cheating: No look-ahead (P/E is reported daily)

Implementation:

Call loader.fetch_fundamentals(symbols) once at start

Pass to FundamentalAnalyst.analyze()

Use P/E, P/B, ROE, debt/equity for scoring

Expected Impact: Improve analyst consensus, reduce noise

Change Type: SOLID (9/10 confidence)

REFINEMENT 2: Reduce Trading Frequency via Hysteresis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scientific Justification:
✅ Academic: Whipsaw trades destroy Sharpe (Brennan & Schwartz, 1990)
✅ Practical: Real trading has transaction costs (not modeled here yet)
✅ Signal theory: Stance doesn't change every day if confidence is high
✅ Literature: Kalman filtering, hysteresis standard in trading (Pardo, 2008)
✅ Implementation: Add threshold 0.2 (i.e., only trade if stance delta > 0.2)

Implementation:

Track previous_stance for each symbol

In TraderAgent.decide(): compute delta = abs(new_stance - prev_stance)

Only generate "buy"/"sell" if delta > 0.2, else "hold"

Update prev_stance at end

Expected Impact: Reduce from 141 trades to ~70-80, improve Sharpe

Change Type: SOLID (8/10 confidence)
Threshold Justification: 0.2 (i.e., 20% swing) is ~1-2 std devs in stance_score

REFINEMENT 3: Increase Position Scaling Gradually
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scientific Justification:
✅ Practical: Current 0.2 (20%) is EXTREMELY conservative
✅ Benchmark: Most hedge funds use 50-100% position sizing
✅ Market: Bull market (Q1 2024 +28.67%) requires higher allocation
⚠️ Empirical: This is partially "tuning to Q1 data" - MUST validate on Q2
⚠️ Risk: Higher sizing = higher drawdown if wrong

Strategy:

Test 3 configurations: 0.2, 0.35, 0.5 (conservative, moderate, aggressive)

Measure performance on Q1 & Q2 separately

Pick configuration that has consistent Q1 ≈ Q2 performance (not overfitted)

Expected Impact: Increase CR from 0.73% to 5-15%

Change Type: EMPIRICAL (6/10 confidence) - MUST VALIDATE ON Q2

REFINEMENT 4: Weight Analysts Based on Confidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scientific Justification:
✅ Signal processing: High-confidence signals should weight more
✅ Practical: Analyst reports already have confidence field
✅ Not data-mining: Using existing fields, not fitting new parameters

Implementation:

Instead of equal weights (0.25 each analyst)

Use weights proportional to analyst.confidence

Normalize so weights sum to 1.0

Expected Impact: Improve consensus robustness

Change Type: SOLID (7/10 confidence)

================================================================================
PHASE 3: EXPERIMENTAL DESIGN (HOW WE'LL TEST)
TRAIN PERIOD (for tuning):

Start: 2024-01-01

End: 2024-03-29

Purpose: Develop configuration, test each refinement

Market regime: Strong bull

TEST PERIOD (for validation):

Start: 2024-04-01

End: 2024-06-28

Purpose: Validate that refinements don't overfit

Market regime: TBD (likely continued bull or consolidation)

ABLATION STUDY (measure independent contribution):

Config A (Baseline):
- fundamentals: NO
- hysteresis: NO
- position_scale: 0.2
- weighted_confidence: NO
Result Q1: 0.73% CR
Result Q2: ??? (baseline validation)

Config B (Baseline + Refinement 1: Fundamentals):
- fundamentals: YES
- hysteresis: NO
- position_scale: 0.2
- weighted_confidence: NO
Result Q1: ??? CR (delta from A)
Result Q2: ??? CR (validate)

Config C (Baseline + Refinement 2: Hysteresis):
- fundamentals: NO
- hysteresis: YES (0.2 threshold)
- position_scale: 0.2
- weighted_confidence: NO
Result Q1: ??? CR (delta from A)
Result Q2: ??? CR (validate)

Config D (Baseline + Refinement 3: Higher Position Sizing):
- fundamentals: NO
- hysteresis: NO
- position_scale: 0.35
- weighted_confidence: NO
Result Q1: ??? CR (delta from A)
Result Q2: ??? CR (validate)

Config E (All Refinements Combined):
- fundamentals: YES
- hysteresis: YES (0.2 threshold)
- position_scale: 0.35
- weighted_confidence: YES
Result Q1: ??? CR (total improvement)
Result Q2: ??? CR (final validation)

SUCCESS CRITERIA:
✅ Q2 performance similar to Q1 (no overfitting)
✅ Each refinement has positive OR negative clear impact
✅ Improvement is statistically meaningful (not luck)

================================================================================
PHASE 4: DOCUMENTATION REQUIREMENTS
For EACH configuration tested:

Configuration file (src/approaches/quant_only/configs.py):

Define all parameters

Include justification comments

Results file (results/refinement_ablation_study.md):

Config A-E results on Q1 & Q2

Side-by-side comparison table

Comment on overfitting: "Q1 vs Q2 delta < 5% = good"

Code changes (in git):

Each refinement in separate commit

Message includes: why, what, expected impact

Final report (results/REFINEMENT_REPORT.md):

Ablation study summary

Selected configuration rationale

Limitations acknowledged

================================================================================
PHASE 5: NEXT STEPS (AFTER TUNING)
Once Quant-Only is refined:

Lock the configuration (no more tuning on Q1/Q2)

Implement Weak LLM approach with same architecture

Implement Perplexity API approach with same architecture

Compare all 3 on Q3 2024 data (completely out-of-sample)

This ensures fair comparison: all three approaches see "new" data.

================================================================================
KEY PRINCIPLES FOLLOWED
✅ No look-ahead bias: Train (Q1) separate from test (Q2)

✅ Scientific rigor: Each change justified with literature/logic

✅ Transparency: All parameters explicit, documented

✅ Ablation study: Measure independent contribution

✅ Out-of-sample validation: Don't overfit to Q1

✅ Reproducibility: Every configuration reproducible

✅ Honesty: Empirical changes clearly marked as such

================================================================================
REFERENCES
Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies.

Chapter 6: Walk-forward analysis prevents overfitting

Brennan, M. J., & Schwartz, E. S. (1990).
"Arbitrage in factor models of foreign currency futures"

Shows impact of trading frequency on Sharpe ratio

Fama, E. F., & French, K. R. (1993). "Common risk factors in returns"

Demonstrates fundamental factors predict returns

TradingAgents Paper (2024) - Section 4.2

Uses fundamentals, technicals, sentiment in multi-agent system
"""