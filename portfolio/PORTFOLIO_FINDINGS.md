# Multi-asset Black-Litterman portfolio — results

Universe: BTC, SPY, QQQ, TLT, GLD (2017-2026, total return). Structure A: long-only,
no leverage (weights sum <=1, rest cash). Risk-parity (inverse-vol) prior + per-asset
views = the strategy signals, with view confidence reflecting each asset's VERIFIED edge
(BTC 1.0, TLT 0.8, GLD 0.4, QQQ 0.35, SPY 0.30 - efficient equities held mostly via the
prior). Band rebalancing, per-asset fees (BTC 0.2%, ETFs 0.1%).

## Result (full sample / out-of-sample)
| portfolio | full Sharpe | full NAV | full MaxDD | val | test |
|-----------|-------------|----------|------------|-----|------|
| Equal-weight 25% (the naive baseline) | 1.14 | 5.34 | -38% | 0.18 | 1.37 |
| Risk-parity (no views) | 1.16 | 3.28 | -28% | 0.18 | 1.51 |
| **Black-Litterman** | **1.30** | 3.36 | **-21%** | **0.59** | **1.85** |
| best single asset (QQQ) | 0.95 | 5.56 | -35% | | |

BL beats the naive 25% equal-weight, risk-parity, and EVERY single asset on Sharpe, with
roughly HALF the drawdown (-21% vs -38%). Layer attribution: dollar-equal -> risk-parity
halves drawdown; views (BL) lift Sharpe 1.16 -> 1.30 and cut DD further.

## Robustness
- Beats risk-parity on full Sharpe in 22/27 (VIEW_SCALE x COVWIN x CAP) combos; the
  misses are weak-view ties. Drawdown -20..-23% in ALL combos.
- Views on only the verified-edge assets (BTC+TLT) still give Sharpe 1.26 -> the Sharpe
  lift is driven by the real edges, not the efficient-equity views (honest, as expected).
- Causal: covariance from trailing returns, signal at t held into t+1; no lookahead.

## Why it beats the 25%/35% approach
1. RISK-balanced not DOLLAR-balanced (BTC's 63% vol no longer dominates) -> drawdown halved.
2. Uses the per-asset signals as BL views -> Sharpe lift from BTC/TLT edges.
3. Correlation-aware (SPY/QQQ 0.93 redundancy handled; TLT/GLD low-corr diversifiers).
4. Dynamic target + band rebalancing, not a mechanical 35%->25% threshold.

Files: portfolio/panel.py (returns+signals), portfolio/bl.py (BL optimizer + backtest).
