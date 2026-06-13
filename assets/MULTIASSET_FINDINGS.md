# Per-asset long-only strategies (for the Black-Litterman portfolio) — honest results

Literature-grounded (Moreira-Muir vol management; Moskowitz-Ooi-Pedersen TSMOM; bond
carry), tested vs each asset's total-return buy-and-hold (0.1%/side fee, train 2002-13 /
val 2013-19 / test 2019-26), with adversarial verification of every claimed winner.

## Verified outcomes
| asset | strategy | full Sh vs BH | full DD vs BH | verdict |
|-------|----------|---------------|---------------|---------|
| TLT (20y+ UST) | vol-managed (Moreira-Muir) | 0.42 vs 0.33 | -42% vs -48% | **SURVIVED** — genuine, robust (param plateau, 14/25 yrs, survives dropping 2009+2022) |
| SPY (S&P 500) | vol-managed | 0.75 vs 0.64 | -29% vs -55% | REFUTED as Sharpe edge — win is the in-sample 2008 crash; OOS dSharpe -0.02; lower vol at LOWER return |
| QQQ (Nasdaq 100) | vol-managed x trend | 0.65 vs 0.52 | -29% vs -83% | REFUTED as Sharpe edge — regime-dependent, stat-insignificant, underperforms post-2010/OOS |
| IEF (7-10y UST) | vol-managed + tsmom | 0.59 vs 0.55 | -21% vs -24% | REFUTED — edge thins on a wider param grid |

## The honest lesson
Liquid, EFFICIENT markets (SPY/QQQ) cannot be out-Sharpe'd by timing: vol management
halves drawdowns but does NOT add risk-adjusted return out-of-sample (the apparent edge
is concentrated in in-sample crises). Only the LESS-efficient / structurally-driven
assets give a real timing edge: BTC (factor_composite), TLT (vol-managed bonds), gold
(vol-managed). This matches market-efficiency priors.

## Implication for the portfolio
Per-asset standalone alpha is NOT where the value is for SPY/QQQ. The portfolio's edge
comes from DIVERSIFICATION across low-correlation assets + portfolio-level risk
management. So the BL "views":
- BTC: factor_composite score (real directional edge).
- TLT (or IEF): vol-managed bond signal (verified edge) + a diversifier that rallies in risk-off.
- Gold: vol-managed (beats buy-hold; low correlation to everything).
- SPY/QQQ: held with vol-managed RISK control (drawdown reduction is real & robust) and the
  equity risk premium as a bullish prior; NOT aggressively timed (no robust timing edge).
Next: build the BL layer (risk-parity prior + these views -> long-only weights) and show
the portfolio-level Sharpe/drawdown beats the naive 25% equal-weight allocation.
