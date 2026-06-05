# Research Findings — BTC Daily Strategy

Running log of honest out-of-sample conclusions. The governing rule: a strategy
is only "better" if it beats buy-and-hold robustly out-of-sample, selected
without peeking at the test window.

## Headline conclusion (as of the latest round)

**No strategy found so far robustly beats buy-and-hold on BOTH Sharpe and NAV.**
The vol-targeted trend ensemble offers genuinely lower drawdown (and higher
full-sample Sharpe), but it gives up NAV in bull markets, and on a year-by-year
basis buy-and-hold has the best risk-adjusted return.

## Most promising lead: factor mining (deterministic)

Pivoting from "more models on the same features" to MINING NEW FACTORS found the
strongest signal yet. Factors ranked by IC stability across train AND validation
(same sign, both above noise) surfaced economically-grounded winners:
  - **BTC halving-cycle phase** (halving_cos): robust on all 3 horizons, IC
    ~0.2-0.33 at 20-day. The single best factor — a genuinely BTC-specific edge.
  - return kurtosis, volatility regime, MVRV valuation z-score, 120-day momentum,
    exchange net-flow (sell pressure).

The IC-weighted composite (`factor_composite` strategy), mapped to a long-biased
vol-targeted weight, is DETERMINISTIC (no seed luck) and low-turnover (~89 trades):
  - **Test (2024-2026): Sharpe 1.01 / NAV 2.02 vs buy-and-hold 0.58 / 1.50** —
    beats BH on both, with a better worst year.
  - **All-8-year mean Sharpe 0.85 vs BH 0.77** — beats BH.
  - **BUT train+val-only years: 0.99 vs BH 1.12** — still just under. It gives up
    the explosive 2019/2023 bull-year Sharpe (vol-targeting caps it) while winning
    the choppy test years. Regime-dependent, not yet a clean disciplined win.
This is the live thread: mine more factors / refine the mapping to clear the
train+val bar without test-peeking.

## What works

- **Volatility targeting + diversified trend ensemble** — lower drawdown
  (-51% full sample vs buy-and-hold -83%), higher full-sample Sharpe
  (1.07 vs 0.99), positive Sharpe in every fixed window. Its weakness is NAV:
  vol-targeting trades away 2017-2021 bull upside for risk control.
- **No-trade band (fee control)** — raising the band 0.20 -> 0.30 cut trades
  ~44% and fees ~35% while *improving* validation Sharpe and holding drawdown.
  Trading less is materially better given the 0.5% per-trade fee. (0.40 cut more
  fees but blew full-sample drawdown out to -67%.)

## What does NOT work (tested, documented, not hidden)

- **Standalone next-day ML** (500 experiments): mean OOS test Sharpe -0.28;
  top-validation configs were test-negative (val 1.29 -> test -0.34). Of 36
  configs that beat buy-and-hold on test, zero had validation Sharpe > 0.7, so
  none were selectable without peeking.
- **ML gate hybrid** (round 2, 400 experiments): looked promising on the lumped
  test window (logistic/h20/on-chain gate: test Sharpe 0.80), but **year-by-year
  it is not robust** — mean yearly Sharpe 0.30 (worst of the three; buy-and-hold
  0.77, ensemble 0.57), with a catastrophic 2019 (-0.92). The aggregate test
  number was a single-window artifact. This is the key lesson: select on
  fold-by-fold robustness, not aggregate-window Sharpe.
- **Intraday daily-bar stops** — redundant with vol-targeting; BTC drawdowns are
  multi-day grinds, not single-day crashes a daily stop catches.
- **MVRV valuation gating** — 2024-2025 ran persistently high MVRV without a
  top, so de-risking exited a rising market.
- **Fold-robust ML selection** (round 3, 400 experiments): selecting on MIN
  per-year Sharpe over 2019-2023 found exactly ONE config beating buy-and-hold's
  per-year bar (fold-mean 1.15 vs 1.12) — and it still failed on the held-out
  test (0.46 vs 0.58). Robustness on the selectable years does not transfer to
  the held-out years.
- **Reinforcement learning** (REINFORCE policy gradient, MPS, fee in the reward):
  trained only on 2017-2021, the agent learned a sensible low-turnover long-
  biased policy on its own (mean-yearly Sharpe 0.69, 6/8 positive years) — MORE
  robust than the trend ensemble (0.57), but still below buy-and-hold (0.77). It
  is too cautious in the test bull (test NAV 1.10 vs 1.50). Competitive, not
  better.
  - **Reward design matters**: a log-utility (growth-optimal) reward genuinely
    beats the raw-PnL reward (mean-yearly 0.69 -> ~0.8, less cautious, higher
    NAV), consistently across action sets. This is a real RL improvement.
  - **But the apparent BH-beating result was seed luck.** logutil with fine
    actions looked like it beat BH (sweep: test Sh 0.70 / NAV 1.79). Verification
    killed it: on the SELECTABLE years (2019-2023) it only TIES BH (mean-yearly
    1.12 vs 1.12), and a FRESH seed batch (5-9) gave test 0.48 / NAV 1.31 —
    *below* BH. The win did not reproduce. Lesson: re-run on fresh seeds and
    check train+val folds before believing any "win".

## Overall verdict after ~1,300 ML configs + RL

On daily BTC with these features, **nothing robustly beats buy-and-hold out-of-
sample** once fees and per-year robustness are enforced. The 0.5% per-trade fee
is a high hurdle and daily price moves are close to unpredictable. The deployable
choice is a risk preference, not an alpha:
  - buy-and-hold: best year-by-year Sharpe, but -83% drawdowns;
  - vol-targeted trend ensemble (band 0.30): ~half the drawdown and higher full-
    sample Sharpe at similar return, but gives up bull-market NAV.
Further upside almost certainly requires a different data regime (intraday
microstructure, order flow, sentiment) rather than more models on daily bars.

## Methodology notes

- Year-by-year (per-fold) evaluation is mandatory before trusting any edge.
  Aggregate validation/test Sharpe hides window-composition luck.
- ML results are reproducible on MPS (identical across re-runs with fixed seeds).
- Full-sample Sharpe (rewards low drawdown) and mean-yearly Sharpe (rewards
  consistency) can disagree — report both.
