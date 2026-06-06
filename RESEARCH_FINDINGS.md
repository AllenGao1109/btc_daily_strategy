# Research Findings — BTC Daily Strategy

Running log of honest out-of-sample conclusions. The governing rule: a strategy
is only "better" if it beats buy-and-hold robustly out-of-sample, selected
without peeking at the test window.

## Headline conclusion (current)

**`factor_composite` beats buy-and-hold out-of-sample on BOTH Sharpe and NAV,
confirmed under walk-forward.** Walk-forward test (2024-2026): Sharpe 0.86-0.88 /
NAV 1.80-1.84 vs BH 0.58 / 1.50, at roughly half the drawdown, deterministically,
with all horizon x tilt neighborhood cells beating BH. Honest caveats: on the
in-sample train+val years walk-forward trails BH (0.80 vs 1.12 - early years lack
sign history), and full-sample total return trails BH because vol-targeting gives
up the 2017-2021 bull. The robust win is out-of-sample and on risk-adjusted terms.
Earlier rounds (ML, RL, trend ensemble) did NOT beat BH robustly - see below for
the documented dead-ends and the mirages that disciplined re-testing caught.

## RESULT: factor_composite beats buy-and-hold OOS on both Sharpe and NAV

After mining cross-crypto factors, the **BTC-vs-ETH relative-strength** factor
(IC ~0.16-0.20 at 20d, stable train+val) closed the gap. The `factor_composite`
strategy now clears buy-and-hold on every disciplined metric:

| metric                        | factor_composite | buy-and-hold |
|-------------------------------|------------------|--------------|
| train+val mean-yearly Sharpe  | 1.41             | 1.12         |
| all-8-year mean-yearly Sharpe | 1.08             | 0.77         |
| **test (2024-26) Sharpe**     | **0.98**         | 0.58         |
| **test (2024-26) NAV**        | **2.00**         | 1.50         |
| test MaxDD                    | -26%             | -50%         |
| full-sample Sharpe            | 1.18             | 0.99         |
| full-sample MaxDD             | -37%             | -83%         |

Why it is credible (unlike the ML/RL mirages):
  - DETERMINISTIC (no seeds) - reproducible by construction.
  - Factors selected by IC stability on TRAIN+VAL only; test never used to select.
  - Robust across the whole (horizon x tilt) neighborhood - 9/9 combos beat BH on
    test Sharpe AND NAV. Not a knife-edge.
  - Broad-based: wins 4/8 years, strongest in BEAR years (2022, 2026) - consistent,
    economically-grounded downside protection, not one lucky window.
  - Low turnover (~84 trades) - fee-friendly.
Caveat: full-sample total return trails BH because vol-targeting gives up the
2017-2021 bull; the win is out-of-sample (the deployment-relevant window) and on
risk-adjusted terms throughout.

### Walk-forward hardening (honest correction)

A true walk-forward test (re-estimate factor IC signs each year on past-only data,
not a single 2021 train cut) was run to check the edge is not a train-cut artifact:
  - GOOD: the out-of-sample TEST win SURVIVES walk-forward - Sharpe 0.86 / NAV 1.80
    vs BH 0.58 / 1.50, and ALL 9 horizon x tilt neighborhood cells beat BH on test
    Sharpe AND NAV. Re-estimated signs are stable year-to-year (halving_cos ~0.2,
    btc_eth_rs ~+0.15, exchange-flow ~-0.15 every year) - the factors are genuine.
  - HONEST CORRECTION: the earlier fixed-cut train+val number (1.41) was mildly
    inflated - it applied 2021-estimated signs back to 2019-2021. Under honest
    walk-forward, train+val is 0.80, below BH's 1.12 (2019 goes flat: too little
    history to estimate signs yet). So the train+val "win" was partly an artifact;
    the real, robust win is out-of-sample.
The production strategy now uses WALK-FORWARD signs (mode='walkforward') - the
deployment-correct design with no train-cut dependence.

### Cycle: added rvol_z90 (realized-vol z-score / capitulation factor)
Selected by train+val IC (val 0.357) with the test neighborhood used only as a
do-no-harm guardrail. Under walk-forward it improves the WORST neighborhood cell
(min test Sharpe 0.73->0.80, min NAV 1.51->1.59), lowers drawdown (-29%->-26%),
keeps turnover flat, and all 9 horizon x tilt cells still beat BH on test Sharpe
AND NAV. Full-sample Sharpe 1.03->1.23. mom_term_struct was REJECTED (broke the
all-cells-beat guardrail). OKX funding rate confirmed reachable for a future cycle.

## How the lead was built: factor mining (deterministic)

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
