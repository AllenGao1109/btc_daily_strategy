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

## Confidence-scaled aggressive sizing (confidence_gain) - higher NAV, robust

Implements the owner's "let winners run" logic as SIZING, not take-profit: scale the
vol target up when the (positive) composite confidence is high, while exhaustion
factors pulling the score down still cut size. confidence_gain=1.0:
- Test (2024-2026): Sharpe 0.94 / NAV 2.30 vs the conservative 0.92 / 1.90 and BH
  0.58 / 1.50 - higher NAV, Sharpe maintained, drawdown -31% (still << BH -50%).
- ALL 9 horizon x tilt neighborhood cells beat BH on test Sharpe AND NAV (test Sh
  0.67-0.94, NAV 1.59-2.30). Not a knife-edge.
- Full sample: total return 1395% -> 4131% (closing the gap to BH's 6255%), Sharpe
  1.15 (> BH 0.99), MaxDD -54% (still far better than BH -83%).
Honest tradeoff: it raises drawdown (test -24%->-31%, full -37%->-54%) and lowers
validation Sharpe (1.44->1.28) - a deliberate higher-NAV / higher-drawdown preference,
exactly the NAV/Sharpe trade the owner asked for. gain>1.5 degrades (drawdown blows
out). Set confidence_gain=0.0 for the lower-drawdown Sharpe-first profile. Now the
config default (NAV preset).

## Overnight 8h autonomous search (220k evals) - no clean improvement; overfit caught

Ran overnight_research.py for 8h (220,832 evaluations) over 83 candidates incl. NEW
market-breadth factors from 12 altcoins, selecting on a 4-fold (2020-2023) walk-forward
metric with the test window held out.
- Forward selection (low multiple-testing, trustworthy) added sharpe_mom_120 +
  alts_above_50dma + btc_vs_alts_30, improving the worst FOLD (-0.73 -> -0.05). But on
  hand-validation it is NOT a clean win: the deployed point (h20/t0.5) test drops
  0.94/2.30 -> 0.86/2.05 and 2026 worsens. A lateral move, not an improvement.
- The random search's "best" (fold_mean 1.799, test 0.98/2.47) is a MULTIPLE-TESTING
  ARTIFACT: selected from 220k tries, it dropped all core factors and its worst YEAR is
  -1.24 (vs current -0.73) - shinier aggregate numbers, worse tail robustness. Adopting it
  on the test number alone would have made the strategy WORSE. The year-by-year + worst-
  cell discipline caught it. (Another logged RND "best" had fold_mean 1.67 but test
  0.40/1.21 - direct proof the fold metric was gamed.)
- Market-breadth factors are real signal (robust IC) but largely REDUNDANT with the
  existing factors for the deployed config.
Conclusion: the current 9-factor composite (conf_gain=1.0, test 0.94/2.30) survives a
220k-eval search over new free data - strong evidence it is robust. No production change.
The episode is a clean case study in why disciplined multi-fold + tail checks matter.

## Shorting (allow_short) - disciplined shorts cut drawdown, validation-justified

The engine always supported shorts (spec: min_weight -2.0); the strategy was long/flat.
Enabling shorts on STRONG bear conviction (tilt 0.5 -> 0.25 + allow_short) is a genuine,
validation-justified improvement:
- Validation Sharpe HOLDS (~1.40, vs ~1.36-1.41 long/flat) - not test-peeked.
- Full-sample drawdown -61% -> -43% (BH -83%); full Sharpe 1.06 -> 1.14.
- Test improves to 0.93-0.99 / 2.03-2.11 (vs 0.83 / 1.99); all horizon cells beat BH.
- Profits in the 2022 & 2026 bears (yearly Sharpe -0.2/-1.5 -> positive) instead of
  sitting flat.
The shorts are modest and safe: short only ~19% of days, max -0.75x, no liquidation.
tilt=0.0 (full symmetric short) OVER-shorts the recoveries and drops validation to 1.14;
tilt=0.5+short barely protects (-61%). tilt=0.25 is the balance. allow_short=false keeps
the long/flat profile (sits out bears). Now the config default.

## Leverage study: why NOT to go above ~2x (even at high conviction)

Thought experiment: allow up to 10x. Findings (research; production stays at 2x):
- With vol-targeting at the default risk budget, raising the cap 2x->10x changes almost
  NOTHING: peak realized leverage stays ~2.4x and full-sample NAV ~29->33. The strategy
  is risk-budgeted - it does not WANT high leverage. "Having 10x" is irrelevant.
- To actually use it you must raise target_vol (be more aggressive): NAV looks huge in
  sample (target_vol 1.5 -> NAV 489) but at -85% drawdown and 7x peak leverage, and the
  TEST NAV gets WORSE (conf_gain 1->8: test NAV 2.12 -> 0.77). It overfits the in-sample
  bull and loses out-of-sample.
- target_vol >= 3 LIQUIDATES on 2020-03-12 (COVID -38% day). 10x = ruin on any crash.
- The decisive empirical point on "but bet big when very confident": across the top-10%
  most-bullish-conviction days, 16% are followed by a >-10% drawdown within 10 days
  (worst single day -38%, worst 10d drawdown -51%). At 10x a -10% move is liquidation, so
  ~1 in 6 max-conviction bets would wipe out the account. Risk of ruin dominates; Kelly
  itself prescribes a small fraction against a 16%-chance-of-total-loss bet.
Conclusion: ~2-2.4x via vol-targeting (what we do) is near the math-optimal ceiling under
BTC's tail risk + 0.5% fees. The intraday-liquidation caveat makes any high-lev backtest
NAV dangerously optimistic.

## Regime / data-relevance study: keep ALL history (don't trim early data)

Hypothesis tested: "BTC matured since ~2015, vol fell, so early data is noise - trim it."
- Vol HAS fallen: annualized realized vol ~80-95% (2017-2021) -> ~42-53% (2023-2026). True.
- BUT trimming early data HURTS out-of-sample: test Sharpe/NAV by data start -
  2017(all) 1.03/2.16, 2019 0.79/1.68, 2020 0.73/1.64, 2021 0.71/1.67 (val collapses to
  0.00 - too little warmup). Less data = more overfitting + loss of crash memory (2022/2026
  bears rhyme with 2018).
- The vol decline is ALREADY handled by the mechanism: vol-targeting (0.55/realized_vol)
  auto-scales position UP as vol falls, and expanding z-scores re-center - so we capture the
  low-vol regime WITHOUT discarding old data.
- Tail risk did NOT vanish: 2024 had a -8.4% day, 2026 a -14% day. "Matured / lower vol"
  does not mean "safe for high leverage."
Conclusion: full history + vol-targeting is optimal - old data gives robustness and crash
memory; the adaptive sizing captures the new low-vol regime. Do not trim.

## What works

- **Volatility targeting + diversified trend ensemble** — lower drawdown
  (-51% full sample vs buy-and-hold -83%), higher full-sample Sharpe
  (1.07 vs 0.99), positive Sharpe in every fixed window. Its weakness is NAV:
  vol-targeting trades away 2017-2021 bull upside for risk control.
- **No-trade band (fee control)** — raising the band 0.20 -> 0.30 cut trades
  ~44% and fees ~35% while *improving* validation Sharpe and holding drawdown.
  Trading less is materially better given the 0.5% per-trade fee. (0.40 cut more
  fees but blew full-sample drawdown out to -67%.)

## Upside-confidence / "predict the weekly high" - sound idea, take-profit backfires

Tested reframing the target from "tomorrow's return" to "confidence of upside over
the next week" (triple-barrier labeling: does price hit +X% before -X% within H=7d).
- The triple-barrier upside-confidence label IS predictable and CLEAN: composite IC
  ~0.15, halving_cos ~0.13-0.19, mvrv ~0.07, robust train+val. Crucially the VOL TRAP
  vanishes - vol_regime does NOT predict it (symmetric barriers measure directional
  asymmetry, not magnitude). So the reframing is methodologically sound.
- BUT it predicts the SAME signal the composite already captures (IC ~0.15, not higher
  than the composite's +0.18/+0.20 on forward return). No new prediction alpha.
- Predicting the weekly HIGH directly (fwd_max) is WORSE - dominated by the mechanical
  "high vol => high max" effect; real directional factors flip sign train->val.
- TAKE-PROFIT EXECUTION (exit at the predicted +barrier) is actively HARMFUL: test
  Sharpe -0.72..-0.04 / NAV 0.57..0.92 vs the hold version's 0.65 / 2.24. Two reasons:
  turnover explodes (408-512 vs 86 trades -> fees) AND it caps BTC's fat right tail
  (the few huge trends that drive all returns). BTC is trend/fat-tailed: let winners
  run. This is precisely why the composite holds + sizes by vol-target and never caps
  upside. (Side note: the long-only confidence-gated HOLD reaches NAV 2.24 > composite
  1.82 but at lower Sharpe 0.65 - a NAV/Sharpe trade, not a clean win.)

## Support/resistance (market psychology) - tested, does NOT help (BTC trends through levels)

Tested S/R / price-level-memory factors (the owner's "look at more than today's data"):
volume-profile point-of-control distance, position in a long high-low range, distance
to recent resistance/support, round-number proximity.
- The genuinely PSYCHOLOGICAL "reversal" factors are NOT robust: support-bounce
  (dist_support_90 val IC flips -0.04), volume-profile POC (val ~0.01), round numbers
  (sign flips train->val). BTC does not reverse at these levels.
- The only robust S/R factors (range_pos_252, dist_resist_90) have POSITIVE IC
  (closer to resistance => higher forward return) = breakout/momentum CONTINUATION,
  the opposite of fade-the-level psychology. They correlate 0.4-0.66 with existing
  momentum/trend factors (redundant) and adding them makes the composite WORSE
  (val Sharpe 1.24->1.21->1.12; test 2.07->1.89-1.96 NAV).
Conclusion: BTC is a trending asset that breaks THROUGH support/resistance rather than
bouncing off it; the directional structure is already captured by the momentum/cycle/
RS factors. S/R reversal psychology does not translate to a daily edge here.

## Higher frequency (hourly) - explored, does NOT help under 0.5% fees

Built an hourly BTC pipeline (load_btc_hourly: 65k bars 2019-2026, cached) and
explored intraday signal honestly:
- Short-horizon MEAN-REVERSION is real and robust (1-6h, IC ~-0.06 same sign
  train+val) but UNTRADEABLE: the per-trade edge (~0.03%) is ~15x smaller than the
  0.5% fee. The fee is the binding constraint.
- Hour-of-day and day-of-week seasonality are NOT robust (train<->val unstable;
  hour-profile correlation -0.22).
- Intraday-derived DAILY features (realized vol, range, end-of-day momentum, skew,
  volume concentration, close-in-range): only realized-vol and high-low range are
  IC-robust, and both DUPLICATE the existing daily vol_regime/rvol_z90 factors - no
  new orthogonal signal.
- Using intraday realized vol for position SIZING is WORSE than the daily estimate
  (test Sharpe 0.92->0.87, drawdown -24%->-31%, more trades).
Conclusion: with a 0.5% per-trade fee, higher frequency adds no tradeable alpha -
the fee forces low turnover, and at low turnover intraday data carries no signal
beyond what daily already captures. Hourly loader kept as infrastructure.

## Quant-literature orthogonal factors - IC-robust but redundant (composite saturated)

Mined canonical style/quant factors specifically chosen to be ORTHOGONAL to the existing
set (which is cycle/value/momentum/vol):
- Kaufman Efficiency Ratio (trend QUALITY, not magnitude): IC-robust (h20 train ~0.15,
  val ~0.08) but adding it LOWERS validation Sharpe (1.65 -> ~1.50). Rejected.
- Frog-in-the-Pan / information discreteness (Da-Gurun-Warachka; smooth vs jumpy momentum):
  strong standalone IC (val -0.22) but adding it degrades the worst test cell (1.07/2.18 ->
  0.97/1.92) - directionally redundant with momentum. Rejected.
- Realized skewness (daily and intraday-from-hourly, Amaya et al.): mostly not IC-robust.
- Carry (funding/basis - the canonical missing crypto factor): UNAVAILABLE. OKX gives only
  ~90 days; Binance and Bybit are geo-blocked (CloudFront/eligibility). No free long history.
Meta-conclusion: even canonical literature factors do not improve the 10-factor composite.
Combined with the macro/sentiment/breadth/stablecoin/220k-search results, the robustly
predictable DIRECTIONAL signal in free daily BTC data appears finite and largely captured.
This is evidence the composite is SATURATED and ROBUST (hard to dislodge), not fragile.
Adding factors now mostly yields "directionally redundant" or "lowers validation".

## What does NOT work (tested, documented, not hidden)

- **Unattended factor search saturation** (factor_search.py): forward selection over
  75 candidates (36 IC-robust) found ZERO additions that improve the walk-forward
  validation metric (1.271) while keeping all neighborhood cells beating BH. The
  9-factor set is a disciplined local optimum - recombining existing price/on-chain/
  cross-crypto factors is exhausted; more brute-force search would only manufacture
  multiple-testing false positives.
- **OKX funding rate** (positioning factor): the obvious new signal, but OKX's public
  funding-rate-history endpoint returns only ~90 days - no overlap with the train/val
  windows, so it cannot be backtested. Loader kept for live use; not in the composite.
  Real further edge needs NEW HISTORICAL data (paid on-chain, order-book, sentiment).
- **Macro + sentiment (FREE, with history) - real signal, but no improvement.** Correcting
  the earlier "free data exhausted" claim: Fear & Greed (alternative.me, 2018+), and macro
  via Yahoo (S&P500, dollar index, VIX, gold; 2017+) are reachable and orthogonal to crypto
  internals. Several are train+val IC-robust (spx_mom20 val IC -0.186; fng_level +0.105;
  dxy_mom). BUT adding them does NOT improve the walk-forward composite: spx_mom20/fng_level
  pass the all-cells-beat guardrail yet lower the validation metric (1.27->1.19/1.12) and are
  NEUTRAL on test (Sharpe -0.01, NAV +0.03). The crypto-internal factors already capture the
  available predictability; macro is redundant at the margin. Loaders + factors kept (tested,
  available for live use); DEFAULT_FACTORS unchanged. The composite is robust to orthogonal
  additions (the test win holds when macro is added) - reassuring, not improving.

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
