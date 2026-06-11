# Research Findings — BTC Daily Strategy

Running log of honest out-of-sample conclusions. The governing rule: a strategy
is only "better" if it beats buy-and-hold robustly out-of-sample, selected
without peeking at the test window.

## Headline conclusion (as of the latest round)

**No strategy found so far robustly beats buy-and-hold on BOTH Sharpe and NAV.**
The vol-targeted trend ensemble offers genuinely lower drawdown (and higher
full-sample Sharpe), but it gives up NAV in bull markets, and on a year-by-year
basis buy-and-hold has the best risk-adjusted return.

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

## RESULT: CNN equity Fear & Greed improves the composite; crypto F&G does not

Tested the two sentiment indexes (this answers the "different data regime"
hypothesis below — sentiment was the predicted next frontier):
  - **alternative.me crypto Fear & Greed** (0-100 daily, 2018-02+)
  - **CNN Business equity Fear & Greed** (0-100 US trading days, 2011+), as a
    cross-asset risk-appetite proxy

Factor mining verdict (IC stability on train+val, test never used):
  - **`cnnfg_z_60` (CNN F&G 60d z-score) is robust on ALL 3 horizons** with a
    consistently NEGATIVE IC (h20: train -0.13 / val -0.20) — i.e. contrarian:
    stretched-high equity sentiment precedes weak BTC returns. It ranks 3rd of
    54 factors, behind only halving_cos and vol_regime, and it is *exogenous*
    (not derived from BTC price), hence orthogonal to the price/on-chain block.
  - **Every crypto F&G factor failed robustness** (train/val sign flips). As
    suspected, the crypto index is built largely from BTC price/volatility, so
    it adds nothing beyond the existing price factors.

Adding `cnnfg_z_60` to the production `factor_composite` (selection on
train+val only: val Sharpe 1.05 -> 1.12, train+val-years mean 0.98 -> 1.03;
test checked only afterwards):

| window (this data snapshot)   | PROD   | PROD + cnnfg_z_60 |
|-------------------------------|--------|-------------------|
| train Sharpe / NAV            | 0.96 / 4.37 | 0.98 / 4.66  |
| validation Sharpe / NAV       | 1.05 / 1.79 | 1.12 / 1.85  |
| test (2024-26) Sharpe / NAV   | 1.02 / 2.05 | **1.04 / 2.16** |
| trades                        | 86     | 84                |

Consistent improvement across all three windows with *fewer* trades —
`cnnfg_z_60` is now in `DEFAULT_FACTORS`. Caveats:
  - CNN history is community-archived (whit3rabbit/fear-greed-data splices the
    2011-2021 archive with the live CNN endpoint); pre-2021 values are a
    reconstruction.
  - The BTC-equity correlation regime that powers this factor is post-2020;
    monitor per-year IC for decay.
  - Crypto F&G archive ends ~2026-04 in this environment (API blocked); the
    last weeks are forward-filled. Irrelevant to the verdict (factor rejected).
  - This round ran on CoinMetrics mirror prices with synthesized OHLC (see
    `fetch_mirror_data.py`); PROD baseline numbers shift slightly vs the
    CryptoCompare snapshot (1.02/2.05 vs the 0.98/2.00 in config.yaml notes)
    but the comparison is internally consistent.

Band sensitivity (0.20 vs the production 0.30 no-trade band, same snapshot):

| band | strat  | train       | validation  | test        | trades | fees%cap | maxDD | worst yr |
|------|--------|-------------|-------------|-------------|--------|----------|-------|----------|
| 0.30 | PROD   | 0.96 / 4.37 | 1.05 / 1.79 | 1.02 / 2.05 | 86     | 96%      | -53%  | -1.57    |
| 0.30 | PROD+S | 0.98 / 4.66 | 1.12 / 1.85 | 1.04 / 2.16 | 84     | 113%     | -56%  | -1.54    |
| 0.20 | PROD   | 1.12 / 5.34 | 1.14 / 1.87 | 1.07 / 2.10 | 160    | 177%     | -40%  | -1.53    |
| 0.20 | PROD+S | 1.09 / 5.20 | 1.27 / 2.09 | 0.92 / 1.88 | 167    | 186%     | -47%  | -0.94    |

At 0.20 the sentiment composite posts the best validation (1.27/2.09), the
best worst-year (2022: -0.94 — the contrarian equity-sentiment de-risking
bites harder with a tighter band) and the best mean-yearly Sharpe (0.91), but
test softens to 0.92/1.88 (still beats BH on both) at double the turnover.
Keeping band 0.30: the fee prior (0.5%/trade) is an a-priori reason to prefer
half the trades, and test agrees afterwards. PROD+S beats BH on Sharpe AND NAV
at BOTH bands — the sentiment conclusion is not band-sensitive. The 0.20+S
configuration is the designated alternate if the fee assumption ever drops
(e.g. 0.1% maker-taker), on the strength of its worst-year/consistency profile.

## ROUND: miner / exchange-supply / stablecoin / macro factors — ZERO adopted

Mined four new free, industry-recognized factor families (73 factors total now,
26 pass the train+val IC gate). Loaders: `src/macro.py` (FRED + DXY, with
release-lag re-stamping — H.15 yields and HY OAS are published t+1 and get an
extra day of lag to avoid a ~21h leak), `load_stablecoin_mcap` (USDT+USDC),
ONCHAIN_METRICS extended with HashRate/IssTotUSD/SplyExNtv/SplyCur.

IC-gate results (train+val sign stability):
  - PASSED: exsply_ratio (3H, score 0.283 — 2nd best overall), exsply_z_180
    (3H), stable_growth_30/90, dgs10_chg_60, rrp_chg_30, exsply_chg_30.
  - FAILED: hash ribbons, Puell multiple (famous, but val sign flips), SSR,
    VIX level/z, 10y-2y curve, DXY momentum (1H only).
  - NOT TESTABLE: HY OAS — the mirror archive only covers 2023+, zero train
    coverage. (NUPL was skipped a priori: 1 - 1/MVRV is a monotonic transform
    of MVRV, identical Spearman IC.)

Composite marginals looked spectacular on the selectable windows — every
passing factor lifted validation Sharpe (1.12 -> up to 1.49) and repaired the
2022 worst-year (-1.54 -> as good as +0.40). **All of it was 2022-23 regime
fitting.** The factor-level OOS diagnostic (h20 IC, computed for reporting
only, after deciding not to swap the default) showed every candidate's sign
FLIPS in 2024-26:

| factor            | train 17-21 | val 22-23 | test 24-26 |
|-------------------|-------------|-----------|------------|
| exsply_ratio      | -0.14       | -0.35     | **+0.19**  |
| stable_growth_30  | +0.21       | +0.22     | **-0.11**  |
| dgs10_chg_60      | -0.08       | -0.17     | **+0.11**  |
| rrp_chg_30        | -0.10       | -0.24     | +0.02      |

Economic post-mortem: all four key off 2022 events (hiking cycle, Terra/FTX).
The exchange-supply story structurally broke in the ETF era — since 2024-01,
coins leaving exchanges flow into ETF custodians, inverting the old
"self-custody = bullish" reading. Verdict: **DEFAULT_FACTORS unchanged.**
The lesson compounds the ML one: an IC gate on train+val windows is necessary
but not sufficient when train+val contains one dominant macro regime; prefer
factors whose yearly IC is broad-based, not event-concentrated.

Monitoring note: even adopted factors show 2024-25 decay (cnnfg_z_60 test-
window IC +0.08 vs -0.20 in val; 2026 back to -0.17). Re-run the yearly-IC
diagnostic as test years accumulate.

## CONFIG CHANGE (owner, 2026-06): fee 0.5% -> 0.2%; band re-selected to 0.20

With the fee assumption halved-plus (0.5% -> 0.2% per side, closer to real
spot taker tiers), the no-trade band was re-selected on train+val only
(grid: bands 0.30/0.20/0.10 x {BH, ENS, PROD, PROD+S} at fee 0.002):

| band | strat  | validation  | tv-mean | trades | test (after selection) |
|------|--------|-------------|---------|--------|------------------------|
| 0.30 | PROD+S | 1.15 / 1.90 | 1.06    | 84     | 1.07 / 2.22            |
| 0.20 | PROD+S | **1.31 / 2.15** | 1.32 | 167    | **0.97 / 1.96** (selected) |
| 0.10 | PROD+S | 1.25 / 2.06 | 1.39    | 457    | 1.02 / 2.06            |

Selection: 0.20 band (best validation Sharpe; 0.10's higher tv-mean loses the
fee-prior tie-break at ~3x the trades). Test confirms the selected config
beats buy-and-hold on both Sharpe and NAV (0.97/1.96 vs 0.76/1.81). On
record: 0.30's test (1.07/2.22) was again higher — the 2024-26 window keeps
rewarding slower trading. Acting on that observation would be test-peeking;
instead it is visible in the monitor: under the new config the trailing-365d
edge reads +0.00 (vs +0.18 for the old config), i.e. the selected config is
flat vs BH over the last 12 months and closer to the tripwire's soft leg.
The trend ensemble is fee-disqualified even at 0.2% (159-408 trades, fees
442-677% of initial capital, val Sharpe <= PROD+S at every band).

## ROUND: credit / gold / semiconductors — all fail at the gate

Hypothesis batch #2 of the exogenous-factor campaign (the 2022-23 validation
window was a RATES-vol crisis, so credit and real-asset rotation seemed like
the right fear gauges). Full-history HY OAS restored via an archive mirror
(`load_hy_oas`, 1996+, release-lag adjusted) — closing the earlier
"not testable" gap with a definitive verdict:

  - **HY credit spread (hyoas_z_60 / chg_20): train/val SIGN FLIP** (h20:
    -0.10 train vs +0.15 val). Credit-risk appetite did not translate across
    crypto regimes. Previously "untestable", now tested and FAILED.
  - **Gold momentum / BTC-gold relative strength: sign flips too** (gld_mom
    -0.20 train vs +0.15 val). The debasement-rotation story does not survive
    the gate.
  - smh_rs_60 (semis vs QQQ): 1 horizon only, breadth 2+/2- — below the
    >=2-horizon bar every adopted factor met; not taken to the marginal test
    (no exceptions, that is multiplicity discipline).
  - MOVE index (the *right* vol for 2022): ICE-proprietary, no free
    full-history archive found — genuinely unavailable, not untested-by-laziness.

Exogenous campaign scoreboard after two hypothesis batches (yen carry /
ARKK / miner equities / gold / credit / semis): 8 mechanisms tested, 2 gate
passes (yen family), 0 adoptions. The owner's CNN Fear & Greed remains the
only adopted exogenous factor. The bar that keeps rejecting candidates is
not the IC gate — it is (a) family duplication against 9 incumbents and
(b) the validation-Sharpe marginal in the composite. Both are working as
designed.

## ROUND: exogenous cross-asset factors (yen carry / ARKK / miner equities)

Hypothesis-first round: BTC's marginal buyers leave footprints in other
markets first. Tested three mechanisms (loader `src/crossasset.py`; USDJPY
from the FRED mirror — market-observable at its stamp, so the weekly H.10
*release* lag is not an information lag; ARKK/QQQ/RIOT/MARA daily closes):

  - **Yen carry stress (jpy_vol_z_90): the best REJECTED factor so far.**
    Passes everything except the final criterion: gate 2 horizons (score
    0.234), breadth 4+/0- (broad, not event-fit), extra-lag clean, corr
    < 0.28 with every incumbent factor (genuinely new information). In the
    production blend it improves train (1.09 -> 1.32), full-sample MaxDD
    (-49% -> -38%) and test (1.08/2.17 -> 1.14/2.26, seen only after
    selection) — but validation-window Sharpe drops 1.47 -> 1.29, and
    validation Sharpe is the selection criterion every prior adoption used.
    Changing the rule after seeing the test column would be test-peeking by
    rule-shopping. NOT adopted; **first in line for re-test after any split
    roll.**
  - jpy_mom_60: passes the gate but corr 0.70 with dxy_mom_60 (same
    dollar-vs-funding-currency family) — excluded as duplication.
  - ARKK-vs-QQQ speculative appetite: 1 horizon only, breadth 1+/2- — weak,
    rejected.
  - Miner-equity relative strength vs BTC (RIOT/MARA): train/val sign flip —
    the "stock market prices miners ahead of BTC" hypothesis is falsified.

## ROUND: on-chain behavior factors (SOPR / CVD / whale / miner / basis) — ZERO adopted

Literature sweep pointed at behavior/microstructure data as the remaining
free, untested dimension. New source: CryptoQuant CSV archive + CME basis
(public research-repo mirror, all series with pre-2022 coverage; loader
`src/cryptoquant.py`, 12 new factors, 88 total). Leakage controls added for
this round and now permanent:
  - EXTRA-LAG check in factor_mining: every gate-passing factor's h20 IC is
    recomputed with one additional day of lag; a collapse (>50% drop or sign
    flip) flags publication-timing leakage. This round: all 35 gate-passers
    clean — none lives off borderline same-day information.
  - Documented vendor-revision caveat: entity-based series (whale, miner,
    exchange flows) are recomputed with today's wallet labels, biasing
    historical ICs optimistically. (The two failures below make the point
    moot here.)

Gate results: lth_sopr_z_365 PASSES strongly (3 horizons, score 0.281, 3rd
overall; slow-cycle breadth profile like halving/MVRV), cvd_chg_30 passes
(2 horizons, breadth 3+/1-). btc_dominance_mom passes but is the same
mechanism as the adopted btc_eth_rs_30 (corr ~1) — excluded as family
duplication. FAILED: whale ratio (train/val mismatch), miner-to-exchange
flow (dead), aSOPR level, CME basis (negligible IC).

Composite marginals (selection on train+val, production blend):
  - +lth_sopr_z_365: validation COLLAPSES 1.47 -> 1.05, worst-year -0.52 ->
    -1.85. Cause: corr 0.79 with mvrv_z_365 double-weights the valuation
    family, and its positive sign leans into late-cycle distribution — 2022
    punishes it. A factor can have top-3 standalone IC and still be net
    harmful inside the composite.
  - +cvd_chg_30: no marginal value (val 1.40 vs 1.47; corr 0.61 with mom_20
    — the order-flow information is already priced into the momentum block).

**DEFAULT_FACTORS unchanged.** Two rounds in a row the layered pipeline
(IC gate -> breadth -> extra-lag -> composite marginal on train+val) has
correctly rejected everything; standalone factor IC without mechanism
novelty is not enough. Remaining untested-for-coverage reasons: NUPL/Puell/
dormancy families, DVOL, stablecoin exchange ratio, ETF flows (all start
2020-12+; parked until a split roll).

## Statistical significance of the edge (honest sizing of the claim)

`significance.py` (paired circular block bootstrap, B=10k, fixed seed; plus
deflated Sharpe over an assumption grid), at the current config:

  - PROD+S vs buy-and-hold Sharpe edge: validation +1.12, 95% CI [+0.01,
    +2.18] — but validation is the selection window, so this significance is
    contaminated by construction. The clean windows: **test +0.21, CI
    [-0.41, +0.84], p(<=0)=0.25; full-sample +0.10, CI [-0.54, +0.69]**.
    The out-of-sample edge is positive but statistically indistinguishable
    from zero on ~2.4 years of daily data.
  - The sentiment factor's marginal (PROD+S vs PROD): not significant in any
    window; at the current config its test-window contribution is -0.15
    (val +0.13). Kept by the selection rule; on watch.
  - Deflated Sharpe of the test result (SR 0.97, skew 0.74, kurt 10):
    P[true skill] ranges **0.13 (N=1500 trials, wide null) to 0.63 (N=100,
    tight null)**. After everything this project has tried, the honest
    statement is: the strategy is *consistent with* skill, not *evidence of*
    skill.

Every "beats buy-and-hold" claim in this file now carries this caveat. What
the strategy DOES robustly deliver is the risk profile (test MaxDD -31% to
-49% vs BH -84% full-sample), which is a portfolio-construction property,
not a forecasting claim.

## RESULT: strategy-level blend adopted — composite 75% / trend-ensemble 25%

With factor mining exhausted, the remaining diversification was at the
strategy layer. Blending RAW target weights (engine trades the netted blend)
of the two mechanically different survivors, selected on train+val only at
fee 0.2% / band 0.20:

| blend          | validation  | tv-mean | minYr | trades | test (after) |
|----------------|-------------|---------|-------|--------|--------------|
| composite only | 1.31 / 2.15 | 1.32    | -0.88 | 167    | 0.97 / 1.96  |
| **C75/E25**    | **1.47 / 2.32** | 1.37 | -0.52 | **145** | **1.08 / 2.17** |
| C50/E50        | 1.21 / 1.97 | 1.15    | -1.91 | 164    | 0.96 / 1.97  |
| any BH blend   | <= 0.88     |         |       |        |              |

C75/E25 improves validation, worst-year AND trade count simultaneously —
weight averaging nets opposing trades, so the diversification is better than
free. Test (checked after selection) agrees: 1.08/2.17. Adopted as production
(`composite_blend`, blend=0.75). BH-containing blends die on 2022 validation.
The significance caveats above apply unchanged — the blend's improvement over
the composite alone is well inside the bootstrap noise band; the adoption
rationale is the selection rule + the diversification prior + lower turnover,
not a significance claim.

## Standing tools: breadth gate + adopted-factor decay monitor

Implemented the two process fixes from the round above:
  - `factor_mining.py` now reports yearly-IC **breadth** (yr_support/yr_oppose
    over 2019-2023 at h20). It would have flagged dgs10_chg_60 (1+/3-) and
    rrp_chg_30 (3+/2-) BEFORE any composite test. Limits: exsply_ratio scores
    a clean 3+/0- — breadth cannot catch structural regime breaks (ETF era),
    only event concentration; and slow cycle factors (halving_cos 2+/2-,
    mvrv_z_365 1+/3-) legitimately score poorly within years — judged by
    mechanism, not auto-rejected.
  - `factor_monitor.py` — standing decay monitor for DEFAULT_FACTORS
    (reporting only, never for selection). First run (test = 2024-01 ->
    2026-05): **5/9 adopted factors flagged** — vol_regime, mom_120,
    cnnfg_z_60 DECAYED (test-window sign flip); kurt_30, btc_eth_rs_30 WEAK
    (|IC| < 0.03). halving_cos, mvrv_z_365, mvrv_mom_30, ex_netflow_to_mcap
    hold. The composite still beats BH on test (1.04/2.16 vs 0.76/1.81) on
    the strength of the holders + IC-weighting + vol targeting, but the
    factor base is eroding in the post-ETF regime.

DECISION (2026-06, delegated to and taken by the research agent): **HOLD —
do not roll the split.** Evidence: the composite's edge is diluted, not dead.
Trailing-365d Sharpe edge over buy-and-hold across the test window: mean
+0.06, currently **+0.18**, never below -0.26; test MaxDD -31% vs BH -49%;
in the 2026 drawdown the composite is losing materially less (-0.42 vs
-0.60 trailing Sharpe) — the downside-protection profile it was selected
for is delivering. Rolling now would consume 2024-25 and leave only ~5
months of clean out-of-sample. Factor-IC fatigue alone does not justify
that trade.

To keep this from becoming indefinite discretion, the roll condition is
PRE-REGISTERED in `factor_monitor.py` (decided while ahead, not in a
drawdown panic):
  ROLL if trailing-365d Sharpe edge < -0.30, or edge < 0 with >=5 factors
  flagged — on two monitor runs >= 60 days apart (run log committed at
  results/factor_monitor_log.csv). Until it fires, no "what would 2024-25
  select" analysis is run at all — looking is consuming.
First run: edge +0.18, 5/9 flags -> HOLD.

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
