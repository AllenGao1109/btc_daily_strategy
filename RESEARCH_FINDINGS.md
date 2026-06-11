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
