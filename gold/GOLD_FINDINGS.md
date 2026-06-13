# Gold (GC=F) strategy research — honest conclusions

Literature-grounded search across 6 strategy families (mean-reversion, momentum/trend,
macro/real-rates, risk-regime/hedge, seasonality, value/cross-asset), each researched
against the academic/practitioner literature, implemented, and backtested vs gold
buy-and-hold (0.2%/side fee, chronological train 2000-2013 / val 2013-2019 / test
2019-2026). Every signal that beat buy-and-hold OOS was then adversarially verified.

## Headline: nothing robustly beats simply HOLDING gold out-of-sample.

Gold buy-and-hold: full-sample Sharpe ~0.69, NAV ~15x (2000-2026). Every family failed
to beat it robustly. All three "winners" were REFUTED on verification:
- Risk-regime vol-target "boost": the safe-haven (SPX<200dma) boost is value-destroying
  OOS (monotonic: more boost -> worse test); only generic vol-targeting nudged the
  Sharpe, at LOWER NAV and WORSE drawdown than buy-and-hold.
- Gold/commodities relative-strength (gold/SPGSCI>MA200): genuinely robust to params/
  fees/proxy, but the "win" is concentrated in the 2019+ test bull; it LOSES on train
  and full-sample. A trend filter dressed as value, not risk-adjusted alpha.
- Combined RS breadth: loses full-sample (0.54 vs 0.69), "win" only on cherry-picked test.

## Why gold resists timing (recurring across all families)
1. 2019-2026 has been a strong, smooth bull (Sharpe ~1.0) - any timing that sits in cash
   forfeits return and cannot out-Sharpe a full-time long.
2. 2013-2018 was a CHOPPY bear - it whipsaws trend/momentum systems (val Sharpe -0.35..-1.0).
3. Erb & Harvey "The Golden Dilemma" (NBER w18706, 2013): gold has ~zero real long-run
   return and mean-reverts toward a value anchor over DECADES, not tradeable horizons.
   The value sign (sell when expensive) shorts the secular bull and loses.
4. Macro drivers (real rates, dollar) are CONTEMPORANEOUS, not next-day predictive.
5. Seasonality (Baur 2012 autumn effect; Halloween/winter) is real but fragile and
   defensive; day-of-week is a fee trap (2400+ trades).

## The only defensible use: DEFENSE, not alpha
Several signals (rate-regime gate, Halloween-long, gold/commodities RS) reduce drawdown
~5-40% with low turnover, but at lower NAV and no robust Sharpe gain vs holding gold.

## Verdict
Gold is best HELD, not timed - the opposite of BTC (whose huge drawdowns/inefficiency
reward vol-targeting + factor timing). The productive use of gold here is as a low-
correlation DIVERSIFIER in a BTC portfolio, not a standalone timed strategy.

Literature: Erb & Harvey NBER w18706; Baur 2012 "Seasonality of Gold - Autumn Effect"
(SSRN 1989593); Moskowitz-Ooi-Pedersen TSMOM; In Gold We Trust calendar anomalies;
CFTC disaggregated COT (Managed-Money, 2006-2026). Scripts: gold/*.py.
