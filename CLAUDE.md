# BTC Daily Long/Short Leveraged Strategy Research Harness

> This file is the governing spec for the project. The full, authoritative
> version was provided by the project owner. This copy captures the rules the
> code must obey. See README.md for usage.

## 0. Project Mission
Build a reproducible, leakage-free BTC daily-frequency trading strategy research
framework. **Research and backtesting only** — no live trading, no exchange order
execution, no real account. The first milestone is a correct research harness,
not a profitable strategy.

Priority order: correctness > no lookahead > reproducibility > transparent risk
reporting > transaction costs > long/short & leverage accounting > out-of-sample
validation > simple baselines > strategy improvement.

## 1. Trading Scope
- BTC only, daily, UTC. Margin/futures-style exposure, backtest only.
- Directions: long / flat / short. Max absolute leverage 2.0x.
- `target_weight` in [-2.0, +2.0]; `net_exposure = target_weight`,
  `gross_leverage = abs(target_weight)`. Clip to [-2, 2] and warn on clip.

## 2. No Lookahead (most important rule)
Day `t` position is decided by a signal using only data up to day `t-1`.
`target_weight_today = raw_signal.shift(1)`. Never use day-`t` close to decide a
day-`t` trade. Applies to long, short, leveraged, and flat alike.

## 3. Position Persistence
The portfolio is stateful; positions persist across days. Default policy
`signal_change_or_risk_control`: rebalance only if the target changes or risk
control (leverage breach) fires; otherwise carry the position forward. Actual
leverage can drift with price and must be detected/handled, not ignored.
Alternative: `daily_target_rebalance` (rebalances daily, pays drift fees).

## 4. Transaction Costs
`fee_rate = 0.005`, applied to traded notional only.
`turnover = abs(target_weight - actual_weight_before)`;
`traded_notional = turnover * equity_before_trade`; `fee = traded_notional * 0.005`.
No fee on unchanged positions. A `+2 -> -2` flip is 4x turnover. Main reported
performance includes fees.

## 5. Backtest Timing Model
Per day t: carry equity/exposure from t-1 -> set target from lagged signal ->
compare to actual weight before rebalance -> rebalance only if required ->
trade notional -> deduct fee/slippage/funding/borrow -> hold through day t ->
apply day-t BTC return -> update equity & exposure -> check leverage/liquidation
-> record full daily state -> carry into t+1.

## 6. Leveraged Long/Short Accounting
`actual_weight = exposure_notional / equity`; `gross_leverage = abs(actual_weight)`.
`btc_return = close_t / close_{t-1} - 1`. `pnl = exposure_after_rebalance * btc_return`
(shorts profit when BTC falls). `exposure_notional_end = exposure_after * (1 + btc_return)`.
`equity_end = equity_after_cost + pnl`. Equity <= 0 -> liquidate (equity/exposure 0,
stop trading, mark failure).

## 7. Leverage, Margin, Liquidation
`max_leverage = 2.0`. Track gross leverage at all times. Price-driven breaches:
default `delever_next_day` (flag + delever on next rebalance, pay fees); alt
`liquidate` (close immediately). The chosen behavior is written into the report.
First version: daily close-to-close only; intraday liquidation NOT modeled and
stated as such.

## 8. Funding and Borrow Cost
Configurable daily rates, default 0.0. Long borrow on notional above equity,
short borrow on abs short notional, optional perp funding on full abs notional.
Report states explicitly whether financing is included. Fees are always charged.

## 9-28. Structure, modules, strategies, tests, report
See README.md and the source tree. Required baselines: cash, buy-and-hold 1x,
buy-and-hold 2x, SMA long/flat, SMA long/short, momentum long/short, strong
trend leverage. Tests (pytest) cover no-lookahead, signal lag, persistence,
fees (incl. partial & sign-flip), long/short & leveraged PnL, max-leverage clip,
leverage breach, liquidation, metrics, determinism, validation. Validation is
chronological (train -> validation -> test); never tune on the test set.
Metrics annualize with 365 days/year.

## 29. Before coding
Restate: timing model, transaction-cost model, persistence model, long/short
accounting, leverage/liquidation assumptions, funding/borrow assumptions. Then
implement in the development order, run tests, and summarize build/assumptions/
passing tests/limitations.
