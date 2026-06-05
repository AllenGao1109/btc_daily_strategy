# BTC Daily Long/Short Leveraged Strategy Research Harness

A reproducible, leakage-free research framework for backtesting daily-frequency
BTC long/short/leveraged strategies.

> **Research and backtesting only.** This project does not place real orders,
> does not connect to any exchange account, and is not a live trading bot. The
> first milestone is a *correct* harness, not a profitable strategy.

## What it does

- Loads daily BTC OHLCV data (UTC), with deterministic caching and validation.
- Computes timing-correct features and applies an explicit 1-day execution lag
  so day `t` positions only use information available through day `t-1`.
- Runs a stateful long/short/leveraged backtest engine with proper transaction
  costs (0.5% on traded notional only), leverage drift tracking, leverage-breach
  handling, optional funding/borrow costs, and equity-zero liquidation.
- Computes a full metric suite and generates a markdown report plus charts,
  comparing every strategy against cash, buy-and-hold 1x/2x, and SMA baselines.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended. Core dependencies: pandas, numpy, matplotlib, pyyaml,
pytest. `ccxt` (data download) and `scipy` are optional.

## Data source

By default the loader uses `ccxt` (`coinbase`, `BTC/USD`, `1d`) and caches raw
data in `data/raw/` and cleaned data in `data/processed/`. If `ccxt` is not
installed or you are offline, you can:

- place your own OHLCV CSV in `data/raw/BTC-USD_1d.csv`, or
- run with deterministic synthetic data via `--synthetic` (for testing only —
  this is **not** real market data).

## How to run the backtest

```bash
# Use the strategy named in config.yaml against all benchmarks:
python -m src.run_backtest --config config.yaml

# Override the strategy:
python -m src.run_backtest --config config.yaml --strategy momentum_long_short

# Offline / CI with synthetic data:
python -m src.run_backtest --config config.yaml --synthetic
```

## How to run the tests

```bash
pytest
```

The suite covers no-lookahead, signal lag, position persistence, transaction
costs (including partial rebalance and sign-flip), long/short and leveraged PnL,
max-leverage clipping, leverage-breach delever, equity-zero liquidation, metric
correctness, determinism, and validation splits.

## How to generate a report

The report is generated automatically by `run_backtest`. It is written to
`reports/<timestamp>_<strategy>_report.md` with charts under `reports/figures/`.

## Assumptions

### Strategy
- Strategies emit a *target BTC exposure weight* in `[-2.0, +2.0]` only. They do
  not compute PnL, charge fees, touch equity, or see future rows. All execution
  mechanics live in the backtest engine.

### Long/short and leverage
- `+2.0 = 200% long`, `0.0 = flat`, `-1.0 = 100% short`, `-2.0 = 200% short`.
- Equity-based notional accounting; `pnl = exposure_notional * btc_return`.
  Shorts profit when BTC falls. Max absolute (gross) leverage is **2.0x**.
- Leverage drifts with price between rebalances. Breaches above 2.0x are flagged
  and, by default, delevered on the next rebalance (`delever_next_day`);
  alternatively `liquidate`.
- Default rebalance policy `signal_change_or_risk_control`: trade only when the
  lagged target weight changes or risk control fires. Optional
  `daily_target_rebalance` rebalances to target every day (and pays drift fees).

### Transaction costs
- `fee = abs(target_exposure_notional - previous_exposure_notional) * 0.005`.
  Charged only on traded notional, never on the full portfolio, never on hold
  days. A `+2.0 -> -2.0` flip is a 4x-turnover trade.

### Funding / borrow costs
- Configurable daily rates, **default 0.0**. When 0.0, financing is effectively
  excluded; the report states this explicitly. Transaction fees are always
  charged regardless.

## Known limitations

- **Intraday liquidation risk is not modeled.** The backtest is daily
  close-to-close; real leveraged positions could be liquidated intraday even if
  the daily-close backtest survives.
- Funding/borrow rates default to 0.0, so short/leverage financing is not priced
  unless you configure rates.
- The provided strategies are baselines for testing the harness, not validated
  profitable strategies. Do not optimize parameters on the test set.
- Synthetic data is a seeded random walk for offline/CI use only.

## Project layout

```
src/            config, data, features, strategies/, backtest, metrics,
                validation, plotting, report, run_backtest
tests/          pytest suite (correctness, costs, leverage, liquidation, ...)
config.yaml     all knobs (exposure bounds, fees, policies, validation windows)
reports/        generated markdown reports and figures/
```

## Development order (followed)

structure → config → data → validation → features → strategy interface →
baseline strategies → backtest engine → costs → leverage/liquidation → metrics →
tests → report. Parameter search and walk-forward come only after the engine is
tested.
