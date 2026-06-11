"""Research workbench: evaluate strategies out-of-sample across windows.

This module is the iteration harness used to *improve* strategies responsibly:
  - One full-history backtest per strategy (correct warmup, persistent state).
  - Per-window metrics are derived by slicing that single run and rebasing the
    equity curve to the window start. Sharpe/Sortino/vol use the daily returns
    directly (scale-invariant), so no warmup or equity-reset artifacts.
  - The TEST window is reported but must never be used to choose parameters.

Nothing here looks ahead: signals are generated on full history and lagged by
the engine; slicing a window only selects rows, it does not change any signal.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from . import metrics as metrics_mod
from .backtest import run_backtest
from .config import BacktestConfig
from .validation import SplitWindow

SignalBuilder = Callable[[pd.DataFrame], pd.Series]


def load_research_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Load the fully-enriched research frame: OHLCV + features + all merges.

    One canonical place for the data assembly that every research script needs
    (on-chain, sentiment, stablecoin, macro, DXY, ETH close), so scripts cannot
    drift apart in what they merge. All merges are causal (ffill of past values
    via merge_onchain) and the engine still applies the execution lag.
    """
    from .cryptoquant import load_cme_basis, load_cryptoquant
    from .data import load_btc_data
    from .features import build_features
    from .macro import load_dxy, load_fred_macro
    from .onchain import (
        ONCHAIN_METRICS,
        load_coinmetrics,
        load_stablecoin_mcap,
        merge_onchain,
    )
    from .sentiment import load_cnn_fear_greed, load_crypto_fear_greed

    df = build_features(load_btc_data(config))
    df = merge_onchain(df, load_coinmetrics(ONCHAIN_METRICS))
    df = merge_onchain(df, load_crypto_fear_greed())
    df = merge_onchain(df, load_cnn_fear_greed())
    df = merge_onchain(df, load_stablecoin_mcap())
    df = merge_onchain(df, load_fred_macro())
    df = merge_onchain(df, load_dxy())
    df = merge_onchain(df, load_cryptoquant())
    df = merge_onchain(df, load_cme_basis())
    eth = load_coinmetrics(["PriceUSD"], asset="eth").rename(
        columns={"PriceUSD": "eth_close"}
    )
    return merge_onchain(df, eth)


def run_full(
    df: pd.DataFrame, raw_signal: pd.Series, bt: BacktestConfig
) -> pd.DataFrame:
    """Run one full-history backtest with the given config."""
    return run_backtest(
        df,
        raw_signal,
        initial_capital=bt.initial_capital,
        fee_rate=bt.fee_rate,
        slippage_rate=bt.slippage_rate,
        execution_lag_days=bt.execution_lag_days,
        min_weight=bt.min_weight,
        max_weight=bt.max_weight,
        max_leverage=bt.max_leverage,
        rebalance_policy=bt.rebalance_policy,
        leverage_breach_action=bt.leverage_breach_action,
        funding_config=bt.funding_config,
        weight_band=bt.weight_band,
    )


def window_metrics(
    result: pd.DataFrame,
    window: SplitWindow | None = None,
    periods_per_year: int = 365,
) -> dict[str, Any]:
    """Compute window metrics from a full-history result, rebased to the window.

    Args:
        result: Full-history daily state table.
        window: Optional window to restrict to; None uses the whole series.
        periods_per_year: Annualization factor.

    Returns:
        A metrics dict for the window (return-based, rebased equity).
    """
    sub = window.slice(result) if window is not None else result
    if len(sub) == 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown": 0.0, "calmar_ratio": 0.0, "num_trades": 0,
                "n_days": 0, "final_nav": 1.0, "avg_gross_leverage": 0.0,
                "exposure_pct": 0.0, "annualized_volatility": 0.0,
                "sortino_ratio": 0.0, "total_fees_paid": 0.0}

    rets = sub["strategy_daily_return"].astype(float)
    # Rebased NAV starting at 1.0 the day before the window.
    nav = (1.0 + rets).cumprod()

    cagr = metrics_mod.cagr(nav, 1.0, periods_per_year)
    max_dd = metrics_mod.max_drawdown(nav)
    return {
        "total_return": float(nav.iloc[-1] - 1.0),
        "final_nav": float(nav.iloc[-1]),
        "cagr": cagr,
        "annualized_volatility": metrics_mod.annualized_volatility(rets, periods_per_year),
        "sharpe_ratio": metrics_mod.sharpe_ratio(rets, periods_per_year),
        "sortino_ratio": metrics_mod.sortino_ratio(rets, periods_per_year),
        "max_drawdown": max_dd,
        "calmar_ratio": metrics_mod.calmar_ratio(cagr, max_dd),
        "num_trades": int((sub["trade_notional"] > 0).sum()),
        "total_fees_paid": float(sub["fee"].sum()),
        "avg_gross_leverage": float(sub["gross_leverage_end"].mean()),
        "exposure_pct": float((sub["exposure_notional_end"].abs() > 0).mean()),
        "n_days": int(len(sub)),
    }


def evaluate(
    df: pd.DataFrame,
    signals: dict[str, pd.Series],
    bt: BacktestConfig,
    splits: dict[str, SplitWindow],
) -> pd.DataFrame:
    """Backtest each named signal once and tabulate per-window metrics.

    Args:
        df: Feature-augmented OHLCV frame.
        signals: Mapping of strategy label -> raw signal Series.
        bt: Backtest config.
        splits: Mapping of window name -> SplitWindow (e.g. train/validation/test).

    Returns:
        A tidy DataFrame with one row per (strategy, window) and metric columns.
    """
    rows: list[dict[str, Any]] = []
    for label, raw in signals.items():
        result = run_full(df, raw, bt)
        for wname, win in splits.items():
            m = window_metrics(result, win)
            m.update({"strategy": label, "window": wname})
            rows.append(m)
    cols = ["strategy", "window", "final_nav", "total_return", "cagr",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "annualized_volatility", "avg_gross_leverage", "exposure_pct",
            "num_trades", "n_days"]
    out = pd.DataFrame(rows)
    return out[cols]


def print_table(table: pd.DataFrame, windows: list[str] | None = None) -> None:
    """Pretty-print an evaluate() table grouped by window."""
    windows = windows or list(table["window"].unique())
    for w in windows:
        sub = table[table["window"] == w]
        if sub.empty:
            continue
        first = sub.iloc[0]
        print(f"\n=== {w}  ({first['n_days']} days) ===")
        print(f"{'strategy':16s} {'NAV':>7s} {'totRet':>9s} {'CAGR':>8s} "
              f"{'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>7s} "
              f"{'gLev':>5s} {'inMkt':>6s} {'trd':>5s}")
        for _, r in sub.iterrows():
            print(f"{r['strategy']:16s} {r['final_nav']:7.2f} "
                  f"{r['total_return']*100:8.1f}% {r['cagr']*100:7.1f}% "
                  f"{r['sharpe_ratio']:7.2f} {r['sortino_ratio']:8.2f} "
                  f"{r['max_drawdown']*100:7.1f}% {r['calmar_ratio']:7.2f} "
                  f"{r['avg_gross_leverage']:5.2f} {r['exposure_pct']*100:5.0f}% "
                  f"{int(r['num_trades']):5d}")
