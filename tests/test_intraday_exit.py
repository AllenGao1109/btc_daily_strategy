"""Intraday take-profit / stop-loss exits using daily High/Low."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import run_backtest
from tests.conftest import make_signal


def _df(rows, start="2020-01-01"):
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    idx = pd.date_range(start, periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    df.index.name = "date"
    return df


def test_long_stop_loss_fills_at_level():
    # Open long at day0 close=100 (entry basis = prev_close = 100). Day1 dips to
    # low=90 (-10%) intraday before closing at 96. With a 5% stop, we exit at 95.
    df = _df([(100, 100, 100, 100), (96, 99, 90, 96)])
    raw = make_signal([1.0, 1.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0,
                       stop_loss_pct=0.05)
    assert bool(res["intraday_exit"].iloc[1])
    assert res["exposure_notional_end"].iloc[1] == 0.0
    # Realized return is the stop fill (-5%), not the close (-4%).
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(-0.05, abs=1e-9)


def test_short_take_profit_fills_at_level():
    # Short at day0 close=100. Day1 falls to low=88 intraday; 10% take-profit
    # closes the short at 90 for a +10% gain even though it closes at 92.
    df = _df([(100, 100, 100, 100), (98, 99, 88, 92)])
    raw = make_signal([-1.0, -1.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0,
                       take_profit_pct=0.10)
    assert bool(res["intraday_exit"].iloc[1])
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(0.10, abs=1e-9)


def test_no_touch_matches_close_to_close():
    # Neither level touched -> behaves exactly like the close-to-close engine.
    df = _df([(100, 100, 100, 100), (101, 103, 99, 102)])
    raw = make_signal([1.0, 1.0], df)
    base = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)
    with_levels = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0,
                               take_profit_pct=0.20, stop_loss_pct=0.20)
    assert not with_levels["intraday_exit"].any()
    assert with_levels["equity_end"].iloc[-1] == pytest.approx(
        base["equity_end"].iloc[-1], abs=1e-9
    )


def test_stop_checked_before_target_when_both_touched():
    # Day1 spans low=90 and high=112: both a 5% stop (95) and 10% target (110)
    # are touched. The stop is assumed to fill first (conservative) -> loss.
    df = _df([(100, 100, 100, 100), (100, 112, 90, 100)])
    raw = make_signal([1.0, 1.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0,
                       take_profit_pct=0.10, stop_loss_pct=0.05)
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(-0.05, abs=1e-9)
