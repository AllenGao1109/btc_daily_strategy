"""20.7-20.10 long/short and leveraged PnL (before costs)."""

from __future__ import annotations

import pytest

from src.backtest import run_backtest
from tests.conftest import make_df, make_signal


def _run(closes, weight):
    df = make_df(closes)
    raw = make_signal([weight] * len(closes), df)
    return run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)


def test_short_profits_when_btc_falls():
    res = _run([100, 90], weight=-1.0)  # -10% day
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(0.10, abs=1e-9)
    assert res["equity_end"].iloc[1] == pytest.approx(11000.0, abs=1e-6)


def test_short_loses_when_btc_rises():
    res = _run([100, 110], weight=-1.0)  # +10% day
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(-0.10, abs=1e-9)
    assert res["equity_end"].iloc[1] == pytest.approx(9000.0, abs=1e-6)


def test_2x_long_doubles_return():
    res = _run([100, 105], weight=2.0)  # +5% day -> +10%
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(0.10, abs=1e-9)


def test_2x_short_gains_on_down_move():
    res = _run([100, 95], weight=-2.0)  # -5% day -> +10%
    assert res["strategy_daily_return"].iloc[1] == pytest.approx(0.10, abs=1e-9)


def test_no_fee_vs_fee_monotonic():
    df = make_df([100, 105, 95, 110])
    raw = make_signal([1, 1, 1, 1], df)
    res_nofee = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)
    res_fee = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    assert res_fee["equity_end"].iloc[-1] <= res_nofee["equity_end"].iloc[-1]
