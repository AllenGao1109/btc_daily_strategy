"""20.4 fee, 20.5 partial rebalance fee, 20.12 sign-flip fee, 20.14 daily rebalance."""

from __future__ import annotations

import pytest

from src.backtest import run_backtest
from tests.conftest import make_df, make_signal


def test_one_buy_one_sell_fee():
    df = make_df([100] * 5)
    raw = make_signal([0, 1, 1, 1, 0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    traded = res["trade_notional"] > 0
    assert traded.sum() == 2  # one buy (day1), one sell (day4)
    assert traded.iloc[1] and traded.iloc[4]
    assert (res["fee"].iloc[2:4] == 0).all()  # hold days pay nothing


def test_partial_rebalance_charges_on_turnover_not_full_portfolio():
    df = make_df([100, 100, 100])
    raw = make_signal([0, 0.5, 1.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    # Each step trades ~0.5 of equity, not the full portfolio (turnover != 1.0).
    assert res["turnover"].iloc[1] == pytest.approx(0.5, abs=1e-6)
    assert res["turnover"].iloc[2] == pytest.approx(0.5, abs=0.02)
    assert res["turnover"].iloc[2] < 1.0


def test_sign_flip_is_four_x_turnover():
    df = make_df([100, 100])
    raw = make_signal([2.0, -2.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    # +2 -> -2 is a 4x turnover flip, ~2% fee of equity-before-trade.
    assert res["turnover"].iloc[1] == pytest.approx(4.0, abs=0.05)
    fee_pct = res["fee"].iloc[1] / res["equity_before_trade"].iloc[1]
    assert fee_pct == pytest.approx(0.02, abs=0.001)


def test_daily_rebalance_charges_drift_correction_fees():
    df = make_df([100, 110, 121, 133.1])
    raw = make_signal([2, 2, 2, 2], df)
    res = run_backtest(
        df,
        raw,
        10000,
        fee_rate=0.005,
        execution_lag_days=0,
        rebalance_policy="daily_target_rebalance",
    )
    # Price moves every day, so daily target rebalancing forces drift trades.
    assert (res["trade_notional"].iloc[1:] > 0).all()
    assert res["fee"].iloc[1:].sum() > 0
