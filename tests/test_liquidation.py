"""20.15 equity-zero liquidation handling."""

from __future__ import annotations

from src.backtest import run_backtest
from tests.conftest import make_df, make_signal


def test_equity_zero_liquidates_and_stops_trading():
    # 2x long into a -60% day: pnl = 20000 * -0.60 = -12000 wipes out equity.
    df = make_df([100, 40, 50, 60])
    raw = make_signal([2, 2, 2, 2], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)

    assert bool(res["liquidated"].iloc[1])
    assert res["equity_end"].iloc[1] == 0.0
    assert res["exposure_notional_end"].iloc[1] == 0.0
    # Trading stops: all subsequent days are flat and dead.
    assert (res["equity_end"].iloc[1:] == 0.0).all()
    assert (res["exposure_notional_end"].iloc[2:] == 0.0).all()
    assert (res["trade_notional"].iloc[2:] == 0.0).all()


def test_liquidate_breach_action_closes_position():
    # A drift breach with the 'liquidate' action closes the position outright.
    df = make_df([100, 90, 90, 90])
    raw = make_signal([2, 2, 2, 2], df)
    res = run_backtest(
        df,
        raw,
        10000,
        fee_rate=0.0,
        execution_lag_days=0,
        leverage_breach_action="liquidate",
    )
    # Breach occurs end of day 1; day 2 closes to flat and marks liquidation.
    assert bool(res["liquidated"].iloc[2])
    assert res["exposure_notional_end"].iloc[2] == 0.0
