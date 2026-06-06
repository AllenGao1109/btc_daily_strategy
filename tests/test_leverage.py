"""20.11 max leverage + leverage drift / breach detection and delever."""

from __future__ import annotations

from src.backtest import run_backtest
from tests.conftest import make_df, make_signal


def test_leverage_breach_flagged_on_drift():
    # 2x long, then BTC rises: equity grows but the held notional grows faster
    # relative to the post-fee equity is not the issue — instead force a drift
    # breach by a sharp DOWN move that lifts gross leverage above 2.0.
    df = make_df([100, 90, 90])
    raw = make_signal([2, 2, 2], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)
    # Day 1: exposure 20000 held into a -10% day. equity_end = 10000 - 2000 = 8000,
    # exposure_end = 20000*0.9 = 18000 -> gross leverage 18000/8000 = 2.25 > 2.0.
    assert res["gross_leverage_end"].iloc[1] > 2.0
    assert bool(res["leverage_breach"].iloc[1])


def test_breach_triggers_delever_next_day():
    df = make_df([100, 90, 90, 90])
    raw = make_signal([2, 2, 2, 2], df)
    res = run_backtest(
        df,
        raw,
        10000,
        fee_rate=0.005,
        execution_lag_days=0,
        rebalance_policy="signal_change_or_risk_control",
        leverage_breach_action="delever_next_day",
    )
    # The breach forces a risk-control rebalance (trade) even though the target
    # weight did not change.
    assert res["trade_notional"].iloc[2] > 0
    # Delever brings leverage from the breach level (~2.25x) back toward the 2.0
    # cap; the rebalancing fee leaves it a hair above 2.0, never far above.
    assert res["gross_leverage_end"].iloc[2] < res["gross_leverage_before_rebalance"].iloc[2]
    assert res["gross_leverage_end"].iloc[2] < 2.05


def test_signal_above_two_is_clipped_in_engine():
    df = make_df([100, 100])
    raw = make_signal([5.0, -5.0], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.0, execution_lag_days=0)
    assert res["target_weight"].max() <= 2.0
    assert res["target_weight"].min() >= -2.0
