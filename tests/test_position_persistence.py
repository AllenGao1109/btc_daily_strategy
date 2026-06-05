"""20.3 Position persistence + 20.6 buy-and-hold fee + 20.13 no daily rebalance."""

from __future__ import annotations

from src.backtest import run_backtest
from src.strategies import buy_and_hold
from tests.conftest import make_df, make_signal


def test_persistent_position_trades_once():
    df = make_df([100] * 5)
    raw = make_signal([0, 1, 1, 1, 1], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    # Only one buy (day index 1); subsequent holds do not trade.
    assert (res["trade_notional"] > 0).sum() == 1
    assert res["fee"].sum() == 50.0  # one 1.0-turnover trade on 10000 * 0.005
    # No daily rebuy: trade only on the entry day.
    assert res["trade_notional"].iloc[1] > 0
    assert (res["trade_notional"].iloc[2:] == 0).all()


def test_buy_and_hold_one_entry_fee():
    df = make_df([100] * 6)
    raw = buy_and_hold.generate_signals(df, {})
    res = run_backtest(df, raw, 10000, fee_rate=0.005)
    # With 1-day lag, entry happens on day 1; exactly one fee thereafter.
    assert (res["trade_notional"] > 0).sum() == 1
    assert res["fee"].sum() == 50.0


def test_no_daily_rebalance_under_signal_change_policy():
    # Constant price, 2x long, fee=0 so leverage stays exactly 2.0 (no breach).
    df = make_df([100] * 5)
    raw = make_signal([2, 2, 2, 2, 2], df)
    res = run_backtest(
        df,
        raw,
        10000,
        fee_rate=0.0,
        execution_lag_days=0,
        rebalance_policy="signal_change_or_risk_control",
    )
    assert (res["trade_notional"] > 0).sum() == 1  # only the initial entry
    assert res["leverage_breach"].sum() == 0
