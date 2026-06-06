"""20.17 weight bounds, 20.18 determinism, and engine state-table integrity."""

from __future__ import annotations

from src.backtest import STATE_COLUMNS, run_backtest
from src.data import generate_synthetic_btc
from src.strategies import sma_long_short, trend_leverage
from tests.conftest import make_df, make_signal


def test_state_table_has_all_columns():
    df = make_df([100, 105, 95])
    raw = make_signal([1, 1, 1], df)
    res = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    expected = set(STATE_COLUMNS) - {"date"}
    assert expected.issubset(set(res.columns))


def test_weight_bounds_respected_after_clipping():
    df = generate_synthetic_btc(n_days=300, seed=7)
    raw = trend_leverage.generate_signals(df, {})
    res = run_backtest(df, raw, 10000, fee_rate=0.005)
    assert res["target_weight"].between(-2.0, 2.0).all()


def test_weight_band_suppresses_small_rebalances():
    # Target nudges by 0.1 each day; a 0.15 band should ignore those moves and
    # only trade once the cumulative drift past the last executed target exceeds
    # the band.
    df = make_df([100] * 6)
    raw = make_signal([0.0, 0.5, 0.6, 0.7, 0.75, 0.8], df)
    no_band = run_backtest(df, raw, 10000, fee_rate=0.005, execution_lag_days=0)
    banded = run_backtest(
        df, raw, 10000, fee_rate=0.005, execution_lag_days=0, weight_band=0.15
    )
    assert (banded["trade_notional"] > 0).sum() < (no_band["trade_notional"] > 0).sum()
    # Banding never trades more and so never costs more in fees.
    assert banded["fee"].sum() <= no_band["fee"].sum()


def test_deterministic_results():
    df = generate_synthetic_btc(n_days=300, seed=11)
    raw = sma_long_short.generate_signals(df, {"sma_slow": 100})
    res_a = run_backtest(df, raw, 10000, fee_rate=0.005)
    res_b = run_backtest(df, raw, 10000, fee_rate=0.005)
    import pandas as pd

    pd.testing.assert_frame_equal(res_a, res_b)


def test_strategy_daily_return_reconstructs_equity():
    df = generate_synthetic_btc(n_days=200, seed=3)
    raw = sma_long_short.generate_signals(df, {"sma_slow": 50})
    res = run_backtest(df, raw, 10000, fee_rate=0.005)
    reconstructed = 10000 * (1 + res["strategy_daily_return"]).cumprod()
    # Equity curve must match compounding the reported daily returns.
    diff = (reconstructed - res["equity_end"]).abs().max()
    assert diff < 1e-6
