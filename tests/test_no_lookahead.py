"""20.1 No-lookahead test.

Changing FUTURE prices must not change today's signal or today's position.
"""

from __future__ import annotations

from src.backtest import run_backtest
from src.data import generate_synthetic_btc
from src.strategies import sma_long_short


def test_future_prices_do_not_change_past_signal():
    df1 = generate_synthetic_btc(n_days=400, seed=1)
    cut = 300  # index position of "today"

    raw1 = sma_long_short.generate_signals(df1, {"sma_slow": 50})

    # Perturb all prices strictly AFTER 'today'.
    df2 = df1.copy()
    df2.iloc[cut + 1 :, df2.columns.get_loc("close")] *= 3.0
    raw2 = sma_long_short.generate_signals(df2, {"sma_slow": 50})

    # Signals up to and including 'today' must be identical.
    assert raw1.iloc[: cut + 1].equals(raw2.iloc[: cut + 1])


def test_future_prices_do_not_change_past_positions():
    df1 = generate_synthetic_btc(n_days=400, seed=2)
    cut = 300

    raw1 = sma_long_short.generate_signals(df1, {"sma_slow": 50})
    df2 = df1.copy()
    df2.iloc[cut + 1 :, df2.columns.get_loc("close")] *= 0.5
    raw2 = sma_long_short.generate_signals(df2, {"sma_slow": 50})

    res1 = run_backtest(df1, raw1, 10000, fee_rate=0.005)
    res2 = run_backtest(df2, raw2, 10000, fee_rate=0.005)

    # Target weights (positions) up to 'today' must be unchanged.
    tw1 = res1["target_weight"].iloc[: cut + 1]
    tw2 = res2["target_weight"].iloc[: cut + 1]
    assert tw1.equals(tw2)
