"""Strong-trend leveraged baseline (the preferred first non-trivial strategy).

Rule (raw, pre-lag):
    close > SMA_slow AND momentum > 0  -> target_weight = +strong_long_weight
    close < SMA_slow AND momentum < 0  -> target_weight = +strong_short_weight
    otherwise                          -> target_weight = 0.0

This is NOT assumed to be profitable. It exists to exercise the harness with a
leveraged long/short signal. The 1x version (long_weight/short_weight = +/-1)
is obtained by overriding the strong weights in params.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Take leveraged directional exposure only on aligned trend+momentum.

    Args:
        df: Frame with a ``close`` column.
        params: ``sma_slow`` (default 200), ``momentum_window`` (default 30),
            ``strong_long_weight`` (default 2.0), ``strong_short_weight``
            (default -2.0).

    Returns:
        Raw target weights in {strong_short_weight, 0.0, strong_long_weight}.
        Days before indicators are defined are NaN and lag-filled to flat.
    """
    params = params or {}
    sma_slow = int(params.get("sma_slow", 200))
    window = int(params.get("momentum_window", 30))
    long_w = float(params.get("strong_long_weight", 2.0))
    short_w = float(params.get("strong_short_weight", -2.0))

    close = df["close"].astype(float)
    sma = close.rolling(sma_slow).mean()
    momentum = close.pct_change(window)

    signal = pd.Series(0.0, index=df.index, name="raw_signal")
    long_mask = (close > sma) & (momentum > 0)
    short_mask = (close < sma) & (momentum < 0)
    signal[long_mask] = long_w
    signal[short_mask] = short_w
    signal[sma.isna() | momentum.isna()] = float("nan")
    return signal
