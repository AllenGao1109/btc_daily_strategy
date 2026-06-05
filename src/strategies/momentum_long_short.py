"""Momentum long/short baseline.

Rule (raw, pre-lag):
    momentum_window-day return > 0  -> target_weight = +1.0
    otherwise                       -> target_weight = -1.0

The backtest engine lags this signal by one day before execution.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Long when trailing momentum is positive, short when negative.

    Args:
        df: Frame with a ``close`` column.
        params: ``momentum_window`` (default 30), ``long_weight`` (default 1.0),
            ``short_weight`` (default -1.0).

    Returns:
        Raw target weights (+long_weight or short_weight). Days before momentum
        is defined are NaN and will be lag-filled to flat by the engine.
    """
    window = int(params.get("momentum_window", 30)) if params else 30
    long_w = float(params.get("long_weight", 1.0)) if params else 1.0
    short_w = float(params.get("short_weight", -1.0)) if params else -1.0
    close = df["close"].astype(float)
    momentum = close.pct_change(window)
    signal = pd.Series(short_w, index=df.index, name="raw_signal")
    signal[momentum > 0] = long_w
    signal[momentum.isna()] = float("nan")
    return signal
