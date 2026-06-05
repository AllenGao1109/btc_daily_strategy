"""SMA long/short baseline.

Rule (raw, pre-lag):
    close > SMA_slow  -> target_weight = +1.0
    otherwise         -> target_weight = -1.0

The backtest engine lags this signal by one day before execution.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Long above the slow SMA, short below it.

    Args:
        df: Frame with a ``close`` column.
        params: ``sma_slow`` (default 200), ``long_weight`` (default 1.0),
            ``short_weight`` (default -1.0).

    Returns:
        Raw target weights (+long_weight or short_weight). Days before the SMA
        is defined are NaN and will be lag-filled to flat by the engine.
    """
    sma_slow = int(params.get("sma_slow", 200)) if params else 200
    long_w = float(params.get("long_weight", 1.0)) if params else 1.0
    short_w = float(params.get("short_weight", -1.0)) if params else -1.0
    close = df["close"].astype(float)
    sma = close.rolling(sma_slow).mean()
    signal = pd.Series(short_w, index=df.index, name="raw_signal")
    signal[close > sma] = long_w
    signal[sma.isna()] = float("nan")
    return signal
