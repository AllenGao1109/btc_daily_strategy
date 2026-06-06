"""Donchian-channel (support/resistance) breakout strategy.

A price-action approach using rolling highs/lows as resistance/support:
  - Go long when close breaks ABOVE the prior N-day high (resistance break).
  - Exit / go short when close breaks BELOW the prior M-day low (support break).
  - Otherwise hold the existing stance (state persists between breakouts), so it
    trades infrequently in range-bound, stable markets.

Optionally vol-targets the position size like ``vol_target_trend`` so exposure
expands in calm trends and contracts in turbulent ones.

Raw signal (pre-lag): a stance in {+1, 0, -1} (or short disabled) carried forward
between breakouts, multiplied by an optional vol-target size. The engine lags the
signal by one day, so the breakout observed at the close of day t drives the day
t+1 position. The prior-N-day high/low explicitly EXCLUDES the current bar to
avoid a same-bar lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate Donchian breakout exposure weights.

    Args:
        df: Frame with ``high``, ``low``, ``close`` columns.
        params: Keys (all optional):
            ``entry_window`` lookback for the breakout high (default 50),
            ``exit_window`` lookback for the breakdown low (default 25),
            ``allow_short`` short on support breaks (default False),
            ``vol_target`` annualized vol target for sizing, or 0 to disable
                (default 0.0 = fixed unit size),
            ``vol_window`` realized-vol lookback (default 30),
            ``max_size`` per-bar leverage cap (default 2.0).

    Returns:
        Raw target weights in [-2, 2]; NaN during warmup.
    """
    params = params or {}
    entry_window = int(params.get("entry_window", 50))
    exit_window = int(params.get("exit_window", 25))
    allow_short = bool(params.get("allow_short", False))
    vol_target = float(params.get("vol_target", 0.0))
    vol_window = int(params.get("vol_window", 30))
    max_size = float(params.get("max_size", 2.0))

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Prior-window extremes EXCLUDING the current bar (shift(1)).
    upper = high.rolling(entry_window).max().shift(1)
    lower = low.rolling(exit_window).min().shift(1)

    n = len(df)
    stance = np.zeros(n)
    cur = 0.0
    close_v = close.to_numpy()
    up_v = upper.to_numpy()
    lo_v = lower.to_numpy()
    for i in range(n):
        u, l = up_v[i], lo_v[i]
        if not np.isnan(u) and close_v[i] > u:
            cur = 1.0  # resistance break -> long
        elif not np.isnan(l) and close_v[i] < l:
            cur = -1.0 if allow_short else 0.0  # support break -> short or flat
        stance[i] = cur
    stance_s = pd.Series(stance, index=df.index)

    if vol_target > 0:
        ret = close.pct_change()
        realized_vol = ret.rolling(vol_window).std() * np.sqrt(365)
        size = (vol_target / realized_vol.replace(0.0, np.nan)).clip(upper=max_size)
        weight = stance_s * size
    else:
        weight = stance_s

    warmup = upper.isna() & lower.isna()
    weight[warmup] = np.nan
    weight.name = "raw_signal"
    return weight.clip(lower=-2.0, upper=2.0)
