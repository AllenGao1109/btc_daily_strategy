"""Volatility-targeted trend strategy.

Idea: take directional exposure based on a trend filter, but *size* the position
so the portfolio targets a constant annualized volatility. In calm uptrends the
position scales up (toward the 2x cap) to grow NAV; in turbulent regimes it
scales down toward flat, which protects Sharpe and drawdown. This typically
improves risk-adjusted return versus fixed-size trend following.

Raw signal (pre-lag), per day t:
    realized_vol = std(daily_return, vol_window) * sqrt(365)     # annualized
    size         = clip(target_vol / realized_vol, 0, max_size)  # vol scaling
    up   = close > SMA(sma_window)
    down = close < SMA(sma_window)
    weight = +size            if up
           = -size*short_scale if down and allow_short
           = 0                 otherwise

Timing/lookahead: realized_vol and SMA at day t use day-t data, but the engine
lags the whole signal by one day before execution, so day t+1's position only
uses information through day t. No lookahead is introduced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate vol-targeted trend exposure weights.

    Args:
        df: Frame with a ``close`` column.
        params: Keys (all optional):
            ``target_vol`` annualized vol target (default 0.50),
            ``vol_window`` realized-vol lookback (default 30),
            ``sma_window`` trend SMA length (default 100),
            ``max_size`` per-bar leverage cap before global clip (default 2.0),
            ``allow_short`` whether to short downtrends (default False),
            ``short_scale`` short-size multiplier (default 0.5).

    Returns:
        Raw target weights in [-2, 2]; NaN during indicator warmup.
    """
    params = params or {}
    target_vol = float(params.get("target_vol", 0.50))
    vol_window = int(params.get("vol_window", 30))
    sma_window = int(params.get("sma_window", 100))
    max_size = float(params.get("max_size", 2.0))
    allow_short = bool(params.get("allow_short", False))
    short_scale = float(params.get("short_scale", 0.5))

    close = df["close"].astype(float)
    ret = close.pct_change()
    realized_vol = ret.rolling(vol_window).std() * np.sqrt(365)
    sma = close.rolling(sma_window).mean()

    # Avoid divide-by-zero; where vol is ~0, allow full size (calm market).
    size = (target_vol / realized_vol.replace(0.0, np.nan)).clip(upper=max_size)

    weight = pd.Series(0.0, index=df.index, name="raw_signal")
    up = close > sma
    down = close < sma
    weight[up] = size[up]
    if allow_short:
        weight[down] = -size[down] * short_scale

    weight[sma.isna() | realized_vol.isna()] = np.nan
    return weight.clip(lower=-2.0, upper=2.0)
