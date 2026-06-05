"""Diversified trend ensemble with volatility targeting.

Robustness rationale: any single trend lookback (or single breakout window) is
regime-dependent — it shines in some markets and gets chopped in others. Blending
several individually-sound trend filters into a continuous score, then sizing the
blended position to a volatility target, diversifies that regime risk. This is the
standard multi-timeframe trend approach used in managed-futures programs, not a
parameter-mined curve fit.

Construction (raw, pre-lag):
    components:
        c_i = 1 if close > SMA(w_i) else 0        for each w_i in sma_windows
        c_d = 1 if Donchian breakout state is long else 0   (optional)
    score = mean(components)                       in [0, 1]
    if allow_short: score = 2*score - 1            mapped to [-1, +1]
    size  = clip(target_vol / realized_vol, 0, max_size)
    weight = score * size                          clipped to [-2, +2]

When every filter agrees on an uptrend in a calm market, the position scales up
toward the leverage cap (grows NAV); when filters disagree or volatility spikes,
exposure shrinks toward flat (protects Sharpe and drawdown). The engine lags the
signal one day, so no same-bar lookahead is introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate vol-targeted diversified trend-ensemble weights.

    Args:
        df: Frame with ``high``, ``low``, ``close`` columns.
        params: Keys (all optional):
            ``sma_windows`` list of trend SMA lengths (default [50, 100, 200]),
            ``use_donchian`` include a Donchian breakout vote (default True),
            ``entry_window``/``exit_window`` Donchian lookbacks (default 50/25),
            ``target_vol`` annualized vol target (default 0.45),
            ``vol_window`` realized-vol lookback (default 30),
            ``max_size`` per-bar leverage cap before global clip (default 2.0),
            ``allow_short`` map score to [-1, 1] and allow shorts (default False).

    Returns:
        Raw target weights in [-2, 2]; NaN during warmup.
    """
    params = params or {}
    sma_windows = list(params.get("sma_windows", [50, 100, 200]))
    use_donchian = bool(params.get("use_donchian", True))
    entry_window = int(params.get("entry_window", 50))
    exit_window = int(params.get("exit_window", 25))
    target_vol = float(params.get("target_vol", 0.45))
    vol_window = int(params.get("vol_window", 30))
    max_size = float(params.get("max_size", 2.0))
    allow_short = bool(params.get("allow_short", False))

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    components: list[pd.Series] = []
    warmup = pd.Series(False, index=df.index)
    for w in sma_windows:
        sma = close.rolling(w).mean()
        components.append((close > sma).astype(float))
        warmup |= sma.isna()

    if use_donchian:
        upper = high.rolling(entry_window).max().shift(1)
        lower = low.rolling(exit_window).min().shift(1)
        stance = np.zeros(len(df))
        cur = 0.0
        cv, uv, lv = close.to_numpy(), upper.to_numpy(), lower.to_numpy()
        for i in range(len(df)):
            if not np.isnan(uv[i]) and cv[i] > uv[i]:
                cur = 1.0
            elif not np.isnan(lv[i]) and cv[i] < lv[i]:
                cur = 0.0
            stance[i] = cur
        components.append(pd.Series(stance, index=df.index))
        warmup |= (upper.isna() & lower.isna())

    score = pd.concat(components, axis=1).mean(axis=1)  # [0, 1]
    if allow_short:
        score = 2.0 * score - 1.0  # [-1, 1]

    ret = close.pct_change()
    realized_vol = ret.rolling(vol_window).std() * np.sqrt(365)
    size = (target_vol / realized_vol.replace(0.0, np.nan)).clip(upper=max_size)
    warmup |= realized_vol.isna()

    weight = (score * size).clip(lower=-2.0, upper=2.0)
    weight[warmup] = np.nan
    weight.name = "raw_signal"
    return weight
