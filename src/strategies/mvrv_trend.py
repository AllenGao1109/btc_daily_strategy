"""Trend ensemble gated by an MVRV valuation tilt (on-chain signal).

Adds a valuation dimension orthogonal to price trend. MVRV (market-value to
realized-value) is high near cycle tops and low near bottoms. Rather than
hardcoding absolute thresholds (which would be fit to this exact history), we
use a TRAILING z-score of MVRV — computed only from past data — so the tilt is
adaptive and lookahead-free.

The tilt multiplies the trend ensemble's exposure:
  - De-risk when MVRV is historically high: multiplier ramps 1 -> 0 as the
    trailing z-score rises from ``z_high`` to ``z_max`` (steps out of overvalued
    longs, protecting Sharpe/drawdown).
  - Lean in when MVRV is historically low: multiplier ramps 1 -> ``boost_max``
    as the z-score falls from ``z_low`` to ``z_min`` (levers up accumulation in
    undervalued regimes, raising NAV — still inside the 2x cap).

Requires the column ``CapMVRVCur`` to be present on ``df`` (merged in by the
research/runner layer). The engine still lags the final signal by one day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import trend_ensemble


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate MVRV-gated trend-ensemble weights.

    Args:
        df: Frame with ``high``/``low``/``close`` and ``CapMVRVCur`` columns.
        params: trend_ensemble params plus valuation knobs:
            ``z_window`` trailing z-score lookback (default 365),
            ``z_high``/``z_max`` overvaluation de-risk ramp (default 1.0 / 2.5),
            ``z_low``/``z_min`` undervaluation boost ramp (default -0.5 / -2.0),
            ``boost_max`` max undervaluation multiplier (default 1.5).

    Returns:
        Raw target weights in [-2, 2]; NaN during warmup.
    """
    params = params or {}
    if "CapMVRVCur" not in df.columns:
        raise KeyError(
            "mvrv_trend requires the 'CapMVRVCur' column. Merge on-chain data "
            "(see src.onchain.merge_onchain) before calling."
        )
    z_window = int(params.get("z_window", 365))
    z_high = float(params.get("z_high", 1.0))
    z_max = float(params.get("z_max", 2.5))
    z_low = float(params.get("z_low", -0.5))
    z_min = float(params.get("z_min", -2.0))
    boost_max = float(params.get("boost_max", 1.5))

    base = trend_ensemble.generate_signals(df, params)

    mvrv = df["CapMVRVCur"].astype(float)
    roll_mean = mvrv.rolling(z_window, min_periods=60).mean()
    roll_std = mvrv.rolling(z_window, min_periods=60).std()
    z = (mvrv - roll_mean) / roll_std

    mult = pd.Series(1.0, index=df.index)
    # Overvaluation de-risk: 1 at z_high -> 0 at z_max.
    over = ((z_max - z) / (z_max - z_high)).clip(lower=0.0, upper=1.0)
    mult = mult.where(z <= z_high, over)
    # Undervaluation boost: 1 at z_low -> boost_max at z_min.
    under = 1.0 + (boost_max - 1.0) * ((z_low - z) / (z_low - z_min)).clip(
        lower=0.0, upper=1.0
    )
    mult = mult.where(z >= z_low, under)
    # Where z is undefined (warmup), apply no tilt.
    mult[z.isna()] = 1.0

    weight = (base * mult).clip(lower=-2.0, upper=2.0)
    weight.name = "raw_signal"
    # Preserve the base strategy's warmup NaNs.
    weight[base.isna()] = np.nan
    return weight
