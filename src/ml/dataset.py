"""Leakage-free supervised dataset construction for daily BTC ML.

Alignment contract (critical for no-lookahead):
    X[t] = features computed using data through the close of day t.
    y[t] = forward return from day t to day t+horizon (the holding-period return
           the model is trying to anticipate).

A model maps X[t] -> a desired stance. That stance is emitted as raw_signal[t],
and the backtest engine lags it by one day, so the position is held on day t+1
onward — using only information available at day t. The last ``horizon`` rows
have undefined y and are excluded from training.

Feature standardization must be fit on TRAINING rows only (see walkforward); this
module just assembles the raw, un-scaled matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Engineered, mostly-stationary feature columns (ratios / returns / z-scores).
# Raw price levels are deliberately excluded — they are non-stationary and would
# leak the regime/era rather than a generalizable pattern.
BASE_FEATURES = [
    "daily_return",
    "log_return",
    "rolling_vol_7",
    "rolling_vol_14",
    "rolling_vol_30",
    "rolling_vol_90",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_100",
    "price_to_sma_200",
    "momentum_7",
    "momentum_14",
    "momentum_30",
    "momentum_90",
    "drawdown_30",
    "drawdown_90",
    "volume_zscore_30",
]

# Optional on-chain features (added only if present on the frame).
ONCHAIN_FEATURES = ["mvrv_z", "adr_growth_30"]


def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive stationary on-chain features from raw MVRV / active-address columns.

    Args:
        df: Frame that may contain ``CapMVRVCur`` and/or ``AdrActCnt``.

    Returns:
        A copy with ``mvrv_z`` (trailing z-score of MVRV) and ``adr_growth_30``
        (30d active-address growth) added where the raw columns exist. Both use
        only trailing data, so they introduce no lookahead.
    """
    out = df.copy()
    if "CapMVRVCur" in out.columns:
        mvrv = out["CapMVRVCur"].astype(float)
        out["mvrv_z"] = (mvrv - mvrv.rolling(365, min_periods=60).mean()) / mvrv.rolling(
            365, min_periods=60
        ).std()
    if "AdrActCnt" in out.columns:
        adr = out["AdrActCnt"].astype(float)
        out["adr_growth_30"] = adr.pct_change(30)
    return out


def select_features(df: pd.DataFrame, use_onchain: bool = False) -> list[str]:
    """Return the available feature columns for modeling.

    Args:
        df: Feature frame.
        use_onchain: Include on-chain features if present.

    Returns:
        Ordered list of column names that exist on ``df``.
    """
    cols = [c for c in BASE_FEATURES if c in df.columns]
    if use_onchain:
        cols += [c for c in ONCHAIN_FEATURES if c in df.columns]
    return cols


def build_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the aligned (X, y) supervised dataset.

    Args:
        df: Feature frame with a ``close`` column.
        feature_cols: Columns to use as model inputs.
        horizon: Forward-return horizon in days (default 1).

    Returns:
        ``(X, y)`` where X is the feature matrix indexed by date and y is the
        forward ``horizon``-day return aligned to X. Rows with any NaN feature
        or undefined forward return are dropped.
    """
    close = df["close"].astype(float)
    fwd_ret = close.shift(-horizon) / close - 1.0  # known only at t+horizon

    data = df[feature_cols].copy()
    data["__y__"] = fwd_ret
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    y = data.pop("__y__")
    return data, y
