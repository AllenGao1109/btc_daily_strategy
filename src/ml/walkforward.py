"""Walk-forward training -> out-of-sample signal generation.

The walk-forward loop is the core anti-overfitting / anti-lookahead device:
  - Expanding window: at each step, fit on all data strictly BEFORE the OOS
    block, predict the block, advance.
  - Standardization stats (mean/std) are fit on the TRAINING rows only and then
    applied to the OOS block, so no future information leaks into scaling.
  - Models are retrained every ``step`` rows; predictions are stitched into a
    single OOS series spanning everything after the initial training window.

The resulting per-day prediction is converted into a raw target weight which is
then fed to the standard backtest engine (which lags it one more day).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ModelFactory = Callable[[], object]  # returns an object with fit()/predict()


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    initial_train: int = 730,
    step: int = 90,
) -> pd.Series:
    """Generate stitched out-of-sample predictions via expanding walk-forward.

    Args:
        X: Feature matrix indexed by date.
        y: Target (forward return) aligned to X.
        model_factory: Zero-arg callable returning a fresh fit/predict model.
        initial_train: Rows in the first training window before any prediction.
        step: OOS block size; the model is retrained every ``step`` rows.

    Returns:
        A Series of OOS predictions indexed like X (NaN for the initial training
        rows that are never predicted out-of-sample).
    """
    n = len(X)
    preds = pd.Series(np.nan, index=X.index, dtype=float)
    x_vals = X.to_numpy(dtype=np.float64)
    y_vals = y.to_numpy(dtype=np.float64)

    i = initial_train
    while i < n:
        end = min(i + step, n)
        x_tr, y_tr = x_vals[:i], y_vals[:i]
        # Standardize on training rows only.
        mu = x_tr.mean(axis=0)
        sd = x_tr.std(axis=0)
        sd[sd == 0] = 1.0
        x_tr_s = (x_tr - mu) / sd
        x_te_s = (x_vals[i:end] - mu) / sd

        model = model_factory()
        model.fit(x_tr_s, y_tr)
        preds.iloc[i:end] = model.predict(x_te_s)
        i = end

    return preds


def predictions_to_weight(
    preds: pd.Series,
    df: pd.DataFrame,
    *,
    mode: str = "vol_target",
    target_vol: float = 0.55,
    vol_window: int = 45,
    max_weight: float = 2.0,
    long_only: bool = False,
    scale: float = 50.0,
) -> pd.Series:
    """Convert OOS predictions into a raw target-weight signal in [-2, 2].

    Args:
        preds: Per-day model predictions (sign = directional call).
        df: Price frame (for realized-vol sizing).
        mode: ``"sign"`` (fixed +/-1 by direction), ``"linear"`` (clip scaled
            prediction), or ``"vol_target"`` (direction sized to a vol target).
        target_vol: Annualized vol target for ``vol_target`` mode.
        vol_window: Realized-vol lookback.
        max_weight: Absolute weight cap (<= 2.0).
        long_only: If True, negative calls go flat instead of short.
        scale: Multiplier for ``linear`` mode.

    Returns:
        Raw target weights aligned to ``df.index`` (NaN where preds is NaN).
    """
    direction = np.sign(preds)
    if long_only:
        direction = direction.clip(lower=0.0)

    if mode == "sign":
        weight = direction * 1.0
    elif mode == "linear":
        weight = (preds * scale).clip(-max_weight, max_weight)
        if long_only:
            weight = weight.clip(lower=0.0)
    elif mode == "vol_target":
        ret = df["close"].astype(float).pct_change()
        realized_vol = ret.rolling(vol_window).std() * np.sqrt(365)
        size = (target_vol / realized_vol.replace(0.0, np.nan)).clip(upper=max_weight)
        weight = direction * size
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    weight = weight.reindex(df.index).clip(lower=-max_weight, upper=max_weight)
    weight.name = "raw_signal"
    return weight
