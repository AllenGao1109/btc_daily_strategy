"""Feature engineering for daily BTC data.

IMPORTANT timing note:
  Raw features here are computed using day ``t`` data (e.g. ``sma_200`` at day
  ``t`` includes the day-``t`` close). That is correct for *describing* the
  market state as of the close of day ``t``.

  These features must NOT be used to take a position on day ``t``. The backtest
  engine applies a 1-day execution lag to every signal. The explicit
  :func:`lag_signal` helper exists so that lagging is always visible and never
  hidden inside a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lag_signal(signal: pd.Series, lag_days: int = 1) -> pd.Series:
    """Shift a signal forward in time so day ``t`` uses day ``t-lag`` information.

    Args:
        signal: Raw signal indexed by date (one value per day).
        lag_days: Number of days to lag. Default 1 (decision uses prior day).

    Returns:
        The lagged signal, same index. The first ``lag_days`` entries are NaN
        and should be filled conservatively (the engine fills with 0 = flat).
    """
    if lag_days < 0:
        raise ValueError(f"lag_days must be >= 0; got {lag_days}.")
    return signal.shift(lag_days)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature set from an OHLCV frame.

    All features are raw (computed using day-``t`` data). Lagging for execution
    is the backtest engine's responsibility, not this function's.

    Args:
        df: Cleaned OHLCV frame indexed by UTC DatetimeIndex.

    Returns:
        A new DataFrame (copy of ``df``) with feature columns appended.
    """
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["daily_return"] = close.pct_change()
    out["log_return"] = np.log(close).diff()

    for w in (7, 14, 30, 90):
        out[f"rolling_vol_{w}"] = out["daily_return"].rolling(w).std()

    for w in (20, 50, 100, 200):
        sma = close.rolling(w).mean()
        out[f"sma_{w}"] = sma
        out[f"price_to_sma_{w}"] = close / sma - 1.0

    for w in (7, 14, 30, 90):
        out[f"momentum_{w}"] = close.pct_change(w)

    for w in (30, 90):
        rolling_high = close.rolling(w).max()
        out[f"rolling_high_{w}"] = rolling_high
        out[f"drawdown_{w}"] = close / rolling_high - 1.0

    vol_mean_30 = volume.rolling(30).mean()
    vol_std_30 = volume.rolling(30).std()
    out["volume_zscore_30"] = (volume - vol_mean_30) / vol_std_30

    return out
