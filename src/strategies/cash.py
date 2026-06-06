"""Cash-only baseline strategy: always flat (target weight 0.0)."""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Return a constant flat target weight (0.0) for every day.

    Args:
        df: OHLCV/feature frame (only the index is used).
        params: Unused.

    Returns:
        A Series of 0.0 indexed like ``df``.
    """
    return pd.Series(0.0, index=df.index, name="raw_signal")
