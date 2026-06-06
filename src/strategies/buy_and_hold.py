"""Buy-and-hold 1x long baseline: constant target weight +1.0.

Under the default ``signal_change_or_risk_control`` policy this pays a single
entry fee (flat -> +1.0 on the first executable day) and no further fees.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Return a constant +1.0 (100% long) target weight for every day.

    Args:
        df: OHLCV/feature frame (only the index is used).
        params: Optional ``weight`` override (default 1.0).

    Returns:
        A Series of the long weight indexed like ``df``.
    """
    weight = float(params.get("weight", 1.0)) if params else 1.0
    return pd.Series(weight, index=df.index, name="raw_signal")
