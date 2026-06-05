"""Buy-and-hold 2x long leveraged benchmark: constant target weight +2.0.

This benchmark deliberately exercises leverage drift, leverage-breach checks,
funding/borrow costs (if configured), and liquidation handling. Under the
default policy it does NOT rebalance daily; actual leverage drifts with price,
and the engine flags any breach above the 2.0 cap and delevers on the next day.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Return a constant +2.0 (200% long) target weight for every day.

    Args:
        df: OHLCV/feature frame (only the index is used).
        params: Optional ``weight`` override (default 2.0).

    Returns:
        A Series of the leveraged long weight indexed like ``df``.
    """
    weight = float(params.get("weight", 2.0)) if params else 2.0
    return pd.Series(weight, index=df.index, name="raw_signal")
