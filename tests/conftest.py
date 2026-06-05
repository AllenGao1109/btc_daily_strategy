"""Shared pytest fixtures and helpers for the backtest test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest

# Make the project root importable as ``src`` when running pytest from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_df(closes: Sequence[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV frame from a list of closes (constant-OHLC bars).

    Args:
        closes: Daily close prices.
        start: First date (UTC midnight).

    Returns:
        A DataFrame with open/high/low/close/volume and a UTC DatetimeIndex.
    """
    idx = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    close = pd.Series([float(c) for c in closes], index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
        }
    )
    df.index.name = "date"
    return df


def make_signal(values: Sequence[float], df: pd.DataFrame) -> pd.Series:
    """Build a raw-signal Series aligned to ``df.index`` from a list of weights."""
    return pd.Series([float(v) for v in values], index=df.index, name="raw_signal")


@pytest.fixture
def const_df() -> pd.DataFrame:
    """A 5-day constant-price frame (BTC return = 0 every day)."""
    return make_df([100.0] * 5)
