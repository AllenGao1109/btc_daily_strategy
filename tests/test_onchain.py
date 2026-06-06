"""On-chain merge causality + mvrv_trend strategy sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.onchain import merge_onchain
from src.strategies import mvrv_trend
from tests.conftest import make_df


def test_merge_onchain_is_causal_no_future_leak():
    # On-chain value steps from 1.0 to 2.0 on 2020-01-04.
    price = make_df([100] * 6, start="2020-01-01")
    oc_idx = pd.to_datetime(["2020-01-01", "2020-01-04"], utc=True)
    oc = pd.DataFrame({"CapMVRVCur": [1.0, 2.0]}, index=oc_idx)
    oc.index.name = "date"

    merged = merge_onchain(price, oc)
    # Before the step date, only the past value (1.0) is visible.
    assert merged.loc["2020-01-03", "CapMVRVCur"] == 1.0
    # From the step date onward, the new value applies (no early leak of 2.0).
    assert merged.loc["2020-01-04", "CapMVRVCur"] == 2.0
    assert merged.loc["2020-01-05", "CapMVRVCur"] == 2.0


def test_mvrv_trend_requires_column():
    df = make_df([100, 101, 102])
    with pytest.raises(KeyError):
        mvrv_trend.generate_signals(df, {})


def test_mvrv_trend_runs_and_respects_bounds():
    n = 500
    df = make_df(list(100 + np.arange(n) * 0.5))
    # Synthetic MVRV oscillating in a plausible range.
    df["CapMVRVCur"] = 1.5 + 0.8 * np.sin(np.arange(n) / 50.0)
    raw = mvrv_trend.generate_signals(
        df, {"sma_windows": [20, 50], "use_donchian": False, "target_vol": 0.5}
    )
    res = run_backtest(df, raw, 10000, fee_rate=0.005)
    assert res["target_weight"].between(-2.0, 2.0).all()
    assert not res["liquidated"].any()
