"""20.2 Signal lag test + weight bounds / clipping (20.11, 20.17)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import prepare_target_weights
from tests.conftest import make_df


def test_signal_lagged_by_one_day():
    df = make_df([100, 100, 100, 100])
    raw = pd.Series([0, 1, 1, 0], index=df.index, dtype=float)
    target = prepare_target_weights(raw, execution_lag_days=1, min_weight=-2, max_weight=2)
    # [0,1,1,0] lagged by 1 -> [NaN,0,1,1] -> fill 0 -> [0,0,1,1]
    assert list(target.values) == [0.0, 0.0, 1.0, 1.0]


def test_max_leverage_clip_positive_and_negative():
    df = make_df([100, 100, 100])
    raw = pd.Series([3.0, -3.0, 5.0], index=df.index, dtype=float)
    target = prepare_target_weights(raw, execution_lag_days=0, min_weight=-2, max_weight=2)
    assert target.max() <= 2.0
    assert target.min() >= -2.0
    assert target.iloc[0] == 2.0
    assert target.iloc[1] == -2.0


def test_warmup_nans_filled_flat():
    df = make_df([100, 100, 100])
    raw = pd.Series([np.nan, np.nan, 1.0], index=df.index)
    target = prepare_target_weights(raw, execution_lag_days=1, min_weight=-2, max_weight=2)
    assert not target.isna().any()
    assert target.iloc[0] == 0.0
