"""Chronological split + walk-forward fold construction (section 18)."""

from __future__ import annotations

import pandas as pd

from src.data import generate_synthetic_btc
from src.validation import make_fixed_split, make_walk_forward_folds


def test_fixed_split_is_chronological_and_non_overlapping():
    cfg = {
        "train_start": "2017-01-01",
        "train_end": "2021-12-31",
        "validation_start": "2022-01-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": None,
    }
    split = make_fixed_split(cfg)
    assert split["train"].end < split["validation"].start
    assert split["validation"].end < split["test"].start


def test_walk_forward_folds_advance_in_time():
    df = generate_synthetic_btc(n_days=365 * 6, seed=5, start="2017-01-01")
    folds = make_walk_forward_folds(df, train_years=3, val_years=1, expanding=True)
    assert len(folds) >= 2
    # Each validation window must start after its training window ends, and
    # successive folds must move forward in time.
    prev_val_start = None
    for train_w, val_w in folds:
        assert train_w.end < val_w.start
        if prev_val_start is not None:
            assert val_w.start > prev_val_start
        prev_val_start = val_w.start


def test_split_slice_selects_correct_rows():
    df = generate_synthetic_btc(n_days=400, seed=9, start="2020-01-01")
    cfg = {
        "train_start": "2020-01-01",
        "train_end": "2020-06-30",
        "validation_start": "2020-07-01",
        "validation_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": None,
    }
    split = make_fixed_split(cfg)
    train = split["train"].slice(df)
    assert train.index.min() >= pd.Timestamp("2020-01-01", tz="UTC")
    assert train.index.max() <= pd.Timestamp("2020-06-30", tz="UTC")
