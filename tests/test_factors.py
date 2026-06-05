"""Factor library: no-lookahead (causality) and IC sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors import build_factors, composite_signal, information_coefficient
from src.data import generate_synthetic_btc


def test_factors_are_causal():
    # Perturbing FUTURE prices must not change past factor values.
    df1 = generate_synthetic_btc(n_days=900, seed=1)
    cut = 700
    f1 = build_factors(df1)
    df2 = df1.copy()
    df2.iloc[cut + 1 :, df2.columns.get_loc("close")] *= 1.5
    f2 = build_factors(df2)
    # Compare factors that depend only on close-derived history.
    for col in ["mom_20", "rsi_14", "zscore_60", "halving_cos", "vol_regime"]:
        a = f1[col].iloc[:cut + 1]
        b = f2[col].iloc[:cut + 1]
        # Equal where both defined (ignore warmup NaNs).
        mask = a.notna() & b.notna()
        assert np.allclose(a[mask], b[mask]), f"{col} leaked future info"


def test_composite_signal_is_causal():
    df1 = generate_synthetic_btc(n_days=1000, seed=2)
    train_end = df1.index[600]
    cut = 750
    s1 = composite_signal(df1, train_end, ["mom_120", "vol_regime", "zscore_60"])
    df2 = df1.copy()
    df2.iloc[cut + 1 :, df2.columns.get_loc("close")] *= 0.6
    s2 = composite_signal(df2, train_end, ["mom_120", "vol_regime", "zscore_60"])
    a, b = s1.iloc[: cut + 1], s2.iloc[: cut + 1]
    mask = a.notna() & b.notna()
    assert np.allclose(a[mask], b[mask]), "composite signal leaked future info"


def test_information_coefficient_range():
    df = generate_synthetic_btc(n_days=500, seed=3)
    f = build_factors(df)["mom_20"]
    fr = df["close"].pct_change().shift(-1)
    ic = information_coefficient(f, fr)
    assert -1.0 <= ic <= 1.0
