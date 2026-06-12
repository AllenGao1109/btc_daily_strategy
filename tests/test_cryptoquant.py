"""CryptoQuant loaders: cache parsing, regime-column coalescing, factor gating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cryptoquant import load_cme_basis, load_cryptoquant
from src.factors import build_factors
from tests.conftest import make_df


def test_cryptoquant_cache_parses_without_network(tmp_path):
    pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "asopr": [1.01, 0.98],
         "sth_sopr": [1.02, 0.97]}
    ).to_csv(tmp_path / "cryptoquant_asopr-sth_sopr.csv", index=False)
    df = load_cryptoquant(["asopr", "sth_sopr"], raw_dir=tmp_path)
    assert list(df.columns) == ["asopr", "sth_sopr"]
    assert df["asopr"].tolist() == [1.01, 0.98]
    assert str(df.index.tz) == "UTC"


def test_cme_basis_cache_parses_without_network(tmp_path):
    pd.DataFrame(
        {"date": ["2020-01-03"], "cme_basis": [0.012]}
    ).to_csv(tmp_path / "cme_basis.csv", index=False)
    df = load_cme_basis(raw_dir=tmp_path)
    assert df["cme_basis"].tolist() == [0.012]


def test_behavior_factors_gated_and_causal():
    n = 250
    rng = np.random.default_rng(3)
    df = make_df(list(100 + np.cumsum(rng.normal(0, 1, n))))
    plain = build_factors(df)
    for col in ["asopr_30", "lth_sopr_z_365", "cvd_chg_30", "whale_90",
                "icdd_z_180", "m2e_z_180", "basis_level"]:
        assert col not in plain.columns

    df["asopr"] = 1.0 + 0.05 * np.sin(np.arange(n) / 9.0)
    df["taker_cvd"] = np.cumsum(rng.normal(0, 1, n))
    df["cme_basis"] = 0.01 + 0.005 * np.cos(np.arange(n) / 7.0)
    full = build_factors(df)
    assert {"asopr_30", "asopr_z_90", "cvd_chg_30", "cvd_z_180",
            "basis_level", "basis_z_90"} <= set(full.columns)

    # Truncation invariance: appending future rows must not change the past.
    trunc = build_factors(df.iloc[:200])
    cols = ["asopr_30", "asopr_z_90", "cvd_chg_30", "cvd_z_180", "basis_z_90"]
    pd.testing.assert_frame_equal(full.loc[trunc.index, cols], trunc[cols])
