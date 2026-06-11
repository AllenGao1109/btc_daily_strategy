"""Cross-asset loader cache parsing and factor gating/causality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crossasset import load_crossasset
from src.factors import build_factors
from tests.conftest import make_df


def test_crossasset_cache_parses_without_network(tmp_path):
    pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "jpy_usd": [108.5, 109.0],
         "arkk_close": [50.0, 51.0], "qqq_close": [210.0, 211.0],
         "riot_close": [1.2, 1.3], "mara_close": [0.9, 0.95]}
    ).to_csv(tmp_path / "crossasset.csv", index=False)
    df = load_crossasset(raw_dir=tmp_path)
    assert df["jpy_usd"].tolist() == [108.5, 109.0]
    assert str(df.index.tz) == "UTC"


def test_crossasset_factors_gated_and_causal():
    n = 220
    rng = np.random.default_rng(5)
    df = make_df(list(100 + np.cumsum(rng.normal(0, 1, n))))
    plain = build_factors(df)
    for col in ["jpy_mom_60", "jpy_vol_z_90", "arkk_rs_60", "miner_rs_60"]:
        assert col not in plain.columns

    df["jpy_usd"] = 110 + np.cumsum(rng.normal(0, 0.5, n))
    df["arkk_close"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df["qqq_close"] = 200 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df["riot_close"] = 5 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    df["mara_close"] = 3 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    df["gld_close"] = 150 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    df["smh_close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    df["aapl_close"] = 180 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    full = build_factors(df)
    cols = ["jpy_mom_60", "jpy_vol_z_90", "arkk_rs_20", "arkk_rs_60",
            "miner_rs_20", "miner_rs_60", "gld_mom_60", "btc_gld_rs_60",
            "smh_rs_60", "btc_aapl_z_365", "btc_aapl_rs_60"]
    assert set(cols) <= set(full.columns)

    trunc = build_factors(df.iloc[:180])
    pd.testing.assert_frame_equal(full.loc[trunc.index, cols], trunc[cols])
