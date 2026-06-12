"""Macro data: release-lag re-stamping, stablecoin aggregation, factor gating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors import build_factors
from src.macro import apply_release_lag
from src.onchain import merge_onchain
from tests.conftest import make_df


def test_release_lagged_series_restamped_to_next_day():
    idx = pd.to_datetime(["2020-01-06", "2020-01-07"], utc=True)  # Mon, Tue
    panel = pd.DataFrame({"DGS10": [1.80, 1.85], "VIXCLS": [14.0, 15.0]}, index=idx)
    out = apply_release_lag(panel)
    # H.15 yield stamped Mon is only public Tue; VIX keeps its value date.
    assert np.isnan(out.loc["2020-01-06", "DGS10"])
    assert out.loc["2020-01-07", "DGS10"] == 1.80
    assert out.loc["2020-01-08", "DGS10"] == 1.85
    assert out.loc["2020-01-06", "VIXCLS"] == 14.0


def test_release_lag_preserves_friday_values_on_weekend_stamp():
    idx = pd.to_datetime(["2020-01-09", "2020-01-10"], utc=True)  # Thu, Fri
    panel = pd.DataFrame({"DGS10": [1.80, 1.85]}, index=idx)
    out = apply_release_lag(panel)
    # Friday's value lands on Saturday (not in the original index) and must
    # survive so merge_onchain can forward-fill it into the next week.
    assert out.loc["2020-01-11", "DGS10"] == 1.85

    price = make_df([100] * 7, start="2020-01-08")
    merged = merge_onchain(price, out)
    assert merged.loc["2020-01-14", "DGS10"] == 1.85


def test_macro_factors_gated_on_columns():
    df = make_df([100 + i for i in range(80)])
    f_plain = build_factors(df)
    for col in ["vix_z_60", "dgs10_chg_60", "rrp_chg_30", "dxy_mom_60",
                "exsply_ratio", "stable_growth_30", "puell_365",
                "hash_ribbon_3060"]:
        assert col not in f_plain.columns

    df["VIXCLS"] = 15.0
    df["HashRate"] = 1e8
    df["SplyExNtv"] = 2e6
    df["SplyCur"] = 19e6
    df["stable_mcap_usd"] = 1e11
    f = build_factors(df)
    assert "vix_z_60" in f.columns
    assert "hash_ribbon_3060" in f.columns
    assert "exsply_ratio" in f.columns
    assert "stable_growth_30" in f.columns
    # Constant supply ratio -> level defined, no spurious values pre-warmup.
    assert (f["exsply_ratio"].dropna() > 0).all()
