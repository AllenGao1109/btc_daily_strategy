"""Sentiment data: merge causality, factor construction, loader cache parsing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors import build_factors
from src.onchain import merge_onchain
from src.sentiment import load_cnn_fear_greed, load_crypto_fear_greed
from tests.conftest import make_df


def test_merge_sentiment_is_causal_no_future_leak():
    # Sentiment steps from 20 to 80 on 2020-01-04 (e.g. a fear -> greed flip).
    price = make_df([100] * 6, start="2020-01-01")
    idx = pd.to_datetime(["2020-01-01", "2020-01-04"], utc=True)
    sent = pd.DataFrame({"cnn_fg": [20.0, 80.0]}, index=idx)
    sent.index.name = "date"

    merged = merge_onchain(price, sent)
    # Before the step date only the past value is visible; gaps ffill from past.
    assert merged.loc["2020-01-03", "cnn_fg"] == 20.0
    assert merged.loc["2020-01-04", "cnn_fg"] == 80.0
    assert merged.loc["2020-01-05", "cnn_fg"] == 80.0


def test_sentiment_factors_only_built_when_columns_present():
    df = make_df([100 + i for i in range(120)])
    f_plain = build_factors(df)
    assert not any(c.startswith(("fng_", "cnnfg_")) for c in f_plain.columns)

    df["fng_value"] = 50.0
    df["cnn_fg"] = 50.0
    f_sent = build_factors(df)
    for col in ["fng_level", "fng_z_60", "fng_mom_10", "fng_extreme_fear",
                "cnnfg_level", "cnnfg_z_60", "cnnfg_mom_10", "cnnfg_extreme_greed"]:
        assert col in f_sent.columns


def test_extreme_flags_nan_before_coverage_not_fabricated_zeros():
    df = make_df([100] * 10)
    fng = pd.Series([np.nan] * 5 + [10.0, 10.0, 90.0, 90.0, 50.0], index=df.index)
    df["fng_value"] = fng
    f = build_factors(df)
    # No coverage yet -> NaN, not a fabricated "not extreme" 0.0.
    assert f["fng_extreme_fear"].iloc[:5].isna().all()
    assert f["fng_extreme_fear"].iloc[5] == 1.0
    assert f["fng_extreme_greed"].iloc[7] == 1.0
    assert f["fng_extreme_greed"].iloc[9] == 0.0


def test_sentiment_factors_are_causal_truncation_invariant():
    # Factor values at day t must not change when future rows are appended.
    n = 200
    rng = np.random.default_rng(7)
    df = make_df(list(100 + np.cumsum(rng.normal(0, 1, n))))
    df["cnn_fg"] = 50 + 30 * np.sin(np.arange(n) / 15.0)
    df["fng_value"] = 50 + 30 * np.cos(np.arange(n) / 10.0)

    full = build_factors(df)
    trunc = build_factors(df.iloc[:150])
    sent_cols = [c for c in full.columns if c.startswith(("fng_", "cnnfg_"))]
    pd.testing.assert_frame_equal(
        full.loc[trunc.index, sent_cols], trunc[sent_cols]
    )


def test_loaders_parse_cached_csv_without_network(tmp_path):
    pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "fng_value": [25, 75]}
    ).to_csv(tmp_path / "crypto_fear_greed.csv", index=False)
    # Cached file is indexed by its first column when re-read.
    fng = load_crypto_fear_greed(raw_dir=tmp_path)
    assert list(fng.columns) == ["fng_value"]
    assert fng["fng_value"].tolist() == [25.0, 75.0]
    assert str(fng.index.tz) == "UTC"

    pd.DataFrame(
        {"date": ["2020-01-03"], "cnn_fg": [42.5]}
    ).to_csv(tmp_path / "cnn_fear_greed.csv", index=False)
    cnn = load_cnn_fear_greed(raw_dir=tmp_path)
    assert cnn["cnn_fg"].tolist() == [42.5]
