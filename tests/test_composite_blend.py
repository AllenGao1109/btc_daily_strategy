"""composite_blend strategy: blend math, bounds, parameter validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import composite_blend, factor_composite, get_strategy, trend_ensemble
from tests.conftest import make_df


def _frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    df = make_df(list(1000 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n)))),
                 start="2019-01-01")
    return df


PARAMS = {"blend": 0.75, "train_end": "2020-06-30"}


def test_blend_is_convex_combination_of_legs():
    df = _frame()
    blended = composite_blend.generate_signals(df, dict(PARAMS))
    comp = factor_composite.generate_signals(df, {"train_end": "2020-06-30"})
    ens = trend_ensemble.generate_signals(df, composite_blend.DEFAULT_ENS_PARAMS)
    expected = 0.75 * comp.reindex(df.index).fillna(0.0) + 0.25 * ens.reindex(
        df.index
    ).fillna(0.0)
    pd.testing.assert_series_equal(blended, expected.rename("raw_signal"))


def test_blend_registered_and_bounded():
    df = _frame()
    sig = get_strategy("composite_blend")(df, dict(PARAMS))
    assert sig.between(-2.0, 2.0).all()
    assert sig.name == "raw_signal"


def test_blend_param_validation():
    df = _frame()
    with pytest.raises(ValueError):
        composite_blend.generate_signals(df, {"blend": 1.5, "train_end": "2020-06-30"})
    with pytest.raises(KeyError):
        # The composite leg still requires train_end (no silent default).
        composite_blend.generate_signals(df, {"blend": 0.5})


def test_blend_extremes_match_single_legs():
    df = _frame()
    only_comp = composite_blend.generate_signals(df, {"blend": 1.0, "train_end": "2020-06-30"})
    comp = factor_composite.generate_signals(df, {"train_end": "2020-06-30"})
    pd.testing.assert_series_equal(
        only_comp, comp.reindex(df.index).fillna(0.0).rename("raw_signal")
    )


def test_elr_overlay_derisk_when_leverage_stretched():
    df = _frame()
    base = composite_blend.generate_signals(df, dict(PARAMS))

    # Flat leverage for 400 days, then a sharp sustained spike: the z-score
    # stretches far above z0 and the overlay must scale exposure down.
    elr = pd.Series(0.2, index=df.index)
    elr.iloc[400:] = 0.6
    df2 = df.copy()
    df2["est_leverage"] = elr
    shaped = composite_blend.generate_signals(df2, dict(PARAMS))

    pd.testing.assert_series_equal(shaped.iloc[:400], base.iloc[:400])
    spike = shaped.iloc[410:440]
    base_spike = base.iloc[410:440]
    mask = base_spike > 0.01
    assert (spike[mask] < base_spike[mask]).all()
    # Floor respected: never scaled below floor * base.
    assert (spike[mask] >= 0.25 * base_spike[mask] - 1e-12).all()

    off = composite_blend.generate_signals(
        df2, {**PARAMS, "elr_overlay": False}
    )
    pd.testing.assert_series_equal(off, base)


def test_elr_overlay_causal_truncation_invariant():
    # Truncation point is kept past train_end + IC horizon so the composite
    # leg's train-window IC signs are identical; what is being tested is that
    # appending future rows never changes past overlay-scaled weights.
    df = _frame()
    rng = np.random.default_rng(13)
    df["est_leverage"] = 0.2 + np.cumsum(rng.normal(0, 0.003, len(df)))
    full = composite_blend.generate_signals(df, dict(PARAMS))
    trunc = composite_blend.generate_signals(df.iloc[:580], dict(PARAMS))
    pd.testing.assert_series_equal(full.iloc[:580], trunc)
