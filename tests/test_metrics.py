"""20.19 metric correctness against a known equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics as m


def _equity(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def test_total_return():
    eq = _equity([10000, 11000, 12000])
    assert m.total_return(eq, 10000) == pytest.approx(0.20)


def test_max_drawdown_known_curve():
    # Peak 120 then trough 60 -> 50% drawdown.
    eq = _equity([100, 120, 90, 60, 80])
    assert m.max_drawdown(eq) == pytest.approx(-0.5)


def test_cagr_one_year_doubling():
    # 365 daily points, doubling over exactly one year -> CAGR ~ 100%.
    vals = np.linspace(10000, 20000, 365)
    eq = _equity(vals)
    assert m.cagr(eq, 10000, 365) == pytest.approx(1.0, rel=0.02)


def test_volatility_constant_return_is_zero():
    # Constant daily return -> ~zero realized volatility.
    rets = pd.Series([0.001] * 100)
    assert m.annualized_volatility(rets, 365) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_known_values():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.02, 1000))
    sharpe = m.sharpe_ratio(rets, 365)
    expected = rets.mean() / rets.std(ddof=1) * np.sqrt(365)
    assert sharpe == pytest.approx(expected)


def test_calmar():
    assert m.calmar_ratio(0.20, -0.10) == pytest.approx(2.0)
