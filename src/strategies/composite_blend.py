"""Composite x trend-ensemble blend: the strategy-level diversification layer.

Blends the RAW target weights of two mechanically different survivors:
  - factor_composite (cycle/valuation/sentiment composite — downside
    protection, the long-time production strategy), and
  - trend_ensemble (vol-targeted price-trend ensemble — pure price risk
    control, fee-heavy standalone but cheap as a minority leg because weight
    averaging nets opposing trades).

Selected on train+validation only (2026-06 grid: validation Sharpe 1.47 vs
1.31 for the composite alone, best worst-year, FEWER trades than either leg
traded separately). See RESEARCH_FINDINGS.md; the statistical-significance
caveats recorded there apply to this strategy like everything else.

The blend is a convex combination of leg weights, so it inherits the legs'
[-2, 2] bounds; the engine still applies the execution lag and leverage caps.
"""

from __future__ import annotations

import pandas as pd

from . import factor_composite, trend_ensemble

DEFAULT_BLEND = 0.75  # weight on the factor composite; 1-x on the ensemble

DEFAULT_ENS_PARAMS = {
    "sma_windows": [50, 100, 200],
    "use_donchian": True,
    "target_vol": 0.55,
    "vol_window": 45,
}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate blended raw target weights.

    Args:
        df: Enriched OHLCV frame (the composite leg needs its merged columns).
        params: Keys:
            ``blend`` weight on the composite leg (default 0.75);
            ``ensemble`` dict of trend_ensemble params (default
            DEFAULT_ENS_PARAMS); every other key is passed to the composite
            leg unchanged (``train_end`` is required by that leg).

    Returns:
        Raw target weight Series named ``raw_signal`` (warmup NaNs filled 0).
    """
    params = dict(params or {})
    blend = float(params.pop("blend", DEFAULT_BLEND))
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1]; got {blend}.")
    ens_params = params.pop("ensemble", DEFAULT_ENS_PARAMS)

    comp = factor_composite.generate_signals(df, params)
    ens = trend_ensemble.generate_signals(df, ens_params)
    out = blend * comp.reindex(df.index).fillna(0.0) + (1.0 - blend) * ens.reindex(
        df.index
    ).fillna(0.0)
    out.name = "raw_signal"
    return out
