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

ELR DE-LEVERAGING OVERLAY (2026-06): system leverage (CryptoQuant Estimated
Leverage Ratio, OI/exchange-reserve) is the library's strongest standalone IC
but adds nothing inside the IC-weighted return composite — it is a RISK
factor. Applied instead as a position overlay: when the 180d z-score of ELR
stretches above ``elr_z0``, exposure is scaled down linearly (slope
``elr_k``), floored at ``elr_floor``. Selected on train+val only across a
2x2 (z0, k) grid: validation Sharpe 1.47 -> 1.64 (every grid corner
improved), validation MaxDD -25% -> -17%; tie-break by fewer trades. Missing
ELR data (pre-2019 or unmerged frames) leaves the multiplier at 1 — the
overlay degrades away gracefully and remains causal (day-t factor, engine
lags execution).
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

# ELR overlay defaults (production). Set ``elr_overlay: false`` to disable.
DEFAULT_ELR_Z0 = 0.5
DEFAULT_ELR_K = 0.5
DEFAULT_ELR_FLOOR = 0.25


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate blended raw target weights with the ELR de-leveraging overlay.

    Args:
        df: Enriched OHLCV frame (the composite leg needs its merged columns;
            the overlay needs ``est_leverage`` and is skipped without it).
        params: Keys:
            ``blend`` weight on the composite leg (default 0.75);
            ``ensemble`` dict of trend_ensemble params (default
            DEFAULT_ENS_PARAMS); ``elr_overlay`` (default True),
            ``elr_z0``/``elr_k``/``elr_floor`` overlay knobs; every other key
            is passed to the composite leg unchanged (``train_end`` is
            required by that leg).

    Returns:
        Raw target weight Series named ``raw_signal`` (warmup NaNs filled 0).
    """
    params = dict(params or {})
    blend = float(params.pop("blend", DEFAULT_BLEND))
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1]; got {blend}.")
    ens_params = params.pop("ensemble", DEFAULT_ENS_PARAMS)
    use_overlay = bool(params.pop("elr_overlay", True))
    z0 = float(params.pop("elr_z0", DEFAULT_ELR_Z0))
    k = float(params.pop("elr_k", DEFAULT_ELR_K))
    floor = float(params.pop("elr_floor", DEFAULT_ELR_FLOOR))

    comp = factor_composite.generate_signals(df, params)
    ens = trend_ensemble.generate_signals(df, ens_params)
    out = blend * comp.reindex(df.index).fillna(0.0) + (1.0 - blend) * ens.reindex(
        df.index
    ).fillna(0.0)

    if use_overlay and "est_leverage" in df.columns:
        elr = df["est_leverage"].astype(float)
        z = (elr - elr.rolling(180).mean()) / elr.rolling(180).std()
        mult = (1.0 - k * (z - z0).clip(lower=0.0)).clip(floor, 1.0)
        out = out * mult.fillna(1.0)

    out.name = "raw_signal"
    return out
