"""Factor-composite strategy: IC-weighted robust factors -> vol-targeted position.

Combines economically-grounded factors whose information coefficient is stable
across train AND validation (BTC halving-cycle phase, return kurtosis, volatility
regime, MVRV valuation z-score, long-horizon momentum, exchange net-flow). The
composite is built from train-only factor signs (no lookahead; the engine also
lags the signal), then mapped to a long-biased, volatility-targeted weight so the
position scales up in cycle/valuation-favorable, calm regimes and down otherwise.

Deterministic — no random seeds — so results are reproducible by construction.

Requires the on-chain columns (CapMVRVCur, FlowInExUSD, ...) to be present on the
frame; the caller merges them (see src.onchain). ``params['train_end']`` is the
last date whose data may inform factor signs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factors import composite_signal

DEFAULT_FACTORS = [
    "halving_cos", "kurt_30", "vol_regime", "mvrv_z_365",
    "mom_120", "mvrv_mom_30", "ex_netflow_to_mcap",
    "btc_eth_rs_30",  # BTC relative strength vs ETH — the factor that cleared BH
]


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate the vol-targeted factor-composite target weights.

    Args:
        df: OHLCV frame with on-chain columns merged in.
        params: Keys:
            ``train_end`` (required) ISO date; factor signs use only data <= it.
            ``factors`` (default DEFAULT_FACTORS), ``horizon`` (default 20),
            ``tilt`` long-bias offset (default 0.5), ``target_vol`` (default 0.55),
            ``vol_window`` (default 45), ``max_weight`` (default 2.0).

    Returns:
        Raw target weights in [0, max_weight] (long/flat), NaN during warmup.
    """
    params = params or {}
    if "train_end" not in params:
        raise KeyError("factor_composite requires params['train_end'].")
    train_end = pd.Timestamp(params["train_end"], tz="UTC")
    factors = params.get("factors", DEFAULT_FACTORS)
    horizon = int(params.get("horizon", 20))
    tilt = float(params.get("tilt", 0.5))
    target_vol = float(params.get("target_vol", 0.55))
    vol_window = int(params.get("vol_window", 45))
    max_w = float(params.get("max_weight", 2.0))

    score = composite_signal(df, train_end, factors, horizon=horizon)

    ret = df["close"].astype(float).pct_change()
    rvol = (ret.rolling(vol_window).std() * np.sqrt(365)).replace(0.0, np.nan)
    vt_size = (target_vol / rvol).clip(upper=max_w)

    weight = ((tilt + score).clip(0.0, max_w) * vt_size).clip(0.0, max_w)
    weight.name = "raw_signal"
    return weight
