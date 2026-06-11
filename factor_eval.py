"""Evaluate the robust-factor composite signal through the engine, year-by-year.

Deterministic (no seeds), so reproducibility is automatic. Factor IC signs come
from the train window only; positions are sized by a volatility target.

Run:  python3 factor_eval.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.factors import composite_signal
from src.features import build_features
from src.metrics import sharpe_ratio
from src.macro import load_dxy, load_fred_macro
from src.onchain import (
    ONCHAIN_METRICS,
    load_coinmetrics,
    load_stablecoin_mcap,
    merge_onchain,
)
from src.research import run_full, window_metrics
from src.sentiment import load_cnn_fear_greed, load_crypto_fear_greed
from src.strategies import get_strategy
from src.validation import make_fixed_split

ROBUST_FACTORS = ["halving_cos", "kurt_30", "vol_regime", "mvrv_z_365",
                  "mom_120", "mvrv_mom_30"]
# Sentiment addition: CNN equity Fear & Greed z-score is the only sentiment
# factor robust on all horizons in factor_mining (negative IC = contrarian).
SENT_FACTORS = ROBUST_FACTORS + ["cnnfg_z_60"]


def yb(res, lo=2019, hi=2026):
    return {yr: (sharpe_ratio(res[res.index.year == yr]["strategy_daily_return"], 365)
            if len(res[res.index.year == yr]) > 30 else float("nan"))
            for yr in range(lo, hi + 1)}


def to_weight(score, df, target_vol=0.55, vol_window=45, long_only=True, max_w=2.0):
    ret = df["close"].astype(float).pct_change()
    rvol = ret.rolling(vol_window).std() * np.sqrt(365)
    size = (target_vol / rvol.replace(0.0, np.nan)).clip(upper=max_w)
    direction = score.clip(lower=0.0) if long_only else score
    w = (direction * size).clip(-max_w, max_w)
    w.name = "raw_signal"
    return w


def main():
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    df = merge_onchain(df, load_coinmetrics(ONCHAIN_METRICS))
    df = merge_onchain(df, load_crypto_fear_greed())
    df = merge_onchain(df, load_cnn_fear_greed())
    df = merge_onchain(df, load_stablecoin_mcap())
    df = merge_onchain(df, load_fred_macro())
    df = merge_onchain(df, load_dxy())
    # ETH close for the cross-crypto factor (CoinMetrics daily price, ffilled).
    eth = load_coinmetrics(["PriceUSD"], asset="eth").rename(
        columns={"PriceUSD": "eth_close"}
    )
    df = merge_onchain(df, eth)
    splits = make_fixed_split(cfg["validation"])
    train_end = pd.Timestamp(cfg["validation"]["train_end"], tz="UTC")

    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
    ens = run_full(df, get_strategy("trend_ensemble")(
        df, {"sma_windows": [50, 100, 200], "use_donchian": True,
             "target_vol": 0.55, "vol_window": 45}), bt)

    score = composite_signal(df, train_end, ROBUST_FACTORS, horizon=20)
    score_s = composite_signal(df, train_end, SENT_FACTORS, horizon=20)

    # Several position mappings; selection is by the train+val years only.
    ret = df["close"].astype(float).pct_change()
    rvol = (ret.rolling(45).std() * np.sqrt(365)).replace(0.0, np.nan)
    vt_size = (0.55 / rvol).clip(upper=2.0)

    def mk(name, w):
        w = w.clip(-2.0, 2.0)
        w.name = "raw_signal"
        return name, run_full(df, w, bt)

    mappings = [
        mk("F_flatVT", (score > 0).astype(float) * vt_size),     # long/flat, vol-target size
        mk("F_lev", score.clip(lower=0.0) * 2.0),                 # signal-scaled leverage 0..2x
        mk("F_levVT", (0.5 + score).clip(0.0, 2.0) * vt_size),   # bull-tilt x vol-target
        # Same mappings with the CNN sentiment factor added to the composite.
        mk("S_flatVT", (score_s > 0).astype(float) * vt_size),
        mk("S_lev", score_s.clip(lower=0.0) * 2.0),
        mk("S_levVT", (0.5 + score_s).clip(0.0, 2.0) * vt_size),
    ]
    # Production factor_composite strategy with / without the sentiment factor.
    from src.strategies.factor_composite import DEFAULT_FACTORS
    prod_params = dict(cfg["strategy"]["params"])
    mappings.append(("PROD", run_full(
        df, get_strategy("factor_composite")(df, prod_params), bt)))
    mappings.append(("PROD+S", run_full(
        df, get_strategy("factor_composite")(
            df, {**prod_params, "factors": DEFAULT_FACTORS + ["cnnfg_z_60"]}), bt)))
    rows = [("BH", bh), ("ENS", ens)] + mappings
    print(f"{'strat':9s} | " + " | ".join(f"{w[:3]}" for w in ["train", "validation", "test"])
          + " || meanYr  minYr  pos  trd  [tv-mean]")
    for name, r in rows:
        cells = []
        for w in ["train", "validation", "test"]:
            m = window_metrics(r, splits[w])
            cells.append(f"Sh{m['sharpe_ratio']:5.2f}/NAV{m['final_nav']:5.2f}")
        d = yb(r)
        vals = [v for v in d.values() if not np.isnan(v)]
        tv = [d[y] for y in range(2019, 2024) if not np.isnan(d[y])]  # train+val years
        full = window_metrics(r, None)
        print(f"{name:8s} | " + " | ".join(cells)
              + f" || {np.mean(vals):5.2f}  {np.min(vals):5.2f}  {sum(v>0 for v in vals)}/{len(vals)}"
              + f"  {full['num_trades']}  [tv-mean {np.mean(tv):.2f}]")

    print()
    for name, r in [("BH", bh)] + mappings:
        d = yb(r)
        print(f"{name:9s} yby:", "  ".join(f"{y}:{d[y]:5.2f}" for y in range(2019, 2027)))


if __name__ == "__main__":
    main()
