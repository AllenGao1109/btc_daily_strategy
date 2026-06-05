"""Train a REINFORCE position-sizer on the train window; evaluate OOS honestly.

Run:  python3 rl_research.py [--epochs N] [--seeds K]
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.features import build_features
from src.metrics import sharpe_ratio
from src.onchain import load_coinmetrics, merge_onchain
from src.research import run_full, window_metrics
from src.strategies import get_strategy
from src.validation import make_fixed_split
from src.ml.dataset import add_onchain_features, select_features
from src.ml.rl import rl_signal
from src.ml.torch_models import get_device


def yearly(res, lo, hi):
    out = {}
    for yr in range(lo, hi + 1):
        sub = res[res.index.year == yr]["strategy_daily_return"]
        out[yr] = sharpe_ratio(sub, 365) if len(sub) > 30 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    df = add_onchain_features(merge_onchain(df, load_coinmetrics(["CapMVRVCur", "AdrActCnt"])))
    splits = make_fixed_split(cfg["validation"])
    cols = select_features(df, use_onchain=True)
    train_end = pd.Timestamp(cfg["validation"]["train_end"], tz="UTC")
    print(f"device={get_device()}  features={len(cols)}  train_end={train_end.date()}  epochs={args.epochs}")

    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
    ens = run_full(df, get_strategy("trend_ensemble")(
        df, {"sma_windows": [50, 100, 200], "use_donchian": True,
             "target_vol": 0.55, "vol_window": 45}), bt)

    # Average the policy over several seeds (variance reduction).
    t0 = time.time()
    sigs = []
    for s in range(args.seeds):
        sigs.append(rl_signal(df, cols, train_end, actions=(0.0, 1.0, 2.0),
                              epochs=args.epochs, seed=s))
    sig = pd.concat(sigs, axis=1).mean(axis=1)
    res = run_full(df, sig, bt)
    print(f"trained {args.seeds} seeds in {time.time()-t0:.0f}s")

    print(f"\n{'strat':10s} | " + " | ".join(f"{w[:3]}" for w in ["train", "validation", "test"]))
    for name, r in [("BH", bh), ("ENS", ens), ("RL", res)]:
        cells = []
        for w in ["train", "validation", "test"]:
            m = window_metrics(r, splits[w])
            cells.append(f"Sh{m['sharpe_ratio']:5.2f}/NAV{m['final_nav']:5.2f}")
        print(f"{name:10s} | " + " | ".join(cells))

    print("\nyear-by-year Sharpe:")
    yb = {n: yearly(r, 2019, 2026) for n, r in [("BH", bh), ("ENS", ens), ("RL", res)]}
    print("year  " + "  ".join(f"{n:>6s}" for n in yb))
    for yr in range(2019, 2027):
        print(f"{yr}  " + "  ".join(f"{yb[n][yr]:6.2f}" for n in yb))
    for n in yb:
        vals = [v for v in yb[n].values() if not np.isnan(v)]
        print(f"{n} mean-yearly Sharpe {np.mean(vals):.2f}, pos {sum(v>0 for v in vals)}/{len(vals)}, "
              f"min {np.min(vals):.2f}")


if __name__ == "__main__":
    main()
