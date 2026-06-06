"""Sweep RL reward designs / action sets; rank by year-by-year robustness vs BH.

Run:  python3 rl_sweep.py [--epochs N] [--seeds K]
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


def fold_stats(res, lo=2019, hi=2026):
    s = []
    for yr in range(lo, hi + 1):
        sub = res[res.index.year == yr]["strategy_daily_return"]
        if len(sub) > 30:
            s.append(sharpe_ratio(sub, 365))
    return float(np.mean(s)), float(np.min(s)), sum(v > 0 for v in s), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    df = add_onchain_features(merge_onchain(df, load_coinmetrics(["CapMVRVCur", "AdrActCnt"])))
    splits = make_fixed_split(cfg["validation"])
    cols = select_features(df, use_onchain=True)
    train_end = pd.Timestamp(cfg["validation"]["train_end"], tz="UTC")

    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
    bm, bmin, bpos, bn = fold_stats(bh)
    bh_test = window_metrics(bh, splits["test"])
    print(f"buy-hold: mean-yearly {bm:.2f}, min {bmin:.2f}, pos {bpos}/{bn}, "
          f"test Sh {bh_test['sharpe_ratio']:.2f}/NAV {bh_test['final_nav']:.2f}\n")

    configs = [
        ("pnl   {0,1,2}", dict(reward_type="pnl", actions=(0.0, 1.0, 2.0))),
        ("logutil {0,1,2}", dict(reward_type="logutil", actions=(0.0, 1.0, 2.0))),
        ("logutil {0,1}", dict(reward_type="logutil", actions=(0.0, 1.0))),
        ("logutil fine", dict(reward_type="logutil", actions=(0.0, 0.5, 1.0, 1.5, 2.0))),
        ("vol_pen {0,1,2}", dict(reward_type="vol_pen", actions=(0.0, 1.0, 2.0), vol_coef=5.0)),
    ]
    print(f"{'config':16s} {'meanYr':>6s} {'minYr':>6s} {'pos':>5s} {'tstSh':>6s} {'tstNAV':>6s}  beats BH?")
    for name, kw in configs:
        t0 = time.time()
        sigs = [rl_signal(df, cols, train_end, epochs=args.epochs, seed=s, **kw)
                for s in range(args.seeds)]
        res = run_full(df, pd.concat(sigs, axis=1).mean(axis=1), bt)
        m, mn, pos, n = fold_stats(res)
        te = window_metrics(res, splits["test"])
        beat = "Y" if (m > bm and te["final_nav"] > bh_test["final_nav"]) else "."
        print(f"{name:16s} {m:6.2f} {mn:6.2f} {pos:3d}/{n} {te['sharpe_ratio']:6.2f} "
              f"{te['final_nav']:6.2f}  {beat}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
