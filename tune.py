"""Disciplined tune of the trend ensemble on TRAIN+VALIDATION only.

Selection objective (robustness-first, not single-window mining):
    maximize  min(train_sharpe, validation_sharpe)
    subject to validation max drawdown better than buy-and-hold 1x and no liquidation.

The held-out TEST window is computed for the single chosen config only, and
reported once for honesty. It is never used to rank configs.

Run:  python3 tune.py
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from itertools import product

warnings.filterwarnings("ignore")

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.features import build_features
from src.research import run_full, window_metrics
from src.strategies import get_strategy
from src.validation import make_fixed_split

GRID = {
    "target_vol": [0.30, 0.40, 0.45, 0.55],
    "vol_window": [20, 30, 45],
    "weight_band": [0.10, 0.15, 0.20],
}


def main():
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    splits = make_fixed_split(cfg["validation"])

    # Buy-and-hold 1x validation drawdown is the drawdown constraint reference.
    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
    bh_val_dd = window_metrics(bh, splits["validation"])["max_drawdown"]

    keys = list(GRID)
    results = []
    for combo in product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        params = {
            "sma_windows": [50, 100, 200],
            "use_donchian": True,
            "target_vol": p["target_vol"],
            "vol_window": p["vol_window"],
        }
        bt_run = replace(bt, weight_band=p["weight_band"])
        raw = get_strategy("trend_ensemble")(df, params)
        res = run_full(df, raw, bt_run)
        tr = window_metrics(res, splits["train"])
        va = window_metrics(res, splits["validation"])
        feasible = (va["max_drawdown"] >= bh_val_dd) and (res["liquidated"].sum() == 0)
        results.append({
            "params": p,
            "robust_sharpe": min(tr["sharpe_ratio"], va["sharpe_ratio"]),
            "train_sharpe": tr["sharpe_ratio"],
            "val_sharpe": va["sharpe_ratio"],
            "val_calmar": va["calmar_ratio"],
            "val_maxdd": va["max_drawdown"],
            "val_trades": va["num_trades"],
            "feasible": feasible,
        })

    feasible = [r for r in results if r["feasible"]] or results
    feasible.sort(key=lambda r: r["robust_sharpe"], reverse=True)

    print(f"BH 1x validation maxDD (constraint floor): {bh_val_dd*100:.1f}%\n")
    print("Top configs by min(train,val) Sharpe:")
    print(f"{'tvol':>5s} {'vwin':>5s} {'band':>5s} {'robust':>7s} "
          f"{'trainS':>7s} {'valS':>6s} {'valCal':>7s} {'valDD':>7s} {'vTrd':>5s}")
    for r in feasible[:8]:
        p = r["params"]
        print(f"{p['target_vol']:5.2f} {p['vol_window']:5d} {p['weight_band']:5.2f} "
              f"{r['robust_sharpe']:7.2f} {r['train_sharpe']:7.2f} {r['val_sharpe']:6.2f} "
              f"{r['val_calmar']:7.2f} {r['val_maxdd']*100:6.1f}% {int(r['val_trades']):5d}")

    best = feasible[0]
    bp = best["params"]
    params = {"sma_windows": [50, 100, 200], "use_donchian": True,
              "target_vol": bp["target_vol"], "vol_window": bp["vol_window"]}
    bt_run = replace(bt, weight_band=bp["weight_band"])
    res = run_full(df, get_strategy("trend_ensemble")(df, params), bt_run)

    print(f"\nCHOSEN (by train+val robustness): {bp}")
    print(f"{'window':12s} {'NAV':>7s} {'CAGR':>8s} {'Sharpe':>7s} "
          f"{'MaxDD':>8s} {'Calmar':>7s} {'trd':>5s}")
    for w in ["train", "validation", "test"]:
        m = window_metrics(res, splits[w])
        print(f"{w:12s} {m['final_nav']:7.2f} {m['cagr']*100:7.1f}% "
              f"{m['sharpe_ratio']:7.2f} {m['max_drawdown']*100:7.1f}% "
              f"{m['calmar_ratio']:7.2f} {int(m['num_trades']):5d}")
    return best


if __name__ == "__main__":
    main()
