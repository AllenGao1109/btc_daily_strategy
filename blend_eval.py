"""Strategy-LEVEL blending: composite x trend-ensemble x buy-and-hold.

Factor mining has hit diminishing returns; the remaining diversification is at
the strategy layer. We blend the RAW TARGET WEIGHTS of mechanically different
survivors (factor composite = cycle/valuation downside protection; vol-target
trend ensemble = price-trend risk control; buy-and-hold = full bull capture)
and let the engine trade the netted blend — averaging weights nets opposing
trades, so a blend can be cheaper than its legs.

Discipline as everywhere else: blend weights are selected on train+validation
only (validation Sharpe, fee-prior tie-break); the test column is reported for
the record and checked only after selection.

Run:  python3 blend_eval.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.metrics import sharpe_ratio
from src.research import load_research_frame, run_full, window_metrics
from src.strategies import get_strategy
from src.validation import make_fixed_split

ENS_PARAMS = {"sma_windows": [50, 100, 200], "use_donchian": True,
              "target_vol": 0.55, "vol_window": 45}


def main() -> None:
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = load_research_frame(cfg)
    splits = make_fixed_split(cfg["validation"])

    comp = get_strategy("factor_composite")(df, dict(cfg["strategy"]["params"]))
    ens = get_strategy("trend_ensemble")(df, ENS_PARAMS)
    bh = pd.Series(1.0, index=df.index, name="raw_signal")

    def blend(parts: list[tuple[float, pd.Series]]) -> pd.Series:
        out = sum(w * s.reindex(df.index).fillna(0.0) for w, s in parts)
        out.name = "raw_signal"
        return out

    candidates = {
        "COMP (prod)":   blend([(1.0, comp)]),
        "ENS":           blend([(1.0, ens)]),
        "C75/E25":       blend([(0.75, comp), (0.25, ens)]),
        "C50/E50":       blend([(0.50, comp), (0.50, ens)]),
        "C25/E75":       blend([(0.25, comp), (0.75, ens)]),
        "C75/B25":       blend([(0.75, comp), (0.25, bh)]),
        "C50/B50":       blend([(0.50, comp), (0.50, bh)]),
        "C/E/B thirds":  blend([(1 / 3, comp), (1 / 3, ens), (1 / 3, bh)]),
        "BH":            bh,
    }

    def yb(res, lo=2019, hi=2026):
        return {yr: (sharpe_ratio(res[res.index.year == yr]["strategy_daily_return"], 365)
                if len(res[res.index.year == yr]) > 30 else float("nan"))
                for yr in range(lo, hi + 1)}

    rows = {}
    print(f"{'blend':13s} | {'train':16s} | {'validation':16s} | {'test':16s} "
          f"| trd fees%  tv-mean minYr  maxDD")
    for name, sig in candidates.items():
        r = run_full(df, sig, bt)
        rows[name] = r
        c = {w: window_metrics(r, splits[w]) for w in ["train", "validation", "test"]}
        full = window_metrics(r, None)
        d = yb(r)
        tv = [d[y] for y in range(2019, 2024) if not np.isnan(d[y])]
        vals = [v for v in d.values() if not np.isnan(v)]
        fees = float(r["fee"].sum()) / bt.initial_capital * 100
        print(f"{name:13s} | "
              + " | ".join(f"Sh{c[w]['sharpe_ratio']:5.2f}/NAV{c[w]['final_nav']:5.2f}"
                           for w in ["train", "validation", "test"])
              + f" | {full['num_trades']:3d} {fees:5.0f}  {np.mean(tv):5.2f}  "
              f"{min(vals):5.2f}  {full['max_drawdown']:.0%}")

    print()
    for name in ["COMP (prod)", "C75/E25", "C50/E50", "C/E/B thirds"]:
        d = yb(rows[name])
        print(f"{name:13s} yby:", "  ".join(f"{y}:{d[y]:5.2f}" for y in sorted(d)))


if __name__ == "__main__":
    main()
