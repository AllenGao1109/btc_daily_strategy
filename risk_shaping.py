"""Risk-shaping research: ELR as a de-leveraging OVERLAY vs as a composite factor.

Motivation (see RESEARCH_FINDINGS): elr_z_180 (system leverage, OI/reserve) is
the library's strongest standalone IC but adds nothing to return selection —
it is a RISK factor. The right place for a risk factor may be a position
overlay (scale exposure down when system leverage is stretched), not the
IC-weighted return composite.

PRE-REGISTERED OBJECTIVE AND SELECTION RULE (decided before running):
  Alternate objective = drawdown-constrained return. Select, on train+val
  only: maximize VALIDATION Calmar (annualized return / |window MaxDD|),
  subject to validation Sharpe >= 1.40 (95% of the incumbent's 1.47);
  tie-break = fewer trades. The test column is reported for the record and
  read only after selection. The production default (max validation Sharpe)
  is NOT changed by this script; the winner is registered as an opt-in
  preset for a drawdown-constrained owner objective.

Overlay form (causal: day-t factor, engine lags execution by 1 day):
  mult_t = clip(1 - k * max(0, elr_z_t - z0), floor, 1)
  weight_t = blend_weight_t * mult_t        (no data -> mult = 1)

Run:  python3 risk_shaping.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.factors import build_factors
from src.metrics import sharpe_ratio
from src.research import load_research_frame, run_full, window_metrics
from src.strategies import get_strategy
from src.strategies.factor_composite import DEFAULT_FACTORS
from src.validation import make_fixed_split

GRID_Z0 = [0.5, 1.0]
GRID_K = [0.25, 0.5]
FLOOR = 0.25
MIN_VAL_SHARPE = 1.40


def main() -> None:
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = load_research_frame(cfg)
    splits = make_fixed_split(cfg["validation"])
    prod_params = dict(cfg["strategy"]["params"])

    blend_w = get_strategy("composite_blend")(df, dict(prod_params))
    elr_z = build_factors(df)["elr_z_180"]

    def overlay(z0: float, k: float) -> pd.Series:
        mult = (1.0 - k * (elr_z - z0).clip(lower=0.0)).clip(FLOOR, 1.0)
        w = blend_w * mult.fillna(1.0)
        w.name = "raw_signal"
        return w

    variants: dict[str, pd.Series] = {"BLEND (prod)": blend_w}
    for z0 in GRID_Z0:
        for k in GRID_K:
            variants[f"OV z0={z0} k={k}"] = overlay(z0, k)
    variants["F +ELR+EPU"] = get_strategy("composite_blend")(
        df, {**prod_params, "factors": DEFAULT_FACTORS + ["elr_z_180", "epu_z_60"]}
    )

    def yb(res, lo=2019, hi=2026):
        return {yr: (sharpe_ratio(res[res.index.year == yr]["strategy_daily_return"], 365)
                if len(res[res.index.year == yr]) > 30 else float("nan"))
                for yr in range(lo, hi + 1)}

    print(f"Selection rule (pre-registered): max validation Calmar s.t. "
          f"val Sharpe >= {MIN_VAL_SHARPE}; tie-break fewer trades.\n")
    print(f"{'variant':16s} | {'val Sh':>6s} {'valDD':>6s} {'valCal':>7s} | "
          f"{'tra Sh':>6s} {'traDD':>6s} | {'test Sh/NAV/DD':>17s} | trd  fullDD minYr")
    best, best_key = None, (-np.inf, np.inf)
    for name, w in variants.items():
        r = run_full(df, w, bt)
        c = {x: window_metrics(r, splits[x]) for x in ["train", "validation", "test"]}
        full = window_metrics(r, None)
        va = c["validation"]
        val_calmar = va["cagr"] / abs(va["max_drawdown"]) if va["max_drawdown"] != 0 else 0.0
        d = yb(r)
        vals = [v for v in d.values() if not np.isnan(v)]
        eligible = va["sharpe_ratio"] >= MIN_VAL_SHARPE
        key = (val_calmar if eligible else -np.inf, full["num_trades"])
        if eligible and (key[0] > best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1])):
            best, best_key = name, key
        print(f"{name:16s} | {va['sharpe_ratio']:6.2f} {va['max_drawdown']:6.0%} "
              f"{val_calmar:7.2f} | {c['train']['sharpe_ratio']:6.2f} "
              f"{c['train']['max_drawdown']:6.0%} | "
              f"{c['test']['sharpe_ratio']:5.2f}/{c['test']['final_nav']:4.2f}/"
              f"{c['test']['max_drawdown']:5.0%} | {full['num_trades']:3d}  "
              f"{full['max_drawdown']:5.0%} {min(vals):5.2f}"
              + ("   <- eligible" if eligible else ""))
    print(f"\nSELECTED (by the pre-registered rule): {best}")


if __name__ == "__main__":
    main()
