"""Autonomous factor mining: rank candidate factors by out-of-sample IC stability.

For each factor and horizon, compute the information coefficient (rank
correlation with forward returns) separately on TRAIN and VALIDATION. A factor is
"robust" only if its IC has the SAME sign and meaningful magnitude on BOTH — a
train-only IC is just in-sample overfitting. The test window is never used here.

BREADTH DIAGNOSTIC (lesson from the macro/exchange-supply round): a window-level
IC gate is necessary but NOT sufficient when train+val is dominated by one macro
regime (2022-23). For each factor we also report, over the selectable years
(2019-2023), how many yearly h20 ICs support vs oppose the window-level sign
(|IC| >= 0.05 ~ one standard error on ~365 obs counts as a vote). Read it with
judgment, not as a hard gate: fast factors should be broad-based
(yr_support >= 3, yr_oppose == 0); slow cycle factors (halving phase, MVRV)
legitimately score poorly within single years because they barely vary inside
one year — judge those by mechanism and multi-year windows instead.

Run:  python3 factor_mining.py
Appends a ranked table to results/factor_ic.csv.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import load_config
from src.factors import build_factors, forward_return, information_coefficient
from src.research import load_research_frame
from src.validation import make_fixed_split

HORIZONS = [1, 5, 20]
ROBUST_IC = 0.03  # |IC| threshold on a single window (daily data is noisy)
BREADTH_YEARS = range(2019, 2024)  # selectable years (train+val, post-warmup)
BREADTH_IC = 0.05  # ~1 s.e. of a Spearman IC on one year of daily data


def yearly_breadth(
    f: pd.Series, fr: pd.Series, window_sign: float
) -> tuple[int, int]:
    """Count selectable years whose yearly IC supports / opposes ``window_sign``."""
    support = oppose = 0
    for yr in BREADTH_YEARS:
        mask = f.index.year == yr
        ic = information_coefficient(f[mask], fr[mask])
        if abs(ic) < BREADTH_IC or window_sign == 0:
            continue
        if np.sign(ic) == window_sign:
            support += 1
        else:
            oppose += 1
    return support, oppose


def main():
    cfg = load_config("config.yaml")
    df = load_research_frame(cfg)
    factors = build_factors(df)
    splits = make_fixed_split(cfg["validation"])
    tr_idx = splits["train"].slice(df).index
    va_idx = splits["validation"].slice(df).index

    rows = []
    for col in factors.columns:
        f = factors[col]
        rec = {"factor": col}
        robust_h = []
        for h in HORIZONS:
            fr = forward_return(df, h)
            ic_tr = information_coefficient(f.loc[tr_idx], fr.loc[tr_idx])
            ic_va = information_coefficient(f.loc[va_idx], fr.loc[va_idx])
            rec[f"ic_tr_h{h}"] = round(ic_tr, 4)
            rec[f"ic_va_h{h}"] = round(ic_va, 4)
            # Robust: same sign on train+val and both above the noise floor.
            if abs(ic_tr) > ROBUST_IC and abs(ic_va) > ROBUST_IC and np.sign(ic_tr) == np.sign(ic_va):
                robust_h.append((h, ic_tr, ic_va))
        rec["n_robust_h"] = len(robust_h)
        # Score = sum of min(|ic_tr|,|ic_va|) over robust horizons (conservative).
        rec["robust_score"] = round(
            sum(min(abs(a), abs(b)) for _, a, b in robust_h), 4
        )
        # Yearly breadth at h20 vs the window-level sign (see module docstring).
        fr20 = forward_return(df, 20)
        sign20 = np.sign(rec["ic_tr_h20"] + rec["ic_va_h20"])
        sup, opp = yearly_breadth(f, fr20, sign20)
        rec["yr_support"] = sup
        rec["yr_oppose"] = opp
        rows.append(rec)

    table = pd.DataFrame(rows).sort_values(
        ["n_robust_h", "robust_score"], ascending=False
    )
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/factor_ic.csv", index=False)

    print(f"Factors: {len(table)}  |  robust on >=1 horizon: {(table['n_robust_h']>0).sum()}")
    print("\n=== top factors by train+val IC stability ===")
    print(f"{'factor':22s} {'nH':>3s} {'score':>6s} | "
          f"{'ic_tr_h5':>8s} {'ic_va_h5':>8s} | {'ic_tr_h20':>9s} {'ic_va_h20':>9s} | breadth")
    for _, r in table.head(15).iterrows():
        breadth = f"{int(r['yr_support'])}+/{int(r['yr_oppose'])}-"
        note = "" if r["yr_oppose"] == 0 else "  <- event-concentrated or slow-cycle?"
        print(f"{r['factor']:22s} {int(r['n_robust_h']):3d} {r['robust_score']:6.3f} | "
              f"{r['ic_tr_h5']:8.3f} {r['ic_va_h5']:8.3f} | {r['ic_tr_h20']:9.3f} {r['ic_va_h20']:9.3f}"
              f" | {breadth:6s}{note}")


if __name__ == "__main__":
    main()
