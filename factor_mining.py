"""Autonomous factor mining: rank candidate factors by out-of-sample IC stability.

For each factor and horizon, compute the information coefficient (rank
correlation with forward returns) separately on TRAIN and VALIDATION. A factor is
"robust" only if its IC has the SAME sign and meaningful magnitude on BOTH — a
train-only IC is just in-sample overfitting. The test window is never used here.

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
from src.data import load_btc_data
from src.factors import build_factors, forward_return, information_coefficient
from src.features import build_features
from src.onchain import load_coinmetrics, merge_onchain
from src.validation import make_fixed_split

HORIZONS = [1, 5, 20]
ROBUST_IC = 0.03  # |IC| threshold on a single window (daily data is noisy)


def main():
    cfg = load_config("config.yaml")
    df = build_features(load_btc_data(cfg))
    df = merge_onchain(df, load_coinmetrics(["CapMVRVCur", "AdrActCnt"]))
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
        rows.append(rec)

    table = pd.DataFrame(rows).sort_values(
        ["n_robust_h", "robust_score"], ascending=False
    )
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/factor_ic.csv", index=False)

    print(f"Factors: {len(table)}  |  robust on >=1 horizon: {(table['n_robust_h']>0).sum()}")
    print("\n=== top factors by train+val IC stability ===")
    print(f"{'factor':22s} {'nH':>3s} {'score':>6s} | "
          f"{'ic_tr_h5':>8s} {'ic_va_h5':>8s} | {'ic_tr_h20':>9s} {'ic_va_h20':>9s}")
    for _, r in table.head(15).iterrows():
        print(f"{r['factor']:22s} {int(r['n_robust_h']):3d} {r['robust_score']:6.3f} | "
              f"{r['ic_tr_h5']:8.3f} {r['ic_va_h5']:8.3f} | {r['ic_tr_h20']:9.3f} {r['ic_va_h20']:9.3f}")


if __name__ == "__main__":
    main()
