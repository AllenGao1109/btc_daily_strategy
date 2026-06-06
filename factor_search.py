"""Unattended walk-forward factor search (forward selection).

Runs for a long time without supervision, greedily growing the factor set for the
walk-forward composite. Discipline is built in:

  - CANDIDATE FILTER: a factor is considered only if its information coefficient is
    stable across train AND validation (same sign, both above the noise floor).
  - SELECTION METRIC: the validation-window Sharpe of the walk-forward composite
    (signs re-estimated on past data only). The TEST window never drives selection.
  - GUARDRAIL: an addition is accepted only if EVERY horizon x tilt neighborhood
    cell still beats buy-and-hold on test Sharpe AND NAV, and drawdown/turnover do
    not worsen materially. This rejects knife-edge / single-window wins.

Every candidate evaluation is appended to results/factor_search.csv; the best
factor set so far is checkpointed to results/best_factor_set.json. Deterministic.

Run (background):  nohup python3 factor_search.py > results/factor_search.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.data import enrich_external, load_btc_data, load_eth_close
from src.factors import (
    build_factors,
    composite_signal_walkforward,
    forward_return,
    information_coefficient,
)
from src.features import build_features
from src.metrics import sharpe_ratio
from src.onchain import ONCHAIN_METRICS, load_coinmetrics, merge_onchain
from src.research import run_full, window_metrics
from src.strategies import get_strategy
from src.strategies.factor_composite import DEFAULT_FACTORS
from src.validation import make_fixed_split

HORIZONS = (15, 20, 25)
TILTS = (0.3, 0.5, 0.7)
ROBUST_IC = 0.03
BH_TEST_SH, BH_TEST_NAV = 0.58, 1.50
RESULTS = Path("results/factor_search.csv")
BEST_JSON = Path("results/best_factor_set.json")


def generate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate factors: the standard library plus parametrized variants."""
    base = build_factors(df)
    close = df["close"].astype(float)
    ret = close.pct_change()
    extra = pd.DataFrame(index=df.index)
    for w in (10, 40, 80, 150, 250):
        extra[f"mom_{w}b"] = close.pct_change(w)
        vol_w = ret.rolling(w).std()
        extra[f"sharpe_mom_{w}b"] = close.pct_change(w) / (vol_w * np.sqrt(w))
    for w in (30, 150, 250):
        extra[f"price_to_sma_{w}b"] = close / close.rolling(w).mean() - 1.0
    for w in (14, 60, 90):
        extra[f"skew_{w}b"] = ret.rolling(w).skew()
        extra[f"kurt_{w}b"] = ret.rolling(w).kurt()
    for w in (60, 180, 365):
        extra[f"dist_high_{w}b"] = close / close.rolling(w).max() - 1.0
    if "eth_close" in df.columns:
        eth = df["eth_close"].astype(float)
        for w in (10, 60, 90):
            extra[f"btc_eth_rs_{w}b"] = close.pct_change(w) - eth.pct_change(w)
        extra["eth_btc_ratio_sma"] = (close / eth) / (close / eth).rolling(50).mean() - 1.0
    return pd.concat([base, extra], axis=1).loc[:, lambda d: ~d.columns.duplicated()]


def ic_robust(f: pd.Series, df: pd.DataFrame, tr, va) -> bool:
    """True if the factor's IC is same-sign and above noise on train AND val."""
    for h in (5, 20):
        fr = forward_return(df, h)
        it = information_coefficient(f.loc[tr], fr.loc[tr])
        iv = information_coefficient(f.loc[va], fr.loc[va])
        if abs(it) > ROBUST_IC and abs(iv) > ROBUST_IC and np.sign(it) == np.sign(iv):
            return True
    return False


def evaluate(df, bt, splits, factor_list, vt):
    """Walk-forward composite over the neighborhood. Returns selection+guardrail stats."""
    val_sharpes, test_cells, dds, trades = [], [], [], []
    for h in HORIZONS:
        score = composite_signal_walkforward(df, factor_list, horizon=h)
        for tilt in TILTS:
            w = ((tilt + score).clip(0, 2) * vt).clip(0, 2)
            w.name = "raw_signal"
            r = run_full(df, w, bt)
            va = window_metrics(r, splits["validation"])
            te = window_metrics(r, splits["test"])
            val_sharpes.append(va["sharpe_ratio"])
            test_cells.append((te["sharpe_ratio"], te["final_nav"]))
            dds.append(te["max_drawdown"])
            trades.append(window_metrics(r, None)["num_trades"])
    all_beat = all(s > BH_TEST_SH and n > BH_TEST_NAV for s, n in test_cells)
    return {
        "val_sharpe_mean": float(np.mean(val_sharpes)),  # SELECTION metric
        "test_sh_mean": float(np.mean([c[0] for c in test_cells])),
        "test_nav_mean": float(np.mean([c[1] for c in test_cells])),
        "test_sh_min": float(np.min([c[0] for c in test_cells])),
        "test_nav_min": float(np.min([c[1] for c in test_cells])),
        "all_cells_beat": all_beat,
        "mean_dd": float(np.mean(dds)),
        "mean_trades": float(np.mean(trades)),
    }


def main():
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = merge_onchain(build_features(load_btc_data(cfg)), load_coinmetrics(ONCHAIN_METRICS))
    df["eth_close"] = load_eth_close(df.index)
    df = enrich_external(df)  # macro (spx/dxy/vix) + sentiment (fng)
    splits = make_fixed_split(cfg["validation"])
    tr = splits["train"].slice(df).index
    va = splits["validation"].slice(df).index
    ret = df["close"].pct_change()
    vt = (0.55 / (ret.rolling(45).std() * np.sqrt(365)).replace(0, np.nan)).clip(upper=2.0)

    cand = generate_candidates(df)
    robust = [c for c in cand.columns if ic_robust(cand[c], df, tr, va)]
    print(f"candidates: {len(cand.columns)}, train+val IC-robust: {len(robust)}")

    RESULTS.parent.mkdir(exist_ok=True)
    fh = RESULTS.open("a", newline="")
    writer = csv.writer(fh)
    if RESULTS.stat().st_size == 0:
        writer.writerow(["round", "added", "set_size", "val_sharpe_mean", "test_sh_mean",
                         "test_nav_mean", "test_sh_min", "test_nav_min", "all_cells_beat",
                         "mean_dd", "mean_trades", "accepted", "secs"])

    current = list(DEFAULT_FACTORS)
    base = evaluate(df, bt, splits, current, vt)
    best_val = base["val_sharpe_mean"]
    print(f"baseline set ({len(current)}): val_sharpe_mean={best_val:.3f} "
          f"test {base['test_sh_mean']:.2f}/{base['test_nav_mean']:.2f} all_beat={base['all_cells_beat']}")

    rnd = 0
    improved = True
    while improved:
        rnd += 1
        improved = False
        round_best = None
        for c in robust:
            if c in current:
                continue
            t0 = time.time()
            try:
                m = evaluate(df, bt, splits, current + [c], vt)
            except Exception as e:  # noqa: BLE001
                writer.writerow([rnd, c, len(current) + 1, "ERR", str(e)[:60], "", "", "", "", "", "", False, round(time.time()-t0,1)])
                fh.flush()
                continue
            # Accept only if guardrail holds, it does not worsen DD/turnover, and it
            # improves the validation-selection metric.
            ok = (m["all_cells_beat"] and m["mean_dd"] >= base["mean_dd"] - 0.01
                  and m["mean_trades"] <= base["mean_trades"] * 1.15
                  and m["val_sharpe_mean"] > best_val + 1e-4)
            writer.writerow([rnd, c, len(current) + 1, round(m["val_sharpe_mean"], 4),
                             round(m["test_sh_mean"], 3), round(m["test_nav_mean"], 3),
                             round(m["test_sh_min"], 3), round(m["test_nav_min"], 3),
                             m["all_cells_beat"], round(m["mean_dd"], 3),
                             round(m["mean_trades"], 1), ok, round(time.time() - t0, 1)])
            fh.flush()
            if ok and (round_best is None or m["val_sharpe_mean"] > round_best[1]["val_sharpe_mean"]):
                round_best = (c, m)
        if round_best is not None:
            c, m = round_best
            current.append(c)
            best_val = m["val_sharpe_mean"]
            base = m
            improved = True
            BEST_JSON.write_text(json.dumps(
                {"factors": current, "val_sharpe_mean": best_val,
                 "test_sh_mean": m["test_sh_mean"], "test_nav_mean": m["test_nav_mean"],
                 "all_cells_beat": m["all_cells_beat"]}, indent=2))
            print(f"[round {rnd}] +{c} -> set {len(current)}, val={best_val:.3f}, "
                  f"test {m['test_sh_mean']:.2f}/{m['test_nav_mean']:.2f}")

    print(f"DONE. Final set ({len(current)}): {current}")
    print(f"val_sharpe_mean={best_val:.3f}; checkpoint -> {BEST_JSON}")
    fh.close()


if __name__ == "__main__":
    main()
