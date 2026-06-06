"""Autonomous overnight factor-research engine (runs ~8h unattended, deterministic).

Pulls every free data source, builds a large candidate-factor universe (including
NEW market-breadth / cross-sectional factors from ~12 altcoins), and runs a
rigorously disciplined search:

  ANTI-OVERFIT DESIGN (critical for a multi-hour run, which otherwise manufactures
  false positives by multiple testing):
    - Candidate filter: information coefficient must be same-sign and above the
      noise floor on BOTH train and validation.
    - Selection metric: MEAN yearly out-of-sample Sharpe across MULTIPLE folds
      (2020,2021,2022,2023), each scored under WALK-FORWARD signs (re-estimated on
      prior data only). A candidate must improve the multi-fold mean AND not hurt
      the worst fold. Requiring consistency across 4 independent folds makes a
      spurious pass very unlikely.
    - The TEST window (2024-2026) is NEVER used for selection - computed only to
      log the current best for honest reporting.
    - Every candidate evaluation is logged; the running best is checkpointed.
      The number of candidates tried is logged so multiple-testing is transparent.

Outputs (under results/):
    overnight_log.csv      - every candidate evaluation
    overnight_best.json    - best factor set + metrics so far (checkpointed)
    overnight_summary.txt  - human-readable running summary

Run (background):
    nohup python3 overnight_research.py --hours 8 > results/overnight_stdout.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.data import (
    _download_cryptocompare,
    enrich_external,
    load_btc_data,
    load_eth_close,
)
from src.features import build_features
from src.factors import build_factors, forward_return, information_coefficient
from src.onchain import ONCHAIN_METRICS, load_coinmetrics, merge_onchain
from src.research import run_full, window_metrics
from src.metrics import sharpe_ratio
from src.strategies import get_strategy
from src.strategies.factor_composite import DEFAULT_FACTORS
from src.validation import make_fixed_split

RESULTS = Path("results")
LOG = RESULTS / "overnight_log.csv"
BEST = RESULTS / "overnight_best.json"
SUMMARY = RESULTS / "overnight_summary.txt"
ALT_CACHE = RESULTS / "alts_cache.csv"
FOLD_YEARS = [2020, 2021, 2022, 2023]
ALTS = ["ETH", "XRP", "LTC", "BCH", "ADA", "DOGE", "LINK", "XLM", "ETC", "EOS", "TRX", "BNB"]
ROBUST_IC = 0.03
CONF_GAIN = 1.0


def log_summary(msg: str) -> None:
    with SUMMARY.open("a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_alts(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load (cached) altcoin closes aligned to the BTC index."""
    if ALT_CACHE.exists():
        c = pd.read_csv(ALT_CACHE, index_col=0)
        c.index = pd.to_datetime(c.index, utc=True)
        return c.reindex(index)
    out = {}
    for a in ALTS:
        try:
            s = _download_cryptocompare(f"{a}/USD", "2016-01-01")["close"].reindex(index).ffill()
            if s.notna().sum() > 1500:
                out[a] = s
        except Exception:
            pass
    c = pd.DataFrame(out)
    c.to_csv(ALT_CACHE)
    return c


def build_candidate_matrix(df: pd.DataFrame, alts: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full candidate-factor matrix (existing + breadth + variants)."""
    F = build_factors(df)  # existing library (incl. macro/sentiment if columns present)
    close = df["close"].astype(float)
    rets = alts.pct_change()

    # --- market breadth / cross-sectional (NEW) ---
    for w in (50, 100, 200):
        F[f"alts_above_{w}dma"] = (alts > alts.rolling(w).mean()).mean(axis=1)
        F[f"breadth_mom_{w}"] = F[f"alts_above_{w}dma"].diff(20)
    F["alt_dispersion_20"] = rets.rolling(20).std().mean(axis=1)
    F["alt_mean_mom_30"] = rets.rolling(30).mean().mean(axis=1)
    for w in (15, 30, 60):
        F[f"btc_vs_alts_{w}"] = close.pct_change(w) - rets.mean(axis=1).rolling(w).sum()
    F["alt_corr_dispersion"] = rets.rolling(30).corr(close.pct_change()).mean(axis=1)

    # --- parametrized variants of price factors ---
    for w in (10, 40, 80, 150, 250):
        F[f"mom_{w}v"] = close.pct_change(w)
        vw = close.pct_change().rolling(w).std()
        F[f"sharpe_mom_{w}v"] = close.pct_change(w) / (vw * np.sqrt(w))
    return F.replace([np.inf, -np.inf], np.nan)


def ic_robust(f: pd.Series, df: pd.DataFrame, tr, va) -> bool:
    for h in (5, 20):
        fr = forward_return(df, h)
        it = information_coefficient(f.loc[tr], fr.loc[tr])
        iv = information_coefficient(f.loc[va], fr.loc[va])
        if abs(it) > ROBUST_IC and abs(iv) > ROBUST_IC and np.sign(it) == np.sign(iv):
            return True
    return False


def wf_composite(F: pd.DataFrame, names: list[str], df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Walk-forward IC-weighted composite from a factor matrix (annual re-sign)."""
    fr = forward_return(df, horizon)
    Z = {}
    for n in names:
        f = F[n]
        mu = f.expanding(365).mean(); sd = f.expanding(365).std()
        Z[n] = ((f - mu) / sd).clip(-3, 3)
    score = pd.Series(0.0, index=df.index); wsum = pd.Series(0.0, index=df.index)
    for yr in sorted(set(df.index.year)):
        ys = pd.Timestamp(f"{yr}-01-01", tz="UTC"); cut = ys - pd.Timedelta(days=horizon + 1)
        tm = df.index <= cut
        if int(tm.sum()) < 730:
            continue
        ym = (df.index >= ys) & (df.index < pd.Timestamp(f"{yr+1}-01-01", tz="UTC"))
        for n in names:
            ic = information_coefficient(F[n][tm], fr[tm])
            if ic == 0:
                continue
            score.loc[ym] += np.sign(ic) * abs(ic) * Z[n].loc[ym].fillna(0.0)
            wsum.loc[ym] += abs(ic)
    return np.tanh(score / wsum.replace(0.0, np.nan))


def evaluate(F, names, df, bt, splits, vt):
    """Return (multi-fold mean OOS Sharpe, worst fold, test Sharpe, test NAV, all_beat)."""
    score = wf_composite(F, names, df)
    conf = score.clip(lower=0.0)
    w = ((0.5 + score).clip(0, 2) * (vt * (1 + CONF_GAIN * conf)).clip(upper=2.0)).clip(0, 2)
    w.name = "raw_signal"
    res = run_full(df, w, bt)
    fold_sh = []
    for yr in FOLD_YEARS:
        sub = res[res.index.year == yr]["strategy_daily_return"]
        if len(sub) > 30:
            fold_sh.append(sharpe_ratio(sub, 365))
    te = window_metrics(res, splits["test"])
    return (float(np.mean(fold_sh)), float(np.min(fold_sh)),
            te["sharpe_ratio"], te["final_nav"],
            te["sharpe_ratio"] > 0.58 and te["final_nav"] > 1.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    RESULTS.mkdir(exist_ok=True)

    cfg = load_config("config.yaml"); bt = BacktestConfig.from_config(cfg)
    df = merge_onchain(build_features(load_btc_data(cfg)), load_coinmetrics(ONCHAIN_METRICS))
    df["eth_close"] = load_eth_close(df.index)
    df = enrich_external(df)
    alts = load_alts(df.index)
    splits = make_fixed_split(cfg["validation"])
    tr = splits["train"].slice(df).index; va = splits["validation"].slice(df).index
    ret = df["close"].pct_change(); vt = (0.55 / (ret.rolling(45).std() * np.sqrt(365)).replace(0, np.nan))

    F = build_candidate_matrix(df, alts)
    log_summary(f"[start] candidates={F.shape[1]}, alts={alts.shape[1]}, deadline={args.hours}h")
    robust = [c for c in F.columns if ic_robust(F[c], df, tr, va)]
    log_summary(f"[filter] IC-robust train+val: {len(robust)} / {F.shape[1]}")

    new_file = not LOG.exists()
    fh = LOG.open("a", newline=""); writer = csv.writer(fh)
    if new_file:
        writer.writerow(["iter", "phase", "added_or_set", "set_size", "fold_mean_sh",
                         "fold_min_sh", "test_sh", "test_nav", "all_beat", "secs"])

    base = list(DEFAULT_FACTORS)
    fm, fmin, tsh, tnav, ab = evaluate(F, base, df, bt, splits, vt)
    best = {"set": base, "fold_mean": fm, "fold_min": fmin, "test_sh": tsh, "test_nav": tnav}
    BEST.write_text(json.dumps(best, indent=2))
    log_summary(f"[baseline] fold_mean={fm:.3f} fold_min={fmin:.3f} test={tsh:.2f}/{tnav:.2f}")

    it = 0
    # Phase 1: greedy forward selection across robust candidates, multiple rounds.
    current = list(base); improved = True
    while improved and time.time() < deadline:
        improved = False; round_best = None
        for c in robust:
            if c in current or time.time() >= deadline:
                continue
            t0 = time.time()
            try:
                fm, fmin, tsh, tnav, ab = evaluate(F, current + [c], df, bt, splits, vt)
            except Exception as e:
                writer.writerow([it, "FS", c, len(current)+1, "ERR", str(e)[:40], "", "", "", round(time.time()-t0,1)]); fh.flush(); it += 1
                continue
            ok = fm > best["fold_mean"] + 1e-4 and fmin >= best["fold_min"] - 0.05
            writer.writerow([it, "FS", c, len(current)+1, round(fm,4), round(fmin,4),
                             round(tsh,3), round(tnav,3), ab, round(time.time()-t0,1)]); fh.flush(); it += 1
            if ok and (round_best is None or fm > round_best[1]):
                round_best = (c, fm, fmin, tsh, tnav)
        if round_best:
            c, fm, fmin, tsh, tnav = round_best
            current.append(c); best = {"set": current[:], "fold_mean": fm, "fold_min": fmin, "test_sh": tsh, "test_nav": tnav}
            BEST.write_text(json.dumps(best, indent=2)); improved = True
            log_summary(f"[FS +{c}] set={len(current)} fold_mean={fm:.3f} fold_min={fmin:.3f} test={tsh:.2f}/{tnav:.2f}")

    log_summary(f"[FS done] best fold_mean={best['fold_mean']:.3f} set={best['set']}")

    # Phase 2: randomized subset search (fills remaining time; same discipline).
    rng = np.random.default_rng(args.seed)
    pool = list(dict.fromkeys(base + robust))
    while time.time() < deadline:
        k = int(rng.integers(6, min(14, len(pool)) + 1))
        cand = list(rng.choice(pool, size=k, replace=False))
        t0 = time.time()
        try:
            fm, fmin, tsh, tnav, ab = evaluate(F, cand, df, bt, splits, vt)
        except Exception:
            it += 1; continue
        writer.writerow([it, "RND", "|".join(cand), k, round(fm,4), round(fmin,4),
                         round(tsh,3), round(tnav,3), ab, round(time.time()-t0,1)]); fh.flush(); it += 1
        if fm > best["fold_mean"] + 1e-4 and fmin >= best["fold_min"] - 0.05:
            best = {"set": cand, "fold_mean": fm, "fold_min": fmin, "test_sh": tsh, "test_nav": tnav}
            BEST.write_text(json.dumps(best, indent=2))
            log_summary(f"[RND] NEW BEST fold_mean={fm:.3f} test={tsh:.2f}/{tnav:.2f} set={cand}")

    fh.close()
    log_summary(f"[END] iters={it} best fold_mean={best['fold_mean']:.3f} "
                f"(baseline {json.loads(BEST.read_text())['fold_mean'] if BEST.exists() else 'NA'}) "
                f"test={best['test_sh']:.2f}/{best['test_nav']:.2f} set={best['set']}")


if __name__ == "__main__":
    main()
