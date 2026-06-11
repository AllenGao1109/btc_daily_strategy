"""Standing decay monitor for the ADOPTED composite factors.

REPORTING ONLY — never use this script to *select* new factors: it reads the
test window, and anything chosen with test data is contaminated. Its job is
the opposite direction: to tell us when a factor we already adopted (on
train+val evidence) stops working out-of-sample, so we can retire it or
re-research with a clean split.

For every factor in the production composite (factor_composite.DEFAULT_FACTORS)
it reports the h20 IC per calendar year and per window (train / validation /
test-to-date), and flags:

  DECAYED   test-window IC sign opposes the train+val sign (and is material)
  WEAK      test-window |IC| < 0.03 (the mining noise floor)
  OK        sign holds at material magnitude

Slow cycle factors (halving phase, MVRV families) vary little within a year;
read their yearly cells loosely and weight the window columns.

PRE-REGISTERED SPLIT-ROLL TRIPWIRE (decided 2026-06, while the composite was
still ahead — i.e. before knowing the outcome). Rolling the validation split
forward (train<=2023, validate 2024-25) consumes the test window as selection
data, so it happens only when the COMPOSITE fails, not merely when factor ICs
look tired:

  ROLL if  trailing-365d Sharpe edge (composite minus buy-and-hold) < -0.30
       or  edge < 0 AND >=5 adopted factors flagged, both on two consecutive
           monitor runs >= 60 days apart.
  Otherwise HOLD. Until the tripwire fires, do NOT run any "what would
  2024-25 select" analysis — looking is consuming.

Each run appends to results/factor_monitor_log.csv (committed, it is decision
state, not regenerable output) so persistence is checkable.

Run:  python3 factor_monitor.py        (re-run as test years accumulate)
Writes results/factor_monitor.csv and appends results/factor_monitor_log.csv.
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
from src.macro import load_dxy, load_fred_macro
from src.onchain import (
    ONCHAIN_METRICS,
    load_coinmetrics,
    load_stablecoin_mcap,
    merge_onchain,
)
from src.config import BacktestConfig
from src.research import run_full
from src.sentiment import load_cnn_fear_greed, load_crypto_fear_greed
from src.strategies import get_strategy
from src.strategies.factor_composite import DEFAULT_FACTORS
from src.validation import make_fixed_split

HORIZON = 20
WEAK_IC = 0.03
EDGE_HARD = -0.30   # trailing-365d Sharpe edge below this -> tripwire leg 1
FLAG_SOFT = 5       # >= this many flags with edge < 0 -> tripwire leg 2
PERSIST_DAYS = 60   # both legs need two runs at least this far apart
LOG_PATH = Path("results/factor_monitor_log.csv")


def verdict(ic_trval: float, ic_test: float) -> str:
    if abs(ic_test) < WEAK_IC:
        return "WEAK"
    if np.sign(ic_test) != np.sign(ic_trval):
        return "DECAYED"
    return "OK"


def main() -> None:
    cfg = load_config("config.yaml")
    df = build_features(load_btc_data(cfg))
    df = merge_onchain(df, load_coinmetrics(ONCHAIN_METRICS))
    df = merge_onchain(df, load_crypto_fear_greed())
    df = merge_onchain(df, load_cnn_fear_greed())
    df = merge_onchain(df, load_stablecoin_mcap())
    df = merge_onchain(df, load_fred_macro())
    df = merge_onchain(df, load_dxy())
    eth = load_coinmetrics(["PriceUSD"], asset="eth").rename(
        columns={"PriceUSD": "eth_close"}
    )
    df = merge_onchain(df, eth)

    factors = build_factors(df)
    fr = forward_return(df, HORIZON)
    splits = make_fixed_split(cfg["validation"])
    tr_idx = splits["train"].slice(df).index
    va_idx = splits["validation"].slice(df).index
    te_idx = splits["test"].slice(df).index
    years = sorted({d.year for d in df.index if d.year >= 2019})

    rows = []
    for name in DEFAULT_FACTORS:
        if name not in factors.columns:
            rows.append({"factor": name, "verdict": "MISSING-DATA"})
            continue
        f = factors[name]
        rec: dict[str, object] = {"factor": name}
        for yr in years:
            mask = f.index.year == yr
            rec[f"ic_{yr}"] = round(information_coefficient(f[mask], fr[mask]), 3)
        ic_tr = information_coefficient(f.loc[tr_idx], fr.loc[tr_idx])
        ic_va = information_coefficient(f.loc[va_idx], fr.loc[va_idx])
        ic_te = information_coefficient(f.loc[te_idx], fr.loc[te_idx])
        rec["ic_train"] = round(ic_tr, 3)
        rec["ic_val"] = round(ic_va, 3)
        rec["ic_test"] = round(ic_te, 3)
        rec["verdict"] = verdict(ic_tr + ic_va, ic_te)
        rows.append(rec)

    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/factor_monitor.csv", index=False)

    year_cols = [c for c in table.columns if c.startswith("ic_2")]
    print(f"Adopted-factor decay monitor (h{HORIZON} IC; test = "
          f"{te_idx.min().date()} -> {te_idx.max().date()})\n")
    header = f"{'factor':20s} " + " ".join(f"{c[3:]:>6s}" for c in year_cols)
    print(header + f" | {'train':>6s} {'val':>6s} {'test':>6s} | verdict")
    for _, r in table.iterrows():
        if r["verdict"] == "MISSING-DATA":
            print(f"{r['factor']:20s} (factor column absent — data not merged)")
            continue
        cells = " ".join(f"{r[c]:6.2f}" for c in year_cols)
        print(f"{r['factor']:20s} {cells} | {r['ic_train']:6.2f} {r['ic_val']:6.2f} "
              f"{r['ic_test']:6.2f} | {r['verdict']}")
    n_bad = int((table["verdict"].isin(["DECAYED", "WEAK"])).sum())
    print(f"\n{n_bad}/{len(table)} adopted factors flagged. Flags do not auto-"
          "retire a factor; the pre-registered tripwire below decides.")

    edge = composite_edge(df, cfg)
    decide_tripwire(df.index.max(), n_bad, edge)


def composite_edge(df: pd.DataFrame, cfg: dict) -> float:
    """Trailing-365d Sharpe of the production composite minus buy-and-hold."""
    bt = BacktestConfig.from_config(cfg)
    comp = run_full(
        df, get_strategy("factor_composite")(df, dict(cfg["strategy"]["params"])), bt
    )
    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)

    def trailing_sharpe(r: pd.DataFrame) -> float:
        x = r["strategy_daily_return"].astype(float).iloc[-365:]
        return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0

    return trailing_sharpe(comp) - trailing_sharpe(bh)


def decide_tripwire(data_end: pd.Timestamp, n_flagged: int, edge: float) -> None:
    """Append this run to the log and print the pre-registered HOLD/ROLL call."""
    prev = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    row = pd.DataFrame(
        [{"data_end": str(data_end.date()), "n_flagged": n_flagged,
          "trailing_edge": round(edge, 3)}]
    )
    log = pd.concat([prev, row], ignore_index=True).drop_duplicates(
        subset="data_end", keep="last"
    )
    LOG_PATH.parent.mkdir(exist_ok=True)
    log.to_csv(LOG_PATH, index=False)

    def fired(r: pd.Series) -> bool:
        return bool(
            r["trailing_edge"] < EDGE_HARD
            or (r["trailing_edge"] < 0 and r["n_flagged"] >= FLAG_SOFT)
        )

    hits = log[log.apply(fired, axis=1)].copy()
    persistent = False
    if len(hits) >= 2:
        dates = pd.to_datetime(hits["data_end"])
        persistent = (dates.max() - dates.min()).days >= PERSIST_DAYS

    print(f"\nTripwire: trailing-365d Sharpe edge {edge:+.2f} "
          f"(hard < {EDGE_HARD}), flags {n_flagged} (soft >= {FLAG_SOFT} with edge < 0)")
    if persistent:
        print("VERDICT: ROLL — condition met on two runs >= "
              f"{PERSIST_DAYS} days apart. Roll the split (train<=2023, "
              "validate 2024-25, test 2026+), re-select factors, document.")
    elif len(hits) > 0:
        print("VERDICT: HOLD (armed) — condition met on this run; needs "
              f"persistence across {PERSIST_DAYS}+ days to fire.")
    else:
        print("VERDICT: HOLD — composite edge intact; factor flags alone do "
              "not consume the test window.")


if __name__ == "__main__":
    main()
