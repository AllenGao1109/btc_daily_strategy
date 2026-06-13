"""Breadth / sector-rotation INDEX-TIMING factors for SPY & QQQ.

Family: defensive vs cyclical leadership. Literature signal definitions:
  1. XLY/XLP momentum (discretionary vs staples risk appetite; rising=risk-on=bullish).
  2. Defensive vs cyclical leadership: XLU/XLK, XLP/XLK, defensive_basket/cyclical_basket
     momentum (rising defensive ratio = risk-off warning = bearish).
  3. Composite defensive-minus-cyclical momentum (avg defensive mom - avg cyclical mom).
  4. Sector dispersion: cross-sectional std of sector trailing returns (high disp -> caution).

Sectors available in harness: xlu,xlp (defensive); xlk,xly,xlf (cyclical).

DISCIPLINE: select on TRAIN+VAL rank-IC stability ONLY (same-sign & |IC|>0.03 on BOTH),
for BOTH spy and qqq. Test is OOS (report, never tune). Compare timing to buy-and-hold.
Run: PYTHONPATH=. python assets/breadth_rotation.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
idx = d.index
tr = (idx >= SPLIT["train"][0]) & (idx < SPLIT["train"][1])
va = (idx >= SPLIT["val"][0])   & (idx < SPLIT["val"][1])
te = (idx >= SPLIT["test"][0])  & (idx < SPLIT["test"][1])

DEF = ["xlu", "xlp"]            # defensive
CYC = ["xlk", "xly", "xlf"]    # cyclical
HORIZON = 20                    # forward 20d return target

def mom(s, n):    return s.pct_change(n)
def zwin(s, n):   return (s - s.rolling(n).mean()) / (s.rolling(n).std() + 1e-12)

def build_factors():
    F = {}
    # ---- 1. XLY/XLP risk-appetite momentum (rising = risk-on = bullish, expect +IC) ----
    rr = d["xly"] / d["xlp"]
    for n in (20, 60, 120):
        F[f"xly_xlp_mom{n}"] = mom(rr, n)

    # ---- 2. Defensive vs cyclical pairwise leadership (rising defensive ratio = bearish, expect -IC) ----
    for n in (20, 60, 120):
        F[f"xlu_xlk_mom{n}"] = mom(d["xlu"] / d["xlk"], n)
        F[f"xlp_xlk_mom{n}"] = mom(d["xlp"] / d["xlk"], n)
        F[f"xlu_spy_mom{n}"] = mom(d["xlu"] / d["spy"], n)   # utilities vs market
        F[f"xlp_spy_mom{n}"] = mom(d["xlp"] / d["spy"], n)

    # ---- 3. Composite defensive-minus-cyclical momentum (rising = defensive leading = bearish, -IC) ----
    for n in (20, 60, 120):
        defm = pd.concat([mom(d[s], n) for s in DEF], axis=1).mean(axis=1)
        cycm = pd.concat([mom(d[s], n) for s in CYC], axis=1).mean(axis=1)
        F[f"def_minus_cyc_mom{n}"] = defm - cycm
        # ratio of baskets
        defbk = pd.concat([d[s] for s in DEF], axis=1).mean(axis=1)
        cycbk = pd.concat([d[s] for s in CYC], axis=1).mean(axis=1)
        F[f"defbk_cycbk_mom{n}"] = mom(defbk / cycbk, n)

    # ---- 4. Sector dispersion: cross-sectional std of trailing sector returns (high disp = caution, expect -IC) ----
    secs = DEF + CYC
    for n in (20, 60):
        rets = pd.concat([mom(d[s], n) for s in secs], axis=1)
        F[f"sector_disp{n}"] = rets.std(axis=1)
    return F

def run_ic():
    F = build_factors()
    rows = []
    fwd = {"spy": forward_return(d["spy"], HORIZON), "qqq": forward_return(d["qqq"], HORIZON)}
    for name, f in F.items():
        r = {"factor": name}
        for tgt in ("spy", "qqq"):
            r[f"{tgt}_tr"] = ic(f, fwd[tgt], tr)
            r[f"{tgt}_va"] = ic(f, fwd[tgt], va)
            r[f"{tgt}_te"] = ic(f, fwd[tgt], te)
        # robust = same-sign & |IC|>0.03 on BOTH train AND val, for BOTH spy AND qqq
        def stable(a, b): return abs(a) > 0.03 and abs(b) > 0.03 and np.sign(a) == np.sign(b)
        r["robust"] = stable(r["spy_tr"], r["spy_va"]) and stable(r["qqq_tr"], r["qqq_va"])
        # also require spy and qqq same sign on train (coherence)
        r["coherent"] = np.sign(r["spy_tr"]) == np.sign(r["qqq_tr"])
        rows.append(r)
    return pd.DataFrame(rows), F

def to_weight(f, sign, n=120):
    """Long/flat timing: z-score of factor, sign-adjusted, map to [0,1.5] long-only.
    sign=+1 => high factor bullish (full long). sign=-1 => high factor bearish (flatten)."""
    z = zwin(f, n) * sign
    w = (0.75 + 0.75 * np.tanh(z)).clip(0, 1.5)   # ~0.75 neutral, scales 0..1.5
    return w

def bt_block(name, f, sign, tgt):
    w = to_weight(f, sign).reindex(d.index).ffill().fillna(0.75)
    out = {}
    for blk, m in (("val", va), ("test", te), ("full", (va | te | tr))):
        c = d[tgt][m]; ww = w[m]
        net, held, turn = backtest(c, ww, band=0.05)
        mt = metrics(net)
        # buy-hold on same window
        bh = metrics(c.pct_change().fillna(0))
        out[blk] = dict(sharpe=mt["sharpe"], nav=mt["nav"], turn=float(turn.sum()),
                        bh_sharpe=bh["sharpe"], bh_nav=bh["nav"])
    return out

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30)
    df, F = run_ic()
    show = df.copy()
    for c in show.columns:
        if c not in ("factor", "robust", "coherent"):
            show[c] = show[c].map(lambda x: f"{x:+.3f}")
    print("=== Rank-IC vs forward 20d return (SPY & QQQ), TRAIN / VAL / TEST ===")
    print(show.to_string(index=False))

    robust = df[df["robust"]]
    print(f"\n=== ROBUST factors (same-sign & |IC|>0.03 on train+val for BOTH spy & qqq): {len(robust)} ===")
    if len(robust):
        print(robust[["factor", "spy_tr", "spy_va", "spy_te", "qqq_tr", "qqq_va", "qqq_te"]].to_string(index=False))

    # Near-robust: stable on spy OR qqq alone (for honest reporting)
    def stable(a, b): return abs(a) > 0.03 and abs(b) > 0.03 and np.sign(a) == np.sign(b)
    df["spy_stable"] = df.apply(lambda r: stable(r["spy_tr"], r["spy_va"]), axis=1)
    df["qqq_stable"] = df.apply(lambda r: stable(r["qqq_tr"], r["qqq_va"]), axis=1)
    near = df[(df["spy_stable"] | df["qqq_stable"]) & ~df["robust"]]
    print(f"\n=== NEAR-robust (stable on spy OR qqq, not both): {len(near)} ===")
    if len(near):
        print(near[["factor", "spy_tr", "spy_va", "qqq_tr", "qqq_va", "spy_stable", "qqq_stable"]].to_string(index=False))

    # Backtest robust (or top near-robust) timing signals vs buy-hold
    cand = robust["factor"].tolist() if len(robust) else near["factor"].tolist()[:4]
    print(f"\n=== Timing backtest vs buy-hold (band 0.05, 0.1% fee) for {len(cand)} candidates ===")
    for name in cand:
        f = F[name]
        # sign from train IC (use spy)
        sgn = int(np.sign(df.set_index("factor").loc[name, "spy_tr"])) or 1
        for tgt in ("spy", "qqq"):
            res = bt_block(name, f, sgn, tgt)
            for blk in ("val", "test"):
                r = res[blk]
                beat = "BEAT" if r["sharpe"] > r["bh_sharpe"] else ""
                print(f"{name:22s} {tgt} {blk:4s} sign{sgn:+d} | "
                      f"Sharpe {r['sharpe']:+.2f} vs BH {r['bh_sharpe']:+.2f}  "
                      f"NAV {r['nav']:.2f} vs {r['bh_nav']:.2f}  turn {r['turn']:.1f} {beat}")
