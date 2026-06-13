"""Follow-up: stress-test the one signal that beat buy-hold OOS - the VIX term
structure 'contango regime filter' (stay long when VIX/VIX3M ratio < threshold,
flat in stress). Check IC sign-stability of the underlying spread, threshold
sensitivity, and whether it genuinely beats buy-hold on BOTH val AND test net of fees.

Run: PYTHONPATH=. python assets/vol_vix_timing2.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load(); idx = d.index
TR = (idx >= SPLIT["train"][0]) & (idx < SPLIT["train"][1])
VA = (idx >= SPLIT["val"][0])   & (idx < SPLIT["val"][1])
TE = (idx >= SPLIT["test"][0])  & (idx < SPLIT["test"][1])
vix, vix3m, spy, qqq = d["vix"], d["vix3m"], d["spy"], d["qqq"]
ratio = vix / vix3m
spread = vix3m - vix   # >0 contango

# --- IC of the term-structure SPREAD vs fwd ret (the regime-filter's underlying) ---
print("Underlying term-structure SPREAD (VIX3M-VIX) rank-IC vs fwd20 ret:")
print("  (contango-long filter = long when spread>0; sign of edge = sign of IC of -spread on held days)")
for h in (10, 20, 40):
    for tgt, c in (("spy", spy), ("qqq", qqq)):
        fwd = forward_return(c, h)
        print(f"  spread h{h:2d} {tgt}: IC tr={ic(spread,fwd,TR):+.3f} va={ic(spread,fwd,VA):+.3f} te={ic(spread,fwd,TE):+.3f}")

def bt(close, w, band=0.05, fee=0.001):
    net, held, turn = backtest(close, w, band=band, fee=fee)
    r = close.pct_change()
    res = {}
    for k, m in (("train", TR), ("val", VA), ("test", TE)):
        ms, mb = metrics(net, m), metrics(r, m)
        res[k] = (ms["sharpe"], mb["sharpe"], ms["nav"], mb["nav"], ms["maxdd"], mb["maxdd"])
    msf, mbf = metrics(net), metrics(r)
    res["full"] = (msf["sharpe"], mbf["sharpe"], msf["nav"], mbf["nav"], msf["maxdd"], mbf["maxdd"])
    res["trades"] = int((turn > 1e-9).sum())
    return res

def show(res, label):
    print(f"\n--- {label} (trades {res['trades']}) ---")
    for k in ("train","val","test","full"):
        s,bs,nv,bnv,dd,bdd = res[k]
        flag = "BEAT" if s>bs else ""
        print(f"  {k:6s} strat Sh {s:+5.2f} nav {nv:6.2f} ddn {dd*100:4.0f}% | b&h Sh {bs:+5.2f} nav {bnv:6.2f} ddn {bdd*100:4.0f}% {flag}")

# --- threshold sensitivity of contango-long filter on SPY and QQQ ---
print("\n\n========== Contango-long regime filter: threshold sensitivity ==========")
for thr in (0.95, 1.00, 1.05):
    w = (ratio < thr).astype(float).reindex(idx).fillna(0.0)
    for tgt, c in (("spy", spy), ("qqq", qqq)):
        show(bt(c, w), f"LONG when ratio<{thr} [{tgt}]")

# --- smoothed version (5d MA of ratio) to reduce whipsaw, with wider band ---
print("\n\n========== Smoothed contango filter (ratio 5d-MA<1.0, band=0.10) ==========")
ratio_s = ratio.rolling(5).mean()
for tgt, c in (("spy", spy), ("qqq", qqq)):
    w = (ratio_s < 1.0).astype(float).reindex(idx).fillna(0.0)
    show(bt(c, w, band=0.10), f"LONG when 5d-MA ratio<1.0 [{tgt}]")

# --- combine with the trend baseline mention: is contango filter additive to plain long? ---
# Report fraction of time invested + simple summary
for tgt, c in (("spy","spy"),("qqq","qqq")):
    cl = spy if tgt=="spy" else qqq
    w = (ratio < 1.0).astype(float).reindex(idx).fillna(0.0).shift(1)
    for k,m in (("train",TR),("val",VA),("test",TE)):
        inv = w[m].mean()
        print(f"{tgt} {k}: invested {inv*100:4.0f}% of days")

print("\nDONE")
