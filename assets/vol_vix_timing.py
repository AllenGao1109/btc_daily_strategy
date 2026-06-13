"""Volatility / VIX term-structure equity-index TIMING factors.

Family: VIX level (contrarian mean-reversion), VIX3M-VIX term structure
(backwardation=stress=contrarian buy), variance risk premium (VIX-realized),
vol-of-vol / VIX momentum, realized-vol regime.

Discipline: select on TRAIN+VAL rank-IC stability ONLY (same-sign, |IC|>0.03 on
BOTH, vs SPY AND QQQ). test is OOS. Backtest robust factors long/flat vs buy-hold.

Run from repo root with PYTHONPATH=.:  python assets/vol_vix_timing.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
idx = d.index
TR = (idx >= SPLIT["train"][0]) & (idx < SPLIT["train"][1])
VA = (idx >= SPLIT["val"][0])   & (idx < SPLIT["val"][1])
TE = (idx >= SPLIT["test"][0])  & (idx < SPLIT["test"][1])

vix, vix3m = d["vix"], d["vix3m"]
spy, qqq = d["spy"], d["qqq"]

# ---- realized vol of SPY (annualized, %) over 20d, for variance-risk-premium ----
ret = spy.pct_change()
rv20 = ret.rolling(20).std() * np.sqrt(252) * 100.0   # in vol points like VIX

# =================== FACTOR DEFINITIONS ===================
# Sign convention: ic(f, fwd) is +ve when HIGH f -> HIGH forward return.
# Contrarian factors (high fear -> high fwd ret) should come out POSITIVE.
F = {}

# 1. VIX level (contrarian: extreme fear -> buy). Raw level.
F["vix_level"] = vix
# 1b. VIX percentile-ish: z-score over 252d (mean-reversion from extremes)
F["vix_z252"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

# 2. VIX term structure spread (VIX3M - VIX). Positive=contango(calm),
#    negative=backwardation(stress). Lit says backwardation -> +fwd ret (contrarian),
#    i.e. LOW spread -> HIGH fwd ret => expect NEGATIVE IC.
F["ts_spread_3m_spot"] = vix3m - vix
# 2b. Term-structure RATIO VIX/VIX3M. >1 = backwardation(stress). Lit: stress->+fwd ret
#    => HIGH ratio -> HIGH fwd ret => expect POSITIVE IC (contrarian).
F["ts_ratio_vix_v3m"] = vix / vix3m

# 3. Variance risk premium: VIX - realized vol (both in vol points). High VRP -> +ret.
F["vrp_vix_minus_rv"] = vix - rv20

# 4. Vol-of-vol / VIX momentum: 20d change in VIX. Spiking VIX (contrarian) -> +fwd ret.
F["vix_mom20"] = vix - vix.shift(20)
F["vix_pctchg20"] = vix.pct_change(20)
# 4b. vol-of-vol: rolling std of daily VIX changes
F["vol_of_vol60"] = vix.diff().rolling(60).std()

# 5. Realized-vol regime: high realized vol (contrarian) -> +fwd ret; or low rv -> calm/+ret
F["rv20"] = rv20
F["rv_z252"] = (rv20 - rv20.rolling(252).mean()) / rv20.rolling(252).std()

# =================== IC TABLE (TRAIN/VAL/TEST vs SPY & QQQ) ===================
HORIZONS = [20]
robust = []   # (name, target, h, it, iv)
rows = []
for h in HORIZONS:
    fwd_spy = forward_return(spy, h)
    fwd_qqq = forward_return(qqq, h)
    for name, f in F.items():
        for tgt, fwd in (("spy", fwd_spy), ("qqq", fwd_qqq)):
            it = ic(f, fwd, TR); iv = ic(f, fwd, VA); ite = ic(f, fwd, TE)
            rob = (abs(it) > 0.03 and abs(iv) > 0.03 and np.sign(it) == np.sign(iv))
            rows.append((name, tgt, h, it, iv, ite, rob))
            if rob:
                robust.append((name, tgt, h, it, iv, ite))

print(f"Panel {idx.min().date()} -> {idx.max().date()} ({len(d)}d)")
print(f"VIX nonNA={vix.notna().sum()}  VIX3M nonNA={vix3m.notna().sum()}  "
      f"VIX3M first valid={vix3m.first_valid_index()}")
print(f"\n{'factor':22s} {'tgt':3s} {'h':>3s} {'IC_tr':>7s} {'IC_va':>7s} {'IC_te':>7s} robust?")
for name, tgt, h, it, iv, ite, rob in rows:
    print(f"{name:22s} {tgt:3s} {h:3d} {it:+7.3f} {iv:+7.3f} {ite:+7.3f} {'ROBUST' if rob else ''}")

print("\n=== ROBUST (train+val same-sign, |IC|>0.03 on both, this target) ===")
if not robust:
    print("  NONE")
else:
    for name, tgt, h, it, iv, ite in robust:
        print(f"  {name} [{tgt} h{h}] IC tr={it:+.3f} va={iv:+.3f} te={ite:+.3f}")

# Also flag factors robust on BOTH spy AND qqq (stronger requirement)
both = {}
for name, tgt, h, it, iv, ite, rob in rows:
    both.setdefault((name, h), {})[tgt] = (it, iv, ite, rob)
print("\n=== ROBUST on BOTH spy AND qqq ===")
strong = []
for (name, h), dd in both.items():
    if dd.get("spy", (0,0,0,False))[3] and dd.get("qqq", (0,0,0,False))[3]:
        strong.append((name, h, dd))
        print(f"  {name} h{h}: spy IC {dd['spy'][0]:+.3f}/{dd['spy'][1]:+.3f}  "
              f"qqq IC {dd['qqq'][0]:+.3f}/{dd['qqq'][1]:+.3f}")
if not strong:
    print("  NONE")

# =================== BACKTEST robust/strong factors as long/flat ===================
def signal_from_factor(f, sign, lo_q, hi_q, lookback=252):
    """Map factor -> long/flat weight in [0,1]. sign=+1: high factor->long.
    Use rolling 252d quantile thresholds (causal). weight=1 when factor in the
    'bullish' tail, else 0. Smoothed via 5d to cut turnover slightly."""
    x = f * sign
    # rolling rank percentile (causal): fraction of last `lookback` obs <= current
    pct = x.rolling(lookback, min_periods=120).apply(
        lambda a: (a[:-1] <= a[-1]).mean(), raw=True)
    w = (pct >= hi_q).astype(float)   # long when factor in bullish extreme
    return w.reindex(idx).fillna(0.0)

def regime_filter_signal(ratio, thresh):
    """Stay LONG when NOT in stress (ratio below thresh = contango). Classic vol-target style.
    But lit says backwardation->buy (contrarian), so also test the contrarian version."""
    return (ratio < thresh).astype(float).reindex(idx).fillna(0.0)

def bt_report(close, w, label, fee=0.001, band=0.05):
    net, held, turn = backtest(close, w, band=band, fee=fee)
    r = close.pct_change()
    out = {"label": label, "trades": int((turn > 1e-9).sum())}
    for k, m in (("train", TR), ("val", VA), ("test", TE)):
        ms = metrics(net, m); mb = metrics(r, m)
        out[k] = (ms["sharpe"] if ms else None, mb["sharpe"] if mb else None,
                  ms["nav"] if ms else None, mb["nav"] if mb else None)
    msf = metrics(net); mbf = metrics(r)
    out["full"] = (msf["sharpe"], mbf["sharpe"], msf["nav"], mbf["nav"])
    return out, net

def print_bt(out):
    print(f"\n--- {out['label']} (trades {out['trades']}) ---")
    for k in ("train", "val", "test", "full"):
        s, bs, nv, bnv = out[k]
        if s is not None:
            print(f"  {k:6s} strat Sh {s:+5.2f} nav {nv:6.2f} | b&h Sh {bs:+5.2f} nav {bnv:6.2f}"
                  f"  {'BEAT' if s>bs else ''}")

# Determine which factors to backtest: union of robust set; also always test the
# canonical term-structure regime filter even if IC marginal (it's the headline lit signal).
print("\n\n================= BACKTESTS =================")

tested = set()
for name, tgt, h, it, iv, ite in robust:
    if (name, tgt) in tested: continue
    tested.add((name, tgt))
    close = spy if tgt == "spy" else qqq
    sign = int(np.sign(it)) or 1
    w = signal_from_factor(F[name], sign, 0.0, 0.70)   # long in top-30% bullish tail
    out, _ = bt_report(close, w, f"{name} long/flat [{tgt}] (top30% tail, sign={sign:+d})")
    print_bt(out)

# Headline canonical term-structure regime filters (test both directions) on SPY & QQQ
ratio = (vix / vix3m)
for tgt in ("spy", "qqq"):
    close = spy if tgt == "spy" else qqq
    # contango-long: long when ratio<1 (calm), flat in stress -> risk-management style
    w_calm = regime_filter_signal(ratio, 1.0)
    out, _ = bt_report(close, w_calm, f"TS regime: LONG when contango (ratio<1) [{tgt}]")
    print_bt(out)
    # contrarian: long when ratio>1 (backwardation/stress) -> buy fear
    w_stress = (ratio > 1.0).astype(float).reindex(idx).fillna(0.0)
    out, _ = bt_report(close, w_stress, f"TS regime: LONG when backwardation (ratio>1) [{tgt}] (contrarian)")
    print_bt(out)

# VIX-level contrarian: long when VIX above rolling 80th pct (extreme fear) - held 20d-ish
for tgt in ("spy", "qqq"):
    close = spy if tgt == "spy" else qqq
    w = signal_from_factor(vix, +1, 0.0, 0.80)   # long only in top-20% VIX tail
    out, _ = bt_report(close, w, f"VIX extreme-fear contrarian (top20% VIX) [{tgt}]")
    print_bt(out)

print("\nDONE")
