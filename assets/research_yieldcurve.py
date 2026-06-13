"""Yield-curve / rates / macro family — EQUITY-INDEX TIMING factor mining.

Family: slope (10y-3m, 10y-5y), curve momentum, 10y level & change, real-rate proxy.
Discipline: select on TRAIN+VAL rank-IC stability ONLY (same-sign, |IC|>0.03 on BOTH),
test reported OOS but never tuned. Compare any timing signal to SPY/QQQ buy-and-hold.

Run: cd /Users/gaozhiyuan/Desktop/btc_daily_strategy && PYTHONPATH=. python assets/research_yieldcurve.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
idx = d.index
TR = (idx >= SPLIT["train"][0]) & (idx < SPLIT["train"][1])
VA = (idx >= SPLIT["val"][0]) & (idx < SPLIT["val"][1])
TE = (idx >= SPLIT["test"][0]) & (idx < SPLIT["test"][1])

# yields are in % (e.g. tnx ~ 4.5 means 4.5%); irx=3m, fvx=5y, tnx=10y, tyx=30y
tnx, irx, fvx, tyx = d["tnx"], d["irx"], d["fvx"], d["tyx"]

# ---------- factor definitions (sign chosen so HIGHER factor => bullish equities a priori) ----------
def z(s, win=252):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()

factors = {}
# 1. Slope 10y-3m. Steep (positive) = expansion/bullish; inverted (negative) = recession risk. Higher=bullish.
factors["slope_10y3m"]      = tnx - irx
# 2. Slope 10y-5y (proxy for 10y-2y; harness lacks 2y). Higher=bullish.
factors["slope_10y5y"]      = tnx - fvx
# 3. Slope z-score (normalize level so cross-regime comparable). Higher=bullish.
factors["slope_10y3m_z"]    = z(tnx - irx)
# 4. Curve momentum: 60d change of slope. Steepening (positive) historically = recovery/bullish.
factors["slope_chg_60d"]    = (tnx - irx) - (tnx - irx).shift(60)
# 5. Curve momentum 20d.
factors["slope_chg_20d"]    = (tnx - irx) - (tnx - irx).shift(20)
# 6. 10y level (Fed-model-ish): high rates => discount-rate drag, a priori bearish => NEGATE so higher=bullish.
factors["tnx_level_neg"]    = -tnx
# 7. 10y rate-of-change 20d: rising yields a priori bearish (discount-rate channel) => NEGATE.
factors["tnx_chg20_neg"]    = -(tnx - tnx.shift(20))
# 8. 10y rate-of-change 60d, negated.
factors["tnx_chg60_neg"]    = -(tnx - tnx.shift(60))
# 9. 10y vs its 252d average (overbought rates), negated => higher=bullish.
factors["tnx_vs_avg_neg"]   = -(tnx - tnx.rolling(252).mean())
# 10. Real-rate proxy: 10y minus 126d gold momentum (gold up ~ inflation/real-rate down). Higher real rate a priori bearish => negate.
real_proxy = tnx - 100.0 * (d["gld"] / d["gld"].shift(126) - 1.0)
factors["real_rate_proxy_neg"] = -real_proxy
# 11. Inversion dummy momentum: how long has curve been inverted (count). More inverted-history => bearish => negate.
inv = (tnx - irx < 0).astype(float)
factors["inversion_persist_neg"] = -inv.rolling(126).sum()
# 12. Steepening-from-inversion reversal (literature-cited "confirmed reversal"):
#     curve currently below 1y avg (was flattening/inverted) AND now rising 60d.
#     Encode as +1 when inverted-and-steepening (bull), else slope_chg.
flat_lvl = (tnx - irx) - (tnx - irx).rolling(252).mean()
factors["reversal_signal"] = np.sign((tnx - irx).shift(0)*0 - 1) * 0  # placeholder, set below
factors["reversal_signal"] = ((tnx - irx) < 0).astype(float) * factors["slope_chg_60d"]

# ---------- IC table on SPY and QQQ, horizons 20 & 60 ----------
def ic_row(f, tgt, h):
    fwd = forward_return(d[tgt], h)
    return ic(f, fwd, TR), ic(f, fwd, VA), ic(f, fwd, TE)

print("="*108)
print("RANK-IC (higher factor a priori = bullish). Robust = same-sign & |IC|>0.03 on TRAIN *and* VAL, both targets ideally.")
print("="*108)
hz = [20, 60]
results = {}
for name, f in factors.items():
    line = f"{name:24s}"
    rob_flags = []
    for tgt in ("spy", "qqq"):
        for h in hz:
            it, iv, ite = ic_row(f, tgt, h)
            robust = (abs(it) > 0.03 and abs(iv) > 0.03 and np.sign(it) == np.sign(iv))
            results[(name, tgt, h)] = (it, iv, ite, robust)
            tag = "*" if robust else " "
            line += f" | {tgt}{h}:{it:+.3f}/{iv:+.3f}/{ite:+.3f}{tag}"
            rob_flags.append(robust)
    line += "  <<ROBUST" if all(rob_flags) else ("  <robust(some)" if any(rob_flags) else "")
    print(line)

print("\nLegend: each cell = IC_train/IC_val/IC_test ; '*' = train+val robust for that target/horizon.\n")

# ---------- summarize robust (train+val) factors across BOTH targets ----------
print("="*108)
print("ROBUST on BOTH spy & qqq (train+val same-sign & |IC|>0.03) for a given horizon:")
print("="*108)
robust_both = []
for name in factors:
    for h in hz:
        s = results[(name, "spy", h)]
        q = results[(name, "qqq", h)]
        if s[3] and q[3] and np.sign(s[0]) == np.sign(q[0]):
            robust_both.append((name, h, s, q))
            print(f"  {name:24s} h={h}d  SPY tr/va/te={s[0]:+.3f}/{s[1]:+.3f}/{s[2]:+.3f}"
                  f"  QQQ tr/va/te={q[0]:+.3f}/{q[1]:+.3f}/{q[2]:+.3f}")
if not robust_both:
    print("  (NONE robust on both targets train+val)")

# ---------- Build long/flat timing signals & backtest vs buy-hold ----------
# Even if not 'robust on both', backtest the strongest candidates + a slope-trend combo + trend baseline.
print("\n" + "="*108)
print("BACKTEST long/flat timing signals vs BUY-HOLD (net of 0.1% fee). weight in [0,1]. Sharpe/NAV/MaxDD per split.")
print("="*108)

def sig_metrics(tgt, weight, band, label):
    close = d[tgt]
    net, held, turn = backtest(close, weight, band)
    r = close.pct_change()
    ntr = int((turn > 1e-9).sum())
    print(f"\n--- {label} on {tgt.upper()} (trades={ntr}) ---")
    out = {}
    for k, (a, b) in SPLIT.items():
        m = (close.index >= a) & (close.index < b)
        ms = metrics(net, m); mb = metrics(r, m)
        if ms and mb:
            out[k] = (ms["sharpe"], ms["nav"], mb["sharpe"], mb["nav"])
            beat = "BEAT" if ms["sharpe"] > mb["sharpe"] else ""
            print(f"  {k:6s} strat S/{ms['sharpe']:5.2f} NAV/{ms['nav']:6.2f} DD/{ms['maxdd']*100:4.0f}%"
                  f"  | hold S/{mb['sharpe']:5.2f} NAV/{mb['nav']:6.2f} DD/{mb['maxdd']*100:4.0f}%  {beat}")
    return out

# Signal A: slope 10y-3m steep => long (slope>0), flat/inverted => flat.
wA = (factors["slope_10y3m"] > 0).astype(float)
# Signal B: slope z-score > -0.5 (long unless deeply inverted relative to recent regime).
wB = (factors["slope_10y3m_z"] > -0.5).astype(float)
# Signal C: 10y NOT rapidly rising (tnx_chg60_neg > threshold) => long when rates falling/stable.
wC = (factors["tnx_chg60_neg"] > -0.3).astype(float)   # -0.3 => allow mild rise
# Signal D: trend baseline (SPY above 200d MA) — the known 'near ceiling' reference.
wD_spy = (d["spy"] > d["spy"].rolling(200).mean()).astype(float)
wD_qqq = (d["qqq"] > d["qqq"].rolling(200).mean()).astype(float)
# Signal E: combo — trend AND not-inverted (rates filter on top of trend).
wE_spy = ((d["spy"] > d["spy"].rolling(200).mean()) & (factors["slope_10y3m"] > 0)).astype(float)
wE_qqq = ((d["qqq"] > d["qqq"].rolling(200).mean()) & (factors["slope_10y3m"] > 0)).astype(float)

for tgt in ("spy", "qqq"):
    sig_metrics(tgt, wA, 0.5, "A: slope10y3m>0 long/flat")
    sig_metrics(tgt, wB, 0.5, "B: slope z>-0.5 long/flat")
    sig_metrics(tgt, wC, 0.3, "C: 10y not rising-fast long/flat")
    sig_metrics(tgt, wD_spy if tgt=="spy" else wD_qqq, 0.5, "D: trend 200dMA (baseline)")
    sig_metrics(tgt, wE_spy if tgt=="spy" else wE_qqq, 0.5, "E: trend200 AND slope>0")

# buy-hold reference Sharpe full + per-split printed above
print("\n" + "="*108)
print("BUY-HOLD reference (full sample + per split):")
for tgt in ("spy", "qqq"):
    r = d[tgt].pct_change()
    mf = metrics(r, (idx>=SPLIT["train"][0]))
    print(f"  {tgt.upper()} full S/{mf['sharpe']:.2f} NAV/{mf['nav']:.2f}")
    for k,(a,b) in SPLIT.items():
        m=(idx>=a)&(idx<b); mb=metrics(r,m)
        print(f"      {k:6s} hold S/{mb['sharpe']:5.2f} NAV/{mb['nav']:6.2f}")
print("="*108)
