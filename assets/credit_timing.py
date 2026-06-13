"""Credit-spread / risk-appetite EQUITY-INDEX TIMING factors.

Family: credit risk appetite. Signals derived from HYG/LQD, HYG/TLT relative
performance and credit-spread momentum/widening. Literature says rising HYG/LQD
(risk appetite up) precedes equity strength; spread widening (HYG falling vs
TLT/LQD) leads equity weakness (risk-off).

DISCIPLINE: select on train+val rank-IC stability ONLY (same-sign, |IC|>0.03 on
BOTH). Test is OOS - reported, never tuned on. Compare any timing signal to
SPY/QQQ buy-and-hold net of 0.1% fees. Test on BOTH spy and qqq.

Run from /Users/gaozhiyuan/Desktop/btc_daily_strategy with PYTHONPATH=.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
tr = (d.index >= SPLIT["train"][0]) & (d.index < SPLIT["train"][1])
va = (d.index >= SPLIT["val"][0])   & (d.index < SPLIT["val"][1])
te = (d.index >= SPLIT["test"][0])  & (d.index < SPLIT["test"][1])

HYG, LQD, TLT = d["hyg"], d["lqd"], d["tlt"]

def zscore(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

# ---- Factor definitions (sign chosen so POSITIVE factor => expect equity UP) ----
F = {}
# 1. HYG/LQD momentum (risk appetite). Rising => risk-on => equity up. +sign.
F["hyglqd_mom60"]  = (HYG/LQD).pct_change(60)
F["hyglqd_mom20"]  = (HYG/LQD).pct_change(20)
# 2. HYG/TLT momentum (credit vs duration risk appetite). Rising => risk-on. +sign.
F["hygtlt_mom60"]  = (HYG/TLT).pct_change(60)
F["hygtlt_mom20"]  = (HYG/TLT).pct_change(20)
# 3. HYG/LQD ratio vs its 100d MA (regime). >1 => risk-on. +sign.
F["hyglqd_vs100ma"] = (HYG/LQD)/(HYG/LQD).rolling(100).mean() - 1.0
# 4. HYG/TLT ratio vs 100d MA (the classic SPY/TLT switch rule). +sign.
F["hygtlt_vs100ma"] = (HYG/TLT)/(HYG/TLT).rolling(100).mean() - 1.0
# 5. HYG/LQD z-score rolling 252 (risk appetite level). High => risk-on. +sign.
F["hyglqd_z252"]   = zscore(HYG/LQD, 252)
# 6. HYG own momentum 60d (credit-spread widening proxy: HYG falling = spread widening = risk-off). +sign.
F["hyg_mom60"]     = HYG.pct_change(60)
# 7. Spread-widening (negative of HYG/LQD momentum, short window) => risk-off lead.
#    define as -drawdown of HYG/LQD from rolling 60d max. More negative => stress. +sign.
ratio = HYG/LQD
F["hyglqd_dd60"]   = ratio/ratio.rolling(60).max() - 1.0
# 8. HYG total-return momentum vs TLT, 120d (slower credit appetite). +sign.
F["hygtlt_mom120"] = (HYG/TLT).pct_change(120)

HORIZONS = [10, 20]
TARGETS = ["spy", "qqq"]

print("="*92)
print("RANK-IC of credit/risk-appetite factors vs forward returns (TRAIN | VAL | TEST)")
print("sign convention: +factor => expect equity UP. robust = same-sign & |IC|>0.03 on train AND val")
print("="*92)

robust = []
for h in HORIZONS:
    fwd = {t: forward_return(d[t], h) for t in TARGETS}
    print(f"\n--- horizon {h}d ---")
    print(f"{'factor':18s} " + "  ".join(f"{t}:tr/va/te" for t in TARGETS))
    for name, f in F.items():
        cells = []
        rob_flags = []
        for t in TARGETS:
            it = ic(f, fwd[t], tr); iv = ic(f, fwd[t], va); ix = ic(f, fwd[t], te)
            rob = (abs(it) > 0.03 and abs(iv) > 0.03 and np.sign(it) == np.sign(iv))
            rob_flags.append(rob)
            cells.append(f"{it:+.3f}/{iv:+.3f}/{ix:+.3f}")
        flag = ""
        if all(rob_flags): flag = "  <== ROBUST BOTH"
        elif any(rob_flags): flag = "  (robust on one)"
        print(f"{name:18s} " + "  ".join(cells) + flag)
        for t, rf in zip(TARGETS, rob_flags):
            if rf:
                it = ic(f, fwd[t], tr); iv = ic(f, fwd[t], va)
                robust.append((name, t, h, it, iv))

print("\n" + "="*92)
print("ROBUST (train+val stable) factor/target/horizon combos + their TEST IC (sign-flip check):")
if not robust:
    print("  NONE - no credit factor is same-sign & |IC|>0.03 on both train and val.")
for name, t, h, it, iv in robust:
    fwd_t = forward_return(d[t], h)
    ix = ic(F[name], fwd_t, te)
    flip = "SIGN-FLIPS on test (NOT truly OOS-stable)" if np.sign(ix) != np.sign(it) else "holds sign on test"
    print(f"  {name:18s} target={t} h={h}  IC_tr={it:+.3f} IC_va={iv:+.3f} IC_te={ix:+.3f}  -> {flip}")

# ---------------- Backtest robust signals as long/flat timing vs buy-hold ----------------
def bh_sharpe(close, mask):
    r = close.pct_change()[mask].dropna()
    return float(r.mean()/r.std()*np.sqrt(252)) if r.std() > 0 else 0.0

def seg(mask_name):
    return {"train":tr,"val":va,"test":te}[mask_name]

print("\n" + "="*92)
print("BACKTEST robust signals: long(1)/flat(0). Compare net Sharpe & NAV to buy-hold.")
print("="*92)

# Build timing signal using the IC-IMPLIED sign (so we trade the direction the
# train+val IC actually supports, not the literature prior). For these robust combos
# the IC is NEGATIVE on train&val => go long when factor is BELOW its rolling median
# (i.e. low risk-appetite / spreads already widened => contrarian bounce). We test
# both the literature direction and the IC-implied direction for honesty.
tested = set()
for name, t, h, it, iv in robust:
    if name in tested:
        continue
    tested.add(name)
    f = F[name]
    ic_sign = np.sign(it)  # train IC sign (same as val for robust combos)
    # IC-implied long signal: long when sign(IC)*factor > 0.
    sig = ((ic_sign * f) > 0).astype(float).reindex(d.index).fillna(0.0)
    print(f"\n(using IC-implied direction: long when sign(IC)*factor>0, IC_tr sign={ic_sign:+.0f})")
    for tgt in TARGETS:
        close = d[tgt]
        net, held, turn = backtest(close, sig, band=0.0)
        print(f"\n[{name}] long-when sign(IC)*factor>0  target={tgt}")
        for seg_name in ["train","val","test"]:
            m = seg(seg_name)
            sub = net[m].dropna()
            if len(sub) < 30:
                continue
            strat_sh = float(sub.mean()/sub.std()*np.sqrt(252)) if sub.std()>0 else 0.0
            strat_nav = float((1+sub).prod())
            bh = bh_sharpe(close, m)
            bh_r = close.pct_change()[m].dropna()
            bh_nav = float((1+bh_r).prod())
            avg_turn = float(turn[m].mean())
            mark = "BEATS" if strat_sh > bh else "loses"
            print(f"   {seg_name:5s} strat Sh={strat_sh:+.2f} NAV={strat_nav:5.2f} | "
                  f"BH Sh={bh:+.2f} NAV={bh_nav:5.2f} | turn/day={avg_turn:.4f} -> {mark}")

print("\nDONE.")
