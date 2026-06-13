"""Macro / real-rates / dollar family signals for gold.

Logic (from literature):
 - Gold = -real-yield asset. Erb-Harvey: corr(real rate, gold) ~ -0.82.
 - Falling 10y yield / falling real yield  -> bullish gold.
 - Weaker dollar -> bullish gold (gold priced in USD).
 - Higher inflation breakevens -> bullish (store of value).
 - Fair-value residual: regress gold on rates+dollar, trade the macro mean-reversion.

We SELECT params on train+val only; test (2019+) is OOS. Compare to gold buy-and-hold.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT, _yahoo

df = load()
c = df['close']
ret = c.pct_change()

# ---- extra Yahoo data: TIP ETF as real-yield proxy ----
try:
    tip = _yahoo('TIP')['close'].reindex(df.index).ffill()
    HAVE_TIP = tip.notna().sum() > 2000
except Exception as e:
    tip = pd.Series(index=df.index, dtype=float); HAVE_TIP = False
    print("TIP unreachable:", repr(e)[:80])

tnx = df['tnx']   # 10y nominal yield (%*10 on Yahoo, but level scale irrelevant for momentum/sign)
dxy = df['dxy']

def show(w, label, band=0.10):
    net, held, turn = backtest(df, w, band=band)
    report(df, net, turn, label)
    # return val+test sharpe for programmatic scan
    out={}
    for k,(a,b) in SPLIT.items():
        m=(df.index>=a)&(df.index<b); mm=metrics(net,m)
        out[k]=(mm['sharpe'],mm['nav']) if mm else (np.nan,np.nan)
    return out, int((turn>1e-9).sum())

# buy-and-hold baseline (sanity)
print("="*70)
print("BASELINE: gold buy-and-hold")
bh={}
for k,(a,b) in SPLIT.items():
    m=(df.index>=a)&(df.index<b); mm=metrics(ret,m)
    bh[k]=(mm['sharpe'],mm['nav'])
    print(f"  {k:6s} Sharpe {mm['sharpe']:.2f}  NAV {mm['nav']:.2f}  DD {mm['maxdd']*100:.0f}%")

# =====================================================================
# SIGNAL 1: 10y-yield momentum (the known-robust factor). Long gold when
# yields are FALLING (negative yield change). Sign: dGold/dYield < 0.
# weight = -sign/scale of yield momentum.
# =====================================================================
print("\n" + "="*70 + "\nSIGNAL 1: 10y yield momentum (long gold when yields falling)")
for N in (20, 40, 60, 90, 120):
    ymom = tnx - tnx.shift(N)           # change in yield over N days
    w = (-np.sign(ymom)).clip(-1,1)     # falling yield -> +1 long
    o,t = show(w, f"S1 yield-mom binary N={N}", band=0.10)

# continuous z-scored version (smoother, lower turnover)
print("--- continuous z-score variants ---")
for N in (40, 60, 90):
    ymom = tnx - tnx.shift(N)
    z = (ymom - ymom.rolling(252).mean())/ymom.rolling(252).std()
    w = (-z).clip(-1.5,1.5)
    o,t = show(w, f"S1 yield-mom z N={N}", band=0.10)

# =====================================================================
# SIGNAL 2: real-yield proxy via TIP momentum. TIP price up => real yields
# down => bullish gold. weight = +sign(TIP momentum).
# =====================================================================
print("\n" + "="*70 + "\nSIGNAL 2: TIP (real-yield proxy) momentum, long gold when TIP rising")
if HAVE_TIP:
    for N in (20, 40, 60, 90, 120):
        tmom = tip/tip.shift(N) - 1
        w = (np.sign(tmom)).clip(-1,1)
        o,t = show(w, f"S2 TIP-mom binary N={N}", band=0.10)
    print("--- continuous ---")
    for N in (40, 60, 90):
        tmom = tip/tip.shift(N) - 1
        z = (tmom - tmom.rolling(252).mean())/tmom.rolling(252).std()
        w = (z).clip(-1.5,1.5)
        o,t = show(w, f"S2 TIP-mom z N={N}", band=0.10)
else:
    print("  SKIP - TIP not available")

# =====================================================================
# SIGNAL 3: dollar (DXY) signal. Weaker dollar -> bullish gold.
# weight = -sign(DXY momentum).
# =====================================================================
print("\n" + "="*70 + "\nSIGNAL 3: dollar (DXY) momentum, long gold when dollar falling")
for N in (20, 40, 60, 90, 120):
    dmom = dxy/dxy.shift(N) - 1
    w = (-np.sign(dmom)).clip(-1,1)
    o,t = show(w, f"S3 DXY-mom binary N={N}", band=0.10)

# =====================================================================
# SIGNAL 4: fair-value residual macro mean-reversion.
# Regress log(gold) on tnx and log(dxy) on a TRAILING window (no lookahead),
# trade the residual: gold cheap vs macro fair value -> long.
# Sign: weight = -sign(residual)  (residual>0 = gold rich -> short/flat).
# =====================================================================
print("\n" + "="*70 + "\nSIGNAL 4: macro fair-value residual (rolling regress logGold ~ tnx + logDXY)")
lg = np.log(c); ldxy = np.log(dxy)
X = pd.DataFrame({'const':1.0,'tnx':tnx,'ldxy':ldxy}, index=df.index)
def rolling_resid(win):
    res = pd.Series(index=df.index, dtype=float)
    Xv=X.values; yv=lg.values
    for i in range(win, len(df)):
        xs=Xv[i-win:i]; ys=yv[i-win:i]
        beta,_,_,_=np.linalg.lstsq(xs,ys,rcond=None)
        res.iloc[i]= yv[i]-Xv[i]@beta   # actual minus fair value, using betas from PAST window
    return res
for win in (252, 504, 756):
    res = rolling_resid(win)
    z = (res - res.rolling(252).mean())/res.rolling(252).std()
    w = (-z).clip(-1.5,1.5)   # gold rich (z>0) -> reduce; cheap -> long
    o,t = show(w, f"S4 FV-resid win={win}", band=0.10)

# =====================================================================
# SIGNAL 5: RATE-REGIME conditioning. Hold gold (long buy-and-hold style)
# ONLY when yields are falling; else flat. This is a long/flat overlay on B&H.
# =====================================================================
print("\n" + "="*70 + "\nSIGNAL 5: rate-regime gate (hold gold only when yields falling)")
for N in (40, 60, 90, 120, 200):
    falling = (tnx < tnx.shift(N)).astype(float)   # 1 when yield below N days ago
    w = falling.clip(0,1)   # long-only gate, weight in {0,1}
    o,t = show(w, f"S5 regime-gate long/flat N={N}", band=0.05)

# combine regime gate with dollar gate (both bullish)
print("--- S5b: yields falling AND dollar falling ---")
for N in (60, 90, 120):
    gate = ((tnx < tnx.shift(N)) & (dxy < dxy.shift(N))).astype(float)
    w = gate
    o,t = show(w, f"S5b dual-gate long/flat N={N}", band=0.05)

print("\nDONE")
