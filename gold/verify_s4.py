"""Independent verification of S4 gold signal.

S4: Vol-targeted base-long + 0.5x below-200dma boost.
weight = clip( volscale(0.12,45d) * (1 + 0.5*[SPX<SMA200]), 0, 2 ); band=0.20
Claimed: val Sharpe -0.19, test 1.03.
"""
import sys
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, SPLIT, FEE

df = load()
c = df['close']
spx = df['spx']
ret = c.pct_change()

def vol_target(target_ann=0.12, win=45):
    realized = ret.rolling(win).std() * np.sqrt(252)
    return (target_ann / realized.replace(0, np.nan))

def s4_weight(target=0.12, volwin=45, boost=0.5, smawin=200, cap=2.0):
    vs = vol_target(target, volwin)
    below = (spx < spx.rolling(smawin).mean()).astype(float)
    w = (vs * (1 + boost * below)).clip(0, cap)
    return w

def seg_metrics(net, label):
    out = {}
    for k,(a,b) in SPLIT.items():
        m = (df.index>=a)&(df.index<b)
        mm = metrics(net, m)
        out[k] = mm
    return out

def show(net, turn, label):
    print(f"\n=== {label} (trades={int((turn>1e-9).sum())}, turnover_sum={turn.sum():.1f}) ===")
    rb = ret
    print(f"{'seg':6s} {'strat Sh/NAV/DD/ann/vol':>40s} | {'B&H Sh/NAV/DD':>22s}")
    for k,(a,b) in SPLIT.items():
        m = (df.index>=a)&(df.index<b)
        ms = metrics(net,m); mb = metrics(rb,m)
        if ms and mb:
            print(f"{k:6s} {ms['sharpe']:5.2f}/{ms['nav']:6.2f}/{ms['maxdd']*100:5.0f}%/{ms['ann']*100:5.1f}/{ms['vol']*100:5.1f} | {mb['sharpe']:5.2f}/{mb['nav']:6.2f}/{mb['maxdd']*100:5.0f}%")
    return {k: metrics(net,(df.index>=a)&(df.index<b)) for k,(a,b) in SPLIT.items()}

print("="*70)
print("BASELINE: buy-and-hold")
for k,(a,b) in SPLIT.items():
    m=(df.index>=a)&(df.index<b); mb=metrics(ret,m)
    print(f"  {k:6s} Sh={mb['sharpe']:.2f} NAV={mb['nav']:.2f} DD={mb['maxdd']*100:.0f}%")
mb_full = metrics(ret); print(f"  full   Sh={mb_full['sharpe']:.2f} NAV={mb_full['nav']:.2f} DD={mb_full['maxdd']*100:.0f}%")

# (a) reproduce
print("\n" + "="*70)
print("(a) REPRODUCE: S4 base params, band=0.20")
w = s4_weight()
net, held, turn = backtest(df, w, band=0.20)
base = show(net, turn, "S4 base (t=0.12,vw=45,boost=0.5,sma=200,band=0.20)")

# (b) parameter sensitivity
print("\n" + "="*70)
print("(b) PARAMETER SENSITIVITY")
print("\n-- target vol --")
for t in [0.08,0.10,0.12,0.14,0.16]:
    w=s4_weight(target=t); net,h,tn=backtest(df,w,band=0.20)
    r=seg_metrics(net,'')
    print(f"  t={t:.2f}: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f} test DD={r['test']['maxdd']*100:4.0f}%")
print("\n-- vol window --")
for vw in [20,30,45,60,90]:
    w=s4_weight(volwin=vw); net,h,tn=backtest(df,w,band=0.20)
    r=seg_metrics(net,'')
    print(f"  vw={vw}: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f} test DD={r['test']['maxdd']*100:4.0f}%")
print("\n-- boost --")
for bo in [0.0,0.25,0.5,0.75,1.0]:
    w=s4_weight(boost=bo); net,h,tn=backtest(df,w,band=0.20)
    r=seg_metrics(net,'')
    print(f"  boost={bo:.2f}: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f} test DD={r['test']['maxdd']*100:4.0f}%")
print("\n-- sma window --")
for sw in [100,150,200,250]:
    w=s4_weight(smawin=sw); net,h,tn=backtest(df,w,band=0.20)
    r=seg_metrics(net,'')
    print(f"  sma={sw}: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f} test DD={r['test']['maxdd']*100:4.0f}%")
print("\n-- band --")
for bd in [0.05,0.10,0.15,0.20,0.30]:
    w=s4_weight(); net,h,tn=backtest(df,w,band=bd)
    r=seg_metrics(net,'')
    print(f"  band={bd:.2f}: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f} test DD={r['test']['maxdd']*100:4.0f}% trades={int((tn>1e-9).sum())}")

# (c) does the boost actually add value, or is it just vol-targeted long?
print("\n" + "="*70)
print("(c) DECOMPOSITION: is the edge from the SPX<200dma boost or just vol-target long?")
w_plain = vol_target().clip(0,2.0)
net_p, h, tn_p = backtest(df, w_plain, band=0.20)
print("\n-- plain vol-target long (NO boost) --")
plain = show(net_p, tn_p, "vol-target long only")
w_full = s4_weight()
net_f, h, tn_f = backtest(df, w_full, band=0.20)
# isolate boost contribution: full minus plain net returns
print("\nBoost-only incremental (full net - plain net) by segment:")
diff = net_f - net_p
for k,(a,b) in SPLIT.items():
    m=(df.index>=a)&(df.index<b)
    d=metrics(diff,m)
    if d: print(f"  {k:6s} incremental ann={d['ann']*100:6.2f}% sharpe(of diff)={d['sharpe']:5.2f}")

# (d) regime breakdown - yearly test-period returns
print("\n" + "="*70)
print("(d) REGIME BREAKDOWN: yearly net returns (S4 base)")
net, held, turn = backtest(df, s4_weight(), band=0.20)
yr = (1+net).groupby(net.index.year).prod()-1
yr_bh = (1+ret).groupby(ret.index.year).prod()-1
for y in sorted(set(net.index.year)):
    if y in yr.index:
        tag = "TEST" if y>=2019 else ("VAL" if y>=2013 else "trn")
        d = (yr[y]-yr_bh[y])*100
        print(f"  {y} [{tag}]: S4={yr[y]*100:7.2f}%  B&H={yr_bh[y]*100:7.2f}%  diff={d:7.2f}%")

# (e) fee sensitivity
print("\n" + "="*70)
print("(e) FEE SENSITIVITY (test Sharpe)")
for fee in [0.0,0.001,0.002,0.004,0.006]:
    net,h,tn=backtest(df,s4_weight(),band=0.20,fee=fee)
    r=seg_metrics(net,'')
    print(f"  fee={fee*100:.1f}%/side: val Sh={r['val']['sharpe']:5.2f} test Sh={r['test']['sharpe']:5.2f}")

# Full-sample comparison
print("\n" + "="*70)
print("FULL SAMPLE S4 vs B&H")
net,h,turn=backtest(df,s4_weight(),band=0.20)
mfull=metrics(net);
print(f"  S4 : Sh={mfull['sharpe']:.2f} NAV={mfull['nav']:.2f} DD={mfull['maxdd']*100:.0f}% ann={mfull['ann']*100:.1f}% vol={mfull['vol']*100:.1f}%")
print(f"  B&H: Sh={mb_full['sharpe']:.2f} NAV={mb_full['nav']:.2f} DD={mb_full['maxdd']*100:.0f}%")
