"""Attribution + refinement of the risk-regime gold signals.

From round 1 the only candidate beating BH on BOTH val AND test SHARPE was a
vol-targeted base-long with a below-200dma boost (S4). But its NAV trailed BH.
Need to answer rigorously:
  (A) Is the Sharpe edge from VOL-TARGETING alone, or from the RISK-REGIME boost?
  (B) Does the regime BOOST add anything on top of vol-targeted base-long?
  (C) Best honest low-turnover candidate.

Bar to beat (gold BH): val Sh -0.22 / NAV 0.76 ; test Sh 0.99 / NAV 3.31 ;
full-sample Sh ~0.69.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT

df=load(); c=df["close"]; r=c.pct_change(); vix=df["vix"]; spx=df["spx"]
spx_200=spx.rolling(200).mean(); spx_below200=(spx<spx_200).astype(float)
vix_ma=vix.rolling(50).mean()

def vt(target=0.12,win=45,cap=1.5):
    rv=r.rolling(win).std()*np.sqrt(252)
    return (target/rv.replace(0,np.nan)).clip(upper=cap)

def fullsh(net):
    m=metrics(net); return m["sharpe"],m["nav"]

def run(weight,label,band=0.10):
    net,held,turn=backtest(df,weight,band=band)
    report(df,net,turn,label)
    fs,fn=fullsh(net)
    print(f"   full-sample: Sh {fs:.2f} / NAV {fn:.2f} | avg held {held.mean():.2f}")
    return net,held,turn

print("="*70); print("(A) ISOLATE vol-targeting: vt base-long, NO regime"); print("="*70)
for tgt in [0.10,0.12,0.14]:
    run((vt(tgt)).clip(0,2), f"VTonly target={tgt}", band=0.10)

print("\n"+"="*70); print("(B) vt base + below200 boost — does regime ADD?"); print("="*70)
for boost in [0.0,0.25,0.5,1.0]:
    w=(vt(0.12)*(1.0+boost*spx_below200)).clip(0,2)
    run(w, f"vt0.12 + {boost}*below200", band=0.10)

print("\n"+"="*70); print("(C) low-turnover regime tilt on FLAT (non-vt) base"); print("="*70)
# Pure asymmetry test, low turnover: long 1, boost in below200, but slow (band)
for band in [0.10,0.25,0.40]:
    w=1.0+0.5*spx_below200
    run(w, f"base1+0.5*below200 band={band}", band=band)

print("\n"+"="*70); print("(D) combine: vt base + below200 boost, wider band (lower turnover)"); print("="*70)
for band in [0.10,0.20,0.30]:
    w=(vt(0.12)*(1.0+0.5*spx_below200)).clip(0,2)
    run(w, f"vt0.12+0.5below200 band={band}", band=band)

print("\n"+"="*70); print("(E) sanity: is below200 boost robust to dma length / lag?"); print("="*70)
for N in [150,200,250]:
    below=(spx<spx.rolling(N).mean()).astype(float)
    w=(vt(0.12)*(1.0+0.5*below)).clip(0,2)
    run(w, f"vt0.12+0.5*(spx<{N}dma)", band=0.10)

print("\nDONE")
