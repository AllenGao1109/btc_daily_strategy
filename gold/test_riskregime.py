"""Risk-regime / hedge-timing signals for GOLD.

Thesis (Baur & Lucey 2010; Baur & McDermott 2010; WGC): gold is a SAFE HAVEN that
pays off specifically in equity risk-off states (VIX spikes, SPX below 200dma /
in drawdown, rising realized vol). In calm regimes gold behaves like a normal
growth asset (positive equity correlation). So TIMING gold to be long only/more
in risk-off should, in theory, harvest the crisis-alpha while sidestepping calm
drawdowns. We test whether this beats simply HOLDING gold, net of fees, OOS.

Discipline: tune on train+val only; test (2019+) is OOS. Compare to gold BH.
All weights are target-exposure Series in [-2,2]; harness lags 1d, fee 0.2%/side,
no-trade band. Prefer LOW turnover (these are regime signals -> few switches).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT

df = load()
c   = df["close"]
r   = c.pct_change()
vix = df["vix"]
spx = df["spx"]

# ---- helper building blocks ---------------------------------------------
def vt_scale(target=0.10, win=45, cap=1.5):
    """vol-target scaler for gold so exposures are comparable to BH (~full)."""
    rv = r.rolling(win).std()*np.sqrt(252)
    return (target/rv.replace(0,np.nan)).clip(upper=cap)

spx_ret = spx.pct_change()
spx_200 = spx.rolling(200).mean()
spx_dd  = spx/spx.cummax()-1.0                  # equity drawdown from running peak
spx_below200 = (spx < spx_200).astype(float)    # 1 when SPX below its 200dma
spx_rv  = spx_ret.rolling(20).std()*np.sqrt(252)
vix_ma  = vix.rolling(50).mean()

def summ(net, turn):
    out={}
    for k,(a,b) in SPLIT.items():
        m=(df.index>=a)&(df.index<b)
        ms=metrics(net,m); mb=metrics(r,m)
        out[k]=(ms,mb)
    out["trades"]=int((turn>1e-9).sum())
    return out

results={}
def run(weight, label, band=0.10):
    net,held,turn=backtest(df,weight,band=band)
    report(df,net,turn,label)
    results[label]=summ(net,turn)
    return net,held,turn

print("#"*70)
print("# BASELINE: gold buy-and-hold (weight=1)")
print("#"*70)
run(pd.Series(1.0,index=df.index),"BH gold (w=1)")

# =========================================================================
# SIGNAL 1: VIX-regime overlay. Long gold always, but ADD exposure in
# risk-off (high VIX) and TRIM in calm. Logic: gold's hedge payoff concentrates
# in high-VIX states; calm states are where gold whipsaws/drifts.
# Base long 1.0, scaled up to 1.5 when VIX high, down to 0.5 when VIX low.
# =========================================================================
print("\n"+"#"*70); print("# SIGNAL 1: VIX-regime exposure tilt"); print("#"*70)
for lo,hi in [(15,25),(15,30),(18,28),(20,35)]:
    # linear ramp of exposure from 0.5 (vix<=lo) to 1.5 (vix>=hi)
    tilt = (0.5 + 1.0*((vix-lo)/(hi-lo)).clip(0,1))
    run(tilt, f"S1 VIX-tilt [{lo},{hi}]", band=0.10)

# =========================================================================
# SIGNAL 2: Risk-OFF binary switch. Hold gold ONLY when equity is risk-off
# (VIX above its own moving-avg OR SPX below 200dma), else flat. Tests the
# pure asymmetry: is gold's return concentrated in risk-off days?
# =========================================================================
print("\n"+"#"*70); print("# SIGNAL 2: risk-off ON/OFF switch"); print("#"*70)
riskoff_vix   = (vix > vix_ma).astype(float)                  # rising vol regime
riskoff_200   = spx_below200
riskoff_either= ((vix>vix_ma)|(spx<spx_200)).astype(float)
riskoff_both  = ((vix>vix_ma)&(spx<spx_200)).astype(float)
run(riskoff_vix,    "S2a long-iff VIX>VIXma",        band=0.10)
run(riskoff_200,    "S2b long-iff SPX<200dma",       band=0.10)
run(riskoff_either, "S2c long-iff (VIX>ma OR <200)", band=0.10)
run(riskoff_both,   "S2d long-iff (VIX>ma AND<200)", band=0.10)

# =========================================================================
# SIGNAL 3: ALWAYS-LONG base + risk-off BOOST (asymmetric). Stay long gold
# (1.0) to keep the secular bull, but lever up to 1.5 only in risk-off.
# Captures crisis-alpha WITHOUT shorting the bull. Low turnover (regime).
# =========================================================================
print("\n"+"#"*70); print("# SIGNAL 3: base-long + risk-off boost"); print("#"*70)
for boost in [0.5, 1.0]:
    w = 1.0 + boost*spx_below200
    run(w, f"S3a base1+{boost}*below200", band=0.10)
for boost in [0.5, 1.0]:
    w = 1.0 + boost*(vix>vix_ma).astype(float)
    run(w, f"S3b base1+{boost}*VIX>ma", band=0.10)
# deeper-stress boost: only when SPX in a real drawdown
for thr,boost in [(-0.05,0.5),(-0.10,1.0),(-0.10,0.5)]:
    w = 1.0 + boost*(spx_dd<thr).astype(float)
    run(w, f"S3c base1+{boost}*(spxDD<{thr})", band=0.10)

# =========================================================================
# SIGNAL 4: vol-targeted base-long + risk-off boost. Same as S3 but exposure
# vol-scaled so risk is steady; boost in risk-off. (controls for vol drift)
# =========================================================================
print("\n"+"#"*70); print("# SIGNAL 4: vol-target base + risk-off boost"); print("#"*70)
vts = vt_scale(0.12, 45, 1.5)
for boost in [0.5, 1.0]:
    w = (vts*(1.0 + boost*spx_below200)).clip(0,2)
    run(w, f"S4 vt-base+{boost}*below200", band=0.10)

# =========================================================================
# SIGNAL 5: VIX z-score continuous tilt around base-long. exposure =
# 1 + k*zscore(vix). Rising/elevated VIX -> more gold. Low band to keep turnover down.
# =========================================================================
print("\n"+"#"*70); print("# SIGNAL 5: VIX z-score tilt"); print("#"*70)
for N in [60,120,252]:
    z=((vix-vix.rolling(N).mean())/vix.rolling(N).std()).clip(-2,2)
    for k in [0.25,0.5]:
        w=(1.0 + k*z).clip(0,2)
        run(w, f"S5 1+{k}*z(vix,{N})", band=0.12)

print("\n\nDONE")
