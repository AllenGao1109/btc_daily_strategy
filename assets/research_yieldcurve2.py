"""Yield-curve/rates family — round 2: rigorous sign-consistency check across train+val+test
for the candidates that survived round 1, plus cleaner timing backtests.

Focus: is ANY rates factor same-SIGN with |IC|>0.03 on train AND val AND test on BOTH spy & qqq?
(test only reported, never tuned — but we want to SEE if the survivors actually generalize.)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load(); idx = d.index
TR=(idx>=SPLIT["train"][0])&(idx<SPLIT["train"][1])
VA=(idx>=SPLIT["val"][0])&(idx<SPLIT["val"][1])
TE=(idx>=SPLIT["test"][0])&(idx<SPLIT["test"][1])
tnx,irx,fvx = d["tnx"],d["irx"],d["fvx"]

cand = {
  "slope_10y3m":        tnx-irx,
  "slope_chg_60d":      (tnx-irx)-(tnx-irx).shift(60),
  "tnx_level_neg":      -tnx,
  "tnx_chg20_neg":      -(tnx-tnx.shift(20)),
  "tnx_chg60_neg":      -(tnx-tnx.shift(60)),
  "tnx_vs_avg_neg":     -(tnx-tnx.rolling(252).mean()),
  "tnx_vs_avg126_neg":  -(tnx-tnx.rolling(126).mean()),
}
print("Sign-consistency table: + means factor predicts higher fwd ret. Want SAME sign on tr/va/te BOTH targets.")
print(f"{'factor':20s}{'tgt/h':8s}{'IC_tr':>8s}{'IC_va':>8s}{'IC_te':>8s}  verdict")
for name,f in cand.items():
    for tgt in ("spy","qqq"):
        for h in (20,60):
            fwd=forward_return(d[tgt],h)
            it,iv,ite=ic(f,fwd,TR),ic(f,fwd,VA),ic(f,fwd,TE)
            allsame = np.sign(it)==np.sign(iv)==np.sign(ite) and min(abs(it),abs(iv),abs(ite))>0.03
            tvsame  = np.sign(it)==np.sign(iv) and min(abs(it),abs(iv))>0.03
            v="ALL3" if allsame else ("tr+va" if tvsame else "")
            print(f"{name:20s}{tgt+str(h):8s}{it:+8.3f}{iv:+8.3f}{ite:+8.3f}  {v}")
    print()

# ---- timing: rate-momentum (tnx_vs_avg126) as continuous weight, low turnover via band ----
def run(tgt, weight, band, label):
    close=d[tgt]; net,held,turn=backtest(close,weight,band); r=close.pct_change()
    ntr=int((turn>1e-9).sum()); row=f"{label:34s}{tgt.upper():4s} trades={ntr:4d}"
    res={}
    for k,(a,b) in SPLIT.items():
        m=(close.index>=a)&(close.index<b); ms=metrics(net,m); mb=metrics(r,m)
        res[k]=(ms['sharpe'],ms['nav'],mb['sharpe'],mb['nav'])
        row+=f" | {k[:2]} S{ms['sharpe']:+.2f}v{mb['sharpe']:+.2f}"
    print(row); return res

print("="*120)
print("TIMING BACKTESTS (rates-momentum overlays vs trend baseline). 'S strat v hold' per split.")
print("="*120)
# weight = long when rates falling vs 126d avg (tnx below avg). Continuous via sigmoid-ish clip.
ratefall = -(tnx-tnx.rolling(126).mean())            # >0 = rates below avg = bullish
wRM = (ratefall>0).astype(float)                      # binary long/flat
wRM_c = (0.5+ (ratefall.clip(-1,1))).clip(0,1)        # continuous 0..1
trend_spy=(d["spy"]>d["spy"].rolling(200).mean()).astype(float)
trend_qqq=(d["qqq"]>d["qqq"].rolling(200).mean()).astype(float)
# overlay: trend long, but go flat if rates rising fast (ratefall<-0.5 => recent sharp rate spike)
ov_spy=(trend_spy* (ratefall>-0.5)).astype(float)
ov_qqq=(trend_qqq* (ratefall>-0.5)).astype(float)
# leverage version: trend gives 1, add 0.5 when rates also falling (mild)
lev_spy=(trend_spy*(1.0+0.5*(ratefall>0))).clip(0,1.5)
lev_qqq=(trend_qqq*(1.0+0.5*(ratefall>0))).clip(0,1.5)

for tgt in ("spy","qqq"):
    tr = trend_spy if tgt=="spy" else trend_qqq
    ov = ov_spy if tgt=="spy" else ov_qqq
    lev= lev_spy if tgt=="spy" else lev_qqq
    run(tgt, wRM, 0.4, "rates<126avg long/flat")
    run(tgt, wRM_c, 0.3, "rates-momentum continuous")
    run(tgt, tr, 0.5, "trend200 baseline")
    run(tgt, ov, 0.5, "trend200 + flat-if-rate-spike")
    run(tgt, lev, 0.5, "trend200 + lev-if-rates-fall")
    print()

print("BUY-HOLD: ", end="")
for tgt in ("spy","qqq"):
    r=d[tgt].pct_change()
    s={k:metrics(r,(idx>=a)&(idx<b))['sharpe'] for k,(a,b) in SPLIT.items()}
    print(f"{tgt.upper()} tr{s['train']:+.2f}/va{s['val']:+.2f}/te{s['test']:+.2f}  ",end="")
print()
