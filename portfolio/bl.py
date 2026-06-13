"""Black-Litterman long-only portfolio over {BTC,SPY,QQQ,TLT,GLD}.

Structure A: long-only, no leverage (weights sum <=1, rest cash). Risk-parity prior +
per-asset views (the strategy signals) with confidence reflecting each asset's verified
edge (BTC/TLT high; SPY/QQQ/GLD low -> held mostly via the prior). Band rebalancing,
per-asset fees. Compared vs equal-weight and risk-parity-no-views.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from portfolio.panel import build

ASSETS=["btc","spy","qqq","tlt","gld"]
CONF={"btc":1.0,"tlt":0.8,"qqq":0.35,"spy":0.30,"gld":0.40}   # view confidence (edge strength)
FEE={"btc":0.002,"spy":0.001,"qqq":0.001,"tlt":0.001,"gld":0.001}
CAP=0.40; COVWIN=126; SHRINK=0.2; DELTA_TS=0.4; TAU=0.05; VIEW_SCALE=1.0
SPLIT={"train":("2017-01-01","2021-01-01"),"val":("2021-01-01","2023-07-01"),"test":("2023-07-01","2027-01-01")}

def cov(Rwin):
    S=np.cov(Rwin.T)*252
    return (1-SHRINK)*S+SHRINK*np.diag(np.diag(S))

def bl_target(Rwin, signals):
    S=cov(Rwin); sig=np.sqrt(np.diag(S)); n=len(ASSETS)
    iv=1.0/sig; w_prior=iv/iv.sum()                      # risk-parity (inverse-vol) prior
    sp=np.sqrt(w_prior@S@w_prior); delta=DELTA_TS/sp     # calibrate risk aversion to prior Sharpe
    Pi=delta*S@w_prior
    Q=Pi+VIEW_SCALE*signals*sig                          # absolute views = equilibrium tilted by signal*vol
    conf=np.array([CONF[a] for a in ASSETS])
    Omega=np.diag(TAU*np.diag(S)/conf)
    tS=TAU*S; tSinv=np.linalg.inv(tS); Oinv=np.linalg.inv(Omega)
    M=np.linalg.inv(tSinv+Oinv); mu=M@(tSinv@Pi+Oinv@Q)
    w=(1/delta)*np.linalg.inv(S)@mu
    w=np.clip(w,0,CAP)
    if w.sum()>1: w=w/w.sum()
    return w

def backtest(panel, weight_fn, band=0.08):
    R=panel[[f"ret_{a}" for a in ASSETS]].values
    S=panel[[f"sig_{a}" for a in ASSETS]].values
    n=len(ASSETS); held=np.zeros(n); rows=[]
    fee=np.array([FEE[a] for a in ASSETS])
    for t in range(COVWIN, len(panel)-1):
        Rwin=R[t-COVWIN:t]                                # trailing returns (causal)
        wt=weight_fn(Rwin, S[t])                          # target using data up to t
        if np.abs(wt-held).sum()>band:
            turn=np.abs(wt-held); cost=(turn*fee).sum(); held=wt.copy()
        else: cost=0.0
        pr=held@R[t+1]-cost                               # held into t+1
        rows.append((panel.index[t+1], pr, held.copy()))
    idx=[r[0] for r in rows]; ret=pd.Series([r[1] for r in rows],index=idx)
    W=pd.DataFrame([r[2] for r in rows],index=idx,columns=ASSETS)
    return ret, W

def perf(ret,a=None,b=None):
    s=ret if a is None else ret[(ret.index>=a)&(ret.index<b)]; s=s.dropna()
    if len(s)<30: return None
    ann=s.mean()*252; vol=s.std()*np.sqrt(252); eq=(1+s).cumprod(); dd=(eq/eq.cummax()-1).min()
    return ann/vol if vol>0 else 0, eq.iloc[-1]/eq.iloc[0], dd
def line(ret,lbl):
    f=perf(ret); cells=" | ".join(f"{k}:{(lambda p:f'{p[0]:.2f}/{p[1]:.2f}/{p[2]*100:.0f}%' if p else 'NA')(perf(ret,a,b))}" for k,(a,b) in SPLIT.items())
    print(f"{lbl:20s} 全:{f[0]:.2f}/{f[1]:5.2f}/{f[2]*100:3.0f}%  ||  {cells}")

def equal_w(Rwin,sig): return np.full(len(ASSETS),1.0/len(ASSETS))
def riskparity(Rwin,sig):
    sig_=np.sqrt(np.diag(cov(Rwin))); iv=1/sig_; w=iv/iv.sum(); return np.clip(w,0,CAP)

if __name__=="__main__":
    panel=build()
    print(f"组合回测 {panel.index[COVWIN].date()} -> {panel.index[-1].date()}  (格式 Sharpe/NAV/MaxDD)\n")
    print(f"{'策略':20s} {'全样本':>16s}  ||  train | val | test")
    for name,fn in [("等权 25%(基准)",equal_w),("风险平价(无观点)",riskparity),("Black-Litterman",bl_target)]:
        ret,W=backtest(panel,fn); line(ret,name)
    # single assets buy-hold for reference
    print()
    for a in ASSETS:
        line(panel[f"ret_{a}"].loc[panel.index[COVWIN]:], f"  持有 {a}")
