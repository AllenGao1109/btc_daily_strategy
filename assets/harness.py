"""Multi-asset research harness (SPX/NDX/UST + macro), total-return ETFs.

Uses Yahoo ADJUSTED close (includes dividends/coupons = true total return) for
tradeable ETFs: SPY (S&P500), QQQ (Nasdaq100), TLT (20y+ UST), IEF (7-10y UST),
GLD (gold). Plus macro (^TNX yield, ^VIX, DX-Y.NYB dollar). Cached. Backtests vs
each asset's own buy-and-hold, 0.1%/side fee, chronological train/val/test.
"""
from __future__ import annotations
import json, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd

CACHE = Path("data/processed/assets"); CACHE.mkdir(parents=True, exist_ok=True)
FEE = 0.001
SPLIT = {"train": ("2002-01-01","2013-01-01"), "val": ("2013-01-01","2019-01-01"),
         "test": ("2019-01-01","2027-01-01")}
TICKERS = {"spy":"SPY","qqq":"QQQ","tlt":"TLT","ief":"IEF","gld":"GLD",
           "tnx":"^TNX","vix":"^VIX","dxy":"DX-Y.NYB"}

def _yahoo_adj(sym):
    u=(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
       "?period1=0&period2=9999999999&interval=1d&events=div%2Csplit")
    r=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read())['chart']['result'][0]
    idx=[pd.Timestamp(t,unit='s',tz='UTC').normalize() for t in r['timestamp']]
    adj=r.get('indicators',{}).get('adjclose',[{}])[0].get('adjclose')
    cl=r['indicators']['quote'][0]['close']
    s=pd.Series(adj if adj else cl, index=idx).dropna()
    return s[~s.index.duplicated()].sort_index()

def load(force=False):
    p=CACHE/"panel.csv"
    if p.exists() and not force:
        d=pd.read_csv(p,index_col=0); d.index=pd.to_datetime(d.index,utc=True); return d
    cols={}
    for name,sym in TICKERS.items():
        try: cols[name]=_yahoo_adj(sym)
        except Exception as e: print(f"[warn] {name}({sym}): {str(e)[:40]}")
    d=pd.DataFrame(cols).ffill(); d.to_csv(p); return d

def backtest(close, weight, band=0.10, fee=FEE):
    r=close.pct_change()
    w=weight.shift(1).reindex(close.index).fillna(0.0).clip(-2,2)
    h=np.zeros(len(w)); wv=w.values
    for i in range(1,len(wv)):
        h[i]=h[i-1] if abs(wv[i]-h[i-1])<band else wv[i]
    held=pd.Series(h,index=w.index); turn=(held-held.shift()).abs().fillna(held.abs())
    return held*r-turn*fee, held, turn

def metrics(net, mask=None):
    s=net if mask is None else net[mask]; s=s.dropna()
    if len(s)<50: return None
    ann=s.mean()*252; vol=s.std()*np.sqrt(252); eq=(1+s).cumprod(); dd=(eq/eq.cummax()-1).min()
    return {"ann":ann,"vol":vol,"sharpe":ann/vol if vol>0 else 0,"maxdd":dd,"nav":eq.iloc[-1]/eq.iloc[0]}

def report(close, net, turn, label):
    r=close.pct_change()
    print(f"\n=== {label} (交易 {int((turn>1e-9).sum())} 次) ===")
    for k,(a,b) in SPLIT.items():
        m=(close.index>=a)&(close.index<b); ms=metrics(net,m); mb=metrics(r,m)
        if ms and mb: print(f"  {k:6s} 策略 {ms['sharpe']:5.2f}/{ms['nav']:6.2f}/{ms['maxdd']*100:4.0f}% | 持有 {mb['sharpe']:5.2f}/{mb['nav']:6.2f}/{mb['maxdd']*100:4.0f}%")
    f=metrics(net); fb=metrics(r); print(f"  {'全样本':6s} 策略 {f['sharpe']:5.2f}/{f['nav']:6.2f}/{f['maxdd']*100:4.0f}% | 持有 {fb['sharpe']:5.2f}/{fb['nav']:6.2f}/{fb['maxdd']*100:4.0f}%")

if __name__=="__main__":
    d=load(force=True)
    print(f"面板: {d.index.min().date()} -> {d.index.max().date()} ({len(d)}天)")
    for c in d.columns:
        s=d[c].dropna(); print(f"  {c:5s}: {s.index.min().date()} ({len(s)}天) last={s.iloc[-1]:.2f}")
    # quick buy-hold characterization of the 3 new assets
    print("\n各资产买入持有(全样本):")
    for a in ['spy','qqq','tlt','ief']:
        m=metrics(d[a].pct_change()); print(f"  {a.upper():4s}: Sharpe {m['sharpe']:.2f}, 年化 {m['ann']*100:.1f}%, 波动 {m['vol']*100:.1f}%, 回撤 {m['maxdd']*100:.0f}%")
