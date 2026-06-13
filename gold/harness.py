"""Shared gold research harness: cached data + disciplined backtest eval.

Data: Yahoo GC=F (gold futures, 2000+) plus macro (^TNX yields, DX-Y.NYB dollar,
^GSPC, ^VIX). Cached under data/processed/gold/. Backtests are vs gold buy-and-hold,
with a 0.2% per-side fee and a no-trade band, split train/val/test chronologically.
"""
from __future__ import annotations
import json, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd

CACHE = Path("data/processed/gold"); CACHE.mkdir(parents=True, exist_ok=True)
FEE = 0.002
SPLIT = {"train": ("2000-01-01", "2013-01-01"),
         "val":   ("2013-01-01", "2019-01-01"),
         "test":  ("2019-01-01", "2027-01-01")}

def _yahoo(sym):
    u=f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1=0&period2=9999999999&interval=1d"
    r=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read())['chart']['result'][0]
    q=r['indicators']['quote'][0]
    return pd.DataFrame({k:q[k] for k in ['open','high','low','close','volume']},
                        index=[pd.Timestamp(t,unit='s',tz='UTC').normalize() for t in r['timestamp']]).dropna(subset=['close'])

def load(force=False):
    p=CACHE/"gold_panel.csv"
    if p.exists() and not force:
        d=pd.read_csv(p,index_col=0); d.index=pd.to_datetime(d.index,utc=True); return d
    g=_yahoo('GC=F'); out=pd.DataFrame(index=g.index)
    for c in ['open','high','low','close','volume']: out[c]=g[c]
    for sym,name in [('^TNX','tnx'),('DX-Y.NYB','dxy'),('^GSPC','spx'),('^VIX','vix')]:
        try: out[name]=_yahoo(sym)['close'].reindex(g.index).ffill()
        except Exception: pass
    out=out.ffill().dropna(subset=['close']); out.to_csv(p); return out

def backtest(df, weight, band=0.10, fee=FEE):
    """weight: target series (signal, pre-lag). Returns net daily returns + held weight."""
    r=df['close'].pct_change()
    w=weight.shift(1).reindex(df.index).fillna(0.0).clip(-2,2)
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

def report(df, net, turn, label):
    r=df['close'].pct_change()
    print(f"\n=== {label} (交易 {int((turn>1e-9).sum())} 次) ===")
    print(f"{'区间':10s} {'策略 Sh/NAV/DD':>22s} | {'买入持有 Sh/NAV/DD':>22s}")
    for k,(a,b) in SPLIT.items():
        m=(df.index>=a)&(df.index<b)
        ms=metrics(net,m); mb=metrics(r,m)
        if ms and mb:
            print(f"{k:10s} {ms['sharpe']:5.2f}/{ms['nav']:5.2f}/{ms['maxdd']*100:4.0f}% | {mb['sharpe']:5.2f}/{mb['nav']:5.2f}/{mb['maxdd']*100:4.0f}%")

if __name__=="__main__":
    df=load(force=True); c=df['close']
    print(f"GC=F: {df.index.min().date()}->{df.index.max().date()} ({len(df)}天), 列: {list(df.columns)}")
    # 用户的想法:均值回归 z-score。position = -z (低于均值买,高于卖)
    for N in (50,100,200):
        z=((c-c.rolling(N).mean())/c.rolling(N).std()).clip(-3,3)
        # 连续版:仓位 = -z * scale,vol-target 到 ~10%
        vt=(0.10/(c.pct_change().rolling(45).std()*np.sqrt(252)).replace(0,np.nan)).clip(upper=2.0)
        w=(-z*0.5).clip(-2,2)  # 纯均值回归方向,固定缩放
        net,held,turn=backtest(df,w,band=0.15)
        report(df,net,turn,f"均值回归(连续) N={N}")
