"""Equity-index timing harness: SPY/QQQ targets + a broad macro/risk/credit panel.

For mining INDEX-TIMING factors (not cross-sectional stock selection): yield curve,
credit spreads, VIX term structure, breadth/sector ratios, momentum, cross-asset.
Total-return ETFs (adjusted close). Cached. forward_return + IC helpers, train/val/test.
"""
from __future__ import annotations
import json, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd

CACHE=Path("data/processed/equity"); CACHE.mkdir(parents=True,exist_ok=True)
SPLIT={"train":("2006-01-01","2015-01-01"),"val":("2015-01-01","2020-01-01"),"test":("2020-01-01","2027-01-01")}
TICKERS={"spy":"SPY","qqq":"QQQ",                       # targets (total return)
         "vix":"^VIX","vix3m":"^VIX3M","tnx":"^TNX","irx":"^IRX","fvx":"^FVX","tyx":"^TYX",
         "dxy":"DX-Y.NYB","hyg":"HYG","lqd":"LQD","tlt":"TLT","gld":"GLD","slv":"SLV",
         "xlu":"XLU","xlk":"XLK","xly":"XLY","xlp":"XLP","xlf":"XLF","cper":"CPER"}

def _y(sym, adj=True):
    u=(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
       "?period1=0&period2=9999999999&interval=1d&events=div%2Csplit")
    r=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read())['chart']['result'][0]
    idx=[pd.Timestamp(t,unit='s',tz='UTC').normalize() for t in r['timestamp']]
    a=r.get('indicators',{}).get('adjclose',[{}])[0].get('adjclose') if adj else None
    cl=r['indicators']['quote'][0]['close']
    s=pd.Series(a if a else cl,index=idx).dropna()
    return s[~s.index.duplicated()].sort_index()

def load(force=False):
    p=CACHE/"equity_panel.csv"
    if p.exists() and not force:
        d=pd.read_csv(p,index_col=0); d.index=pd.to_datetime(d.index,utc=True); return d
    cols={}
    for n,sym in TICKERS.items():
        try: cols[n]=_y(sym, adj=(sym not in ("^VIX","^VIX3M","^TNX","^IRX","^FVX","^TYX","DX-Y.NYB")))
        except Exception as e: print(f"[warn] {n}({sym}): {str(e)[:35]}")
    d=pd.DataFrame(cols).ffill(); d.to_csv(p); return d

def forward_return(close, h):
    return close.shift(-h)/close-1
def ic(f, fwd, mask):
    j=pd.concat([f,fwd],axis=1).replace([np.inf,-np.inf],np.nan)[mask].dropna()
    return float(j.iloc[:,0].rank().corr(j.iloc[:,1].rank())) if len(j)>120 else 0.0

if __name__=="__main__":
    d=load(force=True)
    print(f"股指因子面板: {d.index.min().date()} -> {d.index.max().date()} ({len(d)}天)")
    print("可用列:", [c for c in d.columns if d[c].notna().sum()>500])
    # quick sanity: IC of a few canonical equity-timing factors vs SPY fwd 20d
    tgt="spy"; c=d[tgt]; tr=(d.index<'2015-01-01')&(d.index>='2006-01-01'); va=(d.index>='2015-01-01')&(d.index<'2020-01-01')
    fwd=forward_return(c,20)
    F={}
    if "tnx" in d and "irx" in d: F["收益率曲线 10y-3m"]=d["tnx"]-d["irx"]   # 倒挂(负)预示衰退
    if "vix3m" in d and "vix" in d: F["VIX期限 (3m-spot)"]=d["vix3m"]-d["vix"] # 负=恐慌(短端高)
    if "hyg" in d and "lqd" in d: F["信用 HYG/LQD动量"]=(d["hyg"]/d["lqd"]).pct_change(60)
    if "xlu" in d and "xlk" in d: F["防御/进攻 XLU/XLK"]=(d["xlu"]/d["xlk"]).pct_change(60)
    F["SPY动量200d"]=c.pct_change(200)
    print(f"\n{'因子':22s} {'IC_tr':>7s} {'IC_va':>7s} robust?")
    for n,f in F.items():
        it=ic(f,fwd,tr); iv=ic(f,fwd,va); rob='ROBUST' if abs(it)>0.03 and abs(iv)>0.03 and np.sign(it)==np.sign(iv) else ''
        print(f"{n:22s} {it:+7.3f} {iv:+7.3f} {rob}")
