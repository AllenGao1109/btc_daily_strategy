"""Multi-asset panel: BTC, SPY, QQQ, GLD (risk sleeves) + SGOV cash yield.

No bonds (not tradeable on Robinhood). Each risk asset gets a causal directional view
s in [-1,1] from its own strategy. The residual (1 - sum of risk weights) is parked in
SGOV, modeled as the short T-bill rate (^IRX) accrued daily (SGOV ~ rolling 0-3mo bills,
zero duration). Aligned from 2017 (BTC start). All causal.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json, urllib.request, urllib.parse
import numpy as np, pandas as pd
from pathlib import Path

RISK = ["btc", "spy", "qqq", "gld"]

def _yahoo(sym):
    u=(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
       "?period1=0&period2=9999999999&interval=1d")
    r=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read())['chart']['result'][0]
    return pd.Series({pd.Timestamp(t,unit='s',tz='UTC').normalize():c for t,c in zip(r['timestamp'],r['indicators']['quote'][0]['close']) if c}).sort_index()

def btc_signal():
    from src.config import load_config
    from src.data import load_btc_data, load_eth_close, enrich_external
    from src.features import build_features
    from src.onchain import ONCHAIN_METRICS, load_coinmetrics, merge_onchain
    from src.factors import composite_signal_walkforward
    from src.strategies.factor_composite import DEFAULT_FACTORS
    cfg=load_config("config.yaml")
    df=merge_onchain(build_features(load_btc_data(cfg)), load_coinmetrics(ONCHAIN_METRICS))
    df["eth_close"]=load_eth_close(df.index); df=enrich_external(df)
    sig=composite_signal_walkforward(df, DEFAULT_FACTORS, horizon=20)
    return df["close"].pct_change().rename("btc"), sig.clip(-1,1).rename("btc")

def asset_signals():
    from assets.harness import load
    d=load(); out_r={}; out_s={}
    for a in ["spy","qqq","gld"]:
        c=d[a].dropna(); r=c.pct_change(); out_r[a]=r
        rv=(r.rolling(60).std()*np.sqrt(252)).replace(0,np.nan)
        if a in ("spy","qqq"):       # equities: mild trend-tilt view
            s=np.tanh(3*(c/c.rolling(200).mean()-1))
        else:                        # gold: vol-managed conviction, mild
            w=(0.12/rv).clip(0,1.5); s=0.5*(2*w/1.5-1)
        out_s[a]=s.clip(-1,1)
    return out_r, out_s

def build(force=False):
    p=Path("data/processed/portfolio_panel.csv")
    if p.exists() and not force:
        d=pd.read_csv(p,index_col=0); d.index=pd.to_datetime(d.index,utc=True); return d
    br,bs=btc_signal(); ar,asig=asset_signals()
    rets={"btc":br}; sigs={"btc":bs}; rets.update(ar); sigs.update(asig)
    R=pd.DataFrame({f"ret_{k}":v for k,v in rets.items()})
    S=pd.DataFrame({f"sig_{k}":v for k,v in sigs.items()})
    # SGOV / cash: short T-bill rate (^IRX, annual %) accrued daily, lagged (risk-free, known)
    irx=_yahoo("^IRX").reindex(R.index).ffill()
    cash=(irx.shift(1)/100.0/252.0).rename("ret_cash")
    panel=pd.concat([R,S,cash],axis=1)
    panel=panel[panel.index>=pd.Timestamp("2017-01-01",tz="UTC")]
    panel=panel.dropna(subset=[f"ret_{a}" for a in RISK])
    panel["ret_cash"]=panel["ret_cash"].fillna(0.0)
    panel.to_csv(p); return panel

if __name__=="__main__":
    panel=build(force=True)
    print(f"组合面板(无债券): {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)}天)")
    R=panel[[f'ret_{a}' for a in RISK]]; R.columns=RISK
    ann=R.mean()*252; vol=R.std()*np.sqrt(252)
    print("\n风险资产:")
    for a in RISK: print(f"  {a:4s}: 年化 {ann[a]*100:5.1f}%  波动 {vol[a]*100:4.0f}%  Sharpe {ann[a]/vol[a]:.2f}")
    print(f"  SGOV现金: 年化 {panel['ret_cash'].mean()*252*100:.1f}% (近一年 {panel['ret_cash'].iloc[-252:].mean()*252*100:.1f}%)")
    print("\n相关性:"); print((R.corr()*100).round(0).astype(int).to_string())
    print("\n当前信号:", {a: round(float(panel[f'sig_{a}'].iloc[-1]),2) for a in RISK})
