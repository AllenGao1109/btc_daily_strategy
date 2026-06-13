"""Assemble the multi-asset panel: daily total returns + causal view signals.

Assets: BTC, SPY, QQQ, TLT, GLD. Each gets a directional view s in [-1,1] from its
own strategy (BTC factor_composite; equities trend-tilt; TLT vol-managed; gold
vol-managed). Aligned from 2017 (BTC start). All signals are causal (computable at t).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

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
    sig=composite_signal_walkforward(df, DEFAULT_FACTORS, horizon=20)  # [-1,1]
    return df["close"].pct_change().rename("btc"), sig.clip(-1,1).rename("btc")

def asset_signals():
    from assets.harness import load
    d=load(); out_r={}; out_s={}
    for a in ["spy","qqq","tlt","ief","gld"]:
        c=d[a].dropna(); r=c.pct_change(); out_r[a]=r
        rv=(r.rolling(60).std()*np.sqrt(252)).replace(0,np.nan)
        if a in ("spy","qqq"):       # equities: mild trend-tilt view (Sharpe-neutral but cuts DD)
            s=np.tanh(3*(c/c.rolling(200).mean()-1))
        elif a in ("tlt","ief"):     # bonds: vol-managed directional (TLT verified edge)
            tgt=0.13 if a=="tlt" else 0.075
            w=(tgt/rv).clip(0,1.5); s=(2*w/1.5-1)
        else:                        # gold: vol-managed conviction, mild (no timing edge)
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
    panel=pd.concat([R,S],axis=1)
    panel=panel[panel.index>=pd.Timestamp("2017-01-01",tz="UTC")].dropna(how="all")
    # require all returns present (portfolio starts when all 5 assets trade)
    rc=[c for c in panel.columns if c.startswith("ret_")]
    panel=panel.dropna(subset=rc)
    panel.to_csv(p); return panel

if __name__=="__main__":
    panel=build(force=True)
    assets=["btc","spy","qqq","tlt","gld"]
    print(f"组合面板: {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)}天)")
    print("\n各资产(全面板)年化收益/波动/相关性:")
    R=panel[[f'ret_{a}' for a in assets]]; R.columns=assets
    ann=R.mean()*252; vol=R.std()*np.sqrt(252)
    for a in assets: print(f"  {a:4s}: 年化 {ann[a]*100:5.1f}%  波动 {vol[a]*100:4.0f}%  Sharpe {ann[a]/vol[a]:.2f}")
    print("\n相关性矩阵:"); print((R.corr()*100).round(0).astype(int).to_string())
    print("\n当前信号(最新):")
    for a in assets: print(f"  {a:4s}: {panel[f'sig_{a}'].iloc[-1]:+.2f}")
