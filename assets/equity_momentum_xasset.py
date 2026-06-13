"""Equity-index TIMING factor mining: Momentum / seasonality / cross-asset family.

Signals (from literature):
  TSMOM/trend: price vs 200d-MA (the baseline to beat), 12-1 momentum, 6m return.
  Dual-momentum: SPY absolute momentum (positive 12m return) -> long/flat.
  Seasonality: sell-in-May (Nov-Apr long), turn-of-month.
  Cross-asset: copper/gold (CPER/GLD) growth proxy, dollar (DXY) momentum,
               gold/SPX ratio momentum.

DISCIPLINE: select on train+val rank-IC stability ONLY (same-sign & |IC|>0.03 on BOTH).
test is OOS - reported, never tuned. Compare every timing strategy to BUY-AND-HOLD.
Run: cd ~/Desktop/btc_daily_strategy; PYTHONPATH=. python assets/equity_momentum_xasset.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
idx = d.index
TR = (idx >= SPLIT["train"][0]) & (idx < SPLIT["train"][1])
VA = (idx >= SPLIT["val"][0])   & (idx < SPLIT["val"][1])
TE = (idx >= SPLIT["test"][0])  & (idx < SPLIT["test"][1])
H = 20  # forward horizon (~1 trading month), matches monthly-rebalance momentum lit

def build_factors(tgt: str) -> dict:
    """Return dict name -> factor series. tgt in {'spy','qqq'}. All factors use only
    info available at time t (no look-ahead). Sign convention: higher factor => more bullish."""
    c = d[tgt]
    F = {}
    # ---- TREND / TSMOM ----
    F["trend_200dMA(base)"] = c / c.rolling(200).mean() - 1.0       # >0 above 200d-MA
    F["mom_12m"]            = c.pct_change(252)                       # 12m total return
    F["mom_12_1"]          = c.shift(21).pct_change(231)            # 12m skipping last month
    F["mom_6m"]            = c.pct_change(126)
    F["mom_3m"]            = c.pct_change(63)
    F["ma_50_200"]         = c.rolling(50).mean()/c.rolling(200).mean()-1.0  # golden/death cross
    # ---- DUAL MOMENTUM absolute component (12m return sign) ----
    F["absmom_12m"]        = c.pct_change(252)   # same series; gated to long/flat in bt
    # ---- SEASONALITY ----
    m = idx.month
    F["sell_in_may"]       = pd.Series(np.where((m>=11)|(m<=4),1.0,-1.0), index=idx)  # Nov-Apr bullish
    dom = idx.day
    # turn-of-month: last 3 + first 3 trading days proxy via calendar day-of-month
    F["turn_of_month"]     = pd.Series(np.where((dom>=26)|(dom<=4),1.0,-1.0), index=idx)
    # ---- CROSS-ASSET RISK ----
    if "cper" in d and "gld" in d:
        F["copper_gold_mom"] = (d["cper"]/d["gld"]).pct_change(60)   # growth proxy momentum
        F["copper_gold_lvl"] = (d["cper"]/d["gld"]).rolling(20).mean()/(d["cper"]/d["gld"]).rolling(120).mean()-1.0
    if "dxy" in d:
        F["dollar_mom_inv"]  = -d["dxy"].pct_change(60)              # strong $ = headwind => invert
    if "gld" in d:
        F["gold_spx_inv"]    = -(d["gld"]/c).pct_change(60)          # gold beating eq = risk-off => invert
    # cross-asset risk composite z (avg of standardized risk-on signals)
    return F

ROWS = []
robust = {}  # name -> {'spy':(it,iv), 'qqq':(it,iv)}

for tgt in ("spy","qqq"):
    c = d[tgt]; fwd = forward_return(c, H)
    F = build_factors(tgt)
    for n,f in F.items():
        it = ic(f, fwd, TR); iv = ic(f, fwd, VA); ie = ic(f, fwd, TE)
        ROWS.append((tgt, n, it, iv, ie))
        rob = abs(it)>0.03 and abs(iv)>0.03 and np.sign(it)==np.sign(iv)
        if rob:
            robust.setdefault(n,{})[tgt]=(it,iv,ie)

print("="*78)
print(f"RANK-IC vs forward {H}d return  (train 06-15 / val 15-20 / test 20-26)")
print(f"{'tgt':4s} {'factor':22s} {'IC_tr':>7s} {'IC_va':>7s} {'IC_te':>7s}  flag")
print("-"*78)
last=None
for tgt,n,it,iv,ie in ROWS:
    rob = abs(it)>0.03 and abs(iv)>0.03 and np.sign(it)==np.sign(iv)
    flag = "ROBUST" if rob else ""
    if tgt!=last: print("-"*78); last=tgt
    print(f"{tgt:4s} {n:22s} {it:+7.3f} {iv:+7.3f} {ie:+7.3f}  {flag}")

print("\n"+"="*78)
print("ROBUST factors (same-sign & |IC|>0.03 on BOTH train & val):")
both = [n for n,v in robust.items() if "spy" in v and "qqq" in v]
spy_only = [n for n,v in robust.items() if "spy" in v and "qqq" not in v]
for n,v in robust.items():
    tgts=",".join(v.keys()); print(f"  {n:22s} on [{tgts}]")
print(f"  -> robust on BOTH spy & qqq: {both}")
print(f"  -> robust on spy only:       {spy_only}")

# ---------------- BACKTEST: long/flat timing for robust + baseline ----------------
def to_weight(f, thr=0.0, hi=1.0, lo=0.0):
    """long when factor>thr else flat. shift(1) to trade next bar (no look-ahead)."""
    w = pd.Series(np.where(f>thr, hi, lo), index=f.index).shift(1).fillna(0.0)
    return w

def bh_metrics(close, mask):
    """buy-hold metrics over a mask: feed weight=1 everywhere."""
    w = pd.Series(1.0, index=close.index)
    net,_,_ = backtest(close, w, band=0.1)
    return metrics(net, mask)

def report_bt(tgt, name, weight, band=0.1):
    c = d[tgt]
    net,held,turn = backtest(c, weight, band=band)
    out={}
    for split,mask in (("val",VA),("test",TE),("full",(VA|TE|TR))):
        m = metrics(net, mask); out[split]=m
    return out, net

print("\n"+"="*78)
print("BACKTEST long/flat vs BUY-HOLD (net of 0.1% fee, band=0.1). Sharpe shown.")
print(f"{'tgt':4s} {'strategy':22s} {'val_Shp':>8s} {'test_Shp':>8s} {'full_Shp':>8s} {'turn':>6s}")
print("-"*78)

# baselines: buy-hold, and the plain 200d-MA trend rule
bt_results={}
for tgt in ("spy","qqq"):
    c=d[tgt]; F=build_factors(tgt)
    # buy-hold
    wbh=pd.Series(1.0,index=c.index)
    netbh,_,_=backtest(c,wbh,band=0.1)
    mv=metrics(netbh,VA); mt=metrics(netbh,TE); mf=metrics(netbh,VA|TE|TR)
    bt_results[(tgt,"buy_hold")]=(mv,mt,mf)
    print(f"{tgt:4s} {'buy_hold':22s} {mv['sharpe']:8.2f} {mt['sharpe']:8.2f} {mf['sharpe']:8.2f} {'--':>6s}")
    # plain 200d-MA trend (the view to beat)
    w200=to_weight(F["trend_200dMA(base)"])
    out,_=report_bt(tgt,"trend_200dMA",w200)
    _,h,tn=backtest(c,w200,band=0.1)
    bt_results[(tgt,"trend_200dMA")]=(out["val"],out["test"],out["full"])
    print(f"{tgt:4s} {'trend_200dMA':22s} {out['val']['sharpe']:8.2f} {out['test']['sharpe']:8.2f} {out['full']['sharpe']:8.2f} {float(tn.sum()):6.1f}")
    # 12-1 momentum long/flat
    w121=to_weight(F["mom_12_1"])
    out,_=report_bt(tgt,"mom_12_1",w121); _,_,tn=backtest(c,w121,band=0.1)
    bt_results[(tgt,"mom_12_1")]=(out["val"],out["test"],out["full"])
    print(f"{tgt:4s} {'mom_12_1 L/F':22s} {out['val']['sharpe']:8.2f} {out['test']['sharpe']:8.2f} {out['full']['sharpe']:8.2f} {float(tn.sum()):6.1f}")
    # absolute momentum (dual-momentum gate, 12m>0)
    wabs=to_weight(F["absmom_12m"])
    out,_=report_bt(tgt,"absmom_12m",wabs); _,_,tn=backtest(c,wabs,band=0.1)
    bt_results[(tgt,"absmom_12m")]=(out["val"],out["test"],out["full"])
    print(f"{tgt:4s} {'absmom_12m(dualmom)':22s} {out['val']['sharpe']:8.2f} {out['test']['sharpe']:8.2f} {out['full']['sharpe']:8.2f} {float(tn.sum()):6.1f}")
    # sell-in-may seasonal (long Nov-Apr, flat else)
    wmay=to_weight(F["sell_in_may"],thr=0.0)
    out,_=report_bt(tgt,"sell_in_may",wmay); _,_,tn=backtest(c,wmay,band=0.1)
    bt_results[(tgt,"sell_in_may")]=(out["val"],out["test"],out["full"])
    print(f"{tgt:4s} {'sell_in_may L/F':22s} {out['val']['sharpe']:8.2f} {out['test']['sharpe']:8.2f} {out['full']['sharpe']:8.2f} {float(tn.sum()):6.1f}")
    # COMBO: trend AND seasonal (only long when above 200d-MA AND in Nov-Apr)
    fcombo = ((F["trend_200dMA(base)"]>0) & (F["sell_in_may"]>0)).astype(float)
    wcombo = fcombo.shift(1).fillna(0.0)
    out,_=report_bt(tgt,"trend&season",wcombo); _,_,tn=backtest(c,wcombo,band=0.1)
    bt_results[(tgt,"trend_x_season")]=(out["val"],out["test"],out["full"])
    print(f"{tgt:4s} {'trend & season':22s} {out['val']['sharpe']:8.2f} {out['test']['sharpe']:8.2f} {out['full']['sharpe']:8.2f} {float(tn.sum()):6.1f}")
    print("-"*78)

# Also report full-sample NAV multiple (terminal) for the key strategies vs buy-hold
print("\nTerminal NAV (full sample VA|TE|TR overlap) sanity, & MaxDD:")
print(f"{'tgt':4s} {'strategy':22s} {'NAV':>8s} {'maxDD':>8s}")
for tgt in ("spy","qqq"):
    c=d[tgt]; F=build_factors(tgt)
    full=TR|VA|TE
    for nm,w in (("buy_hold",pd.Series(1.0,index=c.index)),
                 ("trend_200dMA",to_weight(F["trend_200dMA(base)"])),
                 ("trend_x_season",(((F["trend_200dMA(base)"]>0)&(F["sell_in_may"]>0)).astype(float)).shift(1).fillna(0.0))):
        net,_,_=backtest(c,w,band=0.1)
        m=metrics(net,full)
        print(f"{tgt:4s} {nm:22s} {m.get('nav',float('nan')):8.2f} {m.get('maxdd',float('nan')):8.2f}")
