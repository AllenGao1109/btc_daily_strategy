"""Long-only volatility-managed gold (Moreira-Muir style) + trend-filter variants."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, SPLIT
df=load(); c=df['close']; r=c.pct_change()
def show(net,turn,lbl):
    full=metrics(net); print(f"\n{lbl}  [交易 {int((turn>1e-9).sum())} 次]")
    print(f"  {'区间':8s} {'Sh':>5s} {'NAV':>6s} {'回撤':>6s}")
    fb=metrics(r); print(f"  {'全样本':8s} {full['sharpe']:5.2f} {full['nav']:6.2f} {full['maxdd']*100:5.0f}%   (持有: {fb['sharpe']:.2f}/{fb['nav']:.1f}/{fb['maxdd']*100:.0f}%)")
    for k,(a,b) in SPLIT.items():
        m=(df.index>=a)&(df.index<b); ms=metrics(net,m); mb=metrics(r,m)
        print(f"  {k:8s} {ms['sharpe']:5.2f} {ms['nav']:6.2f} {ms['maxdd']*100:5.0f}%   (持有: {mb['sharpe']:.2f}/{mb['nav']:.1f}/{mb['maxdd']*100:.0f}%)")
rv=r.rolling(45).std()*np.sqrt(252)
# 1) 纯波动率管理多头, 不同目标波动 + 杠杆上限
print("="*60); print("黄金持有买入持有 Sharpe ~0.69 是要超过的标杆")
for tv,cap in [(0.12,1.0),(0.12,1.5),(0.15,2.0),(0.10,1.5)]:
    w=(tv/rv).clip(0,cap)
    net,held,turn=backtest(df,w,band=0.10)
    show(net,turn,f"波动率管理多头 target={tv} cap={cap}x")
# 2) 波动率管理 + 慢趋势过滤(只在 200dma 上方持有, 仍按波动定仓)
sma=c.rolling(200).mean()
for tv,cap in [(0.15,2.0),(0.12,1.5)]:
    w=((tv/rv).clip(0,cap))*(c>sma).astype(float)
    net,held,turn=backtest(df,w,band=0.10)
    show(net,turn,f"波动管理 + 200d趋势过滤 target={tv} cap={cap}x")
