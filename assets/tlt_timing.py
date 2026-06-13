"""
TLT (20y+ UST total return) LONG-ONLY timing strategy.

GOAL: beat TLT buy-and-hold (Sharpe ~0.33, MaxDD -48% from the 2022 bond crash)
on a risk-adjusted basis, long-only, low turnover, robust across train/val/test.

LITERATURE TESTED
  (A) TREND / time-series momentum on TLT price (Moskowitz-Ooi-Pedersen 2012).
  (B) CARRY / YIELD-TREND: 10y yield (tnx) level & momentum. Rising yields ->
      bonds fall, so cut exposure when the 10y yield trends up.
  (C) VOL-MANAGEMENT (Moreira-Muir 2017): scale exposure inverse to recent
      realized vol of TLT.
  (D) RISK-REGIME: VIX tilt (bonds rally in equity risk-off).

FINDINGS (params selected on TRAIN+VAL only; TEST 2019-2026 is OOS)
  - Pure price TREND / TSMOM HURTS on TLT: whipsaws + the long 2002-2020 bull
    that buy-hold captured in full -> full Sharpe BELOW buy-hold. (Trend works
    for diversified bond FUTURES baskets in the literature, not single-name TLT
    on/off timing.)
  - HARD yield/trend GATES cut the 2022 drawdown nicely (DD -27..-36%) but
    sacrifice too much Sharpe (cut exposure during the bull too) -> ~tied with
    buy-hold on full Sharpe.
  - VOL-MANAGEMENT is the robust winner: higher Sharpe in EVERY window, lower
    drawdown, very low turnover. This is the single best long-only improvement.

WINNER: vol-managed TLT, window=100d, annual target vol=13%, cap=1.5x.
  weight_t = clip( 0.13 / realized_vol_100d(TLT) , 0 , 1.5 )    (lagged 1d by harness)
                full Sharpe 0.42 vs 0.33 BH (+27%) | NAV 3.05 vs 2.36
                MaxDD -42% vs -48% | only 113 trades over 24y
                train 0.72/0.64  val 0.37/0.29  test 0.06/-0.03  (strat/BH)

NORMALIZED DIRECTIONAL SIGNAL in [-1,1] (the portfolio "view" for Black-Litterman):
  signal_t = clip( 2 * weight_t / 1.5 - 1 , -1 , 1 )
  i.e. exposure 0 -> -1 (defensive/cash), 0.75 -> 0 (neutral), 1.5 -> +1 (max long).
  This is a LONG-ONLY conviction signal: vol low -> lever up (bullish view),
  vol high -> de-risk toward cash. Never short.

Run: cd /Users/gaozhiyuan/Desktop/btc_daily_strategy ; PYTHONPATH=. python assets/tlt_timing.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from assets.harness import load, backtest, metrics, report, SPLIT

# ---------------------------------------------------------------- data & masks
d   = load()
tlt = d['tlt'].dropna()
idx = tlt.index
r   = tlt.pct_change()
tnx = d['tnx'].reindex(idx).ffill()          # 10y yield (%), for diagnostics
m_tr=(idx>=SPLIT['train'][0])&(idx<SPLIT['train'][1])
m_va=(idx>=SPLIT['val'][0])&(idx<SPLIT['val'][1])
m_te=(idx>=SPLIT['test'][0])&(idx<SPLIT['test'][1])

def shp(net,m): mm=metrics(net,m); return mm['sharpe'] if mm else float('nan')

# ------------------------------------------------ WINNER: vol-managed exposure
VT_WIN, TARGET, CAP = 100, 0.13, 1.5

def vol_managed_weight(price=tlt, win=VT_WIN, target=TARGET, cap=CAP):
    """Moreira-Muir vol-managed LONG-ONLY exposure for TLT.
    weight = target_vol / trailing realized vol, clipped to [0, cap]."""
    rv = price.pct_change().rolling(win).std() * np.sqrt(252)
    return (target / rv).clip(0.0, cap)

def directional_signal(weight, cap=CAP):
    """Normalized portfolio view in [-1,1] for Black-Litterman.
    Linearly maps exposure [0,cap] -> [-1,1]; long-only (never shorts)."""
    return (2.0*weight/cap - 1.0).clip(-1.0, 1.0)

# ----------------------------------------------------------------- run & report
w   = vol_managed_weight()
net, held, turn = backtest(tlt, w, band=0.10)
report(tlt, net, turn, "TLT vol-managed (win=100, tgt=13%, cap=1.5)")

f, fb = metrics(net), metrics(r)
print("\nSummary vs buy-and-hold:")
print(f"  strat : tr={shp(net,m_tr):5.2f} va={shp(net,m_va):5.2f} te={shp(net,m_te):5.2f} "
      f"| full={f['sharpe']:.2f} maxdd={f['maxdd']*100:.0f}% nav={f['nav']:.2f} trades={int((turn>1e-9).sum())}")
print(f"  bh    : tr={shp(r,m_tr):5.2f} va={shp(r,m_va):5.2f} te={shp(r,m_te):5.2f} "
      f"| full={fb['sharpe']:.2f} maxdd={fb['maxdd']*100:.0f}% nav={fb['nav']:.2f}")

sig = directional_signal(w)
print(f"\n[-1,1] directional signal: min={sig.min():.2f} max={sig.max():.2f} "
      f"mean={sig.mean():.2f} last={sig.iloc[-1]:.2f}  (date {idx[-1].date()})")
