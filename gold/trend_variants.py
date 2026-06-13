"""Momentum / trend variants for GOLD, applied from the TSMOM/CTA literature.

Literature anchors:
- Moskowitz-Ooi-Pedersen (2012) "Time Series Momentum": sign of trailing ~12m
  return, position vol-scaled to a constant ex-ante vol target. Trend persists
  1-12m then partially reverses.
- Donchian channel breakout (CTA classic): long on N-day high break, flat/short
  on N-day low break; whipsaws in ranges -> add a slow trend filter.
- Dual moving-average with HYSTERESIS: separate (wide) enter vs exit thresholds
  so we don't flip on every micro-cross -> fewer whipsaws, lower turnover.
- Momentum-of-momentum / acceleration: change of trend.
- Momentum + mean-reversion COMBO: 12m trend picks the SIDE, short-horizon
  mean-reversion (buy dips / fade pops) times the ENTRY within that side.

Discipline: parameters chosen on train+val (2000-2019). test (2019+) is OOS.
Run:  PYTHONPATH=. python3 gold/trend_variants.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT, FEE

df = load()
c = df['close']
ret = c.pct_change()

# annualized realized vol estimate (for vol-scaling), ~2-month window
def realized_vol(window=42):
    return ret.rolling(window).std() * np.sqrt(252)

def vol_target(target=0.12, window=42, cap=1.5):
    rv = realized_vol(window).replace(0, np.nan)
    return (target / rv).clip(upper=cap)

# convenience: trading days per "month"
M = 21

def run(weight, label, band=0.10):
    net, held, turn = backtest(df, weight, band=band)
    report(df, net, turn, label)
    return net, held, turn

# baseline buy-and-hold reference printed by report() already.

print("\n########## 1) TSMOM 12-month, vol-scaled (MOP 2012) ##########")
# sign of trailing 12m (252d) return, scaled to constant vol target.
for lb in (200, 252):
    for tgt in (0.10, 0.12, 0.15):
        sig = np.sign(c / c.shift(lb) - 1.0)
        w = (sig * vol_target(tgt)).clip(-2, 2)
        run(w, f"TSMOM lb={lb} volTgt={tgt}", band=0.10)

print("\n########## 2) Dual-MA with WIDE HYSTERESIS (fewer whipsaws) ##########")
# fast/slow MA ratio; go long only when ratio > +h_enter, exit to flat when < h_exit.
# Wide gap between enter and exit = hysteresis. Long-only-or-flat (gold secular bull).
def dual_ma_hyst(fast, slow, enter, exit_, allow_short=False, scale_vol=True, tgt=0.12):
    mf = c.rolling(fast).mean(); ms = c.rolling(slow).mean()
    spread = (mf / ms - 1.0)
    state = np.zeros(len(c))
    s = 0
    sp = spread.values
    for i in range(len(sp)):
        if np.isnan(sp[i]):
            state[i] = 0; continue
        if s <= 0 and sp[i] > enter:
            s = 1
        elif s >= 0 and allow_short and sp[i] < -enter:
            s = -1
        elif s == 1 and sp[i] < exit_:
            s = 0
        elif s == -1 and allow_short and sp[i] > -exit_:
            s = 0
        state[i] = s
    base = pd.Series(state, index=c.index)
    if scale_vol:
        return (base * vol_target(tgt)).clip(-2, 2)
    return base.clip(-2, 2)

for fast, slow in [(20, 100), (50, 200), (20, 200)]:
    # enter when fast 2% above slow; exit only when it falls back to -1% (hysteresis)
    w = dual_ma_hyst(fast, slow, enter=0.02, exit_=-0.01, allow_short=False)
    run(w, f"DualMA-hyst {fast}/{slow} enter+2% exit-1% (long/flat)", band=0.10)

print("\n########## 3) Donchian breakout + slow filter ##########")
def donchian(entry_n, exit_n, filt_n=200, allow_short=False, scale_vol=True, tgt=0.12):
    hi = c.rolling(entry_n).max(); lo = c.rolling(entry_n).min()
    # exit channels (tighter) for trailing stop
    hi_x = c.rolling(exit_n).max(); lo_x = c.rolling(exit_n).min()
    trend = c > c.rolling(filt_n).mean()  # only go long above slow MA
    state = np.zeros(len(c)); s = 0
    cv = c.values; hiv = hi.shift(1).values; lov = lo.shift(1).values
    hixv = hi_x.shift(1).values; loxv = lo_x.shift(1).values; tv = trend.values
    for i in range(len(cv)):
        if np.isnan(hiv[i]):
            state[i] = 0; continue
        if s == 0:
            if cv[i] >= hiv[i] and tv[i]:
                s = 1
            elif allow_short and cv[i] <= lov[i] and not tv[i]:
                s = -1
        elif s == 1:
            if cv[i] <= loxv[i]:
                s = 0
        elif s == -1:
            if cv[i] >= hixv[i]:
                s = 0
        state[i] = s
    base = pd.Series(state, index=c.index)
    if scale_vol:
        return (base * vol_target(tgt)).clip(-2, 2)
    return base.clip(-2, 2)

for en, ex in [(50, 25), (100, 50), (20, 10)]:
    w = donchian(en, ex, filt_n=200, allow_short=False)
    run(w, f"Donchian entry={en} exit={ex} filt200 (long/flat)", band=0.10)

print("\n########## 4) Momentum-of-momentum (acceleration) ##########")
# trend = 12m momentum; gate exposure by whether momentum is RISING (accelerating).
# Position long when 12m mom>0 AND its short-term change>0.
def mom_of_mom(lb=252, accel=63, tgt=0.12):
    mom = c / c.shift(lb) - 1.0
    accel_sig = mom - mom.shift(accel)
    long_ok = (mom > 0)
    boost = (accel_sig > 0).astype(float)  # 1 if accelerating else 0
    base = long_ok.astype(float) * (0.5 + 0.5 * boost)  # 0.5 in decel, 1.0 in accel
    return (base * vol_target(tgt)).clip(-2, 2)

for lb in (200, 252):
    for accel in (42, 63):
        w = mom_of_mom(lb, accel)
        run(w, f"Mom-of-mom lb={lb} accel={accel}", band=0.10)

print("\n########## 5) Momentum (direction) + Mean-Reversion (timing) COMBO ##########")
# 12m trend sets allowed SIDE (long/flat for gold bull). Within an uptrend,
# scale UP exposure when short-term z-score is LOW (buy the dip), trim when high.
def mom_mr_combo(lb=252, z_n=20, base_w=1.0, mr_amp=0.7, tgt=0.12, long_only=True):
    trend_up = (c / c.shift(lb) - 1.0) > 0
    z = ((c - c.rolling(z_n).mean()) / c.rolling(z_n).std()).clip(-3, 3)
    # within uptrend: weight = base - mr_amp*z  (low z -> bigger long; high z -> trim)
    raw = base_w - mr_amp * z
    raw = raw.clip(lower=0.0 if long_only else -2.0, upper=2.0)
    base = raw.where(trend_up, 0.0)
    return (base * vol_target(tgt)).clip(-2, 2)

for z_n in (10, 20, 30):
    for mr_amp in (0.4, 0.7):
        w = mom_mr_combo(lb=252, z_n=z_n, mr_amp=mr_amp)
        run(w, f"Mom+MR combo z_n={z_n} mr_amp={mr_amp}", band=0.08)
