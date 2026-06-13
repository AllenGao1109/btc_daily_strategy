"""Gold seasonality / calendar / positioning signals.

Family: month-of-year (autumn/Halloween), turn-of-month, day-of-week.
Each tested for stability across train/val (skeptical). Low turnover by construction.
Discipline: select on train+val only; test 2019+ is OOS.
"""
import numpy as np, pandas as pd
from gold.harness import load, backtest, report, SPLIT, metrics

df = load()
idx = df.index
r = df['close'].pct_change()


def seg_metrics(net, label):
    """Return dict of per-split sharpe/nav for compact comparison."""
    out = {}
    for k, (a, b) in SPLIT.items():
        m = (idx >= a) & (idx < b)
        out[k] = metrics(net, m)
    return out


def bh_metrics():
    out = {}
    for k, (a, b) in SPLIT.items():
        m = (idx >= a) & (idx < b)
        out[k] = metrics(r, m)
    return out

BH = bh_metrics()
print("BUY-AND-HOLD gold benchmark:")
for k in SPLIT:
    print(f"  {k:6s} Sharpe {BH[k]['sharpe']:5.2f}  NAV {BH[k]['nav']:5.2f}  DD {BH[k]['maxdd']*100:4.0f}%")
print()

# Helper calendar features
dow = pd.Series(idx.dayofweek, index=idx)        # 0=Mon..4=Fri
month = pd.Series(idx.month, index=idx)
# day-of-month rank within month for turn-of-month
dom = pd.Series(idx.day, index=idx)
# trading-day position within month (1=first trading day)
g = pd.Series(1, index=idx).groupby([idx.year, idx.month])
tdpos = g.cumsum()                                # 1..N forward
tdcount = g.transform('sum')
tdpos_back = tdcount - tdpos + 1                  # 1=last trading day


def run(weight, label, band=0.10):
    net, held, turn = backtest(df, weight, band=band)
    report(df, net, turn, label)
    sm = seg_metrics(net, label)
    beat = (sm['val'] and sm['test'] and BH['val'] and BH['test']
            and sm['val']['sharpe'] > BH['val']['sharpe']
            and sm['test']['sharpe'] > BH['test']['sharpe'])
    print(f"   >>> beats B&H OOS (val&test Sharpe): {beat}")
    return net, held, turn, sm


# ============================================================
# SIGNAL 1: Autumn effect (Baur 2012). Long in Sep & Nov (classic).
#   Also test "winter/Halloween": long Nov-Apr, flat May-Oct.
# ============================================================
print("\n" + "="*60 + "\nSIGNAL 1 family: month-of-year seasonality\n" + "="*60)

# 1a: Baur autumn — long only Sep & Nov, else flat
w = pd.Series(0.0, index=idx)
w[month.isin([9, 11])] = 1.0
run(w, "1a Baur autumn long Sep+Nov only")

# 1b: Halloween/winter — long Nov-Apr, flat May-Oct
w = pd.Series(0.0, index=idx)
w[month.isin([11, 12, 1, 2, 3, 4])] = 1.0
run(w, "1b Halloween long Nov-Apr, flat May-Oct")

# 1c: Halloween always-invested tilt — long Nov-Apr at 1.5, May-Oct at 0.5
w = pd.Series(0.5, index=idx)
w[month.isin([11, 12, 1, 2, 3, 4])] = 1.5
run(w, "1c Halloween tilt 1.5/0.5")

# 1d: Halloween long/short — long Nov-Apr, short May-Oct
w = pd.Series(-1.0, index=idx)
w[month.isin([11, 12, 1, 2, 3, 4])] = 1.0
run(w, "1d Halloween L/S Nov-Apr long, May-Oct short")

# 1e: full data-driven monthly tilt (selected on TRAIN ONLY then applied)
# compute mean daily return by month on TRAIN window only
tr_mask = (idx >= SPLIT['train'][0]) & (idx < SPLIT['train'][1])
mret = r[tr_mask].groupby(month[tr_mask]).mean()
print("\n  TRAIN-window mean daily return by month (bps):")
print("   " + "  ".join(f"{m}:{mret.get(m,0)*1e4:+.1f}" for m in range(1, 13)))
# long the months that were positive on train, flat otherwise
pos_months = mret[mret > 0].index.tolist()
w = pd.Series(0.0, index=idx)
w[month.isin(pos_months)] = 1.0
run(w, f"1e long train-positive months {pos_months}")


# ============================================================
# SIGNAL 2: Turn-of-month. Long last 2 trading days + first 3 of next month.
#   Logic: month-end fund flows / rebalancing demand. Expected sign +.
# ============================================================
print("\n" + "="*60 + "\nSIGNAL 2 family: turn-of-month\n" + "="*60)

for (nlast, nfirst) in [(1, 3), (2, 3), (1, 2), (3, 4), (2, 4)]:
    w = pd.Series(0.0, index=idx)
    tom = (tdpos_back <= nlast) | (tdpos <= nfirst)
    w[tom] = 1.0
    run(w, f"2 turn-of-month last{nlast}+first{nfirst} (long, else flat)", band=0.05)

# 2-tilt: always invested, overweight TOM window
w = pd.Series(0.5, index=idx)
tom = (tdpos_back <= 2) | (tdpos <= 3)
w[tom] = 1.5
run(w, "2t TOM tilt 1.5 in window / 0.5 else", band=0.05)


# ============================================================
# SIGNAL 3: Day-of-week. Avoid Monday, favor Friday.
#   IGWT/Kohli: Mon negative, Fri strongest.
# ============================================================
print("\n" + "="*60 + "\nSIGNAL 3 family: day-of-week\n" + "="*60)

# 3a: long all days except Monday (flat Mon)
w = pd.Series(1.0, index=idx)
w[dow == 0] = 0.0
run(w, "3a long, flat on Monday", band=0.05)

# 3b: long only Friday, flat else
w = pd.Series(0.0, index=idx)
w[dow == 4] = 1.0
run(w, "3b long Friday only", band=0.05)

# 3c: long Thu+Fri (capture Fri strength + entry), flat else
w = pd.Series(0.0, index=idx)
w[dow.isin([3, 4])] = 1.0
run(w, "3c long Thu+Fri only", band=0.05)


# ============================================================
# SIGNAL 4: Combine best stable pieces — Halloween (always invested) gated.
# ============================================================
print("\n" + "="*60 + "\nSIGNAL 4: combinations\n" + "="*60)

# 4a: Halloween-long + turn-of-month boost
w = pd.Series(0.0, index=idx)
w[month.isin([11, 12, 1, 2, 3, 4])] = 1.0
tom = (tdpos_back <= 2) | (tdpos <= 3)
w[tom & ~month.isin([11, 12, 1, 2, 3, 4])] = 1.0  # also long TOM in summer
run(w, "4a Halloween-long OR turn-of-month", band=0.05)
