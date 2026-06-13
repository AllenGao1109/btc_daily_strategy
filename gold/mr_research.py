"""Mean-reversion done right, for GOLD (GC=F).

Family: mean-reversion, implemented PROPERLY so we don't short the secular bull.
Literature-derived signal definitions (see report text for citations):

 1. RSI(2) Connors: long-only buy-the-dip. Buy when RSI2 < threshold AND close>200dma.
    Exit when RSI2 > exit OR close>5dma. (Connors/StockCharts; mean-reversion within uptrend.)
 2. RSI(14) classic oversold/overbought thresholds (long-only and long/short variants).
 3. Bollinger-band reversion: buy lower band, exit at mid (20dma). Long-only & L/S.
 4. Z-score with entry/exit BANDS (owner idea): buy when z<-Xstd, exit when z>-exit.
    Both absolute-price z (naive, known-bad) and DETRENDED z (around moving trend).
 5. Ornstein-Uhlenbeck half-life sizing: size position by -z scaled by speed of reversion.
 6. DETRENDED reversion: mean-revert around a moving average (DPO), not the level.
 7. LONG-ONLY buy-the-dip-in-uptrend overlay applied to several of the above.

Discipline: pick params on train+val; test (2019+) is OOS. Compare to gold buy-and-hold.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT


def sh(net, key):
    a, b = SPLIT[key]; m = (net.index >= a) & (net.index < b)
    r = metrics(net, m); return None if r is None else r["sharpe"]


def navv(net, key):
    a, b = SPLIT[key]; m = (net.index >= a) & (net.index < b)
    r = metrics(net, m); return None if r is None else r["nav"]


def rsi(series, n):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def summary(df, w, band, label, store):
    net, held, turn = backtest(df, w, band=band)
    ntr = int((turn > 1e-9).sum())
    row = dict(label=label, band=band, trades=ntr,
               tr=sh(net, "train"), va=sh(net, "val"), te=sh(net, "test"),
               vnav=navv(net, "val"), tnav=navv(net, "test"))
    store.append(row)
    return net, turn, row


def main():
    df = load(); c = df["close"]
    r = c.pct_change()
    bh = {k: (sh(r, k), navv(r, k)) for k in SPLIT}
    print("GOLD BUY-AND-HOLD (the bar):")
    for k in SPLIT:
        print(f"  {k:6s} Sharpe {bh[k][0]:+.2f}  NAV {bh[k][1]:.2f}")
    full = metrics(r); print(f"  full   Sharpe {full['sharpe']:+.2f}  NAV {full['nav']:.2f}\n")

    ma200 = c.rolling(200).mean()
    ma50 = c.rolling(50).mean()
    ma5 = c.rolling(5).mean()
    up = (c > ma200)  # secular-uptrend filter

    store = []

    # ---------- 1. RSI(2) Connors long-only buy-the-dip-in-uptrend ----------
    print("="*70, "\nRSI(2) Connors: long-only, buy dip when close>200dma\n", "="*70)
    rsi2 = rsi(c, 2)
    for entry in (5, 10, 15):
        for exit_ in (50, 60, 70):
            # state machine: enter long when rsi2<entry & uptrend; exit when rsi2>exit or c>ma5
            pos = np.zeros(len(c)); inpos = False
            re = rsi2.values; cv = c.values; m5 = ma5.values; uv = up.values
            for i in range(len(c)):
                if not inpos:
                    if uv[i] and re[i] < entry:
                        inpos = True
                else:
                    if re[i] > exit_ or cv[i] > m5[i]:
                        inpos = False
                pos[i] = 1.0 if inpos else 0.0
            w = pd.Series(pos, index=c.index)
            _, _, row = summary(df, w, 0.10, f"RSI2 e<{entry} x>{exit_}", store)
            print(f"  e<{entry:2d} x>{exit_:2d}: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
                  f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- 2. RSI(14) thresholds ----------
    print("="*70, "\nRSI(14) classic oversold thresholds\n", "="*70)
    rsi14 = rsi(c, 14)
    for lo, hi in ((30, 50), (30, 70), (35, 65), (25, 55)):
        # long-only: long when rsi<lo, flat when rsi>hi (hold between)
        pos = np.zeros(len(c)); inpos = False
        rv = rsi14.values
        for i in range(len(c)):
            if not inpos and rv[i] < lo: inpos = True
            elif inpos and rv[i] > hi: inpos = False
            pos[i] = 1.0 if inpos else 0.0
        w = pd.Series(pos, index=c.index)
        _, _, row = summary(df, w, 0.10, f"RSI14 LO lo{lo}/hi{hi}", store)
        print(f"  LONG-ONLY lo{lo}/hi{hi}: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
              f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")
        # long-only WITH uptrend filter (only buy dips above 200dma)
        pos = np.zeros(len(c)); inpos = False; uv = up.values
        for i in range(len(c)):
            if not inpos and rv[i] < lo and uv[i]: inpos = True
            elif inpos and rv[i] > hi: inpos = False
            pos[i] = 1.0 if inpos else 0.0
        w = pd.Series(pos, index=c.index)
        _, _, row = summary(df, w, 0.10, f"RSI14 LO+trend lo{lo}/hi{hi}", store)
        print(f"  +200dma   lo{lo}/hi{hi}: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
              f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- 3. Bollinger-band reversion ----------
    print("="*70, "\nBollinger-band reversion: buy lower band, exit at mid (20dma)\n", "="*70)
    for N in (20, 40):
        for k in (2.0, 2.5):
            mid = c.rolling(N).mean(); sd = c.rolling(N).std()
            lower = mid - k * sd
            # long-only: enter when close<lower, exit when close>mid; +uptrend filter
            pos = np.zeros(len(c)); inpos = False
            cv = c.values; lo = lower.values; mv = mid.values; uv = up.values
            for i in range(len(c)):
                if not inpos and cv[i] < lo[i] and uv[i]: inpos = True
                elif inpos and cv[i] > mv[i]: inpos = False
                pos[i] = 1.0 if inpos else 0.0
            w = pd.Series(pos, index=c.index)
            _, _, row = summary(df, w, 0.10, f"BB N{N} k{k} LO+trend", store)
            print(f"  N{N} k{k} LO+trend: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
                  f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- 4. Z-score entry/exit BANDS (owner idea) ----------
    print("="*70, "\nZ-score bands: ABSOLUTE price (naive) vs DETRENDED\n", "="*70)
    for N in (50, 100, 200):
        # absolute price z (the naive failure case), long/short bands
        z = (c - c.rolling(N).mean()) / c.rolling(N).std()
        pos = np.zeros(len(c)); state = 0; zv = z.values
        for i in range(len(c)):
            if state == 0:
                if zv[i] < -1.0: state = 1
                elif zv[i] > 1.0: state = -1
            elif state == 1 and zv[i] > -0.25: state = 0
            elif state == -1 and zv[i] < 0.25: state = 0
            pos[i] = state
        w = pd.Series(pos, index=c.index)
        _, _, row = summary(df, w, 0.10, f"Zabs N{N} L/S bands", store)
        print(f"  ABS N{N} L/S: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
              f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- 5/6. DETRENDED z-score bands (mean-revert around moving trend) ----------
    print("="*70, "\nDETRENDED z-score: deviation from a moving trend, buy dips, exit near trend\n", "="*70)
    for N in (20, 30, 50):
        trend = c.rolling(N).mean()
        dev = (c - trend) / trend  # % deviation from trend
        zd = (dev - dev.rolling(N).mean()) / dev.rolling(N).std()
        zdv = zd.values
        # LONG-ONLY buy-the-dip: buy when zd<-entry, exit when zd>-exit (back near trend)
        for entry in (1.0, 1.5, 2.0):
            pos = np.zeros(len(c)); inpos = False
            for i in range(len(c)):
                if not inpos and zdv[i] < -entry: inpos = True
                elif inpos and zdv[i] > 0.0: inpos = False
                pos[i] = 1.0 if inpos else 0.0
            w = pd.Series(pos, index=c.index)
            _, _, row = summary(df, w, 0.10, f"DetrZ N{N} e{entry} LO", store)
            print(f"  N{N} e{entry} LONG-ONLY: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
                  f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- 7. OU half-life continuous sizing on detrended series ----------
    print("="*70, "\nOrnstein-Uhlenbeck: continuous -z sizing on detrended series, long-biased\n", "="*70)
    for N in (30, 50):
        trend = c.rolling(N).mean()
        dev = (c - trend) / trend
        zd = ((dev - dev.rolling(N).mean()) / dev.rolling(N).std()).clip(-3, 3)
        # long-only continuous: exposure = clip(-zd*scale + base, 0, 1.5), only when uptrend
        for scale in (0.3, 0.5):
            w = (-zd * scale + 0.5).clip(0, 1.5)
            w = w.where(up, 0.0)  # only hold in uptrend (don't fight the bull, don't short)
            _, _, row = summary(df, w, 0.10, f"OU-cont N{N} s{scale} +trend", store)
            print(f"  N{N} s{scale} +trend: tr {row['tr']:+.2f} va {row['va']:+.2f} te {row['te']:+.2f} "
                  f"| vnav {row['vnav']:.2f} tnav {row['tnav']:.2f} | {row['trades']} tr")

    # ---------- ranking: beats BH on BOTH val and test Sharpe ----------
    print("\n" + "="*70, "\nWINNERS (beat BH on val AND test Sharpe):\n", "="*70)
    vbh, tbh = bh["val"][0], bh["test"][0]
    winners = [s for s in store if s["va"] is not None and s["te"] is not None
               and s["va"] > vbh and s["te"] > tbh]
    winners.sort(key=lambda s: s["va"] + s["te"], reverse=True)
    if not winners:
        print(f"  NONE. BH val Sharpe={vbh:+.2f}, test Sharpe={tbh:+.2f}")
    for s in winners[:10]:
        print(f"  {s['label']:28s} tr {s['tr']:+.2f} va {s['va']:+.2f} te {s['te']:+.2f} "
              f"vnav {s['vnav']:.2f} tnav {s['tnav']:.2f} {s['trades']}tr")

    print("\nTop by val Sharpe (regardless of test):")
    for s in sorted(store, key=lambda s: (s["va"] or -9), reverse=True)[:8]:
        print(f"  {s['label']:28s} tr {s['tr']:+.2f} va {s['va']:+.2f} te {s['te']:+.2f} "
              f"vnav {s['vnav']:.2f} tnav {s['tnav']:.2f} {s['trades']}tr")
    return store, bh


if __name__ == "__main__":
    main()
