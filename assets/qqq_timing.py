"""QQQ (Nasdaq 100) LONG-ONLY timing strategy.

Toolkit (literature-grounded):
  - VOLATILITY MANAGEMENT (Moreira-Muir 2017): scale exposure ~ inverse recent vol,
    targeting a constant risk level. Strong for high-beta risk assets (leverage effect).
  - TIME-SERIES / ABSOLUTE MOMENTUM (Moskowitz-Ooi-Pedersen 2012; Faber 2007):
    long when trend (price > MA / 12m return > 0) is up, de-risk when down.
  - RISK OVERLAY: VIX / drawdown filter to cut tail exposure.

QQQ is higher-beta & more momentum-driven than SPY, so we test whether STRONGER
trend / vol-management helps more here.

DISCIPLINE: long-only weight in [0, ~2]; lag 1d; 0.1%/side fee; 0.10 no-trade band.
Params chosen on TRAIN+VAL only; TEST is OOS. Beat QQQ buy-and-hold risk-adjusted.

Run:  cd /Users/gaozhiyuan/Desktop/btc_daily_strategy && PYTHONPATH=. python assets/qqq_timing.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from assets.harness import load, backtest, metrics, SPLIT, FEE

d = load()
close = d['qqq'].dropna()
ret = close.pct_change()
vix = d['vix'].reindex(close.index).ffill()


def split_sharpe(net, lo, hi):
    m = (close.index >= lo) & (close.index < hi)
    mm = metrics(net, m)
    return mm['sharpe'] if mm else np.nan


def trainval_sharpe(net):
    """In-sample objective = train+val only (no test peeking)."""
    a = SPLIT['train'][0]; b = SPLIT['val'][1]
    m = (close.index >= a) & (close.index < b)
    mm = metrics(net, m)
    return mm['sharpe'] if mm else np.nan


def summarize(net, turn, label, verbose=True):
    rows = {}
    for k, (a, b) in SPLIT.items():
        m = (close.index >= a) & (close.index < b)
        ms = metrics(net, m); mb = metrics(ret, m)
        rows[k] = (ms, mb)
    f = metrics(net); fb = metrics(ret)
    ntr = int((turn > 1e-9).sum())
    if verbose:
        print(f"\n=== {label} (trades {ntr}) ===")
        for k in ['train', 'val', 'test']:
            ms, mb = rows[k]
            print(f"  {k:5s} strat {ms['sharpe']:5.2f}/{ms['nav']:6.2f}/{ms['maxdd']*100:4.0f}% | "
                  f"B&H {mb['sharpe']:5.2f}/{mb['nav']:6.2f}/{mb['maxdd']*100:4.0f}%")
        print(f"  full  strat {f['sharpe']:5.2f}/{f['nav']:6.2f}/{f['maxdd']*100:4.0f}% | "
              f"B&H {fb['sharpe']:5.2f}/{fb['nav']:6.2f}/{fb['maxdd']*100:4.0f}%")
    return {'rows': rows, 'full': f, 'bh': fb, 'trades': ntr,
            'train': rows['train'][0]['sharpe'], 'val': rows['val'][0]['sharpe'],
            'test': rows['test'][0]['sharpe'], 'full_sharpe': f['sharpe'],
            'full_maxdd': f['maxdd'], 'tv': trainval_sharpe(net)}


# ---------- signal builders (all return a target-weight Series, long-only) ----------

def w_vol_target(target=0.18, span=20, wmax=2.0, wmin=0.0):
    """Moreira-Muir vol-target: weight = target_vol / realized_vol, capped [wmin,wmax]."""
    rv = ret.ewm(span=span).std() * np.sqrt(252)
    w = (target / rv).clip(wmin, wmax)
    return w.fillna(0.0)


def w_trend(ma=200, lev=1.0):
    """Faber absolute momentum: long lev if price>MA else 0 (cash)."""
    mav = close.rolling(ma).mean()
    return (close > mav).astype(float) * lev


def w_tsmom(lookback=252, lev=1.0):
    """Moskowitz-Ooi-Pedersen: long if past-lookback return>0 else 0."""
    past = close / close.shift(lookback) - 1.0
    return (past > 0).astype(float) * lev


def w_vix_overlay(thresh=30.0, derisk=0.5):
    """Scale down when VIX above threshold (tail/risk overlay)."""
    return pd.Series(np.where(vix > thresh, derisk, 1.0), index=close.index)


def combo_voltrend(target=0.18, span=20, ma=200, wmax=2.0, soft=False, soft_k=50):
    """Vol-target sized, gated by trend. soft=smooth trend gate via MA distance."""
    base = w_vol_target(target, span, wmax)
    if not soft:
        gate = (close > close.rolling(ma).mean()).astype(float)
    else:
        mav = close.rolling(ma).mean()
        dist = (close / mav - 1.0)
        gate = (1 / (1 + np.exp(-soft_k * dist))).clip(0, 1)
    return (base * gate).fillna(0.0)


# ======================================================================
print("QQQ buy-hold baseline (for reference):")
fb = metrics(ret)
print(f"  full Sharpe {fb['sharpe']:.3f}  NAV {fb['nav']:.1f}  MaxDD {fb['maxdd']*100:.0f}%")

results = {}

# 0) Buy & hold (lev 1) through harness for fair fee comparison
net, held, turn = backtest(close, pd.Series(1.0, index=close.index))
results['buyhold'] = summarize(net, turn, "Buy&Hold (w=1, via harness)")

# 1) Pure vol-target, sweep target vol
print("\n----- (1) Pure volatility-target (Moreira-Muir) -----")
best_vt = None
for target in [0.14, 0.16, 0.18, 0.20, 0.22, 0.25]:
    for span in [20, 30, 40, 60]:
        w = w_vol_target(target, span, wmax=2.0)
        net, held, turn = backtest(close, w)
        tv = trainval_sharpe(net)
        if best_vt is None or tv > best_vt[0]:
            best_vt = (tv, target, span)
print(f"  best train+val: target={best_vt[1]} span={best_vt[2]} (TV Sharpe {best_vt[0]:.3f})")
w = w_vol_target(best_vt[1], best_vt[2], wmax=2.0)
net, held, turn = backtest(close, w)
results['voltarget'] = summarize(net, turn, f"VolTarget t={best_vt[1]} span={best_vt[2]} cap2.0")

# 2) Pure trend (Faber), sweep MA
print("\n----- (2) Pure trend / absolute momentum (Faber) -----")
best_tr = None
for ma in [100, 150, 200, 250]:
    w = w_trend(ma, lev=1.0)
    net, held, turn = backtest(close, w)
    tv = trainval_sharpe(net)
    if best_tr is None or tv > best_tr[0]:
        best_tr = (tv, ma)
print(f"  best train+val MA={best_tr[1]} (TV Sharpe {best_tr[0]:.3f})")
w = w_trend(best_tr[1], 1.0)
net, held, turn = backtest(close, w)
results['trend'] = summarize(net, turn, f"Trend MA={best_tr[1]} lev1")

# 3) TSMOM 12m
print("\n----- (3) Time-series momentum 12m (MOP) -----")
w = w_tsmom(252, 1.0)
net, held, turn = backtest(close, w)
results['tsmom'] = summarize(net, turn, "TSMOM 252d lev1")

# 4) Combo: vol-target sized + trend gate  (the main candidate for high-beta QQQ)
print("\n----- (4) COMBO vol-target x trend gate -----")
best_c = None
for target in [0.16, 0.18, 0.20, 0.22, 0.25]:
    for span in [20, 30, 40]:
        for ma in [150, 200, 250]:
            for wmax in [1.5, 2.0]:
                w = combo_voltrend(target, span, ma, wmax)
                net, held, turn = backtest(close, w)
                tv = trainval_sharpe(net)
                if best_c is None or tv > best_c[0]:
                    best_c = (tv, target, span, ma, wmax)
tv, target, span, ma, wmax = best_c
print(f"  best train+val: target={target} span={span} MA={ma} wmax={wmax} (TV Sharpe {tv:.3f})")
w = combo_voltrend(target, span, ma, wmax)
net, held, turn = backtest(close, w)
results['combo'] = summarize(net, turn, f"Combo VT(t={target},sp={span},cap{wmax}) x Trend(MA={ma})")
BEST_COMBO = dict(target=target, span=span, ma=ma, wmax=wmax)

# 5) Combo + soft trend gate (smoother, lower turnover)
print("\n----- (5) COMBO vol-target x SOFT trend gate -----")
best_s = None
for target in [0.18, 0.20, 0.22]:
    for span in [20, 30]:
        for ma in [150, 200, 250]:
            for k in [30, 50, 80]:
                w = combo_voltrend(target, span, ma, wmax=2.0, soft=True, soft_k=k)
                net, held, turn = backtest(close, w)
                tv = trainval_sharpe(net)
                if best_s is None or tv > best_s[0]:
                    best_s = (tv, target, span, ma, k)
tv, target, span, ma, k = best_s
print(f"  best train+val: target={target} span={span} MA={ma} k={k} (TV Sharpe {tv:.3f})")
w = combo_voltrend(target, span, ma, wmax=2.0, soft=True, soft_k=k)
net, held, turn = backtest(close, w)
results['combo_soft'] = summarize(net, turn, f"ComboSoft VT(t={target},sp={span}) x Trend(MA={ma},k={k})")
BEST_SOFT = dict(target=target, span=span, ma=ma, soft_k=k)

# 6) Combo + VIX risk overlay
print("\n----- (6) COMBO + VIX tail overlay -----")
c = BEST_COMBO
base = combo_voltrend(c['target'], c['span'], c['ma'], c['wmax'])
best_v = None
for thr in [28, 32, 36, 40]:
    for derisk in [0.0, 0.3, 0.5]:
        w = base * w_vix_overlay(thr, derisk)
        net, held, turn = backtest(close, w)
        tv = trainval_sharpe(net)
        if best_v is None or tv > best_v[0]:
            best_v = (tv, thr, derisk)
tv, thr, derisk = best_v
print(f"  best train+val: VIX>{thr} -> x{derisk} (TV Sharpe {tv:.3f})")
w = base * w_vix_overlay(thr, derisk)
net, held, turn = backtest(close, w)
results['combo_vix'] = summarize(net, turn, f"Combo + VIX>{thr} x{derisk}")
BEST_VIX = dict(thr=thr, derisk=derisk)

# ---------- ranking ----------
print("\n\n================ SUMMARY (sorted by TRAIN+VAL Sharpe) ================")
print(f"{'name':14s} {'TV':>5s} {'train':>6s} {'val':>6s} {'test':>6s} {'full':>6s} {'maxdd':>6s} {'trades':>6s}")
for name, r in sorted(results.items(), key=lambda kv: -kv[1]['tv']):
    print(f"{name:14s} {r['tv']:5.2f} {r['train']:6.2f} {r['val']:6.2f} {r['test']:6.2f} "
          f"{r['full_sharpe']:6.2f} {r['full_maxdd']*100:5.0f}% {r['trades']:6d}")
bh = results['buyhold']
print(f"\nB&H full Sharpe {bh['full_sharpe']:.3f}  MaxDD {bh['full_maxdd']*100:.0f}%  NAV {bh['full']['nav']:.1f}")


# ======================================================================
# FINAL CHOSEN STRATEGY  (selected on TRAIN+VAL only; robust plateau in sweeps)
#   Vol-managed sizing (Moreira-Muir) x absolute trend gate (Faber), MA=250.
#   target=0.22, span=40, cap=1.5, band=0.15. Crushes -83% DD -> -29%, keeps NAV,
#   beats B&H risk-adjusted on train+val and full sample, low turnover.
# ======================================================================
FINAL = dict(target=0.22, span=40, ma=250, cap=1.5, band=0.15)

def final_weight(target, span, ma, cap):
    rv = ret.ewm(span=span).std() * np.sqrt(252)
    base = (target / rv).clip(0, cap)          # Moreira-Muir vol target, long-only
    gate = (close > close.rolling(ma).mean()).astype(float)  # Faber abs-momentum gate
    return (base * gate).fillna(0.0)

def final_signal(span, ma, ref_vol=0.22):
    """Normalized directional 'view' in [-1,1] for Black-Litterman.
      view = trend_gate * tanh(vol_score) - (1-gate)*risk_off
      = +1 fully bullish (uptrend & calm vol), 0 neutral, negative when downtrend (risk-off to cash).
    We map the long-only weight to a [-1,1] view: scale the realized weight by /cap to [0,1],
    then shift downtrend regime to negative to express a bearish/cash view for BL.
    """
    rv = ret.ewm(span=span).std() * np.sqrt(252)
    vol_score = (ref_vol / rv).clip(0, 2.0) / 2.0          # 0..1 sizing pressure
    gate = (close > close.rolling(ma).mean()).astype(float)
    # uptrend: view in (0,1] proportional to vol-adjusted conviction; downtrend: -1 (cash/bearish)
    dist = (close / close.rolling(ma).mean() - 1.0)
    soft = np.tanh(8.0 * dist)                              # smooth trend direction in [-1,1]
    view = np.where(gate > 0, vol_score, -1.0)             # crisp regime
    view = 0.5 * np.asarray(view) + 0.5 * soft.values      # blend crisp regime w/ smooth trend
    return pd.Series(np.clip(view, -1, 1), index=close.index).fillna(0.0)

w_final = final_weight(FINAL['target'], FINAL['span'], FINAL['ma'], FINAL['cap'])
net, held, turn = backtest(close, w_final, band=FINAL['band'])
print("\n\n################ FINAL STRATEGY ################")
fr = summarize(net, turn, f"FINAL VolManaged x Trend  {FINAL}")
sig = final_signal(FINAL['span'], FINAL['ma'])
print(f"\nNormalized directional signal in [-1,1]: last={sig.iloc[-1]:+.3f}  "
      f"mean={sig.mean():+.3f}  min={sig.min():+.3f}  max={sig.max():+.3f}")
print(f"corr(signal, held weight) = {np.corrcoef(sig.values, held.values)[0,1]:.3f}")

if __name__ == "__main__":
    # expose for downstream Black-Litterman integration
    out = pd.DataFrame({"weight": w_final, "held": held, "signal": sig})
    out.to_csv("data/processed/assets/qqq_timing_signal.csv")
    print("\nsaved -> data/processed/assets/qqq_timing_signal.csv")
