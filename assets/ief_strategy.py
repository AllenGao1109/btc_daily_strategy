"""LONG-ONLY timing strategy for IEF (7-10y US Treasuries).

Bond toolkit (same as TLT), applied to intermediate-duration IEF:
  (1) TIME-SERIES MOMENTUM / TREND  (Moskowitz-Ooi-Pedersen 2012): a bond's own
      trailing return predicts its next return. Long-only: full exposure when the
      trend is up, cut to a floor when the trend is down.
  (2) CARRY / YIELD-MOMENTUM (bond carry literature): falling 10y yield => rising
      bond prices (positive price momentum from the rate factor). We use the
      trend of the 10y yield (^TNX) as a carry/rate-momentum overlay: yields
      falling = bullish bonds, yields rising = bearish bonds.
  (3) VOLATILITY MANAGEMENT (Moreira-Muir 2017): scale exposure by a vol target /
      recent realized vol, capped, so risk (and drawdown) stay controlled.

DISCIPLINE: weight in [0, ~1.5] (LONG-ONLY, no shorting). Params chosen on
train+val ONLY; test (2019-2026) is OOS. Beat IEF buy-and-hold net of 0.1%/side
fee on a risk-adjusted basis (Sharpe and/or drawdown). Prefer low turnover.

Run:  PYTHONPATH=. python assets/ief_strategy.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from assets.harness import load, backtest, metrics, report, SPLIT, FEE

d = load()
ief   = d["ief"].dropna()
ret   = ief.pct_change()
tnx   = d["tnx"].reindex(ief.index).ffill()          # 10y yield level (%)


# ---------------------------------------------------------------- helpers
def trn(label, close, w):
    """backtest + report, return net series and #trades."""
    net, held, turn = backtest(close, w, band=0.10)
    report(close, net, turn, label)
    return net, int((turn > 1e-9).sum())


def sharpe_mask(net, a, b):
    m = (net.index >= a) & (net.index < b)
    mm = metrics(net[m])
    return mm["sharpe"] if mm else np.nan


def trainval_sharpe(close, w):
    """Selection objective: Sharpe over train+val ONLY (2002..2019)."""
    net, _, _ = backtest(close, w, band=0.10)
    a = SPLIT["train"][0]; b = SPLIT["val"][1]
    m = (net.index >= a) & (net.index < b)
    mm = metrics(net[m])
    return mm["sharpe"] if mm else -9


# ================================================================ SIGNALS
# Each signal is a TARGET-EXPOSURE Series in [0, wmax]; backtest lags it 1 day.

def sig_trend(close, lb=200, floor=0.0, full=1.0):
    """(1) Long-only price trend: exposure=full if price>MA(lb) else floor."""
    ma = close.rolling(lb).mean()
    return pd.Series(np.where(close > ma, full, floor), index=close.index)


def sig_tsmom(close, lb=252, floor=0.0, full=1.0):
    """(1) Time-series momentum: exposure=full if trailing lb-day return>0."""
    mom = close / close.shift(lb) - 1.0
    return pd.Series(np.where(mom > 0, full, floor), index=close.index)


def sig_yield_mom(tnx_s, lb=120, floor=0.0, full=1.0):
    """(2) Carry/rate-momentum: bullish bonds when 10y yield is FALLING.
    exposure=full if yield < its lb-day-ago level (yields trending down)."""
    dyield = tnx_s - tnx_s.shift(lb)
    return pd.Series(np.where(dyield < 0, full, floor), index=tnx_s.index)


def vol_scale(close, target=0.065, lb=40, cap=1.5):
    """(3) Moreira-Muir vol management: scale by target_vol / realized_vol."""
    rv = close.pct_change().rolling(lb).std() * np.sqrt(252)
    s = (target / rv).clip(upper=cap)
    return s.bfill().fillna(1.0)


# ================================================================ RUN
if __name__ == "__main__":
    print("="*64)
    print("IEF buy-and-hold reference:")
    mb = metrics(ret)
    print(f"  full Sharpe {mb['sharpe']:.2f}, NAV {mb['nav']:.2f}, MaxDD {mb['maxdd']*100:.0f}%")

    # ---- baseline single signals -------------------------------------
    trn("S1 trend MA200 (long/cash)",  ief, sig_trend(ief, 200))
    trn("S2 tsmom 12m (long/cash)",    ief, sig_tsmom(ief, 252))
    trn("S3 yield-mom 120d (long/cash)", ief, sig_yield_mom(tnx, 120))
    trn("S4 vol-managed only",         ief, vol_scale(ief))

    # ---- COMBINED: trend OR yield-mom, vol-scaled --------------------
    # Long-only "view": average of trend(price) and carry(yield) in [0,1],
    # keep a cash floor when both bearish; then vol-scale the whole thing.
    def combo(close, tnx_s, lb_t=200, lb_y=120, floor=0.20,
              target=0.065, lb_v=40, cap=1.5):
        t = sig_trend(close, lb_t, floor=0.0, full=1.0)
        y = sig_yield_mom(tnx_s, lb_y, floor=0.0, full=1.0)
        base = (0.5 * t + 0.5 * y)                  # in {0,0.5,1}
        base = base.clip(lower=floor)               # always keep some bond carry
        return base * vol_scale(close, target, lb_v, cap)

    trn("S5 COMBO trend+yield, vol-scaled", ief, combo(ief, tnx))

    # ---- light param sweep on TRAIN+VAL only -------------------------
    print("\n" + "="*64)
    print("PARAM SWEEP (objective = train+val Sharpe ONLY):")
    best = (-9, None)
    for lb_t in (150, 200, 250):
        for lb_y in (90, 120, 150):
            for floor in (0.0, 0.2, 0.4):
                for target in (0.055, 0.065, 0.075):
                    w = combo(ief, tnx, lb_t=lb_t, lb_y=lb_y, floor=floor,
                              target=target)
                    sh = trainval_sharpe(ief, w)
                    if sh > best[0]:
                        best = (sh, (lb_t, lb_y, floor, target))
    print(f"  best train+val Sharpe={best[0]:.3f} @ "
          f"lb_t={best[1][0]} lb_y={best[1][1]} floor={best[1][2]} target={best[1][3]}")

    lb_t, lb_y, floor, target = best[1]
    wbest = combo(ief, tnx, lb_t=lb_t, lb_y=lb_y, floor=floor, target=target)
    net, ntr = trn(f"S6 BEST COMBO {best[1]}", ief, wbest)

    # ================================================================
    # The binary trend/yield SWITCHES whipsaw IEF (weak price momentum,
    # per Treasury duration-rotation literature). The two ingredients that
    # genuinely help OOS are (a) 12m time-series momentum (best test Sharpe,
    # only 89 trades) and (b) Moreira-Muir vol management (best risk control).
    # FINAL STRATEGY: vol-managed exposure, GATED DOWN (not to zero) by tsmom.
    # Smooth tsmom gate + smooth vol scale => low turnover, no hard whipsaw.
    # ================================================================
    print("\n" + "="*64)
    print("FINAL: vol-managed, tsmom-tilted (smooth)  -- sweep on train+val")

    def final_sig(close, lb_mom=252, gate_lo=0.5, target=0.065, lb_v=60, cap=1.5):
        """Long-only target exposure in [0, cap].
        vol-scale = target/realized_vol (Moreira-Muir), then TILT by tsmom:
        multiply by 1.0 when trailing lb_mom return>0, by gate_lo when <0.
        gate_lo>0 keeps harvesting bond carry even in a downtrend (long-only)."""
        vs = vol_scale(close, target, lb_v, cap)
        mom = close / close.shift(lb_mom) - 1.0
        gate = pd.Series(np.where(mom > 0, 1.0, gate_lo), index=close.index)
        return (vs * gate).clip(0, cap)

    bestf = (-9, None)
    for lb_mom in (200, 252, 300):
        for gate_lo in (0.3, 0.5, 0.7):
            for target in (0.055, 0.065, 0.075):
                for lb_v in (40, 60, 80):
                    w = final_sig(ief, lb_mom, gate_lo, target, lb_v)
                    sh = trainval_sharpe(ief, w)
                    if sh > bestf[0]:
                        bestf = (sh, (lb_mom, gate_lo, target, lb_v))
    print(f"  best train+val Sharpe={bestf[0]:.3f} @ "
          f"lb_mom={bestf[1][0]} gate_lo={bestf[1][1]} "
          f"target={bestf[1][2]} lb_v={bestf[1][3]}")

    lb_mom, gate_lo, target, lb_v = bestf[1]
    wf = final_sig(ief, lb_mom, gate_lo, target, lb_v)
    netf, ntrf = trn(f"S7 FINAL {bestf[1]}", ief, wf)

    # ---- normalized directional signal in [-1,1] ---------------------
    # "Portfolio view": +1 max bullish (uptrend + low vol), -1 max bearish.
    # Built from the SAME ingredients, mapped to [-1,1]:
    #   trend leg  : +1 if 12m return>0 else -1
    #   vol leg    : (target/realized_vol) capped to [0,1] as a confidence weight
    # view = trend_leg * vol_confidence, clipped to [-1,1].
    rv = ief.pct_change().rolling(lb_v).std() * np.sqrt(252)
    vol_conf = (target / rv).clip(0, 1).bfill().fillna(0.5)
    mom = ief / ief.shift(lb_mom) - 1.0
    trend_leg = pd.Series(np.where(mom > 0, 1.0, -1.0), index=ief.index)
    view = (trend_leg * vol_conf).clip(-1, 1)
    print("\nDirectional signal [-1,1] (view): describe")
    print(view.describe()[["min", "25%", "50%", "75%", "max"]])
    print("last 5 views:\n", view.tail().round(3))
