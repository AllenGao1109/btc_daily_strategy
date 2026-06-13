"""Value / cross-asset ratio signals for gold.

Family thesis (Erb & Harvey "Golden Dilemma"): gold has ~zero real long-run
return and tends to mean-revert toward a long-run relative value vs an anchor
(CPI, silver, broad commodities, equities). When gold is EXPENSIVE relative to
the anchor, forward gold returns are below average -> reduce/short. When CHEAP
-> add. We also test relative-strength (momentum of the ratio), the opposite
sign, which is the cross-asset analogue of "commodity value vs momentum".

Anchors used (all Yahoo, full-history where possible):
  - SI=F  silver futures (2000+)   -> gold/silver ratio (GSR)
  - SLV   silver ETF (2006+)       -> robustness check on GSR
  - ^SPGSCI broad commodity index (1984+) -> gold/commodities ratio
  - spx   (in panel)               -> gold/SPX ratio

Discipline: parameters chosen on TRAIN+VAL only. TEST (2019+) is OOS.
Compare to gold buy-and-hold. Net of 0.2%/side fee. Prefer LOW turnover.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from gold.harness import load, backtest, metrics, report, SPLIT, FEE, _yahoo

df = load()
c = df["close"]
idx = df.index


def fetch(sym):
    """Fetch a Yahoo close series aligned/ffilled to the gold index."""
    s = _yahoo(sym)["close"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(idx).ffill()


# ---- anchors -------------------------------------------------------------
silver = fetch("SI=F")          # full history
slv    = fetch("SLV")           # 2006+
gsci   = fetch("^SPGSCI")       # full history
spx    = df["spx"]

GSR      = c / silver           # gold/silver ratio (price ratio, ~scale 50-100)
GSR_slv  = c / slv
GCOM     = c / gsci             # gold / broad commodities
GSPX     = c / spx              # gold / equities

print("anchor coverage (first valid date):")
for nm, s in [("silver SI=F", silver), ("SLV", slv), ("GSCI", gsci), ("SPX", spx)]:
    print(f"  {nm:12s} {s.first_valid_index().date()}  nNaN={int(s.isna().sum())}")


# ---- z-score helper ------------------------------------------------------
def zscore(ratio, win):
    m = ratio.rolling(win).mean()
    sd = ratio.rolling(win).std()
    return ((ratio - m) / sd).clip(-3, 3)


def report_compact(label, net, turn):
    """report() vs buy-hold for train/val/test, return a metrics dict."""
    report(df, net, turn, label)
    r = c.pct_change()
    out = {}
    for k, (a, b) in SPLIT.items():
        m = (idx >= a) & (idx < b)
        ms = metrics(net, m); mb = metrics(r, m)
        out[k] = (ms, mb)
    return out


# Long-run gold buy-and-hold reference
print("\n############ BUY-AND-HOLD reference ############")
bh = report_compact("gold buy-and-hold", c.pct_change(), pd.Series(0.0, index=idx))


# =========================================================================
# SIGNAL 1: Gold/Silver Ratio MEAN-REVERSION (value).
#   Logic: GSR mean-reverts. High GSR => gold expensive vs silver => below-avg
#   forward gold returns => trim/short gold. weight = -z(GSR).
#   This is the classic "GSR > 80 is high" rule, continuous z-score form.
# =========================================================================
print("\n############ SIGNAL 1: GSR mean-reversion (weight = -z) ############")
for win in (252, 504, 756):
    z = zscore(GSR, win)
    w = (-z * 0.5 + 0.5).clip(0, 1.5)   # baseline long-tilt 0.5, MR overlay
    net, held, turn = backtest(df, w, band=0.12)
    report_compact(f"S1 GSR-MR win={win} (long-tilt)", net, turn)


# =========================================================================
# SIGNAL 2: Gold/Silver Ratio RELATIVE-STRENGTH (momentum of ratio).
#   Logic: when GSR is RISING, gold is outperforming silver (risk-off / gold
#   regime) -> stay long gold. weight follows sign/level of ratio momentum.
#   This is the OPPOSITE sign to value; precious-metals literature says GSR
#   trends during stress. weight = +z(GSR) tilt.
# =========================================================================
print("\n############ SIGNAL 2: GSR relative-strength (weight = +z) ############")
for win in (252, 504, 756):
    z = zscore(GSR, win)
    w = (z * 0.5 + 0.5).clip(0, 1.5)
    net, held, turn = backtest(df, w, band=0.12)
    report_compact(f"S2 GSR-RS win={win}", net, turn)


# =========================================================================
# SIGNAL 3: Gold/Commodities value (gold cheap/expensive vs broad GSCI).
#   Logic (Erb-Harvey real-value analogue using a tradable real anchor):
#   gold/commodities mean-reverts. Cheap gold vs commodities => add.
#   weight = -z(gold/GSCI). Test both signs.
# =========================================================================
print("\n############ SIGNAL 3: Gold/Commodities value ############")
for win in (504, 756):
    z = zscore(GCOM, win)
    for sign, tag in [(-1, "MR -z"), (+1, "RS +z")]:
        w = (sign * z * 0.5 + 0.5).clip(0, 1.5)
        net, held, turn = backtest(df, w, band=0.12)
        report_compact(f"S3 G/COM {tag} win={win}", net, turn)


# =========================================================================
# SIGNAL 4: Gold/SPX value mean-reversion.
#   Logic: gold/SPX mean-reverts; low ratio favors gold. weight = -z(gold/spx).
#   Test both signs (RS = gold momentum vs equities, a known risk-off proxy).
# =========================================================================
print("\n############ SIGNAL 4: Gold/SPX ratio ############")
for win in (504, 756):
    z = zscore(GSPX, win)
    for sign, tag in [(-1, "MR -z"), (+1, "RS +z")]:
        w = (sign * z * 0.5 + 0.5).clip(0, 1.5)
        net, held, turn = backtest(df, w, band=0.12)
        report_compact(f"S4 G/SPX {tag} win={win}", net, turn)
