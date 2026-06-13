"""
Equity-index TIMING via VALUATION / Equity-Risk-Premium (Fed-model) family.

Family premise (from literature):
  - Fed model: stocks vs bonds compete; gap = equity_yield - bond_yield. When the gap
    is HIGH, stocks are cheap vs bonds -> higher forward equity returns (positive IC vs fwd ret).
  - CAPE / dividend-yield: high yield (cheap) -> higher long-horizon returns; but CAPE is
    near-useless for SHORT-horizon timing (literature: ~random at 1y).

DATA LIMITATION (honest):
  This panel has NO earnings (no E/P, no forward earnings), NO Shiller CAPE, NO actual
  dividend-yield series, NO FRED. So a *true* Fed-model E/P-minus-yield gap is NOT computable.
  We APPROXIMATE the valuation/ERP idea three ways, all from price+yield data in the panel:
    (A) "Trend-yield gap": use a smoothed long price trend as a slow fair-value anchor.
        Proxy equity carry by the trailing trend slope (annualized) and subtract the bond
        yield -> a Fed-model-style "excess equity trend over the risk-free/bond yield".
    (B) "Inverse-price valuation": z-score of price vs its own long (2y) moving average.
        Far ABOVE trend = expensive (low fwd ret); far BELOW = cheap. Sign expected NEGATIVE.
        This is a cheap stand-in for "price/fundamental" mean-reversion (CAPE-like).
    (C) "Realized earnings-yield proxy": trailing 1y total return is a noisy stand-in for the
        cash an index threw off; gap vs 10y yield. Weak, included for honesty.
  These are PROXIES. We test each one's rank-IC vs SPY AND QQQ fwd returns on TRAIN and VAL,
  and only keep same-sign |IC|>0.03-on-both factors. Then backtest the robust ones long/flat.

Run:  cd /Users/gaozhiyuan/Desktop/btc_daily_strategy && PYTHONPATH=. python assets/equity_valuation_erp.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from assets.equity_harness import load, forward_return, ic, SPLIT
from assets.harness import backtest, metrics

d = load()
# restrict to the documented panel window 2006-2026 for selection (SPLIT starts 2006)
d = d[d.index >= "2006-01-01"]

TR = (d.index >= SPLIT["train"][0]) & (d.index < SPLIT["train"][1])
VA = (d.index >= SPLIT["val"][0])   & (d.index < SPLIT["val"][1])
TE = (d.index >= SPLIT["test"][0])  & (d.index < SPLIT["test"][1])

YDAY = 252


def zscore(s, win):
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - m) / sd


# ----------------------------------------------------------------------------
# Build valuation/ERP-proxy factors. All are point-in-time (no lookahead): every
# component uses only trailing data. Forward returns are the targets.
# ----------------------------------------------------------------------------
def build_factors(px):
    F = {}
    logp = np.log(px)

    # (A) Trend-yield gap (Fed-model style). Trailing 1y total return annualized = crude
    # "equity carry/earnings-yield proxy". Subtract the 10y bond yield (tnx, in %).
    # HIGH gap = equities cheap vs bonds -> expect POSITIVE IC.
    eq_carry_1y = px / px.shift(YDAY) - 1.0            # trailing 1y total return
    F["A_trendyield_gap_10y"] = eq_carry_1y * 100.0 - d["tnx"]
    # variant vs 3m bill (irx) -> closer to a true risk premium over cash
    F["A_trendyield_gap_3m"] = eq_carry_1y * 100.0 - d["irx"]

    # (B) Valuation vs own long trend (CAPE-like mean-reversion). z of price vs 2y MA.
    # Expensive (high z) -> expect NEGATIVE IC. Use 504d (~2y) and 252d windows.
    F["B_pricevsMA_z504"] = -zscore(logp, 504)        # sign-flipped so HIGH=cheap (expect +)
    F["B_pricevsMA_z252"] = -zscore(logp, 252)
    # ratio form: price / 200d MA (classic). HIGH=expensive -> NEGATIVE; flip to expect +
    ma200 = px.rolling(200, min_periods=100).mean()
    F["B_inv_px_over_ma200"] = -(px / ma200 - 1.0)

    # (C) Realized-earnings-yield proxy gap, smoothed (3y avg return annualized) vs 10y.
    # Smoothing mimics CAPE's cyclically-adjusted idea. Expect POSITIVE (cheap->high fwd).
    eq_carry_3y = (px / px.shift(3 * YDAY)) ** (1 / 3) - 1.0
    F["C_smooth_carry_gap_10y"] = eq_carry_3y * 100.0 - d["tnx"]

    # (D) ERP via bond-proxy: equity trailing yield minus TLT (long-bond ETF) trailing yield.
    # TLT total return is a tradable long-bond carry proxy; gap = equity premium over bonds.
    tlt_carry_1y = d["tlt"] / d["tlt"].shift(YDAY) - 1.0
    F["D_erp_vs_tlt_1y"] = eq_carry_1y - tlt_carry_1y

    return F


def run_ic_table(tgt_name):
    px = d[tgt_name]
    fwds = {h: forward_return(px, h) for h in (20, 60)}
    F = build_factors(px)
    rows = []
    for name, f in F.items():
        for h, fwd in fwds.items():
            it = ic(f, fwd, TR); iv = ic(f, fwd, VA); ite = ic(f, fwd, TE)
            robust = (abs(it) > 0.03 and abs(iv) > 0.03 and np.sign(it) == np.sign(iv))
            rows.append((name, h, it, iv, ite, robust))
    return rows


def print_table(tgt_name, rows):
    print(f"\n===== rank-IC vs {tgt_name.upper()} forward returns  (selection on TRAIN+VAL only) =====")
    print(f"{'factor':26s} {'h':>3s} {'IC_tr':>7s} {'IC_va':>7s} {'IC_te':>7s}  robust(tr+va)")
    for name, h, it, iv, ite, rob in rows:
        print(f"{name:26s} {h:>3d} {it:+7.3f} {iv:+7.3f} {ite:+7.3f}  {'ROBUST' if rob else ''}")


def signal_from_factor(px, factor, h=20, lo=0.30, hi=0.70):
    """Long/flat (and lightly levered) timing from a valuation factor's trailing percentile.
    Map factor -> weight in [0,1.5] via its own expanding rank percentile so it is point-in-time.
    Higher factor (cheaper, per our sign convention) -> more long."""
    f = factor.reindex(px.index)
    # expanding percentile rank (point-in-time, no lookahead)
    pct = f.rank(pct=True)  # full-sample rank used only for reporting baseline
    # point-in-time expanding percentile:
    pit = f.expanding(min_periods=252).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    w = pd.Series(0.0, index=px.index)
    w[pit >= hi] = 1.5
    w[(pit < hi) & (pit >= lo)] = 1.0
    w[pit < lo] = 0.0
    return w


def bh_metrics(px, mask):
    r = px.pct_change()
    return metrics(r, mask)


def backtest_robust(tgt_name, name, factor):
    px = d[tgt_name]
    w = signal_from_factor(px, factor)
    net, held, turn = backtest(px, w, band=0.10)
    out = {}
    for seg, m in (("val", VA), ("test", TE), ("full", (TR | VA | TE))):
        mm = metrics(net, m); bh = bh_metrics(px, m)
        out[seg] = (mm, bh, float(turn[m].mean()) if m is not None else np.nan)
    print(f"\n--- BACKTEST {name} on {tgt_name.upper()} (long/flat, 0.1% fee, 10% band) ---")
    for seg in ("val", "test", "full"):
        mm, bh, tn = out[seg]
        if mm is None or bh is None:
            print(f"  {seg:4s}: n/a"); continue
        print(f"  {seg:4s}: strat Sharpe {mm['sharpe']:+.2f} nav {mm['nav']:.2f} dd {mm['maxdd']:+.2f} turn/day {tn:.3f}"
              f"  |  BH Sharpe {bh['sharpe']:+.2f} nav {bh['nav']:.2f}")
    return out


if __name__ == "__main__":
    print("Panel:", d.index.min().date(), "->", d.index.max().date(), f"({len(d)} rows)")
    print("Splits:", {k: (a, b) for k, (a, b) in SPLIT.items()})
    print("\nDATA LIMITATION: no E/P, no Shiller CAPE, no dividend-yield, no FRED in panel.")
    print("All factors below are PRICE/YIELD PROXIES for the valuation/ERP idea (honest).")

    all_rows = {}
    for tgt in ("spy", "qqq"):
        rows = run_ic_table(tgt)
        all_rows[tgt] = rows
        print_table(tgt, rows)

    # A factor is "robust" only if same-sign |IC|>0.03 on BOTH train and val, for BOTH spy & qqq,
    # at the same horizon. Collect such (factor,horizon) pairs.
    print("\n===== ROBUST factors (|IC|>0.03 same-sign on TRAIN & VAL, for BOTH SPY & QQQ) =====")
    robust_pairs = []
    spy_map = {(n, h): (it, iv, ite, rob) for (n, h, it, iv, ite, rob) in all_rows["spy"]}
    qqq_map = {(n, h): (it, iv, ite, rob) for (n, h, it, iv, ite, rob) in all_rows["qqq"]}
    for key in spy_map:
        s = spy_map[key]; q = qqq_map[key]
        if s[3] and q[3] and np.sign(s[0]) == np.sign(q[0]):
            robust_pairs.append(key)
            print(f"  {key[0]:26s} h={key[1]:>3d}  SPY IC tr/va {s[0]:+.3f}/{s[1]:+.3f}  "
                  f"QQQ IC tr/va {q[0]:+.3f}/{q[1]:+.3f}")
    if not robust_pairs:
        print("  NONE. No valuation/ERP proxy is train+val stable on both indices.")

    # Backtest any robust factor vs buy-hold on both indices.
    if robust_pairs:
        print("\n===== BACKTEST robust factors vs BUY-AND-HOLD =====")
        for tgt in ("spy", "qqq"):
            px = d[tgt]; F = build_factors(px)
            for (name, h) in sorted(set(robust_pairs)):
                backtest_robust(tgt, name, F[name])

    # Always also report the simple long-trend benchmark (price>200dMA) for context.
    print("\n===== CONTEXT: simple 200d trend long/flat (NOT a valuation factor) =====")
    for tgt in ("spy", "qqq"):
        px = d[tgt]; ma200 = px.rolling(200, min_periods=100).mean()
        w = (px > ma200).astype(float)
        net, held, turn = backtest(px, w, band=0.10)
        for seg, m in (("val", VA), ("test", TE), ("full", (TR | VA | TE))):
            mm = metrics(net, m); bh = bh_metrics(px, m)
            if mm and bh:
                print(f"  {tgt} {seg:4s}: trend Sharpe {mm['sharpe']:+.2f} nav {mm['nav']:.2f}"
                      f"  | BH Sharpe {bh['sharpe']:+.2f} nav {bh['nav']:.2f}")
