"""Daily multi-asset (Black-Litterman) portfolio report: weights, views, action, chart.

Refreshes data, runs the BL backtest to recover the current HELD weights, computes
today's TARGET weights, and reports the rebalance action plus performance vs the naive
equal-weight benchmark. Bilingual HTML + an equity-curve PNG for the daily email.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

import portfolio.bl as B
from portfolio.panel import build

NAMES_ZH = {"btc":"比特币","spy":"标普500","qqq":"纳指100","tlt":"长期美债","gld":"黄金"}
REBAL_BAND = 0.08


def build_report(force=True) -> dict:
    """Compute the current portfolio target/held weights, views, action and stats."""
    panel = build(force=force)
    ret, W = B.backtest(panel, B.bl_target, band=REBAL_BAND)
    held = W.iloc[-1]
    R = panel[[f"ret_{a}" for a in B.ASSETS]].values
    sig = panel[[f"sig_{a}" for a in B.ASSETS]].values[-1]
    target = pd.Series(B.bl_target(R[-B.COVWIN:], sig), index=B.ASSETS)
    drift = float((target - held).abs().sum())
    action = "REBALANCE" if drift > REBAL_BAND else "HOLD"

    # benchmarks + portfolio stats
    eq_ret, _ = B.backtest(panel, B.equal_w, band=REBAL_BAND)
    def st(s, a=None, b=None):
        p = B.perf(s, a, b); return p if p else (0, 1, 0)
    full = st(ret); eqfull = st(eq_ret)
    recent = ret.iloc[-252:]; rec = st(recent)
    # equity series for chart (last ~2y), portfolio vs equal-weight vs BTC buy-hold
    n = min(504, len(ret))
    eqcurve = {
        "dates": [d.date().isoformat() for d in ret.index[-n:]],
        "port": list((1 + ret.iloc[-n:]).cumprod().values),
        "ew": list((1 + eq_ret.reindex(ret.index).iloc[-n:]).cumprod().values),
    }
    for a in B.ASSETS:   # each base asset's buy-hold over the window
        eqcurve[a] = list((1 + panel[f"ret_{a}"].reindex(ret.index).iloc[-n:]).cumprod().values)
    return {
        "as_of": panel.index[-1].date().isoformat(),
        "assets": B.ASSETS, "views": dict(zip(B.ASSETS, sig.round(2))),
        "target": target.round(3).to_dict(), "held": held.round(3).to_dict(),
        "cash_target": round(float(1 - target.sum()), 3),
        "action": action, "drift": round(drift, 3),
        "stats": {"full_sharpe": round(full[0], 2), "full_nav": round(full[1], 2),
                  "full_dd": round(full[2] * 100), "recent_sharpe": round(rec[0], 2),
                  "ew_sharpe": round(eqfull[0], 2), "ew_dd": round(eqfull[2] * 100)},
        "equity": eqcurve,
    }


def make_chart(rep: dict, path: str) -> str | None:
    eq = rep.get("equity")
    if not eq: return None
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.to_datetime(eq["dates"])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    # base assets (thin), then equal-weight, then the BL portfolio on top (bold)
    acolor = {"btc": "#e0a030", "spy": "#2ca02c", "qqq": "#9467bd", "tlt": "#17becf", "gld": "#bcbd22"}
    for a in ["btc", "spy", "qqq", "tlt", "gld"]:
        if a in eq:
            v = np.asarray(eq[a], float); v = v / v[0]
            ax.plot(d, v, color=acolor.get(a, "#bbb"), lw=1.0, alpha=0.75, label=a.upper())
    for key, color, lbl, lw in [("ew", "#888", "Equal-weight 25%", 1.4), ("port", "#2d6cdf", "BL Portfolio", 2.6)]:
        v = np.asarray(eq[key], float); v = v / v[0]
        ax.plot(d, v, color=color, lw=lw, label=lbl)
    ax.set_title(f"BL portfolio vs base assets & equal-weight - last ~{len(d)//252}y (=1 at start)", fontsize=9.5)
    ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def render_html(rep: dict) -> str:
    s = rep["stats"]
    is_trade = rep["action"] == "REBALANCE"
    th = "padding:6px 10px;border:1px solid #ddd;background:#f4f4f4;font-size:13px"
    td = "padding:6px 10px;border:1px solid #ddd;text-align:center;font-size:13px"
    rows = []
    for a in rep["assets"]:
        tw, hw, v = rep["target"][a], rep["held"][a], rep["views"][a]
        chg = tw - hw
        col = "#1a7f37" if v > 0.15 else ("#c0392b" if v < -0.15 else "#888")
        rows.append(
            f"<tr><td style='{td};text-align:left'>{NAMES_ZH[a]} {a.upper()}</td>"
            f"<td style='{td};color:{col}'>{v:+.2f}</td>"
            f"<td style='{td}'>{hw*100:.1f}%</td><td style='{td};font-weight:bold'>{tw*100:.1f}%</td>"
            f"<td style='{td};color:{'#c0392b' if abs(chg)>0.03 else '#999'}'>{chg*100:+.1f}%</td></tr>")
    rows.append(f"<tr><td style='{td};text-align:left'>SGOV 现金(生息)</td><td style='{td}'>-</td>"
                f"<td style='{td}'>-</td><td style='{td};font-weight:bold'>{rep['cash_target']*100:.1f}%</td><td style='{td}'></td></tr>")
    flag = ("⚠️ 调仓 REBALANCE" if is_trade else "持有 HOLD")
    return (
        f"<div style='font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222;max-width:720px'>"
        f"<h2 style='margin:0 0 4px'>多资产组合日报 · Multi-asset portfolio (Black-Litterman)</h2>"
        f"<p style='margin:2px 0;color:#555'>截至 As of <b>{rep['as_of']}</b> · 今日动作 Action: "
        f"<b style='color:{'#c0392b' if is_trade else '#1a7f37'}'>{flag}</b> (漂移 drift {rep['drift']:.2f})</p>"
        f"<img src='cid:portchart' style='max-width:100%;border:1px solid #eee;border-radius:4px;margin:8px 0'/>"
        f"<table style='border-collapse:collapse;margin-top:6px'><thead><tr>"
        f"<th style='{th}'>资产 Asset</th><th style='{th}'>观点 View</th><th style='{th}'>当前 Held</th>"
        f"<th style='{th}'>目标 Target</th><th style='{th}'>调整 Change</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p style='font-size:13px;color:#555;margin-top:10px'>组合表现 Portfolio: 全样本 Sharpe <b>{s['full_sharpe']}</b> / "
        f"NAV {s['full_nav']} / 回撤 {s['full_dd']}% · 近一年 Sharpe {s['recent_sharpe']}. "
        f"对比 vs 等权25% Equal-weight: Sharpe {s['ew_sharpe']} / 回撤 {s['ew_dd']}%.</p>"
        f"<p style='color:#888;font-size:12px'>长仓、不加杠杆;风险资产看空则退到 SGOV(生息现金,~4-5%)。"
        f"全 Robinhood 可交易。仅研究指引,手动执行。Research guidance - act manually.</p></div>")


if __name__ == "__main__":
    rep = build_report(force=True)
    Path("logs").mkdir(exist_ok=True)
    make_chart(rep, "logs/portfolio_equity.png")
    Path("logs/portfolio_preview.html").write_text(render_html(rep))
    print(f"as_of {rep['as_of']}  action {rep['action']} (drift {rep['drift']})")
    print("target:", {k: f"{v*100:.0f}%" for k, v in rep["target"].items()}, f"cash {rep['cash_target']*100:.0f}%")
    print("views:", rep["views"]); print("stats:", rep["stats"])
    print("saved logs/portfolio_preview.html + logs/portfolio_equity.png")
