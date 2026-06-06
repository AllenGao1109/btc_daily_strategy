"""Multi-band decision guidance for the factor_composite strategy.

Produces, for several no-trade bands (decision-frequency profiles) at once:
  - the historical performance profile (trades/year, out-of-sample Sharpe/NAV/
    drawdown, full-sample Sharpe/drawdown), and
  - the LIVE recommendation as of the latest bar: what the signal targets, what
    each band is currently holding, and whether a rebalance triggers now.

This lets the owner pick a frequency profile (e.g. ~9 vs ~24 decisions/year) and
read the current actionable call for that profile. Research output only - not a
live order router.

Run:  python -m src.guidance --config config.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig, load_config
from . import data as data_mod
from . import features as features_mod
from . import onchain as onchain_mod
from .factors import build_factors, forward_return, information_coefficient
from .research import run_full, window_metrics
from .strategies import get_strategy
from .strategies.factor_composite import DEFAULT_FACTORS
from .validation import make_fixed_split

DEFAULT_BANDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]


def _load_enriched(config: dict) -> pd.DataFrame:
    """Load BTC + features + on-chain + ETH + macro/sentiment (same as the runner)."""
    df = features_mod.build_features(data_mod.load_btc_data(config))
    try:
        df = onchain_mod.merge_onchain(
            df, onchain_mod.load_coinmetrics(onchain_mod.ONCHAIN_METRICS)
        )
        df["eth_close"] = data_mod.load_eth_close(df.index)
        df = data_mod.enrich_external(df)
    except Exception as exc:  # pragma: no cover - network
        print(f"[warn] enrichment skipped: {exc}")
    return df


# Human-friendly labels for the factors (the "market psychology" read).
FACTOR_LABELS = {
    "halving_cos": "Halving cycle", "kurt_30": "Tail risk (kurtosis)",
    "vol_regime": "Volatility regime", "mvrv_z_365": "Valuation (MVRV)",
    "mom_120": "Long momentum", "mvrv_mom_30": "Valuation momentum",
    "ex_netflow_to_mcap": "Exchange flows", "btc_eth_rs_30": "BTC vs ETH strength",
    "rvol_z90": "Realized-vol stress",
}
FACTOR_LABELS_ZH = {
    "halving_cos": "减半周期", "kurt_30": "尾部风险(峰度)",
    "vol_regime": "波动率状态", "mvrv_z_365": "估值(MVRV)",
    "mom_120": "长期动量", "mvrv_mom_30": "估值动量",
    "ex_netflow_to_mcap": "交易所资金流", "btc_eth_rs_30": "BTC对ETH强弱",
    "rvol_z90": "已实现波动压力",
}
LEAN_ZH = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}


def factor_breakdown(df: pd.DataFrame, factor_names: list[str], horizon: int = 20) -> list[dict]:
    """Per-factor current stance: standardized value x walk-forward IC sign.

    Returns factors sorted by absolute contribution, each tagged bullish/bearish -
    a plain-language read of what is driving the model's current positioning.
    """
    F = build_factors(df)
    fr = forward_return(df, horizon)
    mask = df.index <= df.index[-1] - pd.Timedelta(days=horizon + 1)  # causal IC
    out = []
    for n in factor_names:
        if n not in F.columns:
            continue
        f = F[n].replace([np.inf, -np.inf], np.nan)
        z = ((f - f.expanding(365).mean()) / f.expanding(365).std()).clip(-3, 3).iloc[-1]
        ic = information_coefficient(f[mask], fr[mask])
        if np.isnan(z) or ic == 0:
            continue
        contrib = float(np.sign(ic) * z)
        lean = "bullish" if contrib > 0.15 else ("bearish" if contrib < -0.15 else "neutral")
        out.append({
            "factor": FACTOR_LABELS.get(n, n),
            "factor_zh": FACTOR_LABELS_ZH.get(n, n),
            "z": round(float(z), 2),
            "contribution": round(contrib, 2),
            "lean": lean,
            "lean_zh": LEAN_ZH[lean],
        })
    return sorted(out, key=lambda d: abs(d["contribution"]), reverse=True)


def market_context(df: pd.DataFrame) -> dict:
    """Recent price action + support/resistance distances + vol regime."""
    c = df["close"].astype(float)
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    ret = c.pct_change()
    def pct(n):
        return round((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100, 1) if len(c) > n else float("nan")
    rvol = ret.rolling(30).std().iloc[-1] * np.sqrt(365)
    return {
        "ret_7d": pct(7), "ret_30d": pct(30), "ret_90d": pct(90),
        "from_high_90d": round((c.iloc[-1] / hi.rolling(90).max().iloc[-1] - 1) * 100, 1),
        "from_high_365d": round((c.iloc[-1] / hi.rolling(365).max().iloc[-1] - 1) * 100, 1),
        "from_low_90d": round((c.iloc[-1] / lo.rolling(90).min().iloc[-1] - 1) * 100, 1),
        "ann_vol_30d": round(float(rvol) * 100, 0),
    }


def strategy_perf(res: pd.DataFrame) -> dict:
    """Recent realized strategy performance from the primary-band equity curve."""
    eq = res["equity_end"]
    def ret(n):
        return round((eq.iloc[-1] / eq.iloc[-1 - n] - 1) * 100, 1) if len(eq) > n else float("nan")
    ytd_mask = res.index >= pd.Timestamp(f"{res.index[-1].year}-01-01", tz="UTC")
    ytd = (eq.iloc[-1] / eq[ytd_mask].iloc[0] - 1) * 100 if ytd_mask.any() else float("nan")
    dd = (eq / eq.cummax() - 1).iloc[-1] * 100
    return {"ret_30d": ret(30), "ret_90d": ret(90), "ytd": round(float(ytd), 1),
            "current_drawdown": round(float(dd), 1)}


def recent_decisions(res: pd.DataFrame, n: int = 15) -> list[dict]:
    """Extract the last ``n`` actual trades (rebalances) from a backtest result.

    Each decision = a day where turnover > 0: the position moved from
    ``actual_weight_before_rebalance`` to ``target_weight``.
    """
    traded = res[res["turnover"] > 1e-9].tail(n)
    out = []
    for dt, r in traded.iterrows():
        frm, to = float(r["actual_weight_before_rebalance"]), float(r["target_weight"])
        direction = ("做空 Short" if to < -0.05 else "做多 Long" if to > 0.05 else "空仓 Flat")
        out.append({
            "date": dt.date().isoformat(),
            "btc_close": round(float(r["close"]), 0),
            "from_weight": round(frm, 2),
            "to_weight": round(to, 2),
            "change": round(to - frm, 2),
            "direction": direction,
            "turnover": round(float(r["turnover"]), 2),
            "fee_paid": round(float(r["fee"]), 2),
            "equity": round(float(r["equity_end"]), 0),
        })
    return out


def make_summary(meta: dict, row: dict, breakdown: list[dict], ctx: dict, perf: dict) -> str:
    """One-paragraph plain-language headline."""
    t = meta["signal_target"]
    stance = ("net SHORT" if t < -0.05 else "LONG" if t > 0.05 else "FLAT / neutral")
    is_trade = str(row["action_now"]).startswith("REBALANCE")
    drivers = ", ".join(f"{b['factor']} ({b['lean']})" for b in breakdown[:3])
    return (
        f"The model is {stance} (signal target {t:.2f}x). Your band ({row['band']:.2f}) "
        f"says {'REBALANCE now' if is_trade else 'HOLD'} — currently {row['now_holding']:.2f}x, "
        f"last traded {row['days_since_trade']} days ago. BTC ${meta['btc_close']:,.0f}, "
        f"{ctx['ret_30d']:+.0f}% over 30d, {ctx['from_high_365d']:+.0f}% from its 1-yr high. "
        f"Strategy YTD {perf['ytd']:+.0f}%, 90d {perf['ret_90d']:+.0f}%. "
        f"Main drivers: {drivers}."
    )


def make_summary_zh(meta: dict, row: dict, breakdown: list[dict], ctx: dict, perf: dict) -> str:
    """中文版一句话摘要。"""
    t = meta["signal_target"]
    stance = ("净空头" if t < -0.05 else "做多" if t > 0.05 else "空仓/中性")
    is_trade = str(row["action_now"]).startswith("REBALANCE")
    drivers = "、".join(f"{b['factor_zh']}({b['lean_zh']})" for b in breakdown[:3])
    return (
        f"模型当前{stance}(目标仓位 {t:.2f}x)。你的档位({row['band']:.2f})建议"
        f"{'立即换仓' if is_trade else '持有'} —— 当前 {row['now_holding']:.2f}x,"
        f"{row['days_since_trade']} 天前交易过。BTC ${meta['btc_close']:,.0f},"
        f"近 30 天 {ctx['ret_30d']:+.0f}%,距 1 年高点 {ctx['from_high_365d']:+.0f}%。"
        f"策略今年以来 {perf['ytd']:+.0f}%,近 90 天 {perf['ret_90d']:+.0f}%。"
        f"主要驱动:{drivers}。"
    )


def build_guidance(config: dict, bands: list[float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Compute the multi-band guidance table and a live-state summary.

    Args:
        config: Loaded config dict.
        bands: No-trade bands to profile (default DEFAULT_BANDS).

    Returns:
        ``(table, meta)`` - ``table`` has one row per band; ``meta`` holds the
        as-of date, BTC close, and the band-independent signal target.
    """
    bands = bands or DEFAULT_BANDS
    bt = BacktestConfig.from_config(config)
    df = _load_enriched(config)
    raw = get_strategy(config["strategy"]["name"])(df, config["strategy"].get("params", {}))
    splits = make_fixed_split(config["validation"])
    years = (df.index[-1] - df.index[0]).days / 365.25

    as_of = df.index[-1]
    signal_target = float(raw.loc[as_of]) if not np.isnan(raw.loc[as_of]) else float("nan")

    primary_band = float(config["backtest"].get("weight_band", 0.40))
    primary_res = None
    rows = []
    for band in bands:
        res = run_full(df, raw, replace(bt, weight_band=band))
        if abs(band - primary_band) < 1e-9:
            primary_res = res
        full = window_metrics(res, None)
        te = window_metrics(res, splits["test"])
        hold = float(res["actual_weight_end"].iloc[-1])
        traded = res["turnover"] > 0
        last_trade = res.index[traded][-1] if traded.any() else None
        days_since = (as_of - last_trade).days if last_trade is not None else None
        # Pending action: does the latest signal target differ from the held
        # position by more than the band? (band is on the vol-targeted weight.)
        delta = abs(signal_target - hold) if not np.isnan(signal_target) else 0.0
        action = (f"REBALANCE -> {signal_target:.2f}x" if delta > band
                  else f"HOLD at {hold:.2f}x")
        rows.append({
            "band": band,
            "trades_per_yr": round(full["num_trades"] / years, 1),
            "test_sharpe": round(te["sharpe_ratio"], 2),
            "test_nav": round(te["final_nav"], 2),
            "test_maxdd": round(te["max_drawdown"] * 100, 0),
            "full_sharpe": round(full["sharpe_ratio"], 2),
            "full_maxdd": round(full["max_drawdown"] * 100, 0),
            "now_holding": round(hold, 2),
            "days_since_trade": days_since,
            "action_now": action,
        })

    meta = {
        "as_of": as_of.date().isoformat(),
        "btc_close": round(float(df["close"].iloc[-1]), 2),
        "signal_target": round(signal_target, 3),
    }
    # Rich insight: factor breakdown (psychology read), market context, strategy perf.
    try:
        factors = config["strategy"].get("params", {}).get("factors", DEFAULT_FACTORS)
        breakdown = factor_breakdown(df, factors, int(
            config["strategy"].get("params", {}).get("horizon", 20)))
        ctx = market_context(df)
        perf = strategy_perf(primary_res) if primary_res is not None else {}
        primary_row = min(rows, key=lambda r: abs(r["band"] - primary_band))
        # Equity vs buy-and-hold for the chart (recent window).
        equity = {}
        if primary_res is not None:
            bh_res = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
            n = min(730, len(primary_res))  # ~2 years
            sub_s = primary_res["equity_end"].iloc[-n:]
            sub_b = bh_res["equity_end"].reindex(sub_s.index)
            equity = {
                "dates": [d.date().isoformat() for d in sub_s.index],
                "strategy": [round(float(x), 2) for x in sub_s.values],
                "bh": [round(float(x), 2) for x in sub_b.values],
            }
        meta["insight"] = {
            "summary": make_summary(meta, primary_row, breakdown, ctx, perf),
            "summary_zh": make_summary_zh(meta, primary_row, breakdown, ctx, perf),
            "breakdown": breakdown, "context": ctx, "perf": perf,
            "primary_band": primary_band,
            "decisions": recent_decisions(primary_res, 15) if primary_res is not None else [],
            "equity": equity,
        }
    except Exception as exc:  # pragma: no cover - never block the core table
        meta["insight"] = {"summary": f"(insight unavailable: {exc})"}
    return pd.DataFrame(rows), meta


def render_markdown(table: pd.DataFrame, meta: dict) -> str:
    """Render the guidance as a Markdown report."""
    ins = meta.get("insight", {})
    lines = [
        "# BTC factor_composite — multi-band decision guidance",
        "",
    ]
    if ins.get("summary"):
        lines += ["## 摘要 Summary"]
        if ins.get("summary_zh"):
            lines += [ins["summary_zh"], ""]
        lines += [ins["summary"], ""]
    lines += [
        f"- As of: **{meta['as_of']}**  |  BTC close: **${meta['btc_close']:,.0f}**",
        f"- Signal target exposure (band-independent): **{meta['signal_target']:.2f}x**",
        "  - This is the model's desired exposure for the next bar; each band only",
        "    acts on it if the change exceeds the band (fewer bands = fewer trades).",
        "",
        "| Band | Trades/yr | Test Sh | Test NAV | Test DD | Full Sh | Full DD | Holding now | Days since trade | Action now |",
        "|------|-----------|---------|----------|---------|---------|---------|-------------|------------------|------------|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['band']:.2f} | {r['trades_per_yr']} | {r['test_sharpe']} | "
            f"{r['test_nav']} | {r['test_maxdd']:.0f}% | {r['full_sharpe']} | "
            f"{r['full_maxdd']:.0f}% | {r['now_holding']:.2f}x | "
            f"{r['days_since_trade']} | {r['action_now']} |"
        )
    lines += [
        "",
        "Higher band = fewer, more deliberate decisions (lower turnover/fees) at the",
        "cost of slower risk response. Buy-and-hold reference: Test Sharpe 0.58 / NAV",
        "1.50 / DD -50%.",
    ]
    if ins.get("breakdown"):
        lines += ["", "## Why — current factor read (market psychology)"]
        for b in ins["breakdown"]:
            lines.append(f"- {b['factor']}: {b['lean']} (std {b['z']:+.2f}, contrib {b['contribution']:+.2f})")
    if ins.get("context"):
        c = ins["context"]
        lines += ["", "## Market context",
                  f"- Returns: 7d {c['ret_7d']:+.1f}% / 30d {c['ret_30d']:+.1f}% / 90d {c['ret_90d']:+.1f}%",
                  f"- From highs: {c['from_high_90d']:+.1f}% (90d), {c['from_high_365d']:+.1f}% (1yr); "
                  f"above 90d low: {c['from_low_90d']:+.1f}%",
                  f"- Annualized vol (30d): {c['ann_vol_30d']:.0f}%"]
    if ins.get("perf"):
        p = ins["perf"]
        lines += ["", "## Strategy performance (your band)",
                  f"- 30d {p.get('ret_30d', float('nan')):+.1f}% / 90d {p.get('ret_90d', float('nan')):+.1f}% / "
                  f"YTD {p.get('ytd', float('nan')):+.1f}% / current drawdown {p.get('current_drawdown', float('nan')):+.1f}%"]
    lines += ["", "Research output only — act manually."]
    return "\n".join(lines)


def make_equity_chart(meta: dict, path: str) -> str | None:
    """Render a recent equity-curve PNG: strategy (your band) vs buy-and-hold.

    Both normalized to 1.0 at the start of the window. Returns the path, or None
    if no equity data is available.
    """
    eq = meta.get("insight", {}).get("equity") or {}
    if not eq.get("dates"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    dates = pd.to_datetime(eq["dates"])
    s = np.asarray(eq["strategy"], dtype=float)
    b = np.asarray(eq["bh"], dtype=float)
    s = s / s[0]
    b = b / b[0]
    band = meta.get("insight", {}).get("primary_band", 0.20)
    yrs = len(dates) / 365.0
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(dates, s, color="#2d6cdf", lw=2.0, label=f"Strategy (band {band:.2f})")
    ax.plot(dates, b, color="#999", lw=1.5, label="Buy & Hold")
    ax.axhline(1.0, color="#ccc", lw=0.8, ls="--")
    ax.set_title(f"Equity vs Buy & Hold — last ~{yrs:.1f}y (=1 at start)", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_excel(path: str, table: pd.DataFrame, meta: dict) -> str:
    """Write an .xlsx with three sheets: Summary, Band guidance, Recent decisions.

    Args:
        path: Output .xlsx path.
        table: The multi-band guidance table.
        meta: Guidance meta (with insight).

    Returns:
        The written path.
    """
    ins = meta.get("insight", {})
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # Sheet 1: summary / header
        summ = pd.DataFrame({
            "项 Item": ["截至 As of", "BTC 收盘 Close", "信号目标 Signal target",
                        "摘要 Summary (中文)", "Summary (EN)"],
            "值 Value": [meta["as_of"], meta["btc_close"], meta["signal_target"],
                         ins.get("summary_zh", ""), ins.get("summary", "")],
        })
        summ.to_excel(xl, sheet_name="摘要 Summary", index=False)

        # Sheet 2: multi-band guidance
        gt = table.rename(columns={
            "band": "带宽 Band", "trades_per_yr": "交易/年 Trades/yr",
            "test_sharpe": "测试 Sharpe", "test_nav": "测试 NAV", "test_maxdd": "测试回撤% DD",
            "full_sharpe": "全样本 Sharpe", "full_maxdd": "全样本回撤% DD",
            "now_holding": "当前持仓 Holding", "days_since_trade": "距上次交易 Days",
            "action_now": "现有动作 Action",
        })
        gt.to_excel(xl, sheet_name="分带宽指引 Bands", index=False)

        # Sheet 3: recent decisions
        dec = pd.DataFrame(ins.get("decisions", []))
        if not dec.empty:
            dec = dec.rename(columns={
                "date": "日期 Date", "btc_close": "BTC收盘 Close",
                "from_weight": "原仓位 From", "to_weight": "新仓位 To",
                "change": "变化 Change", "direction": "方向 Direction",
                "turnover": "换手 Turnover", "fee_paid": "手续费 Fee", "equity": "权益 Equity",
            })
        dec.to_excel(xl, sheet_name="最近决策 Decisions", index=False)

        # Auto-fit column widths.
        for ws in xl.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    return path


def render_html(table: pd.DataFrame, meta: dict, primary_band: float | None = None) -> str:
    """Render the guidance as an HTML email: one row per band, requested columns.

    Columns: Band | Trades/yr | Test Sh/NAV | Full Sh/DD | Holding now |
    Days since trade | Action. The configured band row is highlighted; a
    REBALANCE action is flagged red, HOLD grey.
    """
    th = ("padding:6px 10px;border:1px solid #ddd;background:#f4f4f4;"
          "text-align:center;font-size:13px")
    td = "padding:6px 10px;border:1px solid #ddd;text-align:center;font-size:13px"
    head = "".join(
        f"<th style='{th}'>{h}</th>" for h in
        ["带宽<br>Band", "交易/年<br>Trades/yr", "测试 Sh/NAV<br>Test", "全样本 Sh/回撤<br>Full",
         "当前持仓<br>Holding", "距上次交易<br>Days since", "现有动作<br>Action"]
    )
    body_rows = []
    for _, r in table.iterrows():
        is_primary = primary_band is not None and abs(r["band"] - primary_band) < 1e-9
        is_trade = str(r["action_now"]).startswith("REBALANCE")
        rowbg = "background:#fff7e6;font-weight:bold" if is_primary else ""
        act_color = "#c0392b" if is_trade else "#888"
        act = str(r["action_now"]).replace("REBALANCE", "换仓 REBALANCE").replace("HOLD", "持有 HOLD")
        body_rows.append(
            f"<tr style='{rowbg}'>"
            f"<td style='{td}'>{r['band']:.2f}</td>"
            f"<td style='{td}'>{r['trades_per_yr']}</td>"
            f"<td style='{td}'>{r['test_sharpe']:.2f} / {r['test_nav']:.2f}</td>"
            f"<td style='{td}'>{r['full_sharpe']:.2f} / {r['full_maxdd']:.0f}%</td>"
            f"<td style='{td}'>{r['now_holding']:.2f}x</td>"
            f"<td style='{td}'>{r['days_since_trade']}</td>"
            f"<td style='{td};color:{act_color}'>{act}</td>"
            f"</tr>"
        )
    ins = meta.get("insight", {})
    summary_box = ""
    if ins.get("summary"):
        summary_box = (
            f"<div style='background:#eef5ff;border-left:4px solid #2d6cdf;padding:10px 14px;"
            f"margin:8px 0;border-radius:4px;font-size:14px;line-height:1.6'>"
            f"<b>📊 摘要 Summary</b><br>{ins.get('summary_zh', '')}"
            f"<br><span style='color:#555'>{ins['summary']}</span></div>"
        )

    extra = ""
    if ins.get("decisions"):
        drows = []
        for d in reversed(ins["decisions"]):  # most recent first
            dcolor = ("#c0392b" if d["to_weight"] < -0.05 else
                      "#1a7f37" if d["to_weight"] > 0.05 else "#888")
            drows.append(
                f"<tr><td style='{td}'>{d['date']}</td>"
                f"<td style='{td}'>${d['btc_close']:,.0f}</td>"
                f"<td style='{td}'>{d['from_weight']:+.2f}x → "
                f"<b style='color:{dcolor}'>{d['to_weight']:+.2f}x</b></td>"
                f"<td style='{td};color:{dcolor}'>{d['direction']}</td>"
                f"<td style='{td}'>{d['turnover']:.2f}</td></tr>"
            )
        extra += (
            f"<h3 style='margin:16px 0 4px'>最近几次决策 / Recent decisions "
            f"(band {ins.get('primary_band', 0.4):.2f})</h3>"
            f"<table style='border-collapse:collapse'><thead><tr>"
            f"<th style='{th}'>日期 Date</th><th style='{th}'>BTC</th>"
            f"<th style='{th}'>仓位变化 Change</th><th style='{th}'>方向 Dir</th>"
            f"<th style='{th}'>换手 Turnover</th></tr></thead>"
            f"<tbody>{''.join(drows)}</tbody></table>"
        )
    if ins.get("breakdown"):
        chips = []
        for b in ins["breakdown"]:
            color = {"bullish": "#1a7f37", "bearish": "#c0392b", "neutral": "#888"}[b["lean"]]
            chips.append(
                f"<tr><td style='{td};text-align:left'>{b['factor_zh']} <span style='color:#999'>"
                f"{b['factor']}</span></td>"
                f"<td style='{td}'>{b['z']:+.2f}</td>"
                f"<td style='{td};color:{color}'>{b['lean_zh']} {b['lean']} ({b['contribution']:+.2f})</td></tr>"
            )
        extra += (
            f"<h3 style='margin:16px 0 4px'>为什么 · 因子市场心理读数 / Factor read</h3>"
            f"<table style='border-collapse:collapse'><thead><tr>"
            f"<th style='{th}'>因子 Factor</th><th style='{th}'>标准值 Std</th>"
            f"<th style='{th}'>倾向(贡献) Lean</th></tr></thead>"
            f"<tbody>{''.join(chips)}</tbody></table>"
        )
    if ins.get("context"):
        c = ins["context"]
        extra += (
            f"<h3 style='margin:16px 0 4px'>行情背景 / Market context</h3>"
            f"<p style='font-size:13px;line-height:1.6;margin:0'>"
            f"涨跌 Returns: 7天 {c['ret_7d']:+.1f}% · 30天 {c['ret_30d']:+.1f}% · 90天 {c['ret_90d']:+.1f}%<br>"
            f"距高点 From highs: {c['from_high_90d']:+.1f}% (90天) · {c['from_high_365d']:+.1f}% (1年) "
            f"— 上方压力 resistance<br>"
            f"距 90 天低点 Above 90d low: {c['from_low_90d']:+.1f}% — 下方支撑 support<br>"
            f"年化波动 Annualized vol (30天): {c['ann_vol_30d']:.0f}%</p>"
        )
    if ins.get("perf"):
        p = ins["perf"]
        extra += (
            f"<h3 style='margin:16px 0 4px'>策略表现(你的档位) / Strategy performance</h3>"
            f"<p style='font-size:13px;line-height:1.6;margin:0'>"
            f"近30天 30d {p.get('ret_30d', float('nan')):+.1f}% · 近90天 90d {p.get('ret_90d', float('nan')):+.1f}% · "
            f"今年以来 YTD {p.get('ytd', float('nan')):+.1f}% · 当前回撤 drawdown {p.get('current_drawdown', float('nan')):+.1f}%</p>"
        )

    return (
        f"<div style='font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222;max-width:760px'>"
        f"<h2 style='margin:0 0 4px'>BTC 每日仓位指引 · factor_composite daily guidance</h2>"
        f"<p style='margin:2px 0;color:#555'>截至 As of <b>{meta['as_of']}</b> · "
        f"BTC 收盘 <b>${meta['btc_close']:,.0f}</b> · "
        f"信号目标仓位 signal target <b>{meta['signal_target']:.2f}x</b></p>"
        f"{summary_box}"
        + (f"<h3 style='margin:16px 0 4px'>近期净值 vs 买入持有 / Recent equity vs B&H</h3>"
           f"<img src='cid:equitychart' style='max-width:100%;border:1px solid #eee;border-radius:4px'/>"
           if ins.get("equity") else "")
        + f"<h3 style='margin:16px 0 4px'>分带宽决策指引 / Decision guidance by band</h3>"
        f"<table style='border-collapse:collapse'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        f"<p style='color:#888;font-size:12px;margin:4px 0 0'>高亮行=你当前的档位 / your band。"
        f"买入持有参考 BH: 测试 Test 0.58/1.50, 全样本回撤 Full DD -83%。</p>"
        f"{extra}"
        f"<p style='color:#888;font-size:12px;margin-top:14px'>仅研究指引,需你手动执行,不会自动下单。"
        f"Research guidance — act manually; does not place orders.</p></div>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bands", default=None, help="comma-separated, e.g. 0.2,0.3,0.4")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    config = load_config(args.config)
    bands = [float(b) for b in args.bands.split(",")] if args.bands else None

    table, meta = build_guidance(config, bands)
    md = render_markdown(table, meta)
    print(md)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"guidance_{meta['as_of']}.md"
    path.write_text(md)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
