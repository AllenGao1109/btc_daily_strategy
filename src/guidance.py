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
from .research import run_full, window_metrics
from .strategies import get_strategy
from .validation import make_fixed_split

DEFAULT_BANDS = [0.20, 0.30, 0.40, 0.50, 0.60]


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

    rows = []
    for band in bands:
        res = run_full(df, raw, replace(bt, weight_band=band))
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
    return pd.DataFrame(rows), meta


def render_markdown(table: pd.DataFrame, meta: dict) -> str:
    """Render the guidance as a Markdown report."""
    lines = [
        "# BTC factor_composite — multi-band decision guidance",
        "",
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
        "1.50 / DD -50%. Research output only.",
    ]
    return "\n".join(lines)


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
        ["Band", "Trades/yr", "Test Sh / NAV", "Full Sh / DD",
         "Holding now", "Days since trade", "Action now"]
    )
    body_rows = []
    for _, r in table.iterrows():
        is_primary = primary_band is not None and abs(r["band"] - primary_band) < 1e-9
        is_trade = str(r["action_now"]).startswith("REBALANCE")
        rowbg = "background:#fff7e6;font-weight:bold" if is_primary else ""
        act_color = "#c0392b" if is_trade else "#888"
        body_rows.append(
            f"<tr style='{rowbg}'>"
            f"<td style='{td}'>{r['band']:.2f}</td>"
            f"<td style='{td}'>{r['trades_per_yr']}</td>"
            f"<td style='{td}'>{r['test_sharpe']:.2f} / {r['test_nav']:.2f}</td>"
            f"<td style='{td}'>{r['full_sharpe']:.2f} / {r['full_maxdd']:.0f}%</td>"
            f"<td style='{td}'>{r['now_holding']:.2f}x</td>"
            f"<td style='{td}'>{r['days_since_trade']}</td>"
            f"<td style='{td};color:{act_color}'>{r['action_now']}</td>"
            f"</tr>"
        )
    return (
        f"<div style='font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222'>"
        f"<h2 style='margin:0 0 4px'>BTC factor_composite — daily guidance</h2>"
        f"<p style='margin:2px 0;color:#555'>As of <b>{meta['as_of']}</b> · "
        f"BTC close <b>${meta['btc_close']:,.0f}</b> · "
        f"signal target <b>{meta['signal_target']:.2f}x</b></p>"
        f"<table style='border-collapse:collapse;margin-top:8px'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        f"<p style='color:#888;font-size:12px;margin-top:10px'>Highlighted row = your "
        f"configured band. Buy-and-hold reference: Test 0.58 / 1.50, Full DD -83%. "
        f"Research guidance — act manually.</p></div>"
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
