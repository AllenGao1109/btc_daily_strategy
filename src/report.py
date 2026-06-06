"""Markdown report generation.

Produces a timestamped markdown report under ``reports/`` with the sections
required by CLAUDE.md section 21. The report makes explicit, precise claims
(beats/does not beat buy-and-hold, fee drag, leverage breaches, liquidation,
intraday risk caveat) rather than vague statements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import metrics as metrics_mod


def _pct(x: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{x * 100:.2f}%"


def _fmt(x: Any) -> str:
    """Format a metric value for a markdown table cell."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build_benchmark_table(metrics_by_name: Mapping[str, dict[str, Any]]) -> str:
    """Build the markdown performance-vs-benchmarks comparison table.

    Args:
        metrics_by_name: Mapping of strategy name -> metrics dict.

    Returns:
        A markdown table string (section 17 columns).
    """
    cols = [
        ("Strategy", None),
        ("Total Return", "total_return"),
        ("CAGR", "cagr"),
        ("Vol", "annualized_volatility"),
        ("Sharpe", "sharpe_ratio"),
        ("Sortino", "sortino_ratio"),
        ("Max DD", "max_drawdown"),
        ("Calmar", "calmar_ratio"),
        ("Total Fees", "total_fees_paid"),
        ("Funding/Borrow", "total_funding_borrow_cost"),
        ("# Trades", "num_trades"),
        ("Avg Net Exp", "avg_net_exposure"),
        ("Avg Gross Lev", "avg_gross_leverage"),
        ("Max Gross Lev", "max_gross_leverage"),
        ("Lev Breaches", "num_leverage_breaches"),
        ("Liquidated", "liquidated"),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    pct_keys = {"total_return", "cagr", "annualized_volatility", "max_drawdown"}
    for name, m in metrics_by_name.items():
        row = [name]
        for _, key in cols[1:]:
            val = m.get(key)
            if key in pct_keys and isinstance(val, (int, float)):
                row.append(_pct(val))
            else:
                row.append(_fmt(val))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _returns_table(series: pd.Series, label: str) -> str:
    """Render a period-returns Series as a markdown table."""
    if len(series) == 0:
        return "_No data._"
    lines = [f"| Period | {label} |", "| --- | --- |"]
    for idx, val in series.items():
        period = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)
        lines.append(f"| {period} | {_pct(float(val))} |")
    return "\n".join(lines)


def _comparison_claims(
    strat: dict[str, Any],
    benchmarks: Mapping[str, dict[str, Any]],
    funding_enabled: bool,
) -> list[str]:
    """Produce explicit, precise comparison statements (section 21)."""
    claims: list[str] = []
    bh1 = benchmarks.get("buy_and_hold")
    bh2 = benchmarks.get("leveraged_buy_and_hold")

    if bh1:
        beats_cagr = strat["cagr"] > bh1["cagr"]
        claims.append(
            f"- After fees, the strategy's CAGR ({_pct(strat['cagr'])}) is "
            f"{'higher' if beats_cagr else 'lower'} than buy-and-hold 1x "
            f"({_pct(bh1['cagr'])})."
        )
        lower_dd = abs(strat["max_drawdown"]) < abs(bh1["max_drawdown"])
        claims.append(
            f"- The strategy's max drawdown ({_pct(strat['max_drawdown'])}) is "
            f"{'smaller' if lower_dd else 'larger'} than buy-and-hold 1x "
            f"({_pct(bh1['max_drawdown'])})."
        )
    if bh2:
        beats_cagr2 = strat["cagr"] > bh2["cagr"]
        claims.append(
            f"- After fees, the strategy's CAGR is "
            f"{'higher' if beats_cagr2 else 'lower'} than buy-and-hold 2x "
            f"({_pct(bh2['cagr'])})."
        )

    fee_drag = strat["total_fees_pct_initial"]
    claims.append(
        f"- Total fees paid equal {_pct(fee_drag)} of initial capital "
        f"({strat['num_trades']} trades). Fees "
        f"{'materially reduce' if fee_drag > 0.1 else 'modestly reduce'} performance."
    )
    claims.append(
        f"- Funding/borrow costs are {'INCLUDED' if funding_enabled else 'effectively EXCLUDED'} "
        f"(total {strat['total_funding_borrow_cost']:.2f}). "
        + ("Configured rates may be 0.0; check assumptions." if funding_enabled else "")
    )
    claims.append(
        f"- Leverage breaches: {strat['num_leverage_breaches']}. "
        f"Liquidation occurred: {'YES' if strat['liquidated'] else 'no'}."
    )
    claims.append(
        f"- Average gross leverage {strat['avg_gross_leverage']:.2f}x, "
        f"max {strat['max_gross_leverage']:.2f}x; time in market {_pct(strat['exposure_pct'])}."
    )
    claims.append(
        "- Intraday liquidation risk is NOT modeled. This daily close-to-close "
        "backtest may survive drawdowns that real leveraged positions would be "
        "liquidated through intraday."
    )
    return claims


def generate_report(
    strategy_name: str,
    strat_metrics: dict[str, Any],
    benchmark_metrics: Mapping[str, dict[str, Any]],
    result: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    timestamp: str,
    figure_paths: Mapping[str, Path] | None = None,
    walk_forward_summary: str | None = None,
) -> Path:
    """Write a full markdown backtest report and return its path.

    Args:
        strategy_name: Name of the primary strategy.
        strat_metrics: Metrics dict for the primary strategy.
        benchmark_metrics: Mapping of benchmark name -> metrics dict (should
            include the primary strategy for the comparison table).
        result: The primary strategy's daily state table.
        config: Full config dict.
        output_dir: Directory to write the report into.
        timestamp: Timestamp string for the filename and header (passed in for
            determinism/reproducibility; the engine does not call the clock).
        figure_paths: Optional mapping of chart key -> path for embedding.
        walk_forward_summary: Optional pre-rendered walk-forward markdown.

    Returns:
        Path to the written markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{timestamp}_{strategy_name}_report.md"
    path = output_dir / fname

    equity = result["equity_end"].astype(float)
    monthly = metrics_mod.monthly_returns(equity)
    yearly = metrics_mod.yearly_returns(equity)
    funding_cfg = config.get("funding", {})
    funding_enabled = bool(funding_cfg.get("enabled", False))
    backtest_cfg = config.get("backtest", {})

    claims = _comparison_claims(strat_metrics, benchmark_metrics, funding_enabled)

    def fig_md(key: str, caption: str) -> str:
        if not figure_paths or key not in figure_paths:
            return ""
        rel = figure_paths[key]
        try:
            rel = rel.relative_to(output_dir)
        except ValueError:
            pass
        return f"\n![{caption}]({rel})\n"

    sections: list[str] = []
    sections.append(
        f"# BTC Daily Long/Short Leveraged Strategy Backtest Report\n\n"
        f"_Strategy: **{strategy_name}** — generated {timestamp} — "
        f"research/backtest only._"
    )

    sections.append(
        "## 1. Summary\n\n"
        + "\n".join(claims)
    )

    sections.append(
        "## 2. Assumptions\n\n"
        f"- Daily frequency, UTC, BTC only.\n"
        f"- Execution lag: {backtest_cfg.get('execution_lag_days', 1)} day "
        f"(signal at day t-1 drives position on day t).\n"
        f"- Fee rate: {_pct(config['portfolio']['fee_rate'])} of traded notional.\n"
        f"- Slippage rate: {_pct(config['portfolio'].get('slippage_rate', 0.0))}.\n"
        f"- Rebalance policy: `{backtest_cfg.get('rebalance_policy')}`.\n"
        f"- Leverage breach action: `{backtest_cfg.get('leverage_breach_action')}`.\n"
        f"- Max gross leverage: {config['exposure'].get('max_leverage', 2.0)}x.\n"
        f"- Funding/borrow enabled: {funding_enabled} "
        f"(long={funding_cfg.get('long_borrow_rate_daily', 0.0)}, "
        f"short={funding_cfg.get('short_borrow_rate_daily', 0.0)}, "
        f"perp={funding_cfg.get('perp_funding_rate_daily', 0.0)} daily).\n"
        f"- Annualization: 365 days/year (crypto).\n"
        f"- Intraday liquidation risk is NOT modeled."
    )

    sections.append(
        "## 3. Data\n\n"
        f"- Source: {config['data'].get('source')} / {config['data'].get('exchange')}.\n"
        f"- Symbol: {config['data'].get('symbol')}, timeframe {config['data'].get('timeframe')}.\n"
        f"- Rows: {len(result)}; range {result.index.min()} to {result.index.max()}."
    )

    sections.append(
        "## 4. Strategy Logic\n\n"
        f"Strategy `{strategy_name}` emits a raw target exposure weight in "
        f"[-2, 2]. The backtest engine lags it by one day before execution. "
        f"See `src/strategies/{strategy_name}.py`."
    )

    sections.append(
        "## 5. Backtest Timing\n\n"
        "For each day t: carry equity/exposure from t-1; set target weight from "
        "the lagged signal; rebalance only if the target changed or risk control "
        "fires; charge fees/slippage/funding on traded notional; apply day-t "
        "close-to-close return; update equity, exposure, leverage; check breaches "
        "and liquidation."
    )

    sections.append(
        "## 6. Long/Short and Leverage Model\n\n"
        "Equity-based notional accounting. `pnl = exposure_notional * btc_return`; "
        "shorts profit when BTC falls. `gross_leverage = |exposure/equity|`, capped "
        f"at {config['exposure'].get('max_leverage', 2.0)}x. Leverage drifts with "
        "price between rebalances and is tracked daily."
    )

    sections.append(
        "## 7. Transaction Costs\n\n"
        f"Fee = traded_notional * {config['portfolio']['fee_rate']}. Charged only "
        f"on changes in exposure notional, never on the whole portfolio and never "
        f"on hold days. A +2 -> -2 flip costs 4x turnover. Total fees paid: "
        f"{strat_metrics['total_fees_paid']:.2f} "
        f"({_pct(strat_metrics['total_fees_pct_initial'])} of initial capital)."
    )

    sections.append(
        "## 8. Funding and Borrow Costs\n\n"
        + (
            f"Funding/borrow modeling is enabled with daily rates "
            f"long={funding_cfg.get('long_borrow_rate_daily', 0.0)}, "
            f"short={funding_cfg.get('short_borrow_rate_daily', 0.0)}, "
            f"perp={funding_cfg.get('perp_funding_rate_daily', 0.0)}. "
            f"Total funding/borrow cost: {strat_metrics['total_funding_borrow_cost']:.2f}. "
            "If rates are 0.0, financing is effectively excluded — do not treat "
            "this as fully realistic for leveraged/short positions."
            if funding_enabled
            else "Funding/borrow costs are DISABLED. The strategy's short and "
            "leveraged returns therefore omit real financing costs."
        )
    )

    sections.append(
        "## 9. Performance vs Benchmarks\n\n"
        + build_benchmark_table(benchmark_metrics)
        + fig_md("equity", "Equity curve vs benchmarks")
    )

    sections.append(
        "## 10. Risk Metrics\n\n"
        f"- Sharpe: {strat_metrics['sharpe_ratio']:.3f}\n"
        f"- Sortino: {strat_metrics['sortino_ratio']:.3f}\n"
        f"- Annualized volatility: {_pct(strat_metrics['annualized_volatility'])}\n"
        f"- Calmar: {strat_metrics['calmar_ratio']:.3f}\n"
        f"- Win rate: {_pct(strat_metrics['win_rate'])}"
    )

    sections.append(
        "## 11. Drawdown Analysis\n\n"
        f"Max drawdown: {_pct(strat_metrics['max_drawdown'])}."
        + fig_md("drawdown", "Drawdown curve")
    )

    sections.append(
        "## 12. Leverage and Liquidation Analysis\n\n"
        f"- Avg gross leverage: {strat_metrics['avg_gross_leverage']:.2f}x\n"
        f"- Max gross leverage: {strat_metrics['max_gross_leverage']:.2f}x\n"
        f"- Leverage breaches: {strat_metrics['num_leverage_breaches']}\n"
        f"- Liquidated: {'YES' if strat_metrics['liquidated'] else 'no'}\n"
        f"- Breach action: `{backtest_cfg.get('leverage_breach_action')}`"
        + fig_md("gross_leverage", "Gross leverage over time")
    )

    sections.append("## 13. Monthly Returns\n\n" + _returns_table(monthly, "Return"))
    sections.append("## 14. Yearly Returns\n\n" + _returns_table(yearly, "Return"))

    sections.append(
        "## 15. Trade Analysis\n\n"
        f"- Number of trades: {strat_metrics['num_trades']}\n"
        f"- Average holding period: {strat_metrics['avg_holding_period']:.1f} days\n"
        f"- Total turnover: {strat_metrics['total_turnover']:.2f}\n"
        f"- Average daily turnover: {strat_metrics['avg_daily_turnover']:.4f}\n"
        + fig_md("turnover", "Turnover over time")
        + fig_md("cumulative_costs", "Cumulative costs")
    )

    sections.append(
        "## 16. Walk-Forward Validation\n\n"
        + (walk_forward_summary or "_Walk-forward validation not run for this report._")
    )

    sections.append(
        "## 17. Test Set Result\n\n"
        "_The test set must be evaluated only after parameters are selected and "
        "never used for tuning. Run `src/run_backtest.py` with the test window to "
        "populate this section._"
    )

    sections.append(
        "## 18. Failure Modes and Caveats\n\n"
        + "\n".join(claims)
        + "\n\n- Performance may depend on a small number of trades; check section 15.\n"
        "- Synthetic data (if used) is not real market data."
    )

    sections.append(
        "## 19. Conclusion\n\n"
        f"On this dataset, `{strategy_name}` "
        f"{'beats' if strat_metrics['cagr'] > benchmark_metrics.get('buy_and_hold', {}).get('cagr', float('inf')) else 'does not beat'} "
        "buy-and-hold 1x in CAGR after fees. Review risk-adjusted metrics, "
        "drawdown, leverage usage, and fee drag above before drawing conclusions. "
        "This is a research harness result, not a recommendation to trade."
    )

    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path
