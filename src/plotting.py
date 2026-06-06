"""Matplotlib charts for backtest reporting.

All charts are saved under ``reports/figures/``. Charts must never be used to
hide poor performance; they are diagnostic, not promotional. Uses the
non-interactive Agg backend so plotting works headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as metrics_mod  # noqa: E402


def _save(fig: "plt.Figure", path: Path) -> Path:
    """Save a figure to ``path`` (creating parents) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curves(
    curves: Mapping[str, pd.Series], out_path: Path, title: str = "Equity curve vs benchmarks"
) -> Path:
    """Plot multiple equity curves on one axis (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, equity in curves.items():
        ax.plot(equity.index, equity.values, label=name, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Equity (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_drawdown(equity: pd.Series, out_path: Path, label: str = "Strategy") -> Path:
    """Plot the drawdown curve for a single equity series."""
    dd = metrics_mod.drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=0.5)
    ax.set_title(f"Drawdown — {label}")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_price_with_position(
    result: pd.DataFrame, out_path: Path, label: str = "Strategy"
) -> Path:
    """Plot BTC close price with the strategy's target exposure overlaid."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(result.index, result["close"], color="black", linewidth=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC close (log)")
    ax1.set_title(f"BTC price with position overlay — {label}")
    ax1.grid(True, alpha=0.3)

    weight = result["actual_weight_end"]
    ax2.fill_between(
        result.index, weight, 0, where=weight >= 0, color="green", alpha=0.5, label="long"
    )
    ax2.fill_between(
        result.index, weight, 0, where=weight < 0, color="red", alpha=0.5, label="short"
    )
    ax2.set_ylabel("Net exposure")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    return _save(fig, out_path)


def plot_target_exposure(result: pd.DataFrame, out_path: Path) -> Path:
    """Plot target exposure weight over time."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(result.index, result["target_weight"], where="post", linewidth=1.0)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title("Target exposure over time")
    ax.set_ylabel("Target weight")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_gross_leverage(
    result: pd.DataFrame, out_path: Path, max_leverage: float = 2.0
) -> Path:
    """Plot actual gross leverage with the cap and any breach markers."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(result.index, result["gross_leverage_end"], linewidth=1.0)
    ax.axhline(max_leverage, color="red", linestyle="--", linewidth=0.8, label="cap")
    breaches = result[result["leverage_breach"]]
    if len(breaches):
        ax.scatter(
            breaches.index,
            breaches["gross_leverage_end"],
            color="red",
            s=12,
            zorder=5,
            label="breach",
        )
    ax.set_title("Actual gross leverage over time")
    ax.set_ylabel("Gross leverage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_turnover(result: pd.DataFrame, out_path: Path) -> Path:
    """Plot daily turnover over time."""
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.bar(result.index, result["turnover"], width=1.0, color="steelblue")
    ax.set_title("Turnover over time")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_cumulative_costs(result: pd.DataFrame, out_path: Path) -> Path:
    """Plot cumulative fees and cumulative funding/borrow costs."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(result.index, result["fee"].cumsum(), label="cumulative fees", linewidth=1.2)
    funding_borrow = (result["funding_cost"] + result["borrow_cost"]).cumsum()
    ax.plot(result.index, funding_borrow, label="cumulative funding/borrow", linewidth=1.2)
    ax.set_title("Cumulative costs")
    ax.set_ylabel("Cost (quote currency)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_rolling(result: pd.DataFrame, out_path: Path, window: int = 90) -> Path:
    """Plot rolling N-day return and rolling N-day volatility."""
    returns = result["strategy_daily_return"]
    roll_ret = (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
    roll_vol = returns.rolling(window).std() * (365**0.5)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(result.index, roll_ret * 100, linewidth=1.0)
    ax1.set_title(f"Rolling {window}-day return")
    ax1.set_ylabel("Return (%)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(result.index, roll_vol * 100, linewidth=1.0, color="darkorange")
    ax2.set_title(f"Rolling {window}-day volatility (annualized)")
    ax2.set_ylabel("Vol (%)")
    ax2.grid(True, alpha=0.3)
    return _save(fig, out_path)


def generate_all_figures(
    result: pd.DataFrame,
    benchmark_curves: Mapping[str, pd.Series],
    figures_dir: Path,
    label: str = "strategy",
    max_leverage: float = 2.0,
) -> dict[str, Path]:
    """Generate the full chart set for one strategy and return their paths.

    Args:
        result: The strategy's daily state table.
        benchmark_curves: Mapping of name -> equity curve (incl. the strategy).
        figures_dir: Output directory for figures.
        label: Strategy label used in filenames/titles.
        max_leverage: Leverage cap for the leverage chart.

    Returns:
        Mapping of chart key -> saved file path.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "equity": plot_equity_curves(benchmark_curves, figures_dir / f"{label}_equity.png"),
        "drawdown": plot_drawdown(
            result["equity_end"], figures_dir / f"{label}_drawdown.png", label
        ),
        "price_position": plot_price_with_position(
            result, figures_dir / f"{label}_price_position.png", label
        ),
        "target_exposure": plot_target_exposure(
            result, figures_dir / f"{label}_target_exposure.png"
        ),
        "gross_leverage": plot_gross_leverage(
            result, figures_dir / f"{label}_gross_leverage.png", max_leverage
        ),
        "turnover": plot_turnover(result, figures_dir / f"{label}_turnover.png"),
        "cumulative_costs": plot_cumulative_costs(
            result, figures_dir / f"{label}_cumulative_costs.png"
        ),
        "rolling": plot_rolling(result, figures_dir / f"{label}_rolling.png"),
    }
    return paths
