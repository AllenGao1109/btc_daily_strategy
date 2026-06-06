"""Performance, risk, cost, and leverage metrics computed from a backtest table.

Conventions:
  - Crypto trades every day: 365 periods per year (configurable).
  - Risk-free rate is 0 in the first version.
  - Max drawdown is computed from the equity curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365


def _equity_curve(result: pd.DataFrame) -> pd.Series:
    """Extract the end-of-day equity curve from a backtest result table."""
    return result["equity_end"].astype(float)


def total_return(equity: pd.Series, initial_capital: float) -> float:
    """Total return over the whole period: equity_end[-1] / initial_capital - 1."""
    if len(equity) == 0:
        return 0.0
    return float(equity.iloc[-1] / initial_capital - 1.0)


def cagr(equity: pd.Series, initial_capital: float, periods_per_year: int) -> float:
    """Compound annual growth rate based on calendar span of the equity curve."""
    if len(equity) < 2:
        return 0.0
    final = float(equity.iloc[-1])
    if final <= 0:
        return -1.0
    n_periods = len(equity)
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((final / initial_capital) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized volatility of daily strategy returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Annualized Sharpe ratio (risk-free rate defaults to 0)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Annualized Sortino ratio using downside deviation."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Per-day drawdown relative to the running peak (<= 0)."""
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (most negative point of the drawdown series)."""
    if len(equity) == 0:
        return 0.0
    return float(drawdown_series(equity).min())


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    if max_dd == 0:
        return 0.0
    return float(cagr_value / abs(max_dd))


def _period_returns(equity: pd.Series, freq: str) -> pd.Series:
    """Resample the equity curve to period-end and compute period returns."""
    if len(equity) == 0:
        return pd.Series(dtype=float)
    period_end = equity.resample(freq).last()
    return period_end.pct_change().dropna()


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-month returns derived from the equity curve."""
    return _period_returns(equity, "ME")


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year returns derived from the equity curve."""
    return _period_returns(equity, "YE")


def compute_metrics(
    result: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Compute the full metric suite from a backtest result table.

    Args:
        result: Daily state table from :func:`src.backtest.run_backtest`.
        initial_capital: Starting equity used for return normalization.
        periods_per_year: Annualization factor (365 for crypto).
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        A dict of named metrics (section 16 of CLAUDE.md).
    """
    equity = _equity_curve(result)
    returns = result["strategy_daily_return"].astype(float)

    cagr_value = cagr(equity, initial_capital, periods_per_year)
    max_dd = max_drawdown(equity)

    # Trade accounting: a "trade" is any day with non-zero traded notional.
    traded_days = result["trade_notional"] > 0
    n_trades = int(traded_days.sum())
    total_fees = float(result["fee"].sum())
    total_slippage = float(result["slippage"].sum())
    total_funding = float(result["funding_cost"].sum())
    total_borrow = float(result["borrow_cost"].sum())

    # Time in market / exposure stats.
    in_market = result["exposure_notional_end"].abs() > 0
    exposure_pct = float(in_market.mean()) if len(result) else 0.0
    avg_net_exposure = float(result["actual_weight_end"].mean()) if len(result) else 0.0
    avg_gross_leverage = (
        float(result["gross_leverage_end"].mean()) if len(result) else 0.0
    )
    max_gross_leverage = (
        float(result["gross_leverage_end"].max()) if len(result) else 0.0
    )
    n_breaches = int(result["leverage_breach"].sum())
    liquidated = bool(result["liquidated"].any())

    # Average holding period: in-market days divided by number of entries.
    n_entries = n_trades  # each rebalance that changes notional counts as a trade
    avg_holding_period = (
        float(in_market.sum() / n_entries) if n_entries > 0 else 0.0
    )

    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    avg_daily_turnover = float(result["turnover"].mean()) if len(result) else 0.0

    monthly = monthly_returns(equity)
    yearly = yearly_returns(equity)

    metrics: dict[str, Any] = {
        "total_return": total_return(equity, initial_capital),
        "cagr": cagr_value,
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, periods_per_year, risk_free_rate),
        "max_drawdown": max_dd,
        "calmar_ratio": calmar_ratio(cagr_value, max_dd),
        "win_rate": win_rate,
        "num_trades": n_trades,
        "avg_holding_period": avg_holding_period,
        "total_turnover": float(result["turnover"].sum()),
        "avg_daily_turnover": avg_daily_turnover,
        "total_fees_paid": total_fees,
        "total_fees_pct_initial": total_fees / initial_capital
        if initial_capital
        else 0.0,
        "total_slippage": total_slippage,
        "total_funding_cost": total_funding,
        "total_borrow_cost": total_borrow,
        "total_funding_borrow_cost": total_funding + total_borrow,
        "exposure_pct": exposure_pct,
        "avg_net_exposure": avg_net_exposure,
        "avg_gross_leverage": avg_gross_leverage,
        "max_gross_leverage": max_gross_leverage,
        "num_leverage_breaches": n_breaches,
        "liquidated": liquidated,
        "best_month": float(monthly.max()) if len(monthly) else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "best_year": float(yearly.max()) if len(yearly) else 0.0,
        "worst_year": float(yearly.min()) if len(yearly) else 0.0,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
    }
    return metrics
