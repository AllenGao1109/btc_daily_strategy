"""Long/short leveraged daily backtest engine.

This engine implements the timing, cost, persistence, leverage, funding, and
liquidation model described in CLAUDE.md sections 5-8 and 15. It consumes a RAW
(pre-lag) target-weight signal and returns a full daily state table.

Accounting model (per day ``t``), following section 15's preferred formulas::

    equity_before_trade            = previous_equity
    actual_weight_before_rebalance = previous_exposure_notional / equity_before_trade
    target_exposure_notional       = target_weight * equity_before_trade   (if rebalancing)
    trade_notional                 = abs(target_exposure_notional - previous_exposure_notional)
    turnover                       = trade_notional / equity_before_trade
    fee                            = trade_notional * fee_rate
    slippage                       = trade_notional * slippage_rate
    equity_after_cost              = equity_before_trade - fee - slippage - funding - borrow
    pnl                            = exposure_notional_after_rebalance * btc_return
    equity_end                     = equity_after_cost + pnl
    exposure_notional_end          = exposure_notional_after_rebalance * (1 + btc_return)

Key invariants:
  - The signal is lagged by ``execution_lag_days`` (default 1). Day ``t`` uses
    only information available at day ``t-1``.
  - Positions persist. Under ``signal_change_or_risk_control`` the engine only
    trades when the lagged target weight changes or risk control fires.
  - Fees are charged ONLY on traded notional, never on the whole portfolio.
  - Leverage drift is tracked; breaches above ``max_leverage`` are flagged and
    handled per ``leverage_breach_action``.
  - Intraday liquidation is NOT modeled (daily close-to-close only).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .features import lag_signal

# Numerical tolerance so floating-point drift does not spuriously flag breaches.
_LEV_EPS = 1e-9

STATE_COLUMNS = [
    "date",
    "close",
    "btc_return",
    "raw_signal",
    "target_weight",
    "previous_exposure_notional",
    "previous_equity",
    "actual_weight_before_rebalance",
    "gross_leverage_before_rebalance",
    "target_exposure_notional",
    "trade_notional",
    "turnover",
    "fee",
    "slippage",
    "funding_cost",
    "borrow_cost",
    "equity_before_trade",
    "equity_after_cost",
    "exposure_notional_after_rebalance",
    "pnl",
    "equity_end",
    "exposure_notional_end",
    "actual_weight_end",
    "gross_leverage_end",
    "leverage_breach",
    "liquidated",
    "rebalanced",
    "strategy_daily_return",
]


def _funding_and_borrow(
    exposure_after: float,
    equity_before_trade: float,
    funding_config: dict[str, Any] | None,
) -> tuple[float, float]:
    """Compute (funding_cost, borrow_cost) for a held exposure.

    funding_cost is perpetual-style funding on the full absolute notional.
    borrow_cost is the sum of long-margin borrow (notional above equity) and
    short borrow (absolute short notional).

    Args:
        exposure_after: Exposure notional held through the day (post-rebalance).
        equity_before_trade: Equity available before the trade.
        funding_config: ``funding`` config section, or None.

    Returns:
        ``(funding_cost, borrow_cost)``, both non-negative.
    """
    if not funding_config or not funding_config.get("enabled", False):
        return 0.0, 0.0

    long_rate = float(funding_config.get("long_borrow_rate_daily", 0.0))
    short_rate = float(funding_config.get("short_borrow_rate_daily", 0.0))
    perp_rate = float(funding_config.get("perp_funding_rate_daily", 0.0))

    funding_cost = abs(exposure_after) * perp_rate

    borrowed_long = max(exposure_after - equity_before_trade, 0.0)
    short_notional = abs(exposure_after) if exposure_after < 0 else 0.0
    borrow_cost = borrowed_long * long_rate + short_notional * short_rate

    return funding_cost, borrow_cost


def prepare_target_weights(
    raw_signal: pd.Series,
    execution_lag_days: int,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    """Lag, clip, and conservatively fill a raw signal into target weights.

    Args:
        raw_signal: Raw per-day target weights (pre-lag), in [-2, 2] ideally.
        execution_lag_days: Days to lag before execution (default 1).
        min_weight: Lower clip bound (>= -2.0).
        max_weight: Upper clip bound (<= +2.0).

    Returns:
        The executable target-weight series: lagged, clipped to bounds, with
        leading/warmup NaNs filled to 0.0 (flat).
    """
    lagged = lag_signal(raw_signal, execution_lag_days)
    finite = lagged.dropna()
    if ((finite < min_weight - _LEV_EPS) | (finite > max_weight + _LEV_EPS)).any():
        warnings.warn(
            f"Signal values outside [{min_weight}, {max_weight}] were clipped "
            "to respect the leverage cap.",
            stacklevel=2,
        )
    target = lagged.clip(lower=min_weight, upper=max_weight)
    return target.fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    raw_signal: pd.Series,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float = 0.0,
    execution_lag_days: int = 1,
    min_weight: float = -2.0,
    max_weight: float = 2.0,
    max_leverage: float = 2.0,
    rebalance_policy: str = "signal_change_or_risk_control",
    leverage_breach_action: str = "delever_next_day",
    funding_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the long/short leveraged daily backtest.

    Args:
        df: OHLCV frame with a ``close`` column, UTC DatetimeIndex.
        raw_signal: Raw (pre-lag) target weights aligned to ``df.index``.
        initial_capital: Starting equity.
        fee_rate: Transaction fee on traded notional (e.g. 0.005).
        slippage_rate: Slippage on traded notional (default 0).
        execution_lag_days: Signal lag in days (default 1).
        min_weight: Minimum target weight (>= -2.0).
        max_weight: Maximum target weight (<= +2.0).
        max_leverage: Maximum allowed gross leverage (<= 2.0).
        rebalance_policy: ``signal_change_or_risk_control`` (default) or
            ``daily_target_rebalance``.
        leverage_breach_action: ``delever_next_day`` (default) or ``liquidate``.
        funding_config: ``funding`` config section, or None.

    Returns:
        A daily state DataFrame indexed by date with the columns in
        ``STATE_COLUMNS`` (minus ``date``, which is the index).
    """
    if rebalance_policy not in {"signal_change_or_risk_control", "daily_target_rebalance"}:
        raise ValueError(f"Unknown rebalance_policy: {rebalance_policy!r}")
    if leverage_breach_action not in {"delever_next_day", "liquidate"}:
        raise ValueError(f"Unknown leverage_breach_action: {leverage_breach_action!r}")

    raw_signal = raw_signal.reindex(df.index)
    target_weights = prepare_target_weights(
        raw_signal, execution_lag_days, min_weight, max_weight
    )

    close = df["close"].astype(float)
    btc_return = close.pct_change().fillna(0.0)

    # Persistent state carried across days.
    prev_equity = float(initial_capital)
    prev_exposure = 0.0
    prev_target_weight = 0.0
    liquidated = False

    records: list[dict[str, Any]] = []

    for date in df.index:
        close_t = float(close.loc[date])
        ret_t = float(btc_return.loc[date])
        raw_sig_t = raw_signal.loc[date]
        target_w = float(target_weights.loc[date])

        equity_before_trade = prev_equity

        # --- Already liquidated: stay flat and dead for the rest of the run. ---
        if liquidated:
            records.append(
                _flat_record(date, close_t, ret_t, raw_sig_t, target_w)
            )
            prev_target_weight = target_w
            continue

        actual_weight_before = (
            prev_exposure / equity_before_trade if equity_before_trade != 0 else 0.0
        )
        gross_lev_before = abs(actual_weight_before)

        signal_changed = not np.isclose(target_w, prev_target_weight)
        risk_breach = gross_lev_before > max_leverage + _LEV_EPS

        force_liquidate_now = False
        if rebalance_policy == "daily_target_rebalance":
            rebalance = True
        else:
            rebalance = signal_changed
            if risk_breach:
                if leverage_breach_action == "liquidate":
                    force_liquidate_now = True
                else:  # delever_next_day
                    rebalance = True

        # --- Determine post-rebalance exposure. ---
        if force_liquidate_now:
            target_exposure_notional = 0.0
        elif rebalance:
            target_exposure_notional = target_w * equity_before_trade
        else:
            # Carry the (drifted) position forward; no trade.
            target_exposure_notional = prev_exposure

        trade_notional = abs(target_exposure_notional - prev_exposure)
        turnover = trade_notional / equity_before_trade if equity_before_trade else 0.0
        fee = trade_notional * fee_rate
        slippage = trade_notional * slippage_rate

        exposure_after = target_exposure_notional
        funding_cost, borrow_cost = _funding_and_borrow(
            exposure_after, equity_before_trade, funding_config
        )

        equity_after_cost = (
            equity_before_trade - fee - slippage - funding_cost - borrow_cost
        )

        # --- Liquidation check after costs (before applying market return). ---
        if equity_after_cost <= 0:
            records.append(
                {
                    "date": date,
                    "close": close_t,
                    "btc_return": ret_t,
                    "raw_signal": raw_sig_t,
                    "target_weight": target_w,
                    "previous_exposure_notional": prev_exposure,
                    "previous_equity": prev_equity,
                    "actual_weight_before_rebalance": actual_weight_before,
                    "gross_leverage_before_rebalance": gross_lev_before,
                    "target_exposure_notional": target_exposure_notional,
                    "trade_notional": trade_notional,
                    "turnover": turnover,
                    "fee": fee,
                    "slippage": slippage,
                    "funding_cost": funding_cost,
                    "borrow_cost": borrow_cost,
                    "equity_before_trade": equity_before_trade,
                    "equity_after_cost": equity_after_cost,
                    "exposure_notional_after_rebalance": exposure_after,
                    "pnl": 0.0,
                    "equity_end": 0.0,
                    "exposure_notional_end": 0.0,
                    "actual_weight_end": 0.0,
                    "gross_leverage_end": 0.0,
                    "leverage_breach": bool(risk_breach),
                    "liquidated": True,
                    "rebalanced": bool(rebalance or force_liquidate_now),
                    "strategy_daily_return": (0.0 / prev_equity - 1.0)
                    if prev_equity
                    else 0.0,
                }
            )
            liquidated = True
            prev_equity = 0.0
            prev_exposure = 0.0
            prev_target_weight = target_w
            continue

        pnl = exposure_after * ret_t
        equity_end = equity_after_cost + pnl
        exposure_notional_end = exposure_after * (1.0 + ret_t)

        end_liquidated = force_liquidate_now
        if equity_end <= 0:
            end_liquidated = True
            equity_end = 0.0
            exposure_notional_end = 0.0

        if equity_end > 0:
            actual_weight_end = exposure_notional_end / equity_end
        else:
            actual_weight_end = 0.0
        gross_leverage_end = abs(actual_weight_end)
        leverage_breach_end = gross_leverage_end > max_leverage + _LEV_EPS

        strategy_daily_return = (
            equity_end / prev_equity - 1.0 if prev_equity else 0.0
        )

        records.append(
            {
                "date": date,
                "close": close_t,
                "btc_return": ret_t,
                "raw_signal": raw_sig_t,
                "target_weight": target_w,
                "previous_exposure_notional": prev_exposure,
                "previous_equity": prev_equity,
                "actual_weight_before_rebalance": actual_weight_before,
                "gross_leverage_before_rebalance": gross_lev_before,
                "target_exposure_notional": target_exposure_notional,
                "trade_notional": trade_notional,
                "turnover": turnover,
                "fee": fee,
                "slippage": slippage,
                "funding_cost": funding_cost,
                "borrow_cost": borrow_cost,
                "equity_before_trade": equity_before_trade,
                "equity_after_cost": equity_after_cost,
                "exposure_notional_after_rebalance": exposure_after,
                "pnl": pnl,
                "equity_end": equity_end,
                "exposure_notional_end": exposure_notional_end,
                "actual_weight_end": actual_weight_end,
                "gross_leverage_end": gross_leverage_end,
                "leverage_breach": bool(leverage_breach_end),
                "liquidated": bool(end_liquidated),
                "rebalanced": bool(rebalance or force_liquidate_now),
                "strategy_daily_return": strategy_daily_return,
            }
        )

        liquidated = end_liquidated
        prev_equity = equity_end
        prev_exposure = exposure_notional_end
        prev_target_weight = target_w

    result = pd.DataFrame.from_records(records).set_index("date")
    return result


def _flat_record(
    date: pd.Timestamp,
    close_t: float,
    ret_t: float,
    raw_sig_t: Any,
    target_w: float,
) -> dict[str, Any]:
    """Build a fully-flat, post-liquidation daily record (all exposure zeroed)."""
    return {
        "date": date,
        "close": close_t,
        "btc_return": ret_t,
        "raw_signal": raw_sig_t,
        "target_weight": target_w,
        "previous_exposure_notional": 0.0,
        "previous_equity": 0.0,
        "actual_weight_before_rebalance": 0.0,
        "gross_leverage_before_rebalance": 0.0,
        "target_exposure_notional": 0.0,
        "trade_notional": 0.0,
        "turnover": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
        "funding_cost": 0.0,
        "borrow_cost": 0.0,
        "equity_before_trade": 0.0,
        "equity_after_cost": 0.0,
        "exposure_notional_after_rebalance": 0.0,
        "pnl": 0.0,
        "equity_end": 0.0,
        "exposure_notional_end": 0.0,
        "actual_weight_end": 0.0,
        "gross_leverage_end": 0.0,
        "leverage_breach": False,
        "liquidated": True,
        "rebalanced": False,
        "strategy_daily_return": 0.0,
    }
