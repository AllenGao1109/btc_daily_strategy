"""Strategy registry.

Each strategy is a module exposing ``generate_signals(df, params) -> pd.Series``
returning RAW target BTC exposure weights in [-2.0, +2.0], BEFORE the execution
lag. Strategies never compute PnL, charge fees, or touch equity/leverage.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import (
    buy_and_hold,
    cash,
    leveraged_buy_and_hold,
    momentum_long_short,
    sma_long_flat,
    sma_long_short,
    trend_leverage,
)

SignalFn = Callable[[pd.DataFrame, dict], pd.Series]

STRATEGY_REGISTRY: dict[str, SignalFn] = {
    "cash": cash.generate_signals,
    "buy_and_hold": buy_and_hold.generate_signals,
    "leveraged_buy_and_hold": leveraged_buy_and_hold.generate_signals,
    "sma_long_flat": sma_long_flat.generate_signals,
    "sma_long_short": sma_long_short.generate_signals,
    "momentum_long_short": momentum_long_short.generate_signals,
    "trend_leverage": trend_leverage.generate_signals,
}


def get_strategy(name: str) -> SignalFn:
    """Look up a strategy's ``generate_signals`` function by name.

    Args:
        name: Registered strategy name.

    Returns:
        The strategy's signal-generating callable.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]
