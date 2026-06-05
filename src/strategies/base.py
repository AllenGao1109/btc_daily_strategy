"""Strategy interface contract and shared helpers.

Every strategy module exposes::

    def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
        '''Return RAW target BTC exposure weights in [-2.0, +2.0].'''

Interpretation of the returned weight:
    +2.0 = 200% long BTC
    +1.0 = 100% long BTC
     0.0 = flat
    -1.0 = 100% short BTC
    -2.0 = 200% short BTC

Hard rules (enforced by convention + :func:`clip_weights`):
  - Output is a TARGET exposure weight, nothing else.
  - Output is clipped to [min_weight, max_weight].
  - Strategies must not compute PnL, charge fees, modify equity, or know
    future rows. Lagging, costs, leverage, funding, and liquidation all live
    in the backtest engine.
"""

from __future__ import annotations

import warnings

import pandas as pd

MIN_WEIGHT = -2.0
MAX_WEIGHT = 2.0


def clip_weights(
    weights: pd.Series,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    *,
    warn: bool = True,
) -> pd.Series:
    """Clip target weights into [min_weight, max_weight].

    Args:
        weights: Raw target weights.
        min_weight: Lower bound (>= -2.0).
        max_weight: Upper bound (<= +2.0).
        warn: If True, emit a warning when any value is clipped.

    Returns:
        The clipped weights (same index, NaNs preserved).
    """
    if min_weight < MIN_WEIGHT or max_weight > MAX_WEIGHT:
        raise ValueError(
            f"Bounds must stay within [{MIN_WEIGHT}, {MAX_WEIGHT}]; "
            f"got [{min_weight}, {max_weight}]."
        )
    finite = weights.dropna()
    if warn and ((finite < min_weight) | (finite > max_weight)).any():
        warnings.warn(
            "Strategy produced target weights outside "
            f"[{min_weight}, {max_weight}]; clipping. This indicates the "
            "strategy tried to exceed the leverage cap.",
            stacklevel=2,
        )
    return weights.clip(lower=min_weight, upper=max_weight)
