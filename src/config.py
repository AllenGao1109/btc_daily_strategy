"""Configuration loading and access helpers.

The config is a plain nested dict loaded from a YAML file. We deliberately keep
it a dict (rather than a heavy schema object) so notebooks and scripts can read
values directly, but we validate the safety-critical fields (leverage bounds,
fee rate, rebalance policy) on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

VALID_REBALANCE_POLICIES = {"signal_change_or_risk_control", "daily_target_rebalance"}
VALID_BREACH_ACTIONS = {"delever_next_day", "liquidate"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML config.

    Args:
        path: Path to a YAML config file. Defaults to the repo ``config.yaml``.

    Returns:
        The parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If safety-critical fields are missing or invalid.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate safety-critical config fields. Raises ValueError on problems."""
    exposure = config.get("exposure", {})
    min_w = exposure.get("min_weight", -2.0)
    max_w = exposure.get("max_weight", 2.0)
    max_lev = exposure.get("max_leverage", 2.0)
    if min_w < -2.0 or max_w > 2.0:
        raise ValueError(
            f"Exposure bounds must stay within [-2.0, 2.0]; got "
            f"[{min_w}, {max_w}]."
        )
    if max_lev > 2.0:
        raise ValueError(f"max_leverage must not exceed 2.0; got {max_lev}.")
    if min_w > max_w:
        raise ValueError(f"min_weight ({min_w}) must be <= max_weight ({max_w}).")

    portfolio = config.get("portfolio", {})
    fee_rate = portfolio.get("fee_rate", 0.0)
    if fee_rate < 0:
        raise ValueError(f"fee_rate must be non-negative; got {fee_rate}.")

    backtest = config.get("backtest", {})
    policy = backtest.get("rebalance_policy", "signal_change_or_risk_control")
    if policy not in VALID_REBALANCE_POLICIES:
        raise ValueError(
            f"rebalance_policy must be one of {sorted(VALID_REBALANCE_POLICIES)}; "
            f"got {policy!r}."
        )
    action = backtest.get("leverage_breach_action", "delever_next_day")
    if action not in VALID_BREACH_ACTIONS:
        raise ValueError(
            f"leverage_breach_action must be one of {sorted(VALID_BREACH_ACTIONS)}; "
            f"got {action!r}."
        )


@dataclass(frozen=True)
class BacktestConfig:
    """Flattened, typed view of the parameters the backtest engine needs.

    This is a convenience adapter so callers do not have to dig through the
    nested config dict. Funding config is passed through as a dict.
    """

    initial_capital: float
    fee_rate: float
    slippage_rate: float
    execution_lag_days: int
    min_weight: float
    max_weight: float
    max_leverage: float
    rebalance_policy: str
    leverage_breach_action: str
    stop_trading_if_equity_zero: bool
    funding_config: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BacktestConfig":
        """Build a BacktestConfig from the nested config dict."""
        portfolio = config.get("portfolio", {})
        exposure = config.get("exposure", {})
        backtest = config.get("backtest", {})
        funding = config.get("funding", {})
        return cls(
            initial_capital=float(portfolio.get("initial_capital", 10000)),
            fee_rate=float(portfolio.get("fee_rate", 0.005)),
            slippage_rate=float(portfolio.get("slippage_rate", 0.0)),
            execution_lag_days=int(backtest.get("execution_lag_days", 1)),
            min_weight=float(exposure.get("min_weight", -2.0)),
            max_weight=float(exposure.get("max_weight", 2.0)),
            max_leverage=float(exposure.get("max_leverage", 2.0)),
            rebalance_policy=str(
                backtest.get("rebalance_policy", "signal_change_or_risk_control")
            ),
            leverage_breach_action=str(
                backtest.get("leverage_breach_action", "delever_next_day")
            ),
            stop_trading_if_equity_zero=bool(
                backtest.get("stop_trading_if_equity_zero", True)
            ),
            funding_config=dict(funding) if funding else {},
        )
