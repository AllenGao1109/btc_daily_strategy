"""Command-line entry point: run a strategy backtest against all benchmarks.

Usage::

    python -m src.run_backtest --config config.yaml
    python -m src.run_backtest --config config.yaml --strategy momentum_long_short
    python -m src.run_backtest --config config.yaml --synthetic   # offline test data

The runner loads data, builds features, runs the chosen strategy plus the five
mandatory benchmarks, computes metrics, generates charts, and writes a markdown
report under ``reports/``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import data as data_mod
from . import features as features_mod
from . import metrics as metrics_mod
from . import plotting, report
from .config import BacktestConfig, load_config
from .strategies import get_strategy

# The five mandatory benchmark strategies (section 17).
BENCHMARKS = [
    "cash",
    "buy_and_hold",
    "leveraged_buy_and_hold",
    "sma_long_flat",
    "sma_long_short",
]


def run_single(
    feat_df: pd.DataFrame,
    strategy_name: str,
    params: dict[str, Any],
    bt_cfg: BacktestConfig,
) -> pd.DataFrame:
    """Run one strategy through the backtest engine and return its state table.

    Args:
        feat_df: Feature-augmented OHLCV frame.
        strategy_name: Registered strategy name.
        params: Strategy parameters.
        bt_cfg: Flattened backtest config.

    Returns:
        The daily state DataFrame from the engine.
    """
    from .backtest import run_backtest

    signal_fn = get_strategy(strategy_name)
    raw_signal = signal_fn(feat_df, params)
    return run_backtest(
        feat_df,
        raw_signal,
        initial_capital=bt_cfg.initial_capital,
        fee_rate=bt_cfg.fee_rate,
        slippage_rate=bt_cfg.slippage_rate,
        execution_lag_days=bt_cfg.execution_lag_days,
        min_weight=bt_cfg.min_weight,
        max_weight=bt_cfg.max_weight,
        max_leverage=bt_cfg.max_leverage,
        rebalance_policy=bt_cfg.rebalance_policy,
        leverage_breach_action=bt_cfg.leverage_breach_action,
        funding_config=bt_cfg.funding_config,
        weight_band=bt_cfg.weight_band,
    )


def run_all(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Run the primary strategy plus all benchmarks and compute everything.

    Args:
        config: Full config dict.
        use_synthetic: If True, use deterministic synthetic data (offline).
        force_reload: If True, bypass the data cache.

    Returns:
        A dict with keys ``results`` (name -> state table), ``metrics``
        (name -> metrics dict), ``feat_df``, ``primary`` (strategy name).
    """
    bt_cfg = BacktestConfig.from_config(config)
    primary = config["strategy"]["name"]
    primary_params = config["strategy"].get("params", {})

    if use_synthetic:
        raw_df = data_mod.generate_synthetic_btc()
    else:
        raw_df = data_mod.load_btc_data(config, force_reload=force_reload)
    feat_df = features_mod.build_features(raw_df)

    # Enrich with on-chain + cross-asset columns for factor strategies. Best-effort:
    # if the data sources are unreachable, factor strategies simply use the subset
    # of factors that are computable from OHLCV alone.
    if not use_synthetic and config.get("data", {}).get("enrich", True):
        try:
            from . import onchain as onchain_mod
            feat_df = onchain_mod.merge_onchain(
                feat_df, onchain_mod.load_coinmetrics(onchain_mod.ONCHAIN_METRICS)
            )
            feat_df["eth_close"] = data_mod.load_eth_close(feat_df.index)
        except Exception as exc:  # noqa: BLE001 - enrichment is optional
            print(f"[warn] on-chain/cross-asset enrichment skipped: {exc}")

    strategies_to_run = list(dict.fromkeys([*BENCHMARKS, primary]))
    results: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name in strategies_to_run:
        params = primary_params if name == primary else {}
        result = run_single(feat_df, name, params, bt_cfg)
        results[name] = result
        metrics[name] = metrics_mod.compute_metrics(result, bt_cfg.initial_capital)

    return {
        "results": results,
        "metrics": metrics,
        "feat_df": feat_df,
        "primary": primary,
        "bt_cfg": bt_cfg,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run a BTC daily strategy backtest.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--strategy", default=None, help="Override the strategy name in the config."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data (offline, for testing).",
    )
    parser.add_argument(
        "--force-reload", action="store_true", help="Bypass the data cache."
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.strategy:
        config["strategy"]["name"] = args.strategy

    bundle = run_all(
        config, use_synthetic=args.synthetic, force_reload=args.force_reload
    )
    results = bundle["results"]
    metrics = bundle["metrics"]
    primary = bundle["primary"]
    bt_cfg: BacktestConfig = bundle["bt_cfg"]

    output_dir = Path(config["report"].get("output_dir", "reports"))
    figures_dir = output_dir / "figures"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    figure_paths = None
    if config["report"].get("include_charts", True) and not args.no_charts:
        benchmark_curves = {
            name: res["equity_end"].astype(float) for name, res in results.items()
        }
        figure_paths = plotting.generate_all_figures(
            results[primary],
            benchmark_curves,
            figures_dir,
            label=primary,
            max_leverage=bt_cfg.max_leverage,
        )

    report_path = report.generate_report(
        strategy_name=primary,
        strat_metrics=metrics[primary],
        benchmark_metrics=metrics,
        result=results[primary],
        config=config,
        output_dir=output_dir,
        timestamp=timestamp,
        figure_paths=figure_paths,
    )

    print(f"Primary strategy: {primary}")
    for name, m in metrics.items():
        print(
            f"  {name:24s} total_return={m['total_return']*100:8.2f}%  "
            f"CAGR={m['cagr']*100:7.2f}%  MaxDD={m['max_drawdown']*100:7.2f}%  "
            f"Sharpe={m['sharpe_ratio']:5.2f}  fees={m['total_fees_paid']:.0f}  "
            f"liq={m['liquidated']}"
        )
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
