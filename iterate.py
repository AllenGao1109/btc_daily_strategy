"""Research iteration workbench.

Loads real BTC data once, builds the signal set (baselines + experimental
strategies), and prints per-window (train/validation/test) metrics. Used to
improve out-of-sample Sharpe/NAV without test-set tuning.

Run from the project root:  python3 iterate.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.features import build_features
from src.research import evaluate, print_table
from src.strategies import get_strategy
from src.validation import make_fixed_split


def build_signals(df, params):
    """Build the {label: raw_signal} mapping to evaluate this iteration."""
    sig = {}
    baseline = {
        "BH_1x": ("buy_and_hold", {}),
        "BH_2x": ("leveraged_buy_and_hold", {}),
        "SMA200_LF": ("sma_long_flat", {"sma_slow": 200}),
        "SMA200_LS": ("sma_long_short", {"sma_slow": 200}),
        "MOM30_LS": ("momentum_long_short", {"momentum_window": 30}),
        "TREND_LEV": ("trend_leverage", params),
    }
    for label, (name, p) in baseline.items():
        sig[label] = get_strategy(name)(df, p)

    experimental = {
        "VT_LF_100": ("vol_target_trend",
                      {"target_vol": 0.5, "vol_window": 30, "sma_window": 100}),
        "VT_LF_200": ("vol_target_trend",
                      {"target_vol": 0.5, "vol_window": 30, "sma_window": 200}),
        "VT_LS_100": ("vol_target_trend",
                      {"target_vol": 0.5, "vol_window": 30, "sma_window": 100,
                       "allow_short": True, "short_scale": 0.5}),
        "DON_50_25": ("donchian_breakout",
                      {"entry_window": 50, "exit_window": 25}),
        "DON_VT_50": ("donchian_breakout",
                      {"entry_window": 50, "exit_window": 25, "vol_target": 0.5}),
        "ENS_LF": ("trend_ensemble",
                   {"sma_windows": [50, 100, 200], "use_donchian": True,
                    "target_vol": 0.45, "vol_window": 30}),
        "ENS_LS": ("trend_ensemble",
                   {"sma_windows": [50, 100, 200], "use_donchian": True,
                    "target_vol": 0.45, "vol_window": 30, "allow_short": True}),
    }
    for label, (name, p) in experimental.items():
        sig[label] = get_strategy(name)(df, p)
    return sig


def main():
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    splits = make_fixed_split(cfg["validation"])
    params = cfg["strategy"].get("params", {})

    signals = build_signals(df, params)
    table = evaluate(df, signals, bt, splits)
    print(f"Data: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} days)")
    print_table(table, windows=["train", "validation", "test"])
    return table


if __name__ == "__main__":
    main()
