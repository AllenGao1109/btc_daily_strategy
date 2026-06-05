"""On-chain iteration: MVRV-gated ensemble + higher-NAV variants.

Merges CoinMetrics MVRV into the feature frame and compares:
  - buy-and-hold 1x / 2x
  - the plain trend ensemble (chosen baseline)
  - MVRV-gated ensemble (de-risk overvalued, lean into undervalued)
  - a higher-NAV variant (raised vol target + MVRV boost)

Run from the project root:  python3 iterate_onchain.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.features import build_features
from src.onchain import load_coinmetrics, merge_onchain
from src.research import evaluate, print_table
from src.strategies import get_strategy
from src.validation import make_fixed_split

ENS = {"sma_windows": [50, 100, 200], "use_donchian": True,
       "target_vol": 0.55, "vol_window": 45}


def main():
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    df = merge_onchain(df, load_coinmetrics(["CapMVRVCur", "AdrActCnt"]))
    splits = make_fixed_split(cfg["validation"])

    signals = {
        "BH_1x": get_strategy("buy_and_hold")(df, {}),
        "BH_2x": get_strategy("leveraged_buy_and_hold")(df, {}),
        "ENS": get_strategy("trend_ensemble")(df, ENS),
        # MVRV de-risk only (defensive): no boost.
        "MVRV_derisk": get_strategy("mvrv_trend")(
            df, {**ENS, "boost_max": 1.0, "z_high": 1.0, "z_max": 2.5}),
        # MVRV de-risk + undervaluation boost (balanced).
        "MVRV_bal": get_strategy("mvrv_trend")(
            df, {**ENS, "boost_max": 1.5, "z_high": 1.0, "z_max": 2.5,
                 "z_low": -0.5, "z_min": -2.0}),
        # Higher-NAV variant: raised vol target + stronger undervaluation boost.
        "MVRV_NAV": get_strategy("mvrv_trend")(
            df, {**ENS, "target_vol": 0.80, "boost_max": 1.8,
                 "z_high": 1.5, "z_max": 3.0, "z_low": -0.3, "z_min": -2.0}),
    }
    table = evaluate(df, signals, bt, splits)
    print(f"Data: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} days)")
    print_table(table, windows=["train", "validation", "test"])
    return table


if __name__ == "__main__":
    main()
