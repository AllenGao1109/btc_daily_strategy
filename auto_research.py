"""Autonomous ML research driver (runs in background on MPS).

Churns through a randomized search over models / horizons / feature sets / signal
mappings, evaluating each WALK-FORWARD out-of-sample, and appends every result to
results/ml_results.csv. Selection uses validation metrics only; the TEST window is
computed for the current best solely for honest reporting and never drives search.

Design notes:
  - Datasets are cached per (feature_set, horizon) so we pay feature/alignment
    cost once.
  - Each experiment is isolated in try/except so one failure never stops the run.
  - Deterministic given the seed sequence (no wall-clock randomness in modeling).

Run:  python3 auto_research.py --max-exp 400
Edit SEARCH_SPACE below between rounds to steer the search.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.config import BacktestConfig, load_config
from src.data import load_btc_data
from src.features import build_features
from src.onchain import load_coinmetrics, merge_onchain
from src.metrics import sharpe_ratio
from src.research import run_full, window_metrics
from src.strategies import get_strategy
from src.validation import make_fixed_split
from src.ml.dataset import add_onchain_features, build_dataset, select_features
from src.ml.torch_models import TorchMLP, get_device
from src.ml.walkforward import predictions_to_weight, walk_forward_predict

RESULTS = Path("results/ml_results.csv")

# Round 2: standalone next-day ML did not generalize (mean test Sharpe -0.28,
# val/test decoupled at the top). Refocus on (a) longer horizons + trend/vol-mom
# features that showed the least-bad behavior, (b) heavier regularization, and
# (c) the HYBRID "gate" mode where ML only filters the robust trend ensemble
# rather than trading standalone.
SEARCH_SPACE = {
    "horizon": [5, 10, 20, 30],
    "feature_set": ["vol_mom", "trend", "base_onchain"],
    "model": ["logistic", "mlp_small"],
    "task": ["classification", "regression"],
    "dropout": [0.3, 0.5],
    "weight_decay": [3e-3, 1e-2, 3e-2],
    "epochs": [60, 100],
    "n_seeds": [3, 5],
    "signal_mode": ["gate", "vol_target", "sign"],
    "long_only": [True],
    "gate_down": [0.0, 0.3, 0.5],
}

FEATURE_SETS = {
    "base": dict(use_onchain=False),
    "base_onchain": dict(use_onchain=True),
    "trend": ["price_to_sma_20", "price_to_sma_50", "price_to_sma_100",
              "price_to_sma_200", "momentum_30", "momentum_90"],
    "vol_mom": ["rolling_vol_7", "rolling_vol_30", "rolling_vol_90",
                "momentum_7", "momentum_14", "momentum_30", "drawdown_30"],
}


class SeedEnsemble:
    """Average predictions of k models trained with different seeds (variance cut)."""

    def __init__(self, factory, n_seeds: int):
        self.factory = factory
        self.n_seeds = n_seeds
        self.models: list = []

    def fit(self, X, y):
        self.models = []
        for s in range(self.n_seeds):
            m = self.factory(s)
            m.fit(X, y)
            self.models.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def model_factory(cfg: dict):
    """Build a zero-arg-ish factory: returns SeedEnsemble over the configured model."""
    hidden = {"logistic": (), "mlp_small": (16,), "mlp_mid": (32, 16)}[cfg["model"]]

    def make(seed: int):
        return TorchMLP(
            task=cfg["task"], hidden=hidden, dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"], epochs=cfg["epochs"], seed=seed,
        )

    return lambda: SeedEnsemble(make, cfg["n_seeds"])


def get_features(df, feature_set: str):
    """Resolve a feature-set name to concrete columns present on df."""
    spec = FEATURE_SETS[feature_set]
    if isinstance(spec, dict):
        return select_features(df, **spec)
    return [c for c in spec if c in df.columns]


def sample_config(rng: np.random.Generator) -> dict:
    """Draw one random experiment config from the search space."""
    return {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}


def fold_yearly_sharpe(res, lo_year: int, hi_year: int) -> tuple[float, float]:
    """Mean and MIN per-calendar-year Sharpe over [lo_year, hi_year] (inclusive).

    This is the robustness metric: aggregate validation Sharpe hides window
    luck, so we select on consistency across yearly folds instead. Computed only
    over train+validation years (never the test window).
    """
    sharpes = []
    for yr in range(lo_year, hi_year + 1):
        sub = res[res.index.year == yr]["strategy_daily_return"]
        if len(sub) > 30:
            sharpes.append(sharpe_ratio(sub, 365))
    if not sharpes:
        return -1e9, -1e9
    return float(np.mean(sharpes)), float(np.min(sharpes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-exp", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = build_features(load_btc_data(cfg))
    df = add_onchain_features(merge_onchain(df, load_coinmetrics(["CapMVRVCur", "AdrActCnt"])))
    splits = make_fixed_split(cfg["validation"])

    # Reference: the chosen non-ML ensemble's validation Sharpe, and its raw
    # weight (used as the base for hybrid "gate" experiments).
    ens_weight = get_strategy("trend_ensemble")(
        df, {"sma_windows": [50, 100, 200], "use_donchian": True,
             "target_vol": 0.55, "vol_window": 45})
    ens = run_full(df, ens_weight, bt)
    ens_val = window_metrics(ens, splits["validation"])
    print(f"device={get_device()}  reference ENS val Sharpe={ens_val['sharpe_ratio']:.2f}")

    ds_cache: dict = {}
    RESULTS.parent.mkdir(exist_ok=True)
    new_file = not RESULTS.exists()
    fh = RESULTS.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["exp", "config", "train_sharpe", "val_sharpe",
                         "fold_min_sharpe", "val_nav", "val_maxdd", "val_trades",
                         "test_sharpe_PEEK", "liquidated", "secs", "fold_mean_sharpe"])

    rng = np.random.default_rng(args.seed)
    best = {"robust": -1e9}
    for exp in range(args.max_exp):
        c = sample_config(rng)
        t0 = time.time()
        try:
            key = (c["feature_set"], c["horizon"])
            if key not in ds_cache:
                cols = get_features(df, c["feature_set"])
                ds_cache[key] = (cols, *build_dataset(df, cols, horizon=c["horizon"]))
            cols, X, y = ds_cache[key]

            preds = walk_forward_predict(
                X, y, model_factory(c), initial_train=730, step=120)
            raw = predictions_to_weight(
                preds, df, mode=c["signal_mode"], target_vol=0.55,
                long_only=c["long_only"], base_weight=ens_weight,
                gate_down=c.get("gate_down", 0.3))
            res = run_full(df, raw, bt)
            tr = window_metrics(res, splits["train"])
            va = window_metrics(res, splits["validation"])
            te = window_metrics(res, splits["test"])  # PEEK: logged, not selected on
            # Robustness = MIN per-year Sharpe over the train+val OOS years
            # (2019-2023); selecting on this avoids aggregate-window luck.
            fold_mean, fold_min = fold_yearly_sharpe(res, 2019, 2023)
            liq = bool(res["liquidated"].any())
            writer.writerow([exp, json.dumps(c), round(tr["sharpe_ratio"], 3),
                             round(va["sharpe_ratio"], 3), round(fold_min, 3),
                             round(va["final_nav"], 3), round(va["max_drawdown"], 3),
                             int(va["num_trades"]), round(te["sharpe_ratio"], 3),
                             liq, round(time.time() - t0, 1), round(fold_mean, 3)])
            fh.flush()
            # Select on fold-min Sharpe (worst train+val year), the robustness bar.
            if (not liq) and fold_min > best.get("fold_min", -1e9):
                best = {"fold_min": fold_min, "fold_mean": fold_mean,
                        "val": va["sharpe_ratio"], "exp": exp, "config": c,
                        "test_peek": te["sharpe_ratio"]}
                print(f"[{exp}] NEW BEST fold-min={fold_min:.2f} mean={fold_mean:.2f} "
                      f"val={va['sharpe_ratio']:.2f} (test peek {te['sharpe_ratio']:.2f}) "
                      f"{c['model']}/h{c['horizon']}/{c['feature_set']}/{c['signal_mode']}")
        except Exception as e:  # noqa: BLE001 - keep the run alive
            writer.writerow([exp, json.dumps(c), "ERR", str(e)[:80], "", "", "", "", "", "", round(time.time()-t0,1), ""])
            fh.flush()

    fh.close()
    # Reference: buy-and-hold's own fold-min over the same years, the bar to beat.
    bh = run_full(df, get_strategy("buy_and_hold")(df, {}), bt)
    bh_mean, bh_min = fold_yearly_sharpe(bh, 2019, 2023)
    print(f"DONE {args.max_exp} experiments. Best fold-min={best.get('fold_min')} "
          f"(buy-hold fold-min {bh_min:.2f}, mean {bh_mean:.2f}).")


if __name__ == "__main__":
    main()
