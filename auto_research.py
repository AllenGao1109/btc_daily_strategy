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
from src.research import run_full, window_metrics
from src.strategies import get_strategy
from src.validation import make_fixed_split
from src.ml.dataset import add_onchain_features, build_dataset, select_features
from src.ml.torch_models import TorchMLP, get_device
from src.ml.walkforward import predictions_to_weight, walk_forward_predict

RESULTS = Path("results/ml_results.csv")

SEARCH_SPACE = {
    "horizon": [1, 3, 5, 10, 20],
    "feature_set": ["base", "base_onchain", "trend", "vol_mom"],
    "model": ["logistic", "mlp_small", "mlp_mid"],
    "task": ["classification", "regression"],
    "dropout": [0.2, 0.3, 0.5],
    "weight_decay": [1e-3, 3e-3, 1e-2],
    "epochs": [60, 100],
    "n_seeds": [1, 3],
    "signal_mode": ["vol_target", "sign"],
    "long_only": [True, False],
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

    # Reference: the chosen non-ML ensemble's validation Sharpe.
    ens = run_full(df, get_strategy("trend_ensemble")(
        df, {"sma_windows": [50, 100, 200], "use_donchian": True,
             "target_vol": 0.55, "vol_window": 45}), bt)
    ens_val = window_metrics(ens, splits["validation"])
    print(f"device={get_device()}  reference ENS val Sharpe={ens_val['sharpe_ratio']:.2f}")

    ds_cache: dict = {}
    RESULTS.parent.mkdir(exist_ok=True)
    new_file = not RESULTS.exists()
    fh = RESULTS.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["exp", "config", "train_sharpe", "val_sharpe",
                         "robust_sharpe", "val_nav", "val_maxdd", "val_trades",
                         "test_sharpe_PEEK", "liquidated", "secs"])

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
                long_only=c["long_only"])
            res = run_full(df, raw, bt)
            tr = window_metrics(res, splits["train"])
            va = window_metrics(res, splits["validation"])
            te = window_metrics(res, splits["test"])  # PEEK: logged, not selected on
            robust = min(tr["sharpe_ratio"], va["sharpe_ratio"])
            liq = bool(res["liquidated"].any())
            writer.writerow([exp, json.dumps(c), round(tr["sharpe_ratio"], 3),
                             round(va["sharpe_ratio"], 3), round(robust, 3),
                             round(va["final_nav"], 3), round(va["max_drawdown"], 3),
                             int(va["num_trades"]), round(te["sharpe_ratio"], 3),
                             liq, round(time.time() - t0, 1)])
            fh.flush()
            if (not liq) and va["sharpe_ratio"] > best.get("val", -1e9):
                best = {"val": va["sharpe_ratio"], "robust": robust, "exp": exp, "config": c,
                        "test_peek": te["sharpe_ratio"]}
                print(f"[{exp}] NEW BEST val Sharpe={va['sharpe_ratio']:.2f} "
                      f"(test peek {te['sharpe_ratio']:.2f}) {c['model']}/{c['task']}/"
                      f"h{c['horizon']}/{c['feature_set']}")
        except Exception as e:  # noqa: BLE001 - keep the run alive
            writer.writerow([exp, json.dumps(c), "ERR", str(e)[:80], "", "", "", "", "", "", round(time.time()-t0,1)])
            fh.flush()

    fh.close()
    print(f"DONE {args.max_exp} experiments. Best val Sharpe={best.get('val')} "
          f"(ENS ref {ens_val['sharpe_ratio']:.2f}).")


if __name__ == "__main__":
    main()
