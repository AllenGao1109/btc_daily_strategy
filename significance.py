"""Statistical significance of the composite's edge over buy-and-hold.

Two questions, answered honestly:

1. SAMPLING NOISE — is the observed Sharpe edge distinguishable from zero on
   this window? Paired circular block bootstrap of daily returns (both series
   resampled with the SAME blocks, preserving their correlation and most
   autocorrelation; block length ~sqrt(T)). Reports the bootstrap CI and the
   one-sided p-value for edge <= 0, per window.

2. SELECTION BIAS — after ~1,300 ML configs, 73 factors and dozens of
   composite variants, how high a Sharpe would the BEST trial show under the
   null of no skill? Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014):
   benchmark SR0 = E[max SR over N independent null trials], then
   PSR(SR0) with skew/kurtosis-adjusted SR variance. N and the cross-trial SR
   dispersion are assumptions, so we report DSR over a GRID of (N, sigma_SR)
   rather than pretending one number is right.

Deterministic: fixed bootstrap seed (results are reproducible, the seed is
part of the method statement). Run:  python3 significance.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import BacktestConfig, load_config
from src.research import load_research_frame, run_full
from src.strategies import get_strategy
from src.strategies.factor_composite import DEFAULT_FACTORS
from src.validation import make_fixed_split

SEED = 20260611
B = 10_000  # bootstrap draws


def ann_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def paired_block_bootstrap(
    a: np.ndarray, b: np.ndarray, n_draws: int = B, seed: int = SEED
) -> np.ndarray:
    """Bootstrap distribution of Sharpe(a) - Sharpe(b), circular blocks."""
    rng = np.random.default_rng(seed)
    t = len(a)
    block = max(10, int(round(np.sqrt(t))))
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, t, size=(n_draws, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_draws, -1)[:, :t] % t
    return np.array([ann_sharpe(a[i]) - ann_sharpe(b[i]) for i in idx])


def normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    from math import erf
    vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / np.sqrt(2.0))))
    return vec(z)


def normal_ppf(p: float) -> float:
    # Acklam rational approximation is overkill; scipy is available via deps.
    from scipy.stats import norm
    return float(norm.ppf(p))


def deflated_sharpe(
    sr_ann: float, t: int, skew: float, kurt: float, n_trials: int, sr_std_ann: float
) -> tuple[float, float]:
    """(SR0_annualized, DSR probability) per Bailey & Lopez de Prado (2014).

    SR inputs/outputs annualized; internally converted to daily units.
    ``kurt`` is the PEARSON kurtosis (normal = 3). ``sr_std_ann`` is the
    cross-trial dispersion of annualized Sharpe under the null.
    """
    gamma = 0.5772156649015329
    sr_std = sr_std_ann / np.sqrt(365)
    sr0 = sr_std * (
        (1 - gamma) * normal_ppf(1 - 1.0 / n_trials)
        + gamma * normal_ppf(1 - 1.0 / (n_trials * np.e))
    )
    sr = sr_ann / np.sqrt(365)
    denom = np.sqrt((1 - skew * sr + (kurt - 1) / 4.0 * sr**2) / (t - 1))
    dsr = float(normal_cdf((sr - sr0) / denom))
    return float(sr0 * np.sqrt(365)), dsr


def main() -> None:
    cfg = load_config("config.yaml")
    bt = BacktestConfig.from_config(cfg)
    df = load_research_frame(cfg)
    splits = make_fixed_split(cfg["validation"])

    prod_params = dict(cfg["strategy"]["params"])
    base_factors = [f for f in DEFAULT_FACTORS if f != "cnnfg_z_60"]
    runs = {
        "PROD+S": run_full(df, get_strategy("factor_composite")(df, prod_params), bt),
        "PROD": run_full(df, get_strategy("factor_composite")(
            df, {**prod_params, "factors": base_factors}), bt),
        "BH": run_full(df, get_strategy("buy_and_hold")(df, {}), bt),
    }

    print(f"Paired circular block bootstrap, B={B}, seed={SEED} "
          "(block ~ sqrt(T) days)\n")
    pairs = [("PROD+S", "BH"), ("PROD+S", "PROD")]
    windows = ["validation", "test", None]
    for a_name, b_name in pairs:
        print(f"=== Sharpe({a_name}) - Sharpe({b_name}) ===")
        for w in windows:
            ra, rb = runs[a_name], runs[b_name]
            if w is not None:
                win = splits[w]
                ra, rb = win.slice(ra), win.slice(rb)
            a = ra["strategy_daily_return"].astype(float).to_numpy()
            b = rb["strategy_daily_return"].astype(float).to_numpy()
            obs = ann_sharpe(a) - ann_sharpe(b)
            dist = paired_block_bootstrap(a, b)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            # One-sided p: how often a zero-or-worse edge appears under the
            # bootstrap centered at the observed estimate -> use shift method.
            p = float((dist - dist.mean() + 0.0 >= obs).mean()) if obs > 0 else 1.0
            label = w or "full"
            print(f"  {label:10s} T={len(a):4d}  edge {obs:+5.2f}  "
                  f"95% CI [{lo:+5.2f}, {hi:+5.2f}]  p(<=0) {p:.3f}"
                  f"  {'*' if lo > 0 else ''}")
        print()

    # Deflated Sharpe of the headline test-window result, over an assumption grid.
    te = splits["test"].slice(runs["PROD+S"])
    x = te["strategy_daily_return"].astype(float)
    sr = ann_sharpe(x.to_numpy())
    skew, kurt = float(x.skew()), float(x.kurt()) + 3.0  # pandas kurt is excess
    print(f"=== Deflated Sharpe of PROD+S on test (SR {sr:.2f}, T={len(x)}, "
          f"skew {skew:.2f}, kurt {kurt:.1f}) ===")
    print(f"{'N trials':>9s} {'sigma_SR':>9s} {'SR0 (ann)':>10s} {'DSR':>6s}")
    for n_trials in [100, 500, 1500]:
        for sr_std in [0.3, 0.5]:
            sr0, dsr = deflated_sharpe(sr, len(x), skew, kurt, n_trials, sr_std)
            print(f"{n_trials:9d} {sr_std:9.1f} {sr0:10.2f} {dsr:6.2f}")
    print("\nReading: DSR is P[true SR > 0] after penalizing the search. "
          "N=1500/sigma=0.5 is the conservative cell (every config ever tried,"
          " wide null dispersion); N=100/sigma=0.3 treats only structurally "
          "distinct strategies as trials.")


if __name__ == "__main__":
    main()
