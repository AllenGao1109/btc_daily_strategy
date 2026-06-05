"""Factor library for autonomous factor mining.

Each factor is a function of the OHLCV (and optionally on-chain) frame returning
a per-day Series, using ONLY past/current data (rolling / expanding / known
structural constants) so no lookahead is introduced. The backtest engine still
lags any resulting signal by one day.

Factor families:
  - structural: BTC halving-cycle phase (a genuinely BTC-specific factor)
  - calendar:   day-of-week / month / turn-of-month effects
  - trend:      multi-horizon and risk-adjusted momentum, MA distances
  - meanrev:    RSI, Bollinger %b, distance from all-time high, z-scores
  - volatility: realized-vol levels, vol-of-vol, vol regime
  - distribution: rolling skew / kurtosis of returns
  - volume:     volume trend, Amihud illiquidity, OBV slope
  - onchain:    derived from CoinMetrics columns if present

The point of mining is to find factors whose forward-return information
coefficient (rank correlation with next-period returns) is stable out-of-sample.
Most will be noise; the harness measures which, if any, are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bitcoin halving dates (UTC). These are scheduled/structural, not lookahead.
HALVINGS = [
    pd.Timestamp("2012-11-28", tz="UTC"),
    pd.Timestamp("2016-07-09", tz="UTC"),
    pd.Timestamp("2020-05-11", tz="UTC"),
    pd.Timestamp("2024-04-20", tz="UTC"),
    pd.Timestamp("2028-04-01", tz="UTC"),  # approximate future halving
]
_HALVING_PERIOD_DAYS = 1458.0  # ~4 years between halvings


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI in [0, 100]."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full candidate-factor matrix from an OHLCV(/on-chain) frame.

    Args:
        df: Frame with open/high/low/close/volume (and optional on-chain cols).

    Returns:
        DataFrame of factors indexed like ``df``. Columns are prefixed by family.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    ret = close.pct_change()
    logret = np.log(close).diff()

    # --- structural: halving cycle ---
    days_since = pd.Series(index=df.index, dtype=float)
    for i, d in enumerate(df.index):
        prev = [h for h in HALVINGS if h <= d]
        days_since.iloc[i] = (d - prev[-1]).days if prev else np.nan
    out["halving_days_since"] = days_since
    out["halving_phase"] = (days_since / _HALVING_PERIOD_DAYS).clip(0, 1)
    # Sine/cosine encoding of cyclical phase (captures non-monotonic cycle shape).
    out["halving_sin"] = np.sin(2 * np.pi * out["halving_phase"])
    out["halving_cos"] = np.cos(2 * np.pi * out["halving_phase"])

    # --- calendar ---
    out["dow"] = df.index.dayofweek.astype(float)
    out["month"] = df.index.month.astype(float)
    out["turn_of_month"] = ((df.index.day <= 3) | (df.index.day >= 28)).astype(float)

    # --- trend / momentum (incl. risk-adjusted) ---
    for w in (5, 20, 60, 120, 200):
        out[f"mom_{w}"] = close.pct_change(w)
        vol_w = ret.rolling(w).std()
        out[f"sharpe_mom_{w}"] = close.pct_change(w) / (vol_w * np.sqrt(w))
    out["ma_dist_fast_slow"] = close.rolling(20).mean() / close.rolling(100).mean() - 1.0
    out["price_accel"] = close.pct_change(20) - close.pct_change(20).shift(20)

    # --- mean reversion ---
    out["rsi_14"] = _rsi(close, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bollinger_pctb"] = (close - bb_mid) / (2 * bb_std)
    ath = close.cummax()
    out["dist_from_ath"] = close / ath - 1.0
    out["zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std()

    # --- volatility ---
    rv = ret.rolling(30).std()
    out["rvol_30"] = rv
    out["vol_of_vol_30"] = rv.rolling(30).std()
    out["vol_regime"] = rv / ret.rolling(180).std()  # short vs long vol

    # --- return distribution ---
    out["skew_30"] = ret.rolling(30).skew()
    out["kurt_30"] = ret.rolling(30).kurt()
    out["autocorr_20"] = ret.rolling(40).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # --- volume / liquidity ---
    out["vol_mom_20"] = vol.pct_change(20)
    out["amihud_20"] = (ret.abs() / vol.replace(0.0, np.nan)).rolling(20).mean()
    obv = (np.sign(ret).fillna(0.0) * vol).cumsum()
    out["obv_slope_20"] = obv.diff(20)

    # --- on-chain (only if the columns are present) ---
    if "CapMVRVCur" in df.columns:
        mvrv = df["CapMVRVCur"].astype(float)
        out["mvrv_level"] = mvrv
        out["mvrv_z_365"] = (mvrv - mvrv.rolling(365, min_periods=60).mean()) / mvrv.rolling(
            365, min_periods=60
        ).std()
        out["mvrv_mom_30"] = mvrv.pct_change(30)
    if "AdrActCnt" in df.columns:
        adr = df["AdrActCnt"].astype(float)
        out["adr_growth_30"] = adr.pct_change(30)
        out["adr_z_90"] = (adr - adr.rolling(90).mean()) / adr.rolling(90).std()
    # NVT-style: market cap / transaction-volume proxy, if available.
    if "CapMrktCurUSD" in df.columns and "TxTfrValAdjUSD" in df.columns:
        out["nvt"] = df["CapMrktCurUSD"].astype(float) / df["TxTfrValAdjUSD"].astype(
            float
        ).rolling(30).mean()

    return out


def composite_signal(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    factor_names: list[str],
    horizon: int = 20,
    min_periods: int = 365,
) -> pd.Series:
    """Build a deterministic long-bias composite score from robust factors.

    Each factor is z-scored with EXPANDING train-and-past stats (no lookahead),
    sign-aligned by its information coefficient measured on the TRAIN window only
    (so neither validation nor test informs the weights), and averaged. The
    result is a per-day score in roughly [-1, 1] after a tanh squash.

    Args:
        df: OHLCV(/on-chain) frame.
        train_end: Last date whose data may inform factor IC signs.
        factor_names: Robust factors to combine.
        horizon: Forward-return horizon used to measure each factor's IC sign.
        min_periods: Minimum history before a z-score is defined.

    Returns:
        Composite score Series aligned to ``df.index`` (NaN during warmup).
    """
    factors = build_factors(df)
    fr = forward_return(df, horizon)
    train_mask = df.index <= train_end

    parts, weights = [], []
    for name in factor_names:
        if name not in factors.columns:
            continue
        f = factors[name].replace([np.inf, -np.inf], np.nan)
        ic = information_coefficient(f[train_mask], fr[train_mask])
        if ic == 0.0:
            continue
        # Expanding standardization uses only past data (causal).
        mu = f.expanding(min_periods=min_periods).mean()
        sd = f.expanding(min_periods=min_periods).std()
        z = ((f - mu) / sd).clip(-3, 3)
        parts.append(np.sign(ic) * z)
        weights.append(abs(ic))  # weight each factor by its train IC magnitude

    if not parts:
        return pd.Series(np.nan, index=df.index, name="composite")
    W = np.asarray(weights)
    W = W / W.sum()
    score = sum(w * p for w, p in zip(W, parts))
    return np.tanh(score).rename("composite")


def forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Forward ``horizon``-day return aligned to each day t (known at t+horizon)."""
    close = df["close"].astype(float)
    return close.shift(-horizon) / close - 1.0


def information_coefficient(
    factor: pd.Series, fwd_ret: pd.Series, method: str = "spearman"
) -> float:
    """Rank correlation (default Spearman IC) between a factor and forward return.

    Args:
        factor: Factor values.
        fwd_ret: Forward returns aligned to ``factor``.
        method: Correlation method (``spearman`` or ``pearson``).

    Returns:
        The IC over rows where both are defined (0.0 if insufficient data).
    """
    joined = pd.concat([factor, fwd_ret], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(joined) < 60:
        return 0.0
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]
    if method == "spearman":
        # Spearman == Pearson on ranks; avoids the scipy dependency.
        a, b = a.rank(), b.rank()
    std_a, std_b = a.std(), b.std()
    if std_a == 0 or std_b == 0:
        return 0.0
    return float(a.corr(b, method="pearson"))
