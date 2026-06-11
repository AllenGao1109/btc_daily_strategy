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

    # --- cross-crypto (only if an ETH reference column is present) ---
    if "eth_close" in df.columns:
        eth = df["eth_close"].astype(float)
        out["btc_eth_rs_30"] = close.pct_change(30) - eth.pct_change(30)
        out["btc_dominance_mom"] = (close / eth).pct_change(30)

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
    # Exchange flows: coins moving TO exchanges = sell pressure (bearish);
    # OUT = accumulation (bullish). Net flow is a classic positioning signal.
    if "FlowInExUSD" in df.columns and "FlowOutExUSD" in df.columns:
        net = df["FlowInExUSD"].astype(float) - df["FlowOutExUSD"].astype(float)
        out["ex_netflow_z90"] = (net - net.rolling(90).mean()) / net.rolling(90).std()
        out["ex_netflow_mom30"] = net.rolling(30).mean() - net.rolling(90).mean()
        if "CapMrktCurUSD" in df.columns:
            out["ex_netflow_to_mcap"] = net.rolling(7).mean() / df["CapMrktCurUSD"].astype(float)
    if "AdrBalCnt" in df.columns:
        bal = df["AdrBalCnt"].astype(float)
        out["holders_growth_30"] = bal.pct_change(30)
        out["holders_z_90"] = (bal - bal.rolling(90).mean()) / bal.rolling(90).std()
    if "TxCnt" in df.columns:
        tx = df["TxCnt"].astype(float)
        out["tx_growth_30"] = tx.pct_change(30)
        out["tx_z_90"] = (tx - tx.rolling(90).mean()) / tx.rolling(90).std()

    # --- miner economics (only if the columns are present) ---
    # Hash ribbons (30/60d hash-rate MA cross: miner capitulation/recovery) and
    # the Puell multiple (issuance value vs its own 1y mean: cycle top/bottom).
    if "HashRate" in df.columns:
        hr = df["HashRate"].astype(float)
        out["hash_ribbon_3060"] = hr.rolling(30).mean() / hr.rolling(60).mean() - 1.0
        out["hash_growth_60"] = hr.rolling(7).mean().pct_change(60)
    if "IssTotUSD" in df.columns:
        iss = df["IssTotUSD"].astype(float)
        out["puell_365"] = iss / iss.rolling(365, min_periods=180).mean()

    # --- exchange supply stock (complements the flow factors above) ---
    if "SplyExNtv" in df.columns and "SplyCur" in df.columns:
        exr = df["SplyExNtv"].astype(float) / df["SplyCur"].astype(float)
        out["exsply_ratio"] = exr
        out["exsply_z_180"] = (exr - exr.rolling(180).mean()) / exr.rolling(180).std()
        out["exsply_chg_30"] = exr.diff(30)

    # --- stablecoin liquidity ---
    # SSR (BTC mcap / stablecoin mcap): low = lots of dry powder vs BTC. Supply
    # growth = net stablecoin issuance, a crypto-native liquidity inflow proxy.
    if "stable_mcap_usd" in df.columns:
        stab = df["stable_mcap_usd"].astype(float)
        out["stable_growth_30"] = stab.pct_change(30)
        out["stable_growth_90"] = stab.pct_change(90)
        if "CapMrktCurUSD" in df.columns:
            ssr = df["CapMrktCurUSD"].astype(float) / stab
            out["ssr_z_365"] = (ssr - ssr.rolling(365, min_periods=180).mean()) / ssr.rolling(
                365, min_periods=180
            ).std()
            out["ssr_mom_30"] = ssr.pct_change(30)

    # --- macro risk appetite / liquidity (see src.macro for release-lag rules) ---
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"].astype(float)
        out["vix_level"] = vix
        out["vix_z_60"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    if "DGS10" in df.columns:
        out["dgs10_chg_60"] = df["DGS10"].astype(float).diff(60)
    if "DFII10" in df.columns:
        out["real10_chg_60"] = df["DFII10"].astype(float).diff(60)
    if "T10Y2Y" in df.columns:
        out["curve_t10y2y"] = df["T10Y2Y"].astype(float)
    if "BAMLH0A0HYM2" in df.columns:
        hy = df["BAMLH0A0HYM2"].astype(float)
        out["hyoas_z_60"] = (hy - hy.rolling(60).mean()) / hy.rolling(60).std()
        out["hyoas_chg_20"] = hy.diff(20)
    if "RRPONTSYD" in df.columns:
        out["rrp_chg_30"] = df["RRPONTSYD"].astype(float).diff(30)
    if "dxy_close" in df.columns:
        out["dxy_mom_60"] = df["dxy_close"].astype(float).pct_change(60)

    # --- on-chain behavior (CryptoQuant archive; see src.cryptoquant for the
    # timing/leakage audit: these must also pass the extra-lag check) ---
    if "asopr" in df.columns:  # realized profit ratio of moved coins
        asopr = df["asopr"].astype(float)
        out["asopr_30"] = asopr.rolling(30).mean() - 1.0
        out["asopr_z_90"] = (asopr - asopr.rolling(90).mean()) / asopr.rolling(90).std()
    if "sth_sopr" in df.columns:  # short-term holders' realized P/L (capitulation)
        sth = df["sth_sopr"].astype(float)
        out["sth_sopr_30"] = sth.rolling(30).mean() - 1.0
        out["sth_sopr_z_90"] = (sth - sth.rolling(90).mean()) / sth.rolling(90).std()
    if "lth_sopr" in df.columns:  # old-coin profit taking (cycle distribution)
        lth = df["lth_sopr"].astype(float)
        out["lth_sopr_z_365"] = (lth - lth.rolling(365, min_periods=180).mean()) / lth.rolling(
            365, min_periods=180
        ).std()
    if "taker_cvd" in df.columns:  # 90d cumulative taker buy-sell delta (order flow)
        cvd = df["taker_cvd"].astype(float)
        out["cvd_chg_30"] = cvd.diff(30)
        out["cvd_z_180"] = (cvd - cvd.rolling(180).mean()) / cvd.rolling(180).std()
    if "whale_ratio" in df.columns:  # top-10 inflows / total inflows (who is selling)
        wr = df["whale_ratio"].astype(float)
        out["whale_90"] = wr.rolling(90).mean()
        out["whale_z_180"] = (wr - wr.rolling(180).mean()) / wr.rolling(180).std()
    if "inflow_cdd" in df.columns:  # coin-age destroyed by exchange inflows (old coins selling)
        icdd = np.log1p(df["inflow_cdd"].astype(float))
        out["icdd_z_180"] = (icdd - icdd.rolling(180).mean()) / icdd.rolling(180).std()
    if "miner_to_ex" in df.columns:  # miner selling pressure (direct, unlike hash ribbons)
        m2e = np.log1p(df["miner_to_ex"].astype(float))
        out["m2e_z_180"] = (m2e - m2e.rolling(180).mean()) / m2e.rolling(180).std()
    if "cme_basis" in df.columns:  # futures carry / institutional positioning
        basis = df["cme_basis"].astype(float)
        out["basis_level"] = basis
        out["basis_z_90"] = (basis - basis.rolling(90).mean()) / basis.rolling(90).std()

    # --- sentiment (only if the columns are present) ---
    # Fear & Greed indexes are 0-100 composites; low = fear. The level tests the
    # contrarian "buy fear / sell greed" hypothesis; z-score and momentum test
    # sentiment *shifts*; the extreme flags isolate the tails. Flags are masked
    # to NaN where the underlying index has no coverage yet (no fabricated 0s).
    if "fng_value" in df.columns:  # crypto F&G (alternative.me, 2018+)
        fng = df["fng_value"].astype(float)
        out["fng_level"] = fng
        out["fng_z_60"] = (fng - fng.rolling(60).mean()) / fng.rolling(60).std()
        out["fng_mom_10"] = fng.diff(10)
        out["fng_extreme_fear"] = (fng <= 25).astype(float).where(fng.notna())
        out["fng_extreme_greed"] = (fng >= 75).astype(float).where(fng.notna())
    if "cnn_fg" in df.columns:  # CNN US-equity F&G (2011+), cross-asset risk appetite
        cnn = df["cnn_fg"].astype(float)
        out["cnnfg_level"] = cnn
        out["cnnfg_z_60"] = (cnn - cnn.rolling(60).mean()) / cnn.rolling(60).std()
        out["cnnfg_mom_10"] = cnn.diff(10)
        out["cnnfg_extreme_fear"] = (cnn <= 25).astype(float).where(cnn.notna())
        out["cnnfg_extreme_greed"] = (cnn >= 75).astype(float).where(cnn.notna())

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
