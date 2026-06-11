"""Bootstrap data/raw caches from GitHub mirrors when primary APIs are blocked.

Some execution environments (CI sandboxes, restricted networks) cannot reach
CryptoCompare/CoinMetrics/alternative.me APIs but CAN reach
raw.githubusercontent.com. This script builds every cache the research scripts
need from public mirrors:

  1. BTC daily "OHLCV" from CoinMetrics' daily GitHub CSV dump
     (https://github.com/coinmetrics/data). CoinMetrics community data has a
     daily close (PriceUSD, UTC) and reported spot volume but NO intraday
     high/low. We synthesize:
         open_t = close_{t-1}   (exact for a 24/7 market on UTC-midnight bars)
         high_t = max(open_t, close_t)
         low_t  = min(open_t, close_t)
     The high/low are therefore LOWER BOUNDS on the true range. Everything in
     the engine and the factor library uses close/volume only; the only
     consumers of high/low are the Donchian channels (donchian_breakout,
     trend_ensemble), which degrade to close-based channels. This is an
     explicit, opt-in approximation — running this script is the opt-in.

  2. The CoinMetrics on-chain metric cache (same numbers as the community API;
     the GitHub dump is published by CoinMetrics itself).

  3. Crypto Fear & Greed (alternative.me archive) and CNN equity Fear & Greed
     (community archive) sentiment caches.

Run:  python3 fetch_mirror_data.py
"""

from __future__ import annotations

import urllib.request

import pandas as pd

from src.crossasset import load_crossasset
from src.cryptoquant import load_cme_basis, load_cryptoquant
from src.data import PROCESSED_DIR, RAW_DIR, clean_data
from src.macro import load_dxy, load_fred_macro
from src.onchain import (
    CM_GITHUB_MIRROR,
    ONCHAIN_METRICS,
    load_coinmetrics,
    load_stablecoin_mcap,
)
from src.sentiment import load_cnn_fear_greed, load_crypto_fear_greed


def build_btc_ohlcv_cache(start: str = "2016-12-01") -> pd.DataFrame:
    """Build data/raw/BTC-USD_1d.csv from the CoinMetrics GitHub daily dump."""
    url = CM_GITHUB_MIRROR.format(asset="btc")
    req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        cm = pd.read_csv(resp, usecols=["time", "PriceUSD", "volume_reported_spot_usd_1d"])
    cm["date"] = pd.to_datetime(cm["time"], utc=True).dt.normalize()
    cm = cm.set_index("date").sort_index()
    cm = cm[cm.index >= pd.Timestamp(start, tz="UTC")]

    close = pd.to_numeric(cm["PriceUSD"], errors="coerce")
    volume = pd.to_numeric(cm["volume_reported_spot_usd_1d"], errors="coerce")
    df = pd.DataFrame(index=cm.index)
    df["close"] = close
    df["open"] = close.shift(1)
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = volume
    df = df.dropna(subset=["open", "close"])
    df = df[["open", "high", "low", "close", "volume"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "BTC-USD_1d.csv")
    # Drop any stale processed cache so load_btc_data re-cleans from this raw.
    processed = PROCESSED_DIR / "BTC-USD_1d.csv"
    if processed.exists():
        processed.unlink()
    return clean_data(df)


def main() -> None:
    btc = build_btc_ohlcv_cache()
    print(
        f"BTC close/volume (CoinMetrics mirror): {btc.index.min().date()} -> "
        f"{btc.index.max().date()}  ({len(btc)} rows; OHLC synthesized from "
        "close — see module docstring)"
    )
    oc = load_coinmetrics(ONCHAIN_METRICS, force_reload=True)
    print(f"On-chain metrics: {oc.index.min().date()} -> {oc.index.max().date()}")
    fng = load_crypto_fear_greed(force_reload=True)
    print(f"Crypto F&G: {fng.index.min().date()} -> {fng.index.max().date()}")
    cnn = load_cnn_fear_greed(force_reload=True)
    print(f"CNN equity F&G: {cnn.index.min().date()} -> {cnn.index.max().date()}")
    stab = load_stablecoin_mcap(force_reload=True)
    print(f"Stablecoin mcap: {stab.index.min().date()} -> {stab.index.max().date()}")
    fred = load_fred_macro(force_reload=True)
    print(f"FRED macro: {fred.index.min().date()} -> {fred.index.max().date()}")
    dxy = load_dxy(force_reload=True)
    print(f"DXY: {dxy.index.min().date()} -> {dxy.index.max().date()}")
    cq = load_cryptoquant(force_reload=True)
    print(f"CryptoQuant behavior: {cq.index.min().date()} -> {cq.index.max().date()}")
    cb = load_cme_basis(force_reload=True)
    print(f"CME basis: {cb.index.min().date()} -> {cb.index.max().date()}")
    ca = load_crossasset(force_reload=True)
    print(f"Cross-asset (JPY/ARKK/QQQ/miners): {ca.index.min().date()} -> {ca.index.max().date()}")


if __name__ == "__main__":
    main()
