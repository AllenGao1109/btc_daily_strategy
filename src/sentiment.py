"""Sentiment-index data loaders: crypto Fear & Greed and CNN (equity) Fear & Greed.

Sources:
  - alternative.me crypto Fear & Greed index (0-100, daily since 2018-02-01).
    Primary: the official free API. Fallback: a public GitHub research archive
    of the same series, for environments where the API host is unreachable.
  - CNN Business Fear & Greed index for US equities (0-100, trading days since
    2011). CNN exposes no full-history endpoint, so the canonical source here is
    the community-maintained archive (whit3rabbit/fear-greed-data) that splices
    the historical record with the live CNN endpoint and is refreshed daily.

Both are *descriptive sentiment as of day t's value date*. As with on-chain
data, merge with :func:`src.onchain.merge_onchain` (forward-fill of past values
only) and let the engine's 1-day execution lag do the rest — no lookahead.

Timing note (CNN): the equity index is stamped on US trading days and its
closing value exists by ~21:00 UTC, before the BTC 00:00 UTC daily close, so
using the day-t value to position on day t+1 is conservative by construction.

Standard library only (urllib + json); results cached under data/raw/.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FNG_API = "https://api.alternative.me/fng/?limit=0&format=json"
# Same alternative.me series, archived in a public research repo (refreshed
# through ~the repo's last update; used only when the API host is blocked).
FNG_MIRROR = (
    "https://raw.githubusercontent.com/moltovy/"
    "Crypto-Research-Paper-Data-Factors-Analysis-/main/"
    "Data/AlternativeMe/fear_greed_index__daily.csv"
)
CNN_ARCHIVE = (
    "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/"
    "fear-greed.csv"
)


def _fetch(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def load_crypto_fear_greed(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the alternative.me crypto Fear & Greed index, cached, UTC-indexed.

    Args:
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.

    Returns:
        DataFrame indexed by UTC midnight date with one float column
        ``fng_value`` (0-100; low = fear).
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "crypto_fear_greed.csv"
    if cache.exists() and not force_reload:
        return _read_cached(cache, "fng_value")

    try:
        payload = json.loads(_fetch(FNG_API).decode("utf-8"))
        rows = payload.get("data", [])
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [int(r["timestamp"]) for r in rows], unit="s", utc=True
                ).normalize(),
                "fng_value": [float(r["value"]) for r in rows],
            }
        ).set_index("date")
    except Exception:  # noqa: BLE001 - fall back to the archived series
        raw = pd.read_csv(io.BytesIO(_fetch(FNG_MIRROR)))
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["date"], utc=True).dt.normalize(),
                "fng_value": pd.to_numeric(raw["fng_value"], errors="coerce"),
            }
        ).set_index("date")

    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    if df.empty:
        raise RuntimeError("Crypto Fear & Greed loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df


def load_cnn_fear_greed(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the CNN (US equity) Fear & Greed index archive, cached, UTC-indexed.

    The series is stamped on US trading days only; weekend/holiday gaps are the
    caller's concern (``merge_onchain`` forward-fills past values, causally).

    Args:
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.

    Returns:
        DataFrame indexed by UTC midnight date with one float column
        ``cnn_fg`` (0-100; low = fear).
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "cnn_fear_greed.csv"
    if cache.exists() and not force_reload:
        return _read_cached(cache, "cnn_fg")

    raw = pd.read_csv(io.BytesIO(_fetch(CNN_ARCHIVE)))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], utc=True).dt.normalize(),
            "cnn_fg": pd.to_numeric(raw["Fear Greed"], errors="coerce"),
        }
    ).set_index("date")
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    if df.empty:
        raise RuntimeError("CNN Fear & Greed loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df


def _read_cached(cache: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(cache, index_col=0)[[column]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "date"
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
