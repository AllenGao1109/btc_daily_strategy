"""On-chain / positioning data loaders (free sources, no API key).

Sources:
  - CoinMetrics community API: daily asset metrics such as ``CapMVRVCur`` (the
    MVRV valuation ratio) and ``AdrActCnt`` (active addresses). Full BTC history.
  - OKX public API: BTC perpetual funding-rate history (positioning sentiment),
    available from when the swap launched (~2019).

These provide signals orthogonal to price-trend. As with price data, the engine
lags all signals by one day, so using a metric dated day t to position on day
t+1 introduces no lookahead. CoinMetrics publishes a metric for day t after that
day closes, consistent with the 1-day execution lag.

Standard library only (urllib + json); results cached under data/raw/.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CM_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def load_coinmetrics(
    metrics: list[str],
    asset: str = "btc",
    start_date: str = "2017-01-01",
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch daily CoinMetrics community metrics, cached, UTC-indexed.

    Args:
        metrics: Metric names, e.g. ``["CapMVRVCur", "AdrActCnt"]``.
        asset: Asset id (default ``btc``).
        start_date: Earliest date (ISO).
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.

    Returns:
        DataFrame indexed by UTC midnight date with one float column per metric.
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"coinmetrics_{asset}_{'-'.join(metrics)}.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    rows: list[dict] = []
    url = (
        f"{CM_BASE}?assets={asset}&metrics={','.join(metrics)}"
        f"&frequency=1d&page_size=10000&start_time={start_date}"
    )
    while url:
        with urllib.request.urlopen(url, timeout=40) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"CoinMetrics returned no data for {metrics}.")
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    df = df.drop(columns=["time", "asset"]).set_index("date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(cache)
    return df


def load_okx_funding(
    inst_id: str = "BTC-USD-SWAP",
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
    max_pages: int = 200,
) -> pd.Series:
    """Fetch OKX perpetual funding-rate history as a daily-summed Series.

    OKX returns ~8h funding settlements; we sum them per UTC day to get a daily
    funding cost/sentiment series.

    Args:
        inst_id: OKX instrument id.
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.
        max_pages: Pagination safety cap.

    Returns:
        Daily funding-rate Series (UTC-indexed), name ``funding_rate``.
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"okx_funding_{inst_id}.csv"
    if cache.exists() and not force_reload:
        s = pd.read_csv(cache, index_col=0)["funding_rate"]
        s.index = pd.to_datetime(s.index, utc=True)
        s.index.name = "date"
        return s

    base = "https://www.okx.com/api/v5/public/funding-rate-history"
    rows: list[dict] = []
    after = None
    for _ in range(max_pages):
        url = f"{base}?instId={inst_id}&limit=100"
        if after is not None:
            url += f"&after={after}"
        with urllib.request.urlopen(url, timeout=40) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        batch = payload.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        after = batch[-1]["fundingTime"]
        if len(batch) < 100:
            break

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    daily = df.set_index("date")["funding_rate"].resample("1D").sum().sort_index()
    daily.name = "funding_rate"
    daily.to_csv(cache)
    return daily


def merge_onchain(df: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    """Left-join on-chain columns onto a price frame, forward-filling gaps.

    On-chain metrics can lag a day or have occasional gaps; forward-filling
    carries the last *known* value (never a future value), preserving causality.

    Args:
        df: Price/feature frame (UTC DatetimeIndex).
        onchain: On-chain frame to merge in.

    Returns:
        A copy of ``df`` with on-chain columns added (ffilled).
    """
    out = df.copy()
    joined = onchain.reindex(out.index.union(onchain.index)).ffill()
    for col in onchain.columns:
        out[col] = joined[col].reindex(out.index)
    return out
