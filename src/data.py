"""Daily BTC OHLCV data loading, caching, and validation.

Data flow:
  raw download (ccxt)  ->  data/raw/<symbol>_<tf>.csv
  cleaned/validated    ->  data/processed/<symbol>_<tf>.csv

The loader is deterministic: if a processed cache exists it is reused unless
``force_reload=True``. ccxt is optional; if it is unavailable or offline you can
either point at a local CSV or use :func:`generate_synthetic_btc` for testing.

We do NOT forward-fill prices. Missing days are reported, not silently patched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _cache_name(symbol: str, timeframe: str) -> str:
    """Build a filesystem-safe cache filename from symbol and timeframe."""
    safe_symbol = symbol.replace("/", "-")
    return f"{safe_symbol}_{timeframe}.csv"


def load_btc_data(
    config: dict[str, Any],
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load daily BTC OHLCV data with deterministic caching and validation.

    Args:
        config: Parsed config dict (uses the ``data`` section).
        force_reload: If True, ignore the processed cache and re-download.
        raw_dir: Override for the raw cache directory.
        processed_dir: Override for the processed cache directory.

    Returns:
        DataFrame indexed by a UTC ``DatetimeIndex`` with columns
        ``[open, high, low, close, volume]``, sorted ascending by date.
    """
    data_cfg = config.get("data", {})
    symbol = data_cfg.get("symbol", "BTC/USD")
    timeframe = data_cfg.get("timeframe", "1d")
    exchange = data_cfg.get("exchange", "coinbase")
    source = data_cfg.get("source", "ccxt")
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")

    raw_dir = raw_dir or RAW_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_path = processed_dir / _cache_name(symbol, timeframe)
    if processed_path.exists() and not force_reload:
        df = _read_cache(processed_path)
    else:
        raw_path = raw_dir / _cache_name(symbol, timeframe)
        if raw_path.exists() and not force_reload:
            df = _read_cache(raw_path)
        else:
            if source == "cryptocompare":
                df = _download_cryptocompare(symbol, start_date)
            else:
                df = _download_ccxt(exchange, symbol, timeframe, start_date)
            df.to_csv(raw_path)
        df = clean_data(df)
        df.to_csv(processed_path)

    df = _slice_dates(df, start_date, end_date)
    validate_data(df)
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    """Read a cached CSV back into a UTC-indexed OHLCV frame."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "date"
    return df


def _slice_dates(
    df: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Restrict the frame to the configured [start_date, end_date] window."""
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
    return df


def _download_cryptocompare(
    symbol: str, start_date: str | None, *, page_limit: int = 2000
) -> pd.DataFrame:
    """Download daily OHLCV from CryptoCompare's free histoday endpoint.

    Uses only the Python standard library (urllib + json), paginating backwards
    via ``toTs`` until ``start_date`` is reached. Returns USD-quoted OHLC with
    BTC-denominated volume. No API key required.

    Args:
        symbol: Pair like ``"BTC/USD"``; split into fsym/tsym.
        start_date: Earliest date to fetch (ISO string) or None for ~max history.
        page_limit: Candles per request (CryptoCompare allows up to 2000).

    Returns:
        Raw OHLCV DataFrame indexed by UTC DatetimeIndex.
    """
    import json
    import urllib.request

    fsym, _, tsym = symbol.partition("/")
    tsym = tsym or "USD"
    start_ts = (
        int(pd.Timestamp(start_date, tz="UTC").timestamp()) if start_date else 0
    )

    rows: list[dict] = []
    to_ts: int | None = None
    # Hard stop on pages to avoid an infinite loop if the API misbehaves.
    for _ in range(20):
        url = (
            "https://min-api.cryptocompare.com/data/v2/histoday"
            f"?fsym={fsym}&tsym={tsym}&limit={page_limit}"
        )
        if to_ts is not None:
            url += f"&toTs={to_ts}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("Response") != "Success":
            raise RuntimeError(
                f"CryptoCompare error: {payload.get('Message', 'unknown')}"
            )
        batch = payload["Data"]["Data"]
        if not batch:
            break
        rows = batch + rows
        earliest = batch[0]["time"]
        # CryptoCompare pads the start with zero-price rows; stop once we have
        # reached the requested start or run out of real history.
        if earliest <= start_ts or all(r["close"] == 0 for r in batch[:5]):
            break
        to_ts = earliest - 1

    df = pd.DataFrame(rows)
    df = df[df["close"] > 0]  # drop zero-price padding rows
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"volumefrom": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _download_ccxt(
    exchange: str, symbol: str, timeframe: str, start_date: str | None
) -> pd.DataFrame:
    """Download OHLCV via ccxt. Raises a helpful error if ccxt is unavailable."""
    try:
        import ccxt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "ccxt is not installed and no local cache was found. Either "
            "`pip install ccxt`, place a CSV in data/raw/, or use "
            "generate_synthetic_btc() for testing."
        ) from exc

    ex = getattr(ccxt, exchange)()
    ex.load_markets()
    since = ex.parse8601(f"{start_date}T00:00:00Z") if start_date else None
    all_rows: list[list[float]] = []
    limit = 1000
    cursor = since
    while True:  # pragma: no cover - network dependent
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
    if not all_rows:
        raise RuntimeError(f"No OHLCV data returned for {symbol} on {exchange}.")
    df = pd.DataFrame(
        all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("date")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw OHLCV frame: UTC index, required columns, sorted, deduped.

    Args:
        df: Raw OHLCV frame, possibly with extra columns or unsorted index.

    Returns:
        A cleaned copy with exactly ``REQUIRED_COLUMNS`` and a sorted unique
        UTC DatetimeIndex. Does not forward-fill missing days.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "date"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    # Normalize daily timestamps to midnight UTC so signal/return alignment is exact.
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run data-quality checks and return a report dict.

    Hard failures (raise ValueError): duplicate dates, unsorted index,
    non-positive prices, missing close prices.
    Soft warnings (recorded in the report, not raised): missing calendar days,
    abnormal volume spikes.

    Args:
        df: A cleaned OHLCV frame.

    Returns:
        A report dict with keys ``missing_days``, ``abnormal_volume_days``,
        ``n_rows``, ``start``, ``end``.
    """
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found in data.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data index is not sorted ascending.")
    price_cols = ["open", "high", "low", "close"]
    if (df[price_cols] <= 0).any().any():
        raise ValueError("Non-positive prices found in OHLC data.")
    if df["close"].isna().any():
        raise ValueError("Missing close prices found.")

    # Soft check: missing calendar days at daily frequency.
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
    missing_days = full_range.difference(df.index)

    # Soft check: abnormal volume (> mean + 10 std over a 30d rolling window).
    vol = df["volume"].astype(float)
    roll_mean = vol.rolling(30, min_periods=5).mean()
    roll_std = vol.rolling(30, min_periods=5).std()
    threshold = roll_mean + 10 * roll_std
    abnormal = df.index[(vol > threshold) & threshold.notna()]

    report = {
        "n_rows": int(len(df)),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing_days": list(missing_days),
        "abnormal_volume_days": list(abnormal),
    }
    return report


def generate_synthetic_btc(
    n_days: int = 800,
    start: str = "2017-01-01",
    seed: int = 42,
    initial_price: float = 1000.0,
) -> pd.DataFrame:
    """Generate a deterministic synthetic BTC OHLCV series for testing/offline use.

    Uses a seeded geometric random walk with mild trend and volatility. This is
    NOT real market data and must only be used for harness testing.

    Args:
        n_days: Number of daily bars.
        start: First date (UTC midnight).
        seed: RNG seed for reproducibility.
        initial_price: Starting close price.

    Returns:
        A cleaned OHLCV DataFrame indexed by UTC DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    # Mild upward drift with regime-like volatility.
    daily_ret = rng.normal(loc=0.0015, scale=0.04, size=n_days)
    close = initial_price * np.exp(np.cumsum(daily_ret))
    prev_close = np.concatenate([[initial_price], close[:-1]])
    open_ = prev_close
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n_days)))
    volume = np.abs(rng.normal(1e4, 2e3, n_days)) + 1.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return clean_data(df)
