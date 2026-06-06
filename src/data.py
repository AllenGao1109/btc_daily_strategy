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


def _http_json(url: str, timeout: int = 25) -> dict:
    """GET a JSON URL with a browser User-Agent (some hosts block default UA)."""
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def load_fear_greed(index: pd.DatetimeIndex) -> pd.Series:
    """Load the Crypto Fear & Greed Index (alternative.me), daily, history to 2018.

    A free market-sentiment series in [0, 100] (0 = extreme fear, 100 = extreme
    greed). Causal: each day's value is known that day; the engine lags any signal.

    Returns:
        Series named ``fng`` aligned to ``index`` (NaN before 2018 / where missing).
    """
    try:
        data = _http_json("https://api.alternative.me/fng/?limit=0&format=json")["data"]
    except Exception:  # pragma: no cover - network
        return pd.Series(np.nan, index=index, name="fng")
    s = pd.Series(
        {pd.Timestamp(int(x["timestamp"]), unit="s", tz="UTC").normalize(): float(x["value"])
         for x in data}
    ).sort_index()
    return s.reindex(index).rename("fng")


def load_yahoo_close(symbol: str, index: pd.DatetimeIndex, name: str) -> pd.Series:
    """Load a daily close series from Yahoo Finance, aligned + forward-filled.

    Used for macro/cross-asset factors (S&P 500, dollar index, VIX, gold). Markets
    are closed on weekends/holidays, so values are forward-filled onto BTC's 24/7
    calendar using only past data (causal). The engine lags any resulting signal.

    Args:
        symbol: Yahoo ticker (e.g. ``"^GSPC"``, ``"DX-Y.NYB"``, ``"^VIX"``).
        index: BTC DatetimeIndex to align to.
        name: Output column name (e.g. ``"spx_close"``).

    Returns:
        Series named ``name`` aligned to ``index`` (forward-filled, NaN warmup).
    """
    import urllib.parse

    sym = urllib.parse.quote(symbol)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           "?period1=1451606400&period2=9999999999&interval=1d")
    try:
        r = _http_json(url)["chart"]["result"][0]
        ts, cl = r["timestamp"], r["indicators"]["quote"][0]["close"]
    except Exception:  # pragma: no cover - network
        return pd.Series(np.nan, index=index, name=name)
    s = pd.Series(
        {pd.Timestamp(t, unit="s", tz="UTC").normalize(): c for t, c in zip(ts, cl) if c is not None}
    ).sort_index()
    return s.reindex(index).ffill().rename(name)


def enrich_external(df: pd.DataFrame, cache_dir: Path | None = None,
                    force_reload: bool = False) -> pd.DataFrame:
    """Add macro + sentiment columns (fng, spx_close, dxy_close, vix_close), cached.

    Fetches the Fear & Greed Index and Yahoo macro series once and caches them to
    ``data/processed/external.csv`` (Yahoo rate-limits, so caching matters). All
    series are causal and forward-filled onto the BTC calendar; the engine lags any
    resulting signal. On a fetch failure the affected column is left NaN and the
    factors that need it are simply skipped.

    Args:
        df: BTC frame to enrich (returned with extra columns).
        cache_dir: Directory for the cache (defaults to data/processed).
        force_reload: Re-fetch even if the cache exists.

    Returns:
        ``df`` with the external columns merged in.
    """
    cache_dir = cache_dir or PROCESSED_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "external.csv"

    if path.exists() and not force_reload:
        ext = _read_cache(path)
    else:
        ext = pd.DataFrame(index=df.index)
        ext["fng"] = load_fear_greed(df.index)
        ext["spx_close"] = load_yahoo_close("^GSPC", df.index, "spx_close")
        ext["dxy_close"] = load_yahoo_close("DX-Y.NYB", df.index, "dxy_close")
        ext["vix_close"] = load_yahoo_close("^VIX", df.index, "vix_close")
        ext.to_csv(path)

    for col in ["fng", "spx_close", "dxy_close", "vix_close"]:
        if col in ext.columns:
            df[col] = ext[col].reindex(df.index)
    return df


def load_okx_funding(index: pd.DatetimeIndex) -> pd.Series:
    """Load daily-aggregated BTC perpetual funding rate from OKX (positioning).

    Funding is paid every 8h; this aggregates to a daily mean. High positive
    funding = crowded longs (typically bearish for forward returns). Paginates
    backward via the ``after`` cursor.

    LIMITATION: OKX's public funding-rate-history endpoint only returns ~90 days,
    so this is usable for LIVE/recent monitoring but NOT for the 2017-2026 backtest
    (no overlap with the train/val windows). Kept for completeness; the research
    composite does not use funding (see RESEARCH_FINDINGS.md). Causal regardless.

    Args:
        index: BTC DatetimeIndex to align to.

    Returns:
        Series named ``funding`` indexed like ``index`` (NaN where unavailable).
    """
    import json
    import urllib.request

    url = "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USD-SWAP&limit=100"
    rows: list[tuple[int, float]] = []
    cursor: str | None = None
    for _ in range(400):  # backstop; ~3/day since 2020 => a few hundred pages
        u = url + (f"&after={cursor}" if cursor else "")
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:  # pragma: no cover - network
            break
        data = payload.get("data", [])
        if not data:
            break
        for d in data:
            rows.append((int(d["fundingTime"]), float(d["fundingRate"])))
        cursor = str(min(int(d["fundingTime"]) for d in data))
        if len(data) < 100:
            break

    if not rows:
        return pd.Series(np.nan, index=index, name="funding")
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): r for t, r in rows})
    daily = s.groupby(s.index.normalize()).mean()
    return daily.reindex(index).rename("funding")


def load_eth_close(index: pd.DatetimeIndex, start_date: str = "2016-01-01") -> pd.Series:
    """Load ETH daily close aligned to ``index`` (for cross-crypto factors).

    Uses the same keyless CryptoCompare source as BTC. Forward-filled onto the BTC
    calendar; introduces no lookahead (only past ETH closes are used downstream).

    Args:
        index: BTC DatetimeIndex to align to.
        start_date: Earliest ETH date to fetch.

    Returns:
        Series named ``eth_close`` indexed like ``index``.
    """
    eth = _download_cryptocompare("ETH/USD", start_date)["close"]
    return eth.reindex(index).ffill().rename("eth_close")


def load_btc_hourly(
    start_date: str = "2019-01-01",
    symbol: str = "BTC/USD",
    cache_dir: Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load hourly BTC OHLCV from CryptoCompare (free, paginated), cached to disk.

    Args:
        start_date: Earliest hour to fetch (ISO date).
        symbol: Pair (split into fsym/tsym).
        cache_dir: Cache directory (defaults to data/processed).
        force_reload: Re-fetch even if cached.

    Returns:
        DataFrame indexed by a UTC hourly ``DatetimeIndex`` with columns
        ``[open, high, low, close, volume]``, sorted ascending, validated.
    """
    cache_dir = cache_dir or PROCESSED_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "BTC-USD_1h.csv"
    if path.exists() and not force_reload:
        df = _read_cache(path)
    else:
        df = _download_cryptocompare(symbol, start_date, frequency="hour")
        df.to_csv(path)
    df = _slice_dates(df, start_date, None)
    # Light validation: positive prices, sorted, unique.
    assert df.index.is_monotonic_increasing, "hourly index not sorted"
    assert not df.index.has_duplicates, "duplicate hourly timestamps"
    assert (df["close"] > 0).all(), "non-positive hourly close"
    return df


def _download_cryptocompare(
    symbol: str, start_date: str | None, *, page_limit: int = 2000,
    frequency: str = "day",
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

    endpoint = "histohour" if frequency == "hour" else "histoday"
    max_pages = 60 if frequency == "hour" else 20  # hourly needs far more pages

    rows: list[dict] = []
    to_ts: int | None = None
    # Hard stop on pages to avoid an infinite loop if the API misbehaves.
    for _ in range(max_pages):
        url = (
            f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
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
