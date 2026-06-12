"""CryptoQuant on-chain behavior metrics + CME basis (public archive mirrors).

Source: a public research archive (moltovy/Crypto-Research-Paper-Data-Factors-
Analysis-) containing CryptoQuant CSV exports (through ~2026-04) and
TradingView daily exports. Free alternatives to these series do not exist with
pre-2022 coverage, which is what makes this archive valuable: factors can be
tested against our 2017-2021 train window.

TIMING / LEAKAGE AUDIT (read before adding factors):
  - CryptoQuant stamps a daily aggregate on its UTC value date; the value is
    complete exactly at 00:00 UTC of the next day, which is also the engine's
    effective entry time for the next-day position (entry at close_t). Real
    availability is minutes-to-hours later. This matches the documented
    CoinMetrics convention used everywhere in this project, but it is
    borderline by hours; therefore every factor built on these series must
    pass the EXTRA-LAG robustness check in factor_mining (IC must survive one
    additional day of lag) before adoption.
  - CME basis closes with the CME session (~21:00/22:00 UTC), well before the
    00:00 UTC entry: no publication-lag issue. Weekend gaps are expected (CME
    closed); merge_onchain forward-fills the Friday value causally.
  - REVISION CAVEAT (cannot be fixed with free data): entity-based series
    (whale ratio, inflow CDD, miner flows, exchange flows) are recomputed by
    the vendor with TODAY'S wallet labels applied to history. Historical
    values in the archive are therefore cleaner than what was knowable at the
    time, which biases ICs optimistically. Treat entity-based passes with
    extra suspicion; label-free series (SOPR family, taker CVD) do not have
    this problem.

Standard library only (urllib); results cached under data/raw/.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/moltovy/"
    "Crypto-Research-Paper-Data-Factors-Analysis-/main/Data"
)

# column-name -> (subpath under Data/, source column name)
CRYPTOQUANT_SERIES: dict[str, tuple[str, str]] = {
    "asopr": (
        "CryptoQuant/BTC/Market Indicator/Bitcoin Adjusted SOPR (aSOPR) - Day.csv",
        "Adjusted SOPR (aSOPR)",
    ),
    "sth_sopr": (
        "CryptoQuant/BTC/Market Indicator/Bitcoin Short Term Holder SOPR - Day.csv",
        "Short Term Holder SOPR",
    ),
    "lth_sopr": (
        "CryptoQuant/BTC/Market Indicator/Bitcoin Long Term Holder SOPR - Day.csv",
        "Long Term Holder SOPR",
    ),
    "taker_cvd": (
        "CryptoQuant/BTC/Market Indicator/"
        "Bitcoin Spot Taker CVD(Cumulative Volume Delta, 90-day) - Day.csv",
        "Spot Taker CVD(Cumulative Volume Delta, 90-day)",
    ),
    "whale_ratio": (
        "CryptoQuant/BTC/Flow Indicator/"
        "Bitcoin Exchange Whale Ratio - All Exchanges - Day.csv",
        "Exchange Whale Ratio",
    ),
    "inflow_cdd": (
        "CryptoQuant/BTC/Flow Indicator/"
        "Bitcoin Exchange Inflow CDD - All Exchanges - Day.csv",
        "Exchange Inflow CDD",
    ),
    "miner_to_ex": (
        "CryptoQuant/BTC/Miner Flows/"
        "Bitcoin Miner to Exchange Flow (Total) - All Miners, All Exchanges - Day.csv",
        "Miner to Exchange Flow (Total)",
    ),
    "cb_premium": (  # Coinbase vs global spot gap: US institutional demand
        "CryptoQuant/BTC/Fund Data/Bitcoin Coinbase Premium Index - Day.csv",
        "Coinbase Premium Index",
    ),
    "kr_premium": (  # Upbit vs global spot gap: Korean retail demand (2020-07+)
        "CryptoQuant/BTC/Fund Data/Bitcoin Korea Premium Index - Day.csv",
        "Korea Premium Index",
    ),
    "funding_rate": (  # aggregated perp funding: leveraged-long crowding
        "CryptoQuant/BTC/Derivatives/Bitcoin Funding Rates - All Exchanges - Day.csv",
        "Funding Rates",
    ),
    "est_leverage": (  # open interest / exchange reserve: system leverage
        "CryptoQuant/BTC/Derivatives/"
        "Bitcoin Estimated Leverage Ratio - All Exchanges - Day.csv",
        "Estimated Leverage Ratio",
    ),
}

CME_BASIS_PATH = "Tradingview/Daily/CME_BTC_futures_over_SPOT_BTC_ratio__daily.csv"


def _fetch(path: str, timeout: int = 60) -> bytes:
    url = f"{ARCHIVE_BASE}/{urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def load_cryptoquant(
    series: list[str] | None = None,
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the CryptoQuant behavior series, cached, UTC-indexed.

    Args:
        series: Subset of ``CRYPTOQUANT_SERIES`` keys (default: all).
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.

    Returns:
        DataFrame indexed by UTC midnight date, one float column per series.
    """
    series = series or list(CRYPTOQUANT_SERIES)
    unknown = [s for s in series if s not in CRYPTOQUANT_SERIES]
    if unknown:
        raise KeyError(f"Unknown CryptoQuant series: {unknown}")
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"cryptoquant_{'-'.join(series)}.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    cols = {}
    for key in series:
        path, src_col = CRYPTOQUANT_SERIES[key]
        raw = pd.read_csv(io.BytesIO(_fetch(path)))
        if src_col in raw.columns:
            vals = pd.to_numeric(raw[src_col], errors="coerce")
        else:
            # Some chart exports split one line into regime-coloured columns
            # (e.g. CVD: Neutral / Taker Buy Dominant / Taker Sell Dominant,
            # exactly one non-null per row): coalesce them back into one series.
            value_cols = [c for c in raw.columns if c != "date"]
            num = raw[value_cols].apply(pd.to_numeric, errors="coerce")
            if num.notna().sum(axis=1).max() > 1:
                raise RuntimeError(
                    f"{path}: expected column {src_col!r} or one-of regime "
                    f"columns, got overlapping {value_cols}"
                )
            vals = num.bfill(axis=1).iloc[:, 0]
        s = pd.Series(
            vals.values,
            index=pd.to_datetime(raw["date"], utc=True).dt.normalize(),
        )
        cols[key] = s[~s.index.duplicated(keep="first")].sort_index()
    df = pd.DataFrame(cols).dropna(how="all")
    if df.empty:
        raise RuntimeError("CryptoQuant loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df


def load_cme_basis(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the CME-futures/spot ratio (daily close), cached, UTC-indexed.

    Returns:
        DataFrame with one float column ``cme_basis`` = futures/spot - 1
        (annualization is left to the factor layer; the front-month tenor
        varies). CME weekend gaps are the caller's concern (ffill at merge).
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "cme_basis.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)[["cme_basis"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    raw = pd.read_csv(io.BytesIO(_fetch(CME_BASIS_PATH)))
    raw.columns = [c.strip().lower() for c in raw.columns]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], utc=True).dt.normalize(),
            "cme_basis": pd.to_numeric(raw["close"], errors="coerce") - 1.0,
        }
    ).set_index("date")
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    if df.empty:
        raise RuntimeError("CME basis loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df
