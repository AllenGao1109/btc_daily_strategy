"""Macro market data loaders: FRED series and the DXY dollar index.

Sources:
  - FRED (St. Louis Fed): keyless ``fredgraph.csv`` endpoint per series, with a
    fallback to a public GitHub research archive of the same panel for
    environments where the FRED host is unreachable.
  - DXY US dollar index: TradingView daily export archived in the same public
    research repo (market price; no publication lag).

PUBLICATION-LAG HANDLING (lookahead control). FRED stamps a series on its
*value date*, not its release date. H.15 treasury yields and ICE BofA credit
spreads for day t are released on day t+1 (~16:15 ET), AFTER the BTC daily
close at 00:00 UTC of t+1 — so using the day-t stamp to position on day t+1
would leak ~21h of future information. Series listed in ``RELEASE_LAGGED`` are
therefore shifted one extra day at load time (the day-t value becomes effective
day t+1). VIX (closes 21:15 UTC same day) and the ON RRP result (released
~17:15 UTC same day) are available before the next BTC close and are not
shifted. The engine's 1-day execution lag then applies on top, as usual.

Standard library only (urllib); results cached under data/raw/.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_MIRROR = (
    "https://raw.githubusercontent.com/moltovy/"
    "Crypto-Research-Paper-Data-Factors-Analysis-/main/"
    "Data/FRED/fred_macro_panel__daily.csv"
)
DXY_MIRROR = (
    "https://raw.githubusercontent.com/moltovy/"
    "Crypto-Research-Paper-Data-Factors-Analysis-/main/"
    "Data/Tradingview/Daily/DXY_us_dollar_index__daily.csv"
)

# Daily, market-priced or same-day-released series we use as factors.
MACRO_SERIES = [
    "VIXCLS",        # CBOE VIX close (same-day, 21:15 UTC)
    "DGS10",         # 10y treasury yield (H.15, released t+1)
    "DGS2",          # 2y treasury yield (H.15, released t+1)
    "T10Y2Y",        # 10y-2y curve (H.15, released t+1)
    "DFII10",        # 10y TIPS real yield (H.15, released t+1)
    "BAMLH0A0HYM2",  # ICE BofA HY OAS (released t+1)
    "RRPONTSYD",     # ON RRP take-up (same-day, ~17:15 UTC)
    "USEPUINDXD",    # news-based Economic Policy Uncertainty (computed from
                     # day-t newspapers, available t+1 -> release-lagged)
]
RELEASE_LAGGED = ["DGS10", "DGS2", "T10Y2Y", "DFII10", "BAMLH0A0HYM2", "USEPUINDXD"]


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def load_fred_macro(
    series: list[str] | None = None,
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load daily FRED macro series, release-lag adjusted, cached, UTC-indexed.

    Args:
        series: FRED series ids (default ``MACRO_SERIES``).
        force_reload: Bypass the cache.
        raw_dir: Override cache directory.

    Returns:
        DataFrame indexed by UTC midnight *effective* date (value date +1 for
        ``RELEASE_LAGGED`` series) with one float column per series.
    """
    series = series or MACRO_SERIES
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"fred_{'-'.join(series)}.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    try:
        cols = {}
        for s in series:
            raw = pd.read_csv(io.BytesIO(_fetch(FRED_CSV.format(series=s))))
            raw.columns = ["date", s]
            raw["date"] = pd.to_datetime(raw["date"], utc=True)
            cols[s] = raw.set_index("date")[s]
        panel = pd.DataFrame(cols)
    except Exception:  # noqa: BLE001 - fall back to the archived panel
        raw = pd.read_csv(io.BytesIO(_fetch(FRED_MIRROR)))
        raw["date"] = pd.to_datetime(raw["date"], utc=True)
        panel = raw.set_index("date")[series]

    out = apply_release_lag(panel)
    out.to_csv(cache)
    return out


def apply_release_lag(panel: pd.DataFrame) -> pd.DataFrame:
    """Re-stamp ``RELEASE_LAGGED`` columns from value date t to release date t+1.

    Non-lagged columns keep their value-date stamp. The output index is the
    union of all effective dates, so a Friday H.15 value (re-stamped Saturday)
    is preserved and forward-fills correctly at merge time.
    """
    lagged = [c for c in panel.columns if c in RELEASE_LAGGED]
    plain = [c for c in panel.columns if c not in RELEASE_LAGGED]
    shifted = panel[lagged].copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    idx = panel.index.union(shifted.index)
    out = pd.concat(
        [panel[plain].reindex(idx), shifted.reindex(idx)], axis=1
    )[list(panel.columns)]
    out = out.apply(pd.to_numeric, errors="coerce").sort_index().dropna(how="all")
    out.index.name = "date"
    return out


HY_OAS_ARCHIVE = (
    "https://raw.githubusercontent.com/maaurocp/Trading_Protocol/main/"
    "data/raw/fred_BAMLH0A0HYM2.csv"
)


def load_hy_oas(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load full-history ICE BofA HY OAS (1996+), release-lag adjusted, cached.

    The FRED panel mirror only covers this series from 2023; this archive
    mirror restores the full history so the credit factors are testable on the
    train window. Same series id, same release-lag handling (published t+1).
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "hy_oas.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    raw = pd.read_csv(io.BytesIO(_fetch(HY_OAS_ARCHIVE)))
    raw.columns = ["date", "BAMLH0A0HYM2"]
    panel = pd.DataFrame(
        {"BAMLH0A0HYM2": pd.to_numeric(raw["BAMLH0A0HYM2"], errors="coerce").values},
        index=pd.to_datetime(raw["date"], utc=True).dt.normalize(),
    )
    panel = panel[~panel.index.duplicated(keep="first")].sort_index()
    out = apply_release_lag(panel)
    if out.empty:
        raise RuntimeError("HY OAS loader returned no data.")
    out.to_csv(cache)
    return out


def load_dxy(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the DXY dollar-index daily close (market price, no release lag).

    Returns:
        DataFrame indexed by UTC midnight date with one column ``dxy_close``.
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "dxy_close.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)[["dxy_close"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    raw = pd.read_csv(io.BytesIO(_fetch(DXY_MIRROR)))
    raw.columns = [c.strip().lower() for c in raw.columns]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], utc=True).dt.normalize(),
            "dxy_close": pd.to_numeric(raw["close"], errors="coerce"),
        }
    ).set_index("date")
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    if df.empty:
        raise RuntimeError("DXY loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df
