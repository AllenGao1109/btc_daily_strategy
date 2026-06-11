"""Cross-asset behavior data: yen carry, speculative-growth appetite, miner equities.

The exogenous-factor thesis: BTC's marginal buyers leave footprints in OTHER
markets first — the yen funds global leverage (carry unwinds hit BTC, e.g.
2024-08), ARKK-vs-QQQ prices retail speculative appetite, and miner equities
are the stock market's real-time pricing of leveraged BTC exposure.

Sources (all free):
  - USDJPY: FRED H.10 noon rates, mirrored daily in the public
    `datasets/exchange-rates` GitHub repo. TIMING NOTE: the H.10 *release* is
    weekly, but the exchange rate itself is market-observable in real time at
    its stamp (noon ET, hours before the 00:00 UTC entry). Using the archival
    series at its value date is therefore causally sound — unlike
    release-only data (CPI etc.), the information existed publicly at the
    stamp. Values are not revised.
  - ARKK/QQQ/RIOT/MARA daily closes: TradingView exports in the public
    research archive used elsewhere in this project (US close 20/21:00 UTC,
    before the 00:00 UTC entry; weekend gaps ffilled causally at merge).

Standard library only (urllib); results cached under data/raw/.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FX_URL = (
    "https://raw.githubusercontent.com/datasets/exchange-rates/master/data/daily.csv"
)
TV_BASE = (
    "https://raw.githubusercontent.com/moltovy/"
    "Crypto-Research-Paper-Data-Factors-Analysis-/main/Data/Tradingview/Daily"
)
TV_SERIES = {
    "arkk_close": "ARKK_innovation_etf__daily.csv",
    "qqq_close": "QQQ_nasdaq100_etf__daily.csv",
    "riot_close": "RIOT_riot_miner_stock__daily.csv",
    "mara_close": "MARA_marathon_miner_stock__daily.csv",
    "gld_close": "GLD_gold_etf__daily.csv",
    "smh_close": "SMH_vaneck_semiconductor_etf__daily.csv",
}


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def load_crossasset(
    *,
    force_reload: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load USDJPY + cross-asset closes, cached, UTC-indexed.

    Returns:
        DataFrame with float columns ``jpy_usd`` (JPY per USD), ``arkk_close``,
        ``qqq_close``, ``riot_close``, ``mara_close``. Trading-day stamps;
        gaps are the caller's concern (merge_onchain ffills causally).
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "crossasset.csv"
    if cache.exists() and not force_reload:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    fx = pd.read_csv(io.BytesIO(_fetch(FX_URL)))
    fx = fx[fx["Country"] == "Japan"]
    jpy = pd.Series(
        pd.to_numeric(fx["Exchange rate"], errors="coerce").values,
        index=pd.to_datetime(fx["Date"], utc=True),
        name="jpy_usd",
    )
    cols = {"jpy_usd": jpy[~jpy.index.duplicated(keep="first")].sort_index()}

    for name, fname in TV_SERIES.items():
        raw = pd.read_csv(io.BytesIO(_fetch(f"{TV_BASE}/{urllib.parse.quote(fname)}")))
        raw.columns = [c.strip().lower() for c in raw.columns]
        s = pd.Series(
            pd.to_numeric(raw["close"], errors="coerce").values,
            index=pd.to_datetime(raw["date"], utc=True).dt.normalize(),
            name=name,
        )
        cols[name] = s[~s.index.duplicated(keep="first")].sort_index()

    df = pd.DataFrame(cols).dropna(how="all")
    if df.empty:
        raise RuntimeError("Cross-asset loader returned no data.")
    df.index.name = "date"
    df.to_csv(cache)
    return df
