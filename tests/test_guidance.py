"""Multi-band guidance rendering (network-free)."""

from __future__ import annotations

import pandas as pd

from src.guidance import DEFAULT_BANDS, render_markdown


def test_render_markdown_has_a_row_per_band():
    table = pd.DataFrame([
        {"band": 0.30, "trades_per_yr": 15.0, "test_sharpe": 0.94, "test_nav": 2.30,
         "test_maxdd": -31, "full_sharpe": 1.15, "full_maxdd": -54, "now_holding": 0.0,
         "days_since_trade": 69, "action_now": "HOLD at 0.00x"},
        {"band": 0.40, "trades_per_yr": 8.6, "test_sharpe": 0.83, "test_nav": 1.99,
         "test_maxdd": -33, "full_sharpe": 1.06, "full_maxdd": -61, "now_holding": 0.25,
         "days_since_trade": 215, "action_now": "HOLD at 0.25x"},
    ])
    meta = {"as_of": "2026-06-05", "btc_close": 63268.0, "signal_target": 0.0}
    md = render_markdown(table, meta)
    assert "multi-band decision guidance" in md
    assert "2026-06-05" in md
    assert "| 0.30 |" in md and "| 0.40 |" in md
    assert md.count("HOLD at") == 2


def test_default_bands_sorted_and_in_range():
    assert DEFAULT_BANDS == sorted(DEFAULT_BANDS)
    assert all(0.0 < b <= 2.0 for b in DEFAULT_BANDS)
