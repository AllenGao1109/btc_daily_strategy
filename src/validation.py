"""Chronological validation: fixed train/validation/test split and walk-forward.

Rules (CLAUDE.md section 18):
  - NEVER use a random split. Always chronological.
  - The test set is evaluated only after parameters are chosen and must not be
    used for tuning.
  - Walk-forward reports each fold separately, not just the average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

# A scorer takes a backtest result table + initial capital and returns a score.
ScorerFn = Callable[[pd.DataFrame, float], float]
# A runner takes (df slice, params) and returns a backtest result table.
RunnerFn = Callable[[pd.DataFrame, dict], pd.DataFrame]


@dataclass(frozen=True)
class SplitWindow:
    """A named chronological date window [start, end] (inclusive)."""

    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the rows of ``df`` inside this window."""
        mask = pd.Series(True, index=df.index)
        if self.start is not None:
            mask &= df.index >= self.start
        if self.end is not None:
            mask &= df.index <= self.end
        return df[mask]


def _ts(value: str | None) -> pd.Timestamp | None:
    """Parse an optional ISO date string into a UTC timestamp."""
    return pd.Timestamp(value, tz="UTC") if value else None


def make_fixed_split(validation_cfg: dict[str, Any]) -> dict[str, SplitWindow]:
    """Build train / validation / test windows from the ``validation`` config.

    Args:
        validation_cfg: The ``validation`` config section.

    Returns:
        Dict with keys ``train``, ``validation``, ``test`` mapping to
        :class:`SplitWindow`.
    """
    return {
        "train": SplitWindow(
            "train",
            _ts(validation_cfg.get("train_start")),
            _ts(validation_cfg.get("train_end")),
        ),
        "validation": SplitWindow(
            "validation",
            _ts(validation_cfg.get("validation_start")),
            _ts(validation_cfg.get("validation_end")),
        ),
        "test": SplitWindow(
            "test",
            _ts(validation_cfg.get("test_start")),
            _ts(validation_cfg.get("test_end")),
        ),
    }


def make_walk_forward_folds(
    df: pd.DataFrame,
    train_years: int = 3,
    val_years: int = 1,
    expanding: bool = True,
) -> list[tuple[SplitWindow, SplitWindow]]:
    """Build rolling/expanding walk-forward (train, validation) fold windows.

    Args:
        df: The full dataset (used to determine the year range).
        train_years: Number of years in each training window.
        val_years: Number of years in each validation window.
        expanding: If True, training start stays fixed (expanding window);
            if False, training window rolls forward (fixed length).

    Returns:
        A list of ``(train_window, val_window)`` tuples, one per fold.
    """
    if len(df) == 0:
        return []
    years = sorted({d.year for d in df.index})
    folds: list[tuple[SplitWindow, SplitWindow]] = []
    first_year = years[0]
    cursor = first_year + train_years
    while cursor + val_years - 1 <= years[-1]:
        train_start_year = first_year if expanding else cursor - train_years
        train_start = pd.Timestamp(f"{train_start_year}-01-01", tz="UTC")
        train_end = pd.Timestamp(f"{cursor - 1}-12-31", tz="UTC")
        val_start = pd.Timestamp(f"{cursor}-01-01", tz="UTC")
        val_end = pd.Timestamp(f"{cursor + val_years - 1}-12-31", tz="UTC")
        folds.append(
            (
                SplitWindow(f"train_{train_start_year}_{cursor - 1}", train_start, train_end),
                SplitWindow(f"val_{cursor}_{cursor + val_years - 1}", val_start, val_end),
            )
        )
        cursor += val_years
    return folds


def grid_search(
    df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    runner: RunnerFn,
    scorer: ScorerFn,
    initial_capital: float,
    train_window: SplitWindow,
    val_window: SplitWindow,
) -> list[dict[str, Any]]:
    """Evaluate a parameter grid on a train/validation split.

    Parameters are *applied* using train+validation data (so indicators have
    warmup history), but scored only on the validation window. The test set is
    never touched here.

    Args:
        df: Full feature dataset.
        param_grid: Mapping of param name -> list of candidate values.
        runner: Function ``(df_slice, params) -> result_table``.
        scorer: Function ``(result_table, initial_capital) -> score``.
        initial_capital: Starting equity.
        train_window: Training window (for warmup history).
        val_window: Validation window (scored).

    Returns:
        A list of ``{"params": ..., "score": ...}`` dicts sorted by descending
        score.
    """
    from itertools import product

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    # Run on data from train start through validation end so warmup is correct.
    eval_start = train_window.start
    eval_end = val_window.end
    eval_slice = SplitWindow("eval", eval_start, eval_end).slice(df)

    results: list[dict[str, Any]] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = runner(eval_slice, params)
        # Score only the validation portion.
        val_result = val_window.slice(result)
        score = scorer(val_result, initial_capital)
        results.append({"params": params, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
