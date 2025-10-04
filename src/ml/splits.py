from __future__ import annotations

import pandas as pd


def compute_cutoff_dates(dates: pd.DatetimeIndex, *, holdout_last_days: int, max_horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Given available trading dates (sorted), return (cutoff_train_end, end_date_effective).
    cutoff ensures there is room for max_horizon days ahead within dates.
    holdout_last_days excludes last N trading days from training.
    """
    dates = dates.sort_values().unique()
    if len(dates) < (holdout_last_days + max_horizon + 10):
        raise ValueError("Not enough trading days for requested holdout/horizon")
    end_date = dates[-1]
    cutoff_idx = len(dates) - holdout_last_days - max_horizon - 1
    cutoff_idx = max(0, cutoff_idx)
    cutoff_train_end = dates[cutoff_idx]
    return pd.Timestamp(cutoff_train_end), pd.Timestamp(end_date)


def advance_trading_days(dates: pd.DatetimeIndex, current: pd.Timestamp, *, step_days: int) -> pd.Timestamp:
    dates = dates.sort_values().unique()
    try:
        i = dates.get_loc(pd.Timestamp(current))
    except KeyError:
        # find nearest previous
        i = dates.searchsorted(pd.Timestamp(current), side="right") - 1
        i = max(0, i)
    nxt = i + step_days
    if nxt >= len(dates):
        return dates[-1]
    return dates[nxt]
