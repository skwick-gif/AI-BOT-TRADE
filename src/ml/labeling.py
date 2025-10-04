from __future__ import annotations

import pandas as pd


def add_future_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    price = out["adj_close"].fillna(out["close"]) if "adj_close" in out.columns else out["close"]
    for h in horizons:
        out[f"fut_ret_{h}"] = price.shift(-h) / price - 1.0
    return out


def add_classification_labels(df: pd.DataFrame, horizons: list[int], tau: float) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        r = out[f"fut_ret_{h}"]
        lbl = pd.Series("HOLD", index=out.index)
        lbl = lbl.where(r.abs() < tau, other=None)
        lbl = lbl.where(~(r >= tau), other="UP")
        lbl = lbl.where(~(r <= -tau), other="DOWN")
        out[f"y_h{h}"] = lbl.fillna("HOLD")
    return out


def add_regression_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"y_h{h}"] = out[f"fut_ret_{h}"]
    return out
