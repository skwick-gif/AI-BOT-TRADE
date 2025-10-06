from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .config import TrainingConfig
from .features.technical import build_technical_features
from .labeling import add_future_returns, add_classification_labels, add_regression_labels


BRONZE_DIR_DEFAULT = Path("data/bronze/daily")


def load_bronze(dir_path: Path | str = BRONZE_DIR_DEFAULT, tickers: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load per-ticker Parquet files from bronze directory into DataFrames.

    Returns a dict of {ticker: df} with columns [ticker, date, open, high, low, close, adj_close, volume, source].
    """
    p = Path(dir_path)
    # Fallback: if daily folder is missing or empty, try data/bronze directly
    if (not p.exists()) or (not any(p.glob("*.parquet"))):
        alt = Path("data/bronze")
        if alt.exists() and any(alt.glob("*.parquet")):
            print(f"[ML] Falling back to bronze dir: {alt}")
            p = alt
    out: Dict[str, pd.DataFrame] = {}
    if not p.exists():
        return out
    wanted = {t.upper() for t in tickers} if tickers else None
    for fp in p.glob("*.parquet"):
        t = fp.stem.upper()
        if wanted and t not in wanted:
            continue
        df = pd.read_parquet(fp)
        # enforce schema
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Ensure ticker/source columns exist (new stock_data converter may omit them)
        if "ticker" not in df.columns:
            df.insert(0, "ticker", t)
        if "source" not in df.columns:
            df["source"] = "stock_data"
        out[t] = df
    print(f"[ML] Loaded {len(out)} tickers from {p}.")
    for k, v in out.items():
        print(f"[ML] Ticker {k}: {len(v)} rows.")
    return out


def build_symbol_dataset(df: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """For a single symbol daily OHLCV DataFrame, build technical features and labels.

    Returns a DataFrame including original columns plus features and y_h* labels.
    Rows with insufficient history for features remain but may contain NaN; downstream steps can dropna.
    """
    feat_cfg = {
        "ema_windows": cfg.features.ema_windows,
        "rsi_window": cfg.features.rsi_window,
        "macd": cfg.features.macd,
        "atr_window": cfg.features.atr_window,
        "bbands_window": cfg.features.bbands_window,
        "ret_windows": cfg.features.ret_windows,
    }
    out = build_technical_features(df, feat_cfg)
    out = add_future_returns(out, cfg.horizons)
    if cfg.target.type == "classification":
        out = add_classification_labels(out, cfg.horizons, cfg.target.tau)
    else:
        out = add_regression_labels(out, cfg.horizons)
    return out


def build_pooled_dataset(bronze: Dict[str, pd.DataFrame], cfg: TrainingConfig) -> pd.DataFrame:
    """Build pooled, per-row dataset across tickers.

    We build features and labels per symbol independently and then concatenate rows.
    This avoids cross-symbol alignment/leakage and keeps each row as (ticker, date, features..., y_h*).
    """
    frames = []
    for t, df in bronze.items():
        if df.empty:
            continue
        cur = build_symbol_dataset(df, cfg)
        frames.append(cur)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    # ensure dtypes
    pooled["date"] = pd.to_datetime(pooled["date"], errors="coerce")
    pooled = pooled.dropna(subset=["date"])  # drop invalid dates
    return pooled.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_label_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer feature columns (X) and label columns (y_h*)."""
    reserved_prefixes = ("fut_ret_")
    reserved_cols = {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"}
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    x_cols = [
        c for c in df.columns
        if c not in reserved_cols and c not in y_cols and not c.startswith(reserved_prefixes)
    ]
    return x_cols, y_cols
