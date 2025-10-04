"""
Utilities to convert raw JSON stock files (from 'raw data/') into Parquet files
stored under 'data/bronze/'.

Assumptions:
- Each raw JSON contains structure like:
  {
    "ticker": "AMCR",
    "price": {
      "yahoo": { "daily": [ {date, open, high, low, close, adj_close, volume}, ... ] },
      "alphavantage": { "daily": [ ... ] }
    }
  }

We prioritize source 'yahoo' if present, otherwise fallback to 'alphavantage'.
Output per symbol: data/bronze/daily/<TICKER>.parquet
"""
from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger("raw_to_parquet")


def _normalize_daily_records(records: List[Dict[str, Any]], *, ticker: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    # Normalize columns and types
    rename = {
        "adjClose": "adj_close",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA
    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])  # drop invalid dates
    else:
        raise ValueError("missing 'date' field in records")
    # Sort and set index
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "ticker", ticker)
    df["source"] = source
    return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]]


def convert_file(json_path: str, *, prefer_source: str = "yahoo") -> Optional[pd.DataFrame]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ticker = str(data.get("ticker") or os.path.splitext(os.path.basename(json_path))[0]).upper()
    price = data.get("price") or {}
    preferred = price.get(prefer_source) or {}
    fallback_name = "alphavantage" if prefer_source != "alphavantage" else "yahoo"
    fallback = price.get(fallback_name) or {}

    # choose daily
    recs = None
    chosen_source = None
    if isinstance(preferred, dict) and isinstance(preferred.get("daily"), list) and preferred["daily"]:
        recs = preferred["daily"]
        chosen_source = prefer_source
    elif isinstance(fallback, dict) and isinstance(fallback.get("daily"), list) and fallback["daily"]:
        recs = fallback["daily"]
        chosen_source = fallback_name
    else:
        logger.warning(f"No daily records found for {ticker} in {json_path}")
        return None

    try:
        df = _normalize_daily_records(recs, ticker=ticker, source=chosen_source)
        return df
    except Exception as e:
        logger.error(f"Failed to normalize {json_path}: {e}")
        return None


def convert_all(input_dir: str = "raw data", output_dir: str = "data/bronze/daily", *, prefer_source: str = "yahoo") -> Dict[str, str]:
    """Convert all JSON files in input_dir into per-symbol Parquet files under output_dir.

    Returns a dict of {ticker: output_path} for successfully converted files.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, str] = {}
    for name in os.listdir(input_dir):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(input_dir, name)
        df = convert_file(path, prefer_source=prefer_source)
        if df is None or df.empty:
            continue
        ticker = str(df["ticker"].iloc[0]).upper()
        out_path = os.path.join(output_dir, f"{ticker}.parquet")
        try:
            df.to_parquet(out_path, index=False)
            results[ticker] = out_path
            logger.info(f"Wrote {ticker} -> {out_path} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Failed writing {out_path}: {e}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert raw JSON OHLCV to Parquet")
    p.add_argument("--input", default="raw data", help="Input folder with raw JSON files")
    p.add_argument("--output", default="data/bronze/daily", help="Output folder for Parquet files")
    p.add_argument("--prefer", default="yahoo", choices=["yahoo", "alphavantage"], help="Preferred price source")
    args = p.parse_args()

    res = convert_all(args.input, args.output, prefer_source=args.prefer)
    print(f"Converted {len(res)} files to Parquet under '{args.output}'")
