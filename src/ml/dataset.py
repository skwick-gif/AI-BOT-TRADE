from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import concurrent.futures
import threading

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .features.technical import build_technical_features
from .labeling import add_future_returns, add_classification_labels, add_regression_labels


BRONZE_DIR_DEFAULT = Path("data/bronze/daily")


def load_single_parquet(fp: Path, wanted_tickers: Optional[set] = None) -> Optional[tuple[str, pd.DataFrame]]:
    """Load a single parquet file and return (ticker, df) tuple."""
    t = fp.stem.upper()
    if wanted_tickers and t not in wanted_tickers:
        return None

    try:
        df = pd.read_parquet(fp)
        # enforce schema
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Ensure ticker/source columns exist
        if "ticker" not in df.columns:
            df.insert(0, "ticker", t)
        if "source" not in df.columns:
            df["source"] = "stock_data"
        return (t, df)
    except Exception as e:
        print(f"[ML] Error loading {fp}: {e}")
        return None


def load_bronze(dir_path: Path | str = BRONZE_DIR_DEFAULT, tickers: Optional[Iterable[str]] = None, progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Load per-ticker Parquet files from bronze directory into DataFrames.

    Returns a dict of {ticker: df} with columns [ticker, date, open, high, low, close, adj_close, volume, source].
    progress_callback: optional function to call with progress percentage (0-100)
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

    # Get all parquet files first
    all_files = list(p.glob("*.parquet"))
    wanted = {t.upper() for t in tickers} if tickers else None
    filtered_files = [fp for fp in all_files if not wanted or fp.stem.upper() in wanted]

    total_files = len(filtered_files)
    if progress_callback:
        progress_callback(0)

    if total_files == 0:
        return out

    # Use parallel loading for better performance
    max_workers = min(8, total_files)  # Limit to 8 workers to avoid overwhelming the system

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(load_single_parquet, fp, wanted): fp for fp in filtered_files}

        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                ticker, df = result
                out[ticker] = df

            completed += 1
            if progress_callback:
                progress = int(completed / total_files * 100)
                progress_callback(progress)

    print(f"[ML] Loaded {len(out)} tickers from {p} using parallel processing.")
    if len(out) > 0:
        total_rows = sum(len(v) for v in out.values())
        print(f"[ML] Total rows across all tickers: {total_rows}")
        sample_tickers = list(out.keys())[:5]
        print(f"[ML] Sample of loaded tickers: {sample_tickers}")
        # Show details for first 3 tickers
        for k in list(out.keys())[:3]:
            v = out[k]
            print(f"[ML] Ticker {k}: {len(v)} rows, date range: {v['date'].min()} to {v['date'].max()}")
    return out


def build_symbol_dataset(df: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """For a single symbol daily OHLCV DataFrame, build technical features and labels.

    Returns a DataFrame including original columns plus features and y_h* labels.
    Rows with insufficient history for features remain but may contain NaN; downstream steps can dropna.
    """
    out = df.copy()
    # Technical features
    if cfg.use_technical:
        feat_cfg = {
            "ema_windows": cfg.features.ema_windows,
            "rsi_window": cfg.features.rsi_window,
            "macd": cfg.features.macd,
            "atr_window": cfg.features.atr_window,
            "bbands_window": cfg.features.bbands_window,
            "ret_windows": cfg.features.ret_windows,
        }
        out = build_technical_features(out, feat_cfg)
    # Volume-based features (lightweight): SMA of volume and ratios
    if cfg.use_volume and "volume" in out.columns:
        import pandas as pd  # noqa: F401
        out["vma_20"] = out["volume"].rolling(20, min_periods=1).mean()
        out["vma_5"] = out["volume"].rolling(5, min_periods=1).mean()
        # Safe division to avoid infinity - replace zeros with small value and clip extreme values
        out["vol_rel_20"] = (out["volume"] / out["vma_20"].replace(0, 1e-8)).clip(-1e10, 1e10)
        out["vol_rel_5"] = (out["volume"] / out["vma_5"].replace(0, 1e-8)).clip(-1e10, 1e10)
    # Sentiment features will be joined later in pipeline if enabled (UI-controlled)
    out = add_future_returns(out, cfg.horizons)
    if cfg.target.type == "classification":
        out = add_classification_labels(out, cfg.horizons, cfg.target.tau)
    else:
        out = add_regression_labels(out, cfg.horizons)
    return out


def save_pooled_dataset(pooled: pd.DataFrame, filepath: Path | str):
    """Save pooled dataset to parquet file for faster loading."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata about creation time
    pooled.to_parquet(filepath, index=False)
    
    # Also save a metadata file with timestamp info
    metadata_file = filepath.with_suffix('.metadata.json')
    import json
    from datetime import datetime
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'num_rows': len(pooled),
        'num_cols': len(pooled.columns),
        'columns': list(pooled.columns),
        'date_range': {
            'start': pooled['date'].min().isoformat() if 'date' in pooled.columns else None,
            'end': pooled['date'].max().isoformat() if 'date' in pooled.columns else None
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[ML] Saved pooled dataset to {filepath} ({len(pooled)} rows)")


def load_pooled_dataset(filepath: Path | str) -> Optional[pd.DataFrame]:
    """Load pooled dataset from parquet file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    
    try:
        # Check metadata file for validation
        metadata_file = filepath.with_suffix('.metadata.json')
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if any bronze files are newer than cache creation time
            cache_created = metadata.get('created_at')
            if cache_created:
                from datetime import datetime
                cache_time = datetime.fromisoformat(cache_created)
                
                bronze_dir = Path("data/bronze/daily")
                if not bronze_dir.exists():
                    bronze_dir = Path("data/bronze")
                
                if bronze_dir.exists():
                    for parquet_file in bronze_dir.glob("*.parquet"):
                        file_mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
                        if file_mtime > cache_time:
                            print(f"[ML] Cache is outdated - {parquet_file.name} modified after cache creation")
                            return None
        
        df = pd.read_parquet(filepath)
        print(f"[ML] Loaded pooled dataset from {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[ML] Error loading pooled dataset: {e}")
        return None


def build_pooled_dataset(bronze: Dict[str, pd.DataFrame], cfg: TrainingConfig, cache_file: Optional[Path | str] = None, use_parallel: bool = True) -> pd.DataFrame:
    """Build pooled, per-row dataset across tickers.

    We build features and labels per symbol independently and then concatenate rows.
    This avoids cross-symbol alignment/leakage and keeps each row as (ticker, date, features..., y_h*).
    
    Uses caching to avoid rebuilding the same dataset multiple times.
    If cache_file is provided, will save/load from disk for even faster subsequent runs.
    use_parallel: whether to use parallel processing for feature engineering.
    """
    # Create cache key based on config parameters that affect the dataset
    cache_key = (
        cfg.use_technical,
        cfg.use_volume,
        cfg.use_sentiment,
        tuple(cfg.horizons),
        cfg.target.type,
        getattr(cfg.target, 'tau', None),
        tuple(cfg.features.ema_windows) if cfg.use_technical else (),
        cfg.features.rsi_window if cfg.use_technical else None,
        cfg.features.macd if cfg.use_technical else None,
        cfg.features.atr_window if cfg.use_technical else None,
        cfg.features.bbands_window if cfg.use_technical else None,
        tuple(cfg.features.ret_windows) if cfg.use_technical else (),
    )
    
    # Check disk cache first if provided
    if cache_file:
        cache_path = Path(cache_file)
        if cache_path.exists():
            cached_df = load_pooled_dataset(cache_path)
            if cached_df is not None:
                # Verify the cache is still valid by checking a sample
                if len(cached_df) > 0 and all(col in cached_df.columns for col in ['ticker', 'date']):
                    print(f"[ML] Using disk-cached pooled dataset")
                    return cached_df
    
    # Check memory cache
    if hasattr(build_pooled_dataset, '_cache') and build_pooled_dataset._cache.get('key') == cache_key:
        cached_df = build_pooled_dataset._cache.get('data')
        if cached_df is not None:
            print(f"[ML] Using memory-cached pooled dataset ({len(cached_df)} rows)")
            # Save to disk cache if requested
            if cache_file:
                save_pooled_dataset(cached_df, cache_file)
            return cached_df.copy()
    
    print(f"[ML] Building pooled dataset for {len(bronze)} tickers...")
    
    # Build features for each ticker
    if use_parallel and len(bronze) > 4:  # Only use parallel for many tickers
        print(f"[ML] Using parallel feature engineering for {len(bronze)} tickers...")
        max_workers = min(6, len(bronze))  # Limit workers
        
        def process_ticker(ticker_df_tuple):
            t, df = ticker_df_tuple
            if df.empty:
                return None
            return build_symbol_dataset(df, cfg)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            ticker_items = list(bronze.items())
            results = list(executor.map(process_ticker, ticker_items))
            frames = [r for r in results if r is not None and not r.empty]
    else:
        # Sequential processing
        frames = []
        for t, df in bronze.items():
            if df.empty:
                continue
            cur = build_symbol_dataset(df, cfg)
            if not cur.empty:
                frames.append(cur)
    
    if not frames:
        return pd.DataFrame()
    
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    # ensure dtypes
    pooled["date"] = pd.to_datetime(pooled["date"], errors="coerce")
    pooled = pooled.dropna(subset=["date"])  # drop invalid dates
    
    # Clean up infinity and extremely large values that could cause issues
    numeric_cols = pooled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Replace infinity with NaN, then fill with 0
        pooled[col] = pooled[col].replace([np.inf, -np.inf], np.nan)
        # Clip extremely large values
        pooled[col] = pooled[col].clip(-1e10, 1e10)
    
    pooled = pooled.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Cache the result in memory
    if not hasattr(build_pooled_dataset, '_cache'):
        build_pooled_dataset._cache = {}
    build_pooled_dataset._cache['key'] = cache_key
    build_pooled_dataset._cache['data'] = pooled.copy()
    
    # Save to disk cache if requested
    if cache_file:
        save_pooled_dataset(pooled, cache_file)
    
    print(f"[ML] Built pooled dataset with {len(pooled)} rows")
    return pooled


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
