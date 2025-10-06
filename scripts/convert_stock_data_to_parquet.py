#!/usr/bin/env python3
"""
Convert stock_data directory structure to parquet for app consumption.
- Reads per-symbol folders under stock_data/<SYMBOL>/
  - <SYMBOL>_price.csv (custom multi-row header)
  - <SYMBOL>_advanced.json (fundamentals/metadata)
- Writes OHLCV parquet to data/bronze/daily/<SYMBOL>.parquet
- Aggregates fundamentals into data/bronze/fundamentals/<SYMBOL>.json and an optional parquet table

Usage (PowerShell):
  python scripts/convert_stock_data_to_parquet.py --source stock_data --out data/bronze/daily --fund-out data/bronze/fundamentals --aggregate-fund data/bronze/fundamentals.parquet
  # Optional: --limit 100 to process only first 100 symbols
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def parse_price_csv(csv_path: Path, symbol: str) -> pd.DataFrame:
    """Parse the custom price CSV into a normalized DataFrame with columns: date, open, high, low, close, adj_close, volume"""
    # The CSV appears to have 3 header rows:
    # 1: Price,Close,High,Low,Open,Volume
    # 2: Ticker,<SYMBOL>,<SYMBOL>,...
    # 3: Date,,,,,
    # Then rows: YYYY-MM-DD, Price, Close, High, Low, Open, Volume
    try:
        names7 = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        # Skip the three header rows and force column names
        data = pd.read_csv(csv_path, skiprows=3, names=names7, engine='python')
        # Some files might only have 6 data columns (no leading Price/Adj Close)
        if data.shape[1] == 6:
            # Assume columns: Date, Close, High, Low, Open, Volume
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            # Inject Price column equal to Close
            data.insert(1, 'Price', data['Close'])
        elif data.shape[1] != 7:
            # As a last resort, try regular parsing and rename
            tmp = pd.read_csv(csv_path)
            tmp.columns = [str(c).strip().title() for c in tmp.columns]
            # If there's no Price column, synthesize from Close
            if 'Price' not in tmp.columns and 'Close' in tmp.columns:
                tmp.insert(1, 'Price', tmp['Close'])
            # Ensure all needed columns exist
            for col in ['Date','Price','Close','High','Low','Open','Volume']:
                if col not in tmp.columns:
                    tmp[col] = None
            data = tmp[['Date','Price','Close','High','Low','Open','Volume']]

        # Clean types
        def to_float(x):
            if pd.isna(x):
                return None
            s = str(x).replace(',', '').replace('%','').strip()
            if s == '' or s.lower() == 'nan':
                return None
            try:
                return float(s)
            except Exception:
                return None
        def to_int(x):
            f = to_float(x)
            return int(f) if f is not None else 0
        out = pd.DataFrame()
        out['date'] = pd.to_datetime(data['Date'], errors='coerce')
        out['open'] = data['Open'].apply(to_float)
        out['high'] = data['High'].apply(to_float)
        out['low'] = data['Low'].apply(to_float)
        out['close'] = data['Close'].apply(to_float)
        out['adj_close'] = data['Price'].apply(to_float)
        out['volume'] = data['Volume'].apply(to_int)
        out = out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        # Some rows may have None for prices; drop them
        out = out.dropna(subset=['close'])
        return out
    except Exception as e:
        raise RuntimeError(f"Failed parsing price CSV for {symbol}: {e}")


def parse_fundamentals(json_path: Path, symbol: str) -> Dict[str, Any]:
    """Parse and normalize fundamentals JSON; coerce numeric-looking strings; keep text as-is."""
    try:
        raw = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed reading fundamentals for {symbol}: {e}")

    def norm_val(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            s = v.strip().replace(',', '')
            if s.endswith('%'):
                s = s[:-1]
            # Try int then float
            try:
                if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                    return int(s)
                return float(s)
            except Exception:
                return v
        # For lists/dicts or other objects, serialize to string
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    norm = {k.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower(): norm_val(v)
            for k, v in raw.items()}
    norm['symbol'] = symbol
    return norm


def convert_all(source: Path, out_daily: Path, out_fund_dir: Optional[Path], aggregate_fund: Optional[Path], per_symbol_fund_parquet_dir: Optional[Path] = None, limit: Optional[int] = None, only_missing: bool = False) -> None:
    out_daily.mkdir(parents=True, exist_ok=True)
    if out_fund_dir:
        out_fund_dir.mkdir(parents=True, exist_ok=True)
    if per_symbol_fund_parquet_dir:
        per_symbol_fund_parquet_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted([p.name for p in source.iterdir() if p.is_dir()])
    if limit is not None:
        symbols = symbols[:limit]

    agg_rows: List[Dict[str, Any]] = []
    processed = 0
    errors = 0
    for sym in symbols:
        sym_dir = source / sym
        price_csv = sym_dir / f"{sym}_price.csv"
        fund_json = sym_dir / f"{sym}_advanced.json"
        try:
            parquet_path = out_daily / f"{sym}.parquet"
            # Skip price conversion if parquet already exists and only_missing is set
            if only_missing and parquet_path.exists():
                pass
            else:
                if price_csv.exists():
                    df = parse_price_csv(price_csv, sym)
                    if len(df) == 0:
                        raise RuntimeError("no rows after parsing")
                    # Write parquet
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(parquet_path, index=False)
                else:
                    raise FileNotFoundError(f"missing {price_csv.name}")
            # Fundamentals
            if fund_json.exists():
                norm = parse_fundamentals(fund_json, sym)
                if out_fund_dir:
                    (out_fund_dir / f"{sym}.json").write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding='utf-8')
                if per_symbol_fund_parquet_dir:
                    import pandas as pd
                    fp = per_symbol_fund_parquet_dir / f"{sym}.parquet"
                    pd.DataFrame([norm]).to_parquet(fp, index=False)
                agg_rows.append(norm)
            processed += 1
            if processed % 200 == 0:
                print(f"Processed {processed} symbols…")
        except Exception as e:
            errors += 1
            print(f"[WARN] {sym}: {e}")
    # Aggregate fundamentals parquet
    if aggregate_fund and agg_rows:
        try:
            agg_df = pd.DataFrame(agg_rows)
            # Order columns
            cols = ['symbol'] + [c for c in agg_df.columns if c != 'symbol']
            agg_df = agg_df[cols]
            aggregate_fund.parent.mkdir(parents=True, exist_ok=True)
            agg_df.to_parquet(aggregate_fund, index=False)
            print(f"Wrote fundamentals aggregate to {aggregate_fund}")
        except Exception as e:
            print(f"[WARN] Failed to write fundamentals aggregate: {e}")
    print(f"Done. processed={processed} errors={errors}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert stock_data to parquet for app")
    ap.add_argument('--source', type=Path, default=Path('stock_data'))
    ap.add_argument('--out', type=Path, default=Path('data/bronze/daily'))
    ap.add_argument('--fund-out', type=Path, default=Path('data/bronze/fundamentals'))
    ap.add_argument('--aggregate-fund', type=Path, default=Path('data/bronze/fundamentals.parquet'))
    ap.add_argument('--limit', type=int, default=None, help='process only first N symbols (for testing)')
    ap.add_argument('--fund-parquet-dir', type=Path, default=None, help='directory for per-symbol fundamentals parquet outputs')
    ap.add_argument('--only-missing', action='store_true', help='skip symbols that already have parquet output')
    args = ap.parse_args(argv)

    convert_all(args.source, args.out, args.fund_out, args.aggregate_fund, args.fund_parquet_dir, args.limit, args.only_missing)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
