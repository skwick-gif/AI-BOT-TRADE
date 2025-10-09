"""Validate local parquet files in data/bronze/daily for required OHLCV schema.
Writes a CSV report to scripts/parquet_validation_report.csv and prints a summary.
"""
from pathlib import Path
import pandas as pd
import sys

REQUIRED_COLS = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'bronze' / 'daily'
OUT_CSV = Path(__file__).resolve().parent / 'parquet_validation_report.csv'

rows = []
if not DATA_DIR.exists():
    print(f"Data directory not found: {DATA_DIR}")
    sys.exit(1)

parquet_files = sorted(DATA_DIR.glob('*.parquet'))
if not parquet_files:
    print(f"No parquet files found in {DATA_DIR}")
    sys.exit(1)

for p in parquet_files:
    info = {'file': str(p.name)}
    try:
        df = pd.read_parquet(p)
        info['rows'] = len(df)
        info['columns'] = ','.join(df.columns.tolist())
        # normalize column names lower for checks
        cols_lower = [c.lower() for c in df.columns]
        for col in REQUIRED_COLS:
            info[f'has_{col}'] = col in cols_lower
        # date checks
        date_col = None
        for c in df.columns:
            if c.lower() == 'date':
                date_col = c
                break
        if date_col is None:
            # try common alternatives
            for c in df.columns:
                if c.lower() in ('datetime','timestamp','time'):
                    date_col = c
                    break
        info['date_col'] = date_col if date_col is not None else ''
        if date_col:
            try:
                s = pd.to_datetime(df[date_col])
                info['first_date'] = str(s.min())
                info['last_date'] = str(s.max())
            except Exception as e:
                info['first_date'] = 'invalid'
                info['last_date'] = 'invalid'
        else:
            info['first_date'] = ''
            info['last_date'] = ''
        # missingness for required cols
        for col in REQUIRED_COLS:
            # find actual column name case-insensitive
            actual = None
            for c in df.columns:
                if c.lower() == col:
                    actual = c
                    break
            if actual is None:
                info[f'misspct_{col}'] = ''
            else:
                miss = df[actual].isna().mean()
                info[f'misspct_{col}'] = round(float(miss) * 100, 3)
    except Exception as e:
        info['error'] = str(e)
    rows.append(info)

rep = pd.DataFrame(rows)
rep.to_csv(OUT_CSV, index=False)
print('Wrote report to', OUT_CSV)
print(rep[['file','rows','date_col','first_date','last_date']].to_string(index=False))
print('\nSample missingness (first 10 files):')
print(rep[[c for c in rep.columns if c.startswith('misspct_')]].head(10).to_string(index=False))
