#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml.dataset import load_bronze

print('Testing PARQUET loading with progress...')
progress_values = []

def progress_cb(p):
    progress_values.append(p)
    if len(progress_values) % 500 == 0:  # Print every 500 files
        print(f'Progress: {p}% ({len(progress_values)} files processed)')

try:
    bronze = load_bronze('data/bronze/daily', progress_callback=progress_cb)
    print(f'Loaded {len(bronze)} tickers total')
    if len(bronze) > 0:
        total_rows = sum(len(df) for df in bronze.values())
        print(f'Total rows: {total_rows}')
        print('Sample tickers:', list(bronze.keys())[:5])
    print('Test completed successfully!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()