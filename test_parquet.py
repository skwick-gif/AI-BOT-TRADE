#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml.dataset import load_bronze

print('Testing PARQUET loading...')
try:
    bronze = load_bronze('data/bronze/daily')
    print(f'Loaded {len(bronze)} tickers total')
    if len(bronze) > 0:
        total_rows = sum(len(df) for df in bronze.values())
        print(f'Total rows: {total_rows}')
        print('Sample tickers:', list(bronze.keys())[:10])
        print('Test completed successfully!')
    else:
        print('No data loaded!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()