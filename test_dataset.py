#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml.dataset import load_bronze

print('Testing improved PARQUET loading...')
try:
    bronze = load_bronze('data/bronze/daily')
    print(f'\n✅ Successfully loaded {len(bronze)} tickers!')
    if len(bronze) > 0:
        total_rows = sum(len(df) for df in bronze.values())
        print(f'📊 Total rows: {total_rows:,}')
        print(f'📈 Average rows per ticker: {total_rows // len(bronze):,}')
        print(f'🎯 Sample tickers: {list(bronze.keys())[:10]}')
    print('\n✅ Test completed successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()