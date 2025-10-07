"""
Run a small limited scan using stocks_attached.py
This script runs update_price_data and scrape_all_advanced for the first N tickers.
It writes log lines to stdout.

Usage: python run_stocks_small.py --limit 5
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=5)
args = parser.parse_args()

# Path to the stocks_attached.py script
STOCKS_PATH = Path(__file__).parent / 'stocks_attached.py'

try:
    spec = importlib.util.spec_from_file_location('stocks_attached', str(STOCKS_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as e:
    print(f"ERROR: failed to load stocks script at {STOCKS_PATH}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(2)

all_tickers = getattr(mod, 'all_tickers', [])
update_price_data = getattr(mod, 'update_price_data', None)
scrape_all_advanced = getattr(mod, 'scrape_all_advanced', None)

if update_price_data is None or scrape_all_advanced is None:
    print("ERROR: stocks script does not expose required functions (update_price_data, scrape_all_advanced)")
    sys.exit(3)

limit = args.limit if args.limit and args.limit > 0 else 5
tickers = list(all_tickers)[:limit]
print(f"Running small scan for {len(tickers)} tickers: {tickers}", flush=True)

for t in tickers:
    try:
        print('\n' + '='*40, flush=True)
        print(f"Processing: {t}", flush=True)
        try:
            update_price_data(t, getattr(mod, 'START_DATE', '2020-01-01'), getattr(mod, 'DATA_FOLDER', 'stock_data'))
        except Exception as e:
            print(f"update_price_data error for {t}: {e}", flush=True)
        try:
            scrape_all_advanced(t, getattr(mod, 'DATA_FOLDER', 'stock_data'))
        except Exception as e:
            print(f"scrape_all_advanced error for {t}: {e}", flush=True)
        print(f"Finished {t}", flush=True)
    except Exception as e:
        print(f"Unexpected error for {t}: {e}", flush=True)
    
    # Add delay between tickers to avoid rate limiting
    if t != tickers[-1]:  # Don't sleep after the last ticker
        time.sleep(2)

print('\nSmall scan complete.', flush=True)