"""
Small runner to exercise update_price_data and scrape_all_advanced from the provided script
for a short list of tickers (5). This does not run the full scan and does not modify TODO files.
"""
import importlib.util
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name('stocks_backup_run.py')
if not SCRIPT_PATH.exists():
    print(f"Error: {SCRIPT_PATH} not found")
    sys.exit(1)

spec = importlib.util.spec_from_file_location('stocks_backup_run', str(SCRIPT_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# quick internet check
if not mod.check_internet_connection():
    print("No internet connection detected. Aborting small scan.")
    sys.exit(1)

# small set of tickers to test — change here if you prefer others
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

print(f"Running small scan for {len(TICKERS)} tickers: {TICKERS}")
for t in TICKERS:
    try:
        print('\n' + '='*40)
        print(f"Processing {t}")
        # ensure folder exists
        os.makedirs(os.path.join(mod.DATA_FOLDER, t), exist_ok=True)
        # attempt to update price data (should only fetch missing/new entries)
        try:
            mod.update_price_data(t, mod.START_DATE, mod.DATA_FOLDER)
        except Exception as e:
            print(f"update_price_data raised: {e}")
        # attempt to fetch advanced data
        try:
            mod.scrape_all_advanced(t, mod.DATA_FOLDER)
        except Exception as e:
            print(f"scrape_all_advanced raised: {e}")
        print(f"Finished {t}")
    except Exception as e:
        print(f"Unexpected error for {t}: {e}")

print('\nSmall scan complete.')
