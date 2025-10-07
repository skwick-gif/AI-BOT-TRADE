"""
Run a small limited scan using an external stocks.py script (provided by the user).
This script imports the user's stocks.py (from IndicesTrading) and runs update_price_data
and scrape_all_advanced for the first N tickers. It writes log lines to stdout.

Usage: python run_stocks_small.py --limit 5
"""
import argparse
import runpy
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=5)
args = parser.parse_args()

# Path to the user-provided stocks script (IndicesTrading)
STOCKS_PATH = r'c:\MyProjects\IndicesTrading\stocks.py'

try:
    globs = runpy.run_path(STOCKS_PATH)
except Exception as e:
    print(f"ERROR: failed to load stocks script at {STOCKS_PATH}: {e}")
    traceback.print_exc()
    sys.exit(2)

all_tickers = globs.get('all_tickers') or []
update_price_data = globs.get('update_price_data')
scrape_all_advanced = globs.get('scrape_all_advanced')

if update_price_data is None or scrape_all_advanced is None:
    print("ERROR: stocks script does not expose required functions (update_price_data, scrape_all_advanced)")
    sys.exit(3)

limit = args.limit if args.limit and args.limit > 0 else 5
tickers = list(all_tickers)[:limit]
print(f"Running small scan for {len(tickers)} tickers: {tickers}")

for t in tickers:
    try:
        print('\n' + '='*40)
        print(f"Processing: {t}")
        try:
            update_price_data(t, globs.get('START_DATE', '2020-01-01'), globs.get('DATA_FOLDER', 'stock_data'))
        except Exception as e:
            print(f"update_price_data error for {t}: {e}")
        try:
            scrape_all_advanced(t, globs.get('DATA_FOLDER', 'stock_data'))
        except Exception as e:
            print(f"scrape_all_advanced error for {t}: {e}")
        print(f"Finished: {t}")
    except Exception as e:
        print(f"Unexpected error for {t}: {e}")

print('\nSmall scan complete.')
