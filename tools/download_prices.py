"""
Download only price data for all tickers.
Separated from fundamentals for better control and performance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from download_stocks import get_all_tickers, update_price_data, DATA_FOLDER, START_DATE
import argparse

def download_all_prices(limit=None, latest_only=False):
    """Download price data for all tickers"""
    tickers = get_all_tickers()
    print(f"Starting price data download for {len(tickers)} tickers...")
    print(f"Options: limit={limit or 'all'}, latest_only={latest_only}")
    
    if limit:
        tickers = tickers[:int(limit)]
        print(f"Limited to first {len(tickers)} tickers")
    
    success_count = 0
    error_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            update_price_data(ticker, START_DATE, DATA_FOLDER, latest_only=latest_only)
            success_count += 1
        except Exception as e:
            print(f"Error downloading price data for {ticker}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Price data download completed!")
    print(f"Success: {success_count}, Errors: {error_count}, Total: {len(tickers)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download stock price data')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process')
    parser.add_argument('--latest-only', action='store_true', help='Download only latest data (last 5 days)')
    
    args = parser.parse_args()
    download_all_prices(limit=args.limit, latest_only=args.latest_only)
