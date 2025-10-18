"""
Download fundamentals and advanced data for all tickers.
Separated from price data - only needs to run occasionally (weekly/monthly).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from download_stocks import get_all_tickers, scrape_all_data, DATA_FOLDER
import argparse

def download_all_fundamentals(limit=None):
    """Download fundamentals & advanced data for all tickers"""
    tickers = get_all_tickers()
    print(f"Starting fundamentals & advanced data download for {len(tickers)} tickers...")
    print(f"WARNING: This is a SLOW process - fundamentals require web scraping")
    print(f"Rate limiting is in place to avoid blocking")
    
    if limit:
        tickers = tickers[:int(limit)]
        print(f"Limited to first {len(tickers)} tickers")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            # Check if already exists
            json_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}_advanced.json")
            if os.path.exists(json_path):
                import json
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if len(data) >= 10:
                        print(f"  ✓ {ticker} already has sufficient fundamentals data - skipping")
                        skipped_count += 1
                        continue
                except:
                    pass
            
            # Download fundamentals
            if scrape_all_data(ticker, DATA_FOLDER):
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"  ✗ Error downloading advanced data for {ticker}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Fundamentals & advanced data download completed!")
    print(f"Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}, Total: {len(tickers)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download stock fundamentals and advanced data')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process')
    
    args = parser.parse_args()
    download_all_fundamentals(limit=args.limit)
