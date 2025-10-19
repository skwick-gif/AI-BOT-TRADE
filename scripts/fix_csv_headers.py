"""
Fix CSV headers - remove the old 3-line header format and convert to standard pandas format.
Old format:
    Price,Close,High,Low,Open,Volume
    Ticker,XXX,XXX,XXX,XXX,XXX
    Date,,,,,
    
New format:
    Date,Open,High,Low,Close,Volume
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def fix_csv_file(file_path):
    """Fix a single CSV file by removing extra header lines"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if file has the old format (3 header lines)
        if len(lines) < 4:
            return False
            
        # Check if line 2 starts with "Ticker," - this is the old format
        if not lines[1].strip().startswith('Ticker,'):
            return False  # Already in correct format
        
        # Remove lines 1 and 2 (keeping line 0 and 3+)
        # But also replace the header with standard pandas format
        new_lines = ['Date,Open,High,Low,Close,Volume\n'] + lines[3:]
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    stock_data_dir = PROJECT_ROOT / "stock_data"
    
    if not stock_data_dir.exists():
        print(f"Stock data directory not found: {stock_data_dir}")
        return
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Get all ticker directories
    ticker_dirs = [d for d in stock_data_dir.iterdir() if d.is_dir()]
    total = len(ticker_dirs)
    
    print(f"Scanning {total} ticker directories...")
    
    for i, ticker_dir in enumerate(ticker_dirs, 1):
        ticker = ticker_dir.name
        price_file = ticker_dir / f"{ticker}_price.csv"
        
        if not price_file.exists():
            continue
        
        try:
            if fix_csv_file(price_file):
                fixed_count += 1
                if fixed_count % 100 == 0:
                    print(f"[{i}/{total}] Fixed {fixed_count} files so far...")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"CSV Header Fix Completed!")
    print(f"Fixed: {fixed_count}")
    print(f"Skipped (already correct): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total processed: {total}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
