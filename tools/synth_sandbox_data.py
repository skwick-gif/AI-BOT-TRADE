"""Create small synthetic CSV and JSON files for safe tickers in data/sandbox and run the converter to produce parquet outputs.
This is only for UI testing (no network).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent
SANDBOX = ROOT / 'data' / 'sandbox'
SRC = SANDBOX / 'stock_data'
OUT_DAILY = SANDBOX / 'bronze' / 'daily'
OUT_FUND = SANDBOX / 'bronze' / 'fundamentals'
OUT_FUND_PQ = SANDBOX / 'bronze' / 'fundamentals.parquet'
PER_SYMBOL_FUND_PQ = SANDBOX / 'bronze' / 'fund_parquet'

TICKERS = ['AAPL','MSFT','GOOGL','AMZN','TSLA']

# create synthetic price CSVs
SRC.mkdir(parents=True, exist_ok=True)
for t in TICKERS:
    d = SRC / t
    d.mkdir(parents=True, exist_ok=True)
    # 60 days of data
    dates = [datetime.today().date() - timedelta(days=i) for i in range(60,0,-1)]
    opens = np.random.uniform(100,500, size=len(dates)).round(2)
    highs = opens + np.random.uniform(0,10,size=len(dates)).round(2)
    lows = opens - np.random.uniform(0,10,size=len(dates)).round(2)
    closes = (opens + highs + lows) / 3
    adj = closes
    vols = np.random.randint(1000000,5000000,size=len(dates))
    df = pd.DataFrame({
        'Date': [d.isoformat() for d in dates],
        'Price': adj,
        'Close': closes,
        'High': highs,
        'Low': lows,
        'Open': opens,
        'Volume': vols,
    })
    df.to_csv(d / f"{t}_price.csv", index=False)
    # simple fundamentals
    fund = {
        'Market Cap': str(int(np.random.uniform(50e9, 2e12))),
        'PE Ratio (TTM)': round(np.random.uniform(10,50),2),
        'Dividend Yield': f"{round(np.random.uniform(0,5),2)}%",
        'Sector': 'Technology',
        'Industry': 'Software',
        'symbol': t
    }
    with open(d / f"{t}_advanced.json", 'w', encoding='utf-8') as f:
        json.dump(fund, f, ensure_ascii=False, indent=2)

print('Synthetic sandbox stock_data created at', SRC)

# run converter
import runpy
conv = runpy.run_path(str(ROOT / 'scripts' / 'convert_stock_data_to_parquet.py'))
convert_all = conv.get('convert_all')
if convert_all:
    convert_all(SRC, OUT_DAILY, OUT_FUND, OUT_FUND_PQ, PER_SYMBOL_FUND_PQ, None, False)
    print('Conversion finished. Parquet output at', OUT_DAILY)
else:
    print('convert_all not found')
