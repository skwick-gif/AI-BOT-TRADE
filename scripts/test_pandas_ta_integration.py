"""Quick integration test for pandas_ta and repo feature functions.
Creates a small OHLCV DataFrame, runs build_technical_features, and calls get_technicals.
"""
import sys
import importlib
import pandas as pd
import os
from pathlib import Path

print('Testing pandas_ta availability and repo feature functions...')

# Make sure the `src` package is importable when running this script from the repo root
repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
try:
    import pandas_ta as ta
    print('pandas_ta imported, version:', getattr(ta, '__version__', 'unknown'))
except Exception as e:
    print('Failed to import pandas_ta:', e)

# Ensure repo root in sys.path
sys.path.insert(0, '.')
from src.utils.trading_helpers import get_technicals
from src.ml.features.technical import build_technical_features

# create sample OHLCV
dates = pd.date_range(end=pd.Timestamp.today(), periods=40)
df = pd.DataFrame({
    'date': dates,
    'open': 100 + (pd.Series(range(40)).cumsum())*0.1,
    'high': 100 + (pd.Series(range(40)).cumsum())*0.11,
    'low': 100 + (pd.Series(range(40)).cumsum())*0.09,
    'close': 100 + (pd.Series(range(40)).cumsum())*0.1,
    'adj_close': 100 + (pd.Series(range(40)).cumsum())*0.1,
    'volume': 100000 + pd.Series(range(40))*100
})

cfg = {
    'ret_windows': [1,5,10],
    'ema_windows': [5,20],
    'macd': [12,26,9],
    'rsi_window':14,
    'atr_window':14,
    'bbands_window':20
}

out = build_technical_features(df, cfg)
print('Result columns (sample):', [c for c in out.columns if c in ['close','ema_5','sma_5','rsi','atr'] or c.startswith('ret_')][:20])
print('Sample tail for key columns:')
print(out.tail()[['close','ema_5','sma_5','rsi','atr']].fillna('NA'))

print('\nNow testing get_technicals for AAPL (may attempt to download via yfinance)')
try:
    res = get_technicals('AAPL')
    print('get_technicals returned type:', type(res), 'columns count:', len(res.columns) if hasattr(res,'columns') else None)
    print('Columns (sample):', list(res.columns)[:10])
except Exception as e:
    print('get_technicals call failed:', e)
