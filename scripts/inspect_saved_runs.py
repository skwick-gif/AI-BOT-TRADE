import os
import sys
import traceback
from pathlib import Path
try:
    import pandas as pd
except Exception:
    pd = None

repo = Path(__file__).resolve().parents[1]
metrics = repo / 'data' / 'silver' / 'metrics.csv'
preds_dir = repo / 'data' / 'silver' / 'preds'

print('Repository:', repo)
print('metrics.csv exists:', metrics.exists(), 'size:', metrics.stat().st_size if metrics.exists() else 'N/A')
print('preds dir exists:', preds_dir.exists())

files = []
if preds_dir.exists():
    files = sorted(preds_dir.glob('preds_h*.parquet'))
    print('Found', len(files), 'preds parquet files')
    for fp in files:
        try:
            if pd is None:
                print(fp.name, '(pandas not available — cannot read rows)')
                continue
            df = pd.read_parquet(fp)
            n = len(df)
            uniq = df['ticker'].nunique() if 'ticker' in df.columns else 'no ticker col'
            print(fp.name, 'rows=', n, 'unique_tickers=', uniq)
        except Exception as e:
            print(fp.name, 'ERROR reading:', e)

# look for other metrics artifacts
silver = repo / 'data' / 'silver'
for name in ['confusions.csv','model_scores.csv','reports','signals.csv','snapshots','universe.csv']:
    p = silver / name
    if p.exists():
        if p.is_file():
            print(str(p.relative_to(repo)), 'file exists size=', p.stat().st_size)
        else:
            print(str(p.relative_to(repo)), 'dir exists')

# quick check for pooled cache
cache = repo / 'data' / 'cache' / 'pooled_dataset.parquet'
print('pooled_dataset.parquet exists:', cache.exists())

# models dir
models = repo / 'models'
print('models dir exists:', models.exists())

print('\nDone')
