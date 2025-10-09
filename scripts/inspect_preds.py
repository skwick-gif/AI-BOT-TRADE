import pandas as pd
from pathlib import Path
p = Path('data/silver/preds')
for h in [1,5,10]:
    fp = p / f'preds_h{h}.parquet'
    print('\n== H', h, '==')
    if not fp.exists():
        print('file not found')
        continue
    df = pd.read_parquet(fp)
    print('rows=', len(df))
    print('columns=', df.columns.tolist())
    if 'ticker' in df.columns:
        print('unique tickers=', df['ticker'].str.upper().nunique())
        up = df['ticker'].str.upper().unique().tolist()
        sample = up[:10]
        print('example tickers (first 10):', sample)
        mbly_rows = df[df['ticker'].str.upper()=='MBLY']
        if len(mbly_rows):
            print('MBLY rows count:', len(mbly_rows))
            print(mbly_rows.head().to_string())
        else:
            print('MBLY not present in this horizon')
    else:
        print('no ticker column')
