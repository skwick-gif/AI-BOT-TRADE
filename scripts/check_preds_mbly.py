import pandas as pd
from pathlib import Path

def check():
    p = Path('data/silver/preds')
    for h in [1,5,10]:
        fp = p / f'preds_h{h}.parquet'
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                total = len(df)
                mbly = 0
                if 'ticker' in df.columns:
                    mbly = int((df['ticker'].str.upper()=='MBLY').sum())
                print(f'h{h}: file exists, rows={total}, MBLY_rows={mbly}')
            except Exception as e:
                print(f'h{h}: exists but failed to read: {e}')
        else:
            print(f'h{h}: file not found')

if __name__ == '__main__':
    check()
