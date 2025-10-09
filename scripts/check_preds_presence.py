import pandas as pd
from pathlib import Path
for h in [1,5,10]:
    fp = Path(f"data/silver/preds/preds_h{h}.parquet")
    if fp.exists():
        df = pd.read_parquet(fp)
        has = 'MBLY' in set(df['ticker'].str.upper()) if 'ticker' in df.columns else False
        print(h, 'rows', len(df), 'has MBLY?', has)
    else:
        print(h, 'missing')
