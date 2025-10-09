"""Identify problematic parquet files based on the validation CSV.
Flags files that:
 - have fewer than MIN_ROWS (default 250)
 - missing any required columns (date/open/high/low/close/adj_close/volume)
 - have > MAX_MISSING_PCT missing values in any required column
Output:
 - scripts/problematic_parquets.csv (detailed)
 - scripts/problematic_parquets_list.txt (one filename per line)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_CSV = Path(__file__).resolve().parent / 'parquet_validation_report.csv'
OUT_CSV = Path(__file__).resolve().parent / 'problematic_parquets.csv'
OUT_LIST = Path(__file__).resolve().parent / 'problematic_parquets_list.txt'

MIN_ROWS = 250
MAX_MISSING_PCT = 20.0  # percent
REQUIRED_COLS = ['date','open','high','low','close','adj_close','volume']

if not VALIDATION_CSV.exists():
    print('Validation CSV not found:', VALIDATION_CSV)
    raise SystemExit(1)

df = pd.read_csv(VALIDATION_CSV)
problems = []
for _, row in df.iterrows():
    fname = row['file']
    rows = int(row['rows']) if not pd.isna(row['rows']) else 0
    # check missing required cols
    missing_cols = [c for c in REQUIRED_COLS if not bool(row.get(f'has_{c}', False))]
    # check large missingness
    high_missing = []
    for c in REQUIRED_COLS:
        misscol = f'misspct_{c}'
        if misscol in row and not pd.isna(row[misscol]):
            try:
                pct = float(row[misscol])
                if pct > MAX_MISSING_PCT:
                    high_missing.append((c,pct))
            except Exception:
                pass
    reason = []
    if rows < MIN_ROWS:
        reason.append(f'low_rows({rows})')
    if missing_cols:
        reason.append('missing_cols(' + ','.join(missing_cols) + ')')
    if high_missing:
        reason.append('high_missing(' + ','.join([f"{c}:{p}" for c,p in high_missing]) + ')')
    if reason:
        problems.append({
            'file': fname,
            'rows': rows,
            'missing_cols': ','.join(missing_cols) if missing_cols else '',
            'high_missing': ';'.join([f"{c}:{p}" for c,p in high_missing]) if high_missing else '',
            'reasons': ';'.join(reason)
        })

out_df = pd.DataFrame(problems)
out_df.to_csv(OUT_CSV, index=False)
with open(OUT_LIST, 'w', encoding='utf-8') as f:
    for fn in out_df['file'].tolist():
        f.write(fn + '\n')

print('Found', len(out_df), 'problematic files; details in', OUT_CSV)
print('List written to', OUT_LIST)
print('\nSample:')
print(out_df.head(20).to_string(index=False))
