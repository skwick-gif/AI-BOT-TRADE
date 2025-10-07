"""Run a safe small scan using the adapter on a short explicit ticker list and
write outputs to sandbox folders under 'data/sandbox' to avoid touching
production parquet.
"""
import sys
from pathlib import Path

# ensure src on path
SRC = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC))

from services.data_update_service import UpdateConfig
from services import stock_data_adapter

SANDBOX = Path('data/sandbox')
SRC_FOLDER = SANDBOX / 'stock_data'
PARQUET_OUT = SANDBOX / 'bronze' / 'daily'
FUND_JSON = SANDBOX / 'bronze' / 'fundamentals'
FUND_PARQUET = SANDBOX / 'bronze' / 'fundamentals.parquet'
FUND_DIR = SANDBOX / 'bronze' / 'fundamentals_by_symbol'

# Create sandbox folders
PARQUET_OUT.mkdir(parents=True, exist_ok=True)
FUND_JSON.mkdir(parents=True, exist_ok=True)
FUND_DIR.mkdir(parents=True, exist_ok=True)
SRC_FOLDER.mkdir(parents=True, exist_ok=True)

# Small safe ticker list
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

cfg = UpdateConfig(
    source_dir=SRC_FOLDER,
    parquet_out=PARQUET_OUT,
    fund_json_out=FUND_JSON,
    fund_parquet_out=FUND_PARQUET,
    fund_parquet_dir=FUND_DIR,
    start_date="2020-01-01",
    batch_limit=len(TICKERS),
)

print("Running safe small scan for:", TICKERS)
res = stock_data_adapter.process_batch(cfg, cfg.batch_limit, lambda p: print(f"[PROG]{p}"), lambda s: print(f"[LOG]{s}"), lambda: False, tickers=TICKERS)
print("Result:", res)
print("Sandbox parquet folder:", PARQUET_OUT.resolve())
