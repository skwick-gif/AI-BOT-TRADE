"""Run DataUpdateService.run_now synchronously to test the one-shot attached script integration.
This script creates a service with cfg.source_dir pointing to data/sandbox/stock_data so it won't touch production.
"""
from pathlib import Path
import sys
repo = Path(__file__).parent.parent
sys.path.insert(0, str(repo / 'src'))

from services.data_update_service import DataUpdateService, UpdateConfig

cfg = UpdateConfig(
    source_dir=Path('data/sandbox/stock_data'),
    parquet_out=Path('data/sandbox/bronze/daily'),
    fund_json_out=Path('data/sandbox/bronze/fundamentals'),
    fund_parquet_out=Path('data/sandbox/bronze/fundamentals.parquet'),
    fund_parquet_dir=Path('data/sandbox/bronze/fund_parquet'),
    start_date='2020-01-01',
    batch_limit=5,
)

svc = DataUpdateService(cfg)
# run synchronously by calling worker.run directly (no Qt thread here)
w = svc._worker = svc._worker = svc._worker = None
from services.data_update_service import DataUpdateWorker
w = DataUpdateWorker(cfg)
print('Starting worker one-shot run (this may take a while)...')
w.run()
print('Worker finished')
