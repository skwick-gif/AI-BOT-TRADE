"""
Synchronous test runner for DataUpdateWorker.
Runs the worker.run() method directly for a small batch and prints emitted logs to stdout.
"""
import sys
from pathlib import Path

# Add src directory to Python path (same behavior as main.py)
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.data_update_service import DataUpdateWorker, UpdateConfig

# Test config: limit to small batch to avoid long downloads
cfg = UpdateConfig(batch_limit=5)

worker = DataUpdateWorker(cfg)

# Connect worker signals to print functions for headless testing
worker.log.connect(lambda msg: print(f"[LOG] {msg}"))
worker.progress.connect(lambda v: print(f"[PROG] {v}"))
worker.completed.connect(lambda p: print(f"[DONE] {p}"))
worker.error.connect(lambda m: print(f"[ERR] {m}"))

print("Starting synchronous data update test (batch_limit=5)")
worker.run()
print("Test finished")
