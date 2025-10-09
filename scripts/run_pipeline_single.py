# Synchronously run PipelineRunWorker.run() for a single ticker (useful for CI/debug)
import sys
from pathlib import Path
# Ensure project src on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from ui.widgets.ml_widget import PipelineRunWorker
except Exception as e:
    print('Failed to import PipelineRunWorker:', e)
    raise

if __name__ == '__main__':
    params = {
        'tickers': ['MBLY'],
        'holdout': 30,
        'step': 5,
        'lookback': 500,
        'window': 'expanding',
        'models': ['RandomForest'],
        'max_loops': 1,  # keep short for debugging
        'use_parallel': False,
    }
    worker = PipelineRunWorker()
    try:
        worker.run(params)
        print('Pipeline run completed (synchronously)')
    except Exception as e:
        print('Pipeline run failed:', e)
        import traceback
        traceback.print_exc()
