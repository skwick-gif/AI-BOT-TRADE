from pathlib import Path
import sys
# Ensure 'src' directory is on sys.path so package imports (core, ui, ml, etc.) work
project_root = Path('.').resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
	sys.path.insert(0, str(src_path))

from ui.widgets.ml_widget import PipelineRunWorker

w = PipelineRunWorker()
# Attach simple print to completed and error
w.completed.connect(lambda d: print('COMPLETED_PAYLOAD:', d))
w.error_occurred.connect(lambda e: print('ERROR:', e))

params = {'tickers': ['MBLY'], 'holdout': 30, 'step': 5, 'lookback':500, 'window':'expanding', 'models': ['LogisticRegression']}
print('Running worker...')
w.run(params)
print('Worker finished')
