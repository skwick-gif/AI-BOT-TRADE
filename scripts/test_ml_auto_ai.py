import sys, time
sys.path.insert(0, r'C:/Users/eranl/Downloads/AI-BOT-TRADE/src')
from PyQt6.QtWidgets import QApplication
from ui.widgets.ml_widget import MLWidget

app = QApplication([])
ml = MLWidget()
# Ensure Auto AI is checked
try:
    ml.auto_ai_checkbox.setChecked(True)
except Exception:
    pass
# Set the single stock input so on_pipeline_completed treats this as a single-ticker run
try:
    ml.single_stock_input.setText('AAPL')
except Exception:
    pass

# Build a minimal compact payload for AAPL
compact = {
    'symbol': 'AAPL',
    'overall_signal': 'HOLD',
    'price_targets': {1:150.0,5:152.0,10:155.0, 'meta': {1: {'confidence':0.6,'model':'RF','date':'2025-10-08'},5:{'confidence':0.5,'model':'RF','date':'2025-10-08'},10:{'confidence':0.55,'model':'RF','date':'2025-10-08'}}},
    'per_horizon': {1: {'signal':'HOLD','confidence':0.6},5:{'signal':'HOLD','confidence':0.5},10:{'signal':'HOLD','confidence':0.55}}
}

# Simulate pipeline completed signal with compact table
ml.on_pipeline_completed({'compact_table': compact, 'saved_predictions':1, 'best_result':{}})

# Wait up to 35s for AI response to appear in table
for i in range(35):
    item = ml.one_symbol_table.item(0,7)
    txt = item.text() if item else '<no-item>'
    print(f'[{i}] AI cell: {txt}')
    if item and txt not in ('(sending...)','AI timeout','-'):
        print('Final AI cell content (tooltip):', item.toolTip())
        break
    time.sleep(1)

# Clean exit
app.quit()
