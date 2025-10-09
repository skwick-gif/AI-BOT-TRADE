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

# Prepare compact payload and prompt
compact = {
    'symbol': 'AAPL',
    'overall_signal': 'HOLD',
    'price_targets': {1:150.0,5:152.0,10:155.0, 'meta': {1: {'confidence':0.6,'model':'RF','date':'2025-10-08'},5:{'confidence':0.5,'model':'RF','date':'2025-10-08'},10:{'confidence':0.55,'model':'RF','date':'2025-10-08'}}},
    'per_horizon': {1: {'signal':'HOLD','confidence':0.6},5:{'signal':'HOLD','confidence':0.5},10:{'signal':'HOLD','confidence':0.55}}
}
prompt = ml._build_generic_prompt('AAPL', compact)
print('Prompt preview:', prompt[:120])

ml._set_ai_sending_state(prompt, 'AAPL')
ml._auto_ask_ai_async('AAPL', compact, prompt)

# Wait up to 40s
for i in range(40):
    # Process Qt events so timers and thread signals are handled
    try:
        app.processEvents()
    except Exception:
        pass
    item = ml.one_symbol_table.item(0,7)
    txt = item.text() if item else '<no-item>'
    print(f'[{i}] AI cell: {txt}')
    if item and txt not in ('(sending...)','AI timeout','-'):
        print('Final AI cell content (tooltip):', item.toolTip())
        break
    time.sleep(1)

app.quit()
