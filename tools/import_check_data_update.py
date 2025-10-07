import importlib.util
spec = importlib.util.spec_from_file_location('dlg', r'c:\MyProjects\AI-BOT-TRADE\src\ui\dialogs\data_update_dialog.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('import ok')
