import sys
sys.path.insert(0, 'src')
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.config_manager import ConfigManager

app = QApplication(sys.argv)
config = ConfigManager()
print('Creating MainWindow...')
try:
    window = MainWindow(config)
    print('MainWindow created successfully')
    print(f'IBKR service: {hasattr(window, "ibkr_service")}')
    if hasattr(window, 'ibkr_service'):
        print(f'Service type: {type(window.ibkr_service)}')
        print('Testing IBKR connection...')
        connected = window.ibkr_service.connect()
        print(f'Connection result: {connected}')
        if hasattr(window.ibkr_service, 'is_connected'):
            print(f'Is connected: {window.ibkr_service.is_connected()}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()