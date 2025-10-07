@echo off
echo Testing manual IBKR connection (should not block UI)...
echo.
echo This script will test that the connection runs in background
echo and does not freeze the interface.
echo.
pause
python -c "
import sys
import os
sys.path.insert(0, 'src')

from PyQt6.QtWidgets import QApplication
from ui.windows.main_window import MainWindow
from core.config_manager import ConfigManager
import time

app = QApplication(sys.argv)
config = ConfigManager()
window = MainWindow()
window.show()

print('Window shown, starting manual connection test...')
print('If the window freezes, there is a problem.')
print('If the window stays responsive, the fix works!')

# Simulate clicking the connect button
window.connect_ibkr()

# Give it some time to start the connection
time.sleep(2)

print('Connection attempt started in background.')
print('Check if the window is still responsive...')

# Keep the app running briefly to test
app.processEvents()
time.sleep(3)

print('Test completed. Window should have stayed responsive.')
"