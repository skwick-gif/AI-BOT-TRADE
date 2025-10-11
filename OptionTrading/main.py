from PyQt6.QtWidgets import QApplication
import sys
import argparse
from ui.trading_window import TradingWindow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Trading UI")
    parser.add_argument('--offline', action='store_true', help='Run UI without network connections (UI-only)')
    args, extra = parser.parse_known_args()
    app = QApplication(sys.argv)
    win = TradingWindow(offline=args.offline)
    win.show()
    sys.exit(app.exec())
