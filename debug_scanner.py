#!/usr/bin/env python3
"""Debug script to test ScannerWidget loading"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from PyQt6.QtWidgets import QApplication

try:
    app = QApplication(sys.argv)
    
    from ui.widgets.scanner_widget import ScannerWidget
    
    widget = ScannerWidget()
    
    print(f"ScannerWidget created successfully")
    print(f"Has apply_ai_filter: {hasattr(widget, 'apply_ai_filter')}")
    
    if hasattr(widget, 'apply_ai_filter'):
        print(f"apply_ai_filter type: {type(widget.apply_ai_filter)}")
    else:
        print("Available methods:")
        methods = [m for m in dir(widget) if not m.startswith('_') and callable(getattr(widget, m))]
        for method in sorted(methods):
            if 'ai' in method.lower() or 'filter' in method.lower():
                print(f"  - {method}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(f"Traceback: {traceback.format_exc()}")