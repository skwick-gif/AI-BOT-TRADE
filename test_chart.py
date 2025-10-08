"""
Test chart functionality from watchlist
"""

def test_chart_import():
    try:
        from ui.widgets.scanner_widget import ChartDialog
        print("✅ ChartDialog imported successfully")
        print(f"ChartDialog class: {ChartDialog}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ChartDialog: {e}")
        return False

def test_chart_creation():
    if not test_chart_import():
        return False
    
    try:
        from PyQt6.QtWidgets import QApplication
        import sys
        
        # Create application if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        from ui.widgets.scanner_widget import ChartDialog
        dialog = ChartDialog("AAPL")
        print("✅ ChartDialog created successfully")
        dialog.close()
        return True
    except Exception as e:
        print(f"❌ Failed to create ChartDialog: {e}")
        return False

if __name__ == "__main__":
    print("Testing Chart functionality...")
    test_chart_creation()