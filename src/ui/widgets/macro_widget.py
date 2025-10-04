"""
Simple Macro Economic Indicators Widget
Based on the legacy Streamlit implementation - simplified version
"""

import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from datetime import datetime, timedelta


class SimpleLogger:
    """Simple logger"""
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")


class MacroDataThread(QThread):
    """Thread for fetching macro economic data"""
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.logger = SimpleLogger()
    
    def run(self):
        """Fetch macro economic data"""
        try:
            if not self.api_key:
                self.error_occurred.emit("FRED API key not configured")
                return
            
            # Try to import FRED API
            try:
                from fredapi import Fred
            except ImportError:
                self.error_occurred.emit("fredapi package not installed")
                return
            
            fred = Fred(api_key=self.api_key)
            
            # Simple indicators with safer data fetching
            indicators = {
                'DFF': {'name': 'Fed Funds Rate', 'units': '%'},
                'UNRATE': {'name': 'Unemployment Rate', 'units': '%'},
                'CPIAUCSL': {'name': 'CPI', 'units': 'Index'},
            }
            
            data = {}
            
            for series_id, details in indicators.items():
                try:
                    self.logger.info(f"Fetching {series_id}...")
                    series_data = fred.get_series(series_id, limit=1)  # Get only latest
                    
                    if not series_data.empty:
                        latest_value = series_data.iloc[-1]
                        latest_date = series_data.index[-1].strftime('%Y-%m-%d')
                        
                        data[series_id] = {
                            'name': details['name'],
                            'units': details['units'],
                            'value': latest_value,
                            'date': latest_date
                        }
                    
                except Exception as e:
                    self.logger.error(f"Error fetching {series_id}: {e}")
                    data[series_id] = {
                        'name': details['name'],
                        'units': details['units'],
                        'value': None,
                        'date': None,
                        'error': str(e)
                    }
            
            self.data_ready.emit(data)
            
        except Exception as e:
            self.logger.error(f"Error in macro data thread: {e}")
            self.error_occurred.emit(str(e))


class MacroIndicatorCard(QFrame):
    """Individual macro indicator display card"""
    
    def __init__(self, indicator_data):
        super().__init__()
        self.setup_ui(indicator_data)
    
    def setup_ui(self, data):
        """Setup the indicator card UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setFixedHeight(100)
        self.setMinimumWidth(200)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 5px;
            }
            QLabel {
                color: #ffffff;
                background: transparent;
                border: none;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel(data['name'])
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Value
        if data['value'] is not None:
            value_text = f"{data['value']:.2f} {data['units']}"
            value_label = QLabel(value_text)
            value_font = QFont()
            value_font.setPointSize(13)
            value_font.setBold(True)
            value_label.setFont(value_font)
            value_label.setStyleSheet("color: #4CAF50;")
            layout.addWidget(value_label)
            
            # Date
            date_label = QLabel(f"As of {data['date']}")
            date_label.setStyleSheet("color: #888888; font-size: 9px;")
            layout.addWidget(date_label)
            
        else:
            # Error state
            error_label = QLabel("Data unavailable")
            error_label.setStyleSheet("color: #f44336;")
            layout.addWidget(error_label)


class MacroWidget(QWidget):
    """Main Macro Economic Indicators Widget"""
    
    def __init__(self):
        super().__init__()
        self.logger = SimpleLogger()
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        self.setup_ui()
        self.setup_timer()
        
        # Start data fetch
        self.refresh_data()
    
    def setup_ui(self):
        """Setup the main UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title = QLabel("📈 Macroeconomic Indicators")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh Data")
        refresh_button.clicked.connect(self.refresh_data)
        refresh_button.setFixedSize(120, 30)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        main_layout.addWidget(refresh_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Container for indicator cards
        self.indicators_container = QWidget()
        self.indicators_layout = QGridLayout(self.indicators_container)
        self.indicators_layout.setSpacing(10)
        
        main_layout.addWidget(self.indicators_container)
        
        # Status bar
        self.status_label = QLabel("Ready to load data...")
        self.status_label.setStyleSheet("color: #888888; font-size: 10px;")
        main_layout.addWidget(self.status_label)
        
        main_layout.addStretch()
    
    def setup_timer(self):
        """Setup automatic refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        # Refresh every 30 minutes
        self.refresh_timer.start(30 * 60 * 1000)
    
    def refresh_data(self):
        """Refresh macro economic data"""
        if not self.fred_api_key:
            self.status_label.setText("❌ FRED API key not configured")
            self.status_label.setStyleSheet("color: #f44336;")
            self.logger.warning("FRED API key not configured")
            
            # Show demo data
            self.show_demo_data()
            return
        
        self.status_label.setText("🔄 Fetching macro economic data...")
        self.status_label.setStyleSheet("color: #2196f3;")
        
        # Start data fetching thread
        self.data_thread = MacroDataThread(self.fred_api_key)
        self.data_thread.data_ready.connect(self.on_data_ready)
        self.data_thread.error_occurred.connect(self.on_error)
        self.data_thread.start()
    
    def show_demo_data(self):
        """Show demo data when API key is not available"""
        demo_data = {
            'DFF': {'name': 'Fed Funds Rate', 'units': '%', 'value': 5.25, 'date': '2024-10-01'},
            'UNRATE': {'name': 'Unemployment Rate', 'units': '%', 'value': 3.8, 'date': '2024-09-01'},
            'CPIAUCSL': {'name': 'CPI', 'units': 'Index', 'value': 308.2, 'date': '2024-09-01'},
        }
        self.on_data_ready(demo_data)
        self.status_label.setText("📊 Demo data displayed (FRED API key needed for real data)")
        self.status_label.setStyleSheet("color: #ff9800;")
    
    def on_data_ready(self, data):
        """Handle data ready from thread"""
        self.logger.info("Macro economic data received")
        
        # Clear existing indicators
        for i in reversed(range(self.indicators_layout.count())):
            child = self.indicators_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add new indicator cards
        row = 0
        col = 0
        for series_id, indicator_data in data.items():
            card = MacroIndicatorCard(indicator_data)
            self.indicators_layout.addWidget(card, row, col)
            
            col += 1
            if col >= 3:  # 3 columns
                col = 0
                row += 1
        
        # Update status
        successful_indicators = sum(1 for d in data.values() if d.get('value') is not None)
        total_indicators = len(data)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(
            f"✅ Data updated at {current_time} • "
            f"{successful_indicators}/{total_indicators} indicators loaded"
        )
        self.status_label.setStyleSheet("color: #4CAF50;")
    
    def on_error(self, error_message):
        """Handle error from data thread"""
        self.logger.error(f"Error fetching macro data: {error_message}")
        self.status_label.setText(f"❌ Error: {error_message}")
        self.status_label.setStyleSheet("color: #f44336;")
        
        # Show demo data on error
        self.show_demo_data()

    def closeEvent(self, event):
        """Ensure timers and threads are stopped when the widget closes."""
        try:
            # Stop auto refresh timer
            if hasattr(self, 'refresh_timer'):
                self.refresh_timer.stop()
                self.refresh_timer.deleteLater()
            # Stop in-flight data thread if running
            if hasattr(self, 'data_thread') and isinstance(getattr(self, 'data_thread', None), QThread):
                try:
                    if self.data_thread.isRunning():
                        # For QThread subclass, request interruption and quit
                        self.data_thread.requestInterruption()
                        self.data_thread.quit()
                        self.data_thread.wait()
                except Exception:
                    pass
        finally:
            event.accept()