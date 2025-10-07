"""
Simple Macro Economic Indicators Widget
Based on the legacy Streamlit implementation - simplified version
"""

import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QPushButton, QDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from datetime import datetime, timedelta


class GraphDialog(QDialog):
    """Small dialog to show a time series chart using pyqtgraph."""
    def __init__(self, title: str, dates, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 320)
        # Lazy import to avoid hard dependency at import time
        import pyqtgraph as pg
        layout = QVBoxLayout(self)
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setBackground('#2b2b2b')
        pen = pg.mkPen(color=(20, 160, 133), width=2)  # match macro color scheme
        # Convert dates to x axis as ordinal numbers
        import numpy as np
        x = np.array([d.toordinal() for d in dates], dtype=float)
        y = np.array(values, dtype=float)
        plot.plot(x, y, pen=pen)
        plot.setLabel('left', 'Value')
        plot.setLabel('bottom', 'Date')
        layout.addWidget(plot)


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
    
    def __init__(self, indicator_data, series_id: str, on_click=None):
        super().__init__()
        self.series_id = series_id
        self._on_click = on_click
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

    def mousePressEvent(self, event):
        try:
            if callable(self._on_click):
                self._on_click(self.series_id)
        except Exception:
            pass
        return super().mousePressEvent(event)


class MacroWidget(QWidget):
    """Main Macro Economic Indicators Widget"""
    
    def __init__(self):
        super().__init__()
        self.logger = SimpleLogger()
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        self.setup_ui()
        self.setup_timer()
        
        # Don't start data fetch immediately - wait for user interaction or timer
        # self.refresh_data()  # Removed to prevent blocking app startup

        # Default chart range start (per request)
        self.chart_start_date = datetime(2024, 1, 1)
    
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
        """Setup automatic refresh timer - starts only after first manual refresh"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        # Don't start timer automatically - wait for first manual refresh
        # self.refresh_timer.start(30 * 60 * 1000)  # 30 minutes
    
    def refresh_data(self):
        """Refresh macro economic data"""
        if not self.fred_api_key:
            self.status_label.setText("❌ FRED API key not configured")
            self.status_label.setStyleSheet("color: #f44336;")
            self.logger.warning("FRED API key not configured")
            # Clear any existing indicators and do not show demo data
            for i in reversed(range(self.indicators_layout.count())):
                child = self.indicators_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            return
        
        self.status_label.setText("🔄 Fetching macro economic data...")
        self.status_label.setStyleSheet("color: #2196f3;")
        
        # Start data fetching thread
        self.data_thread = MacroDataThread(self.fred_api_key)
        self.data_thread.data_ready.connect(self.on_data_ready)
        self.data_thread.error_occurred.connect(self.on_error)
        self.data_thread.start()
        
        # Start timer after first refresh (if not already started)
        if not self.refresh_timer.isActive():
            self.refresh_timer.start(30 * 60 * 1000)  # 30 minutes
    
    
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
            card = MacroIndicatorCard(indicator_data, series_id, on_click=self.open_indicator_chart)
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
        # Do not show demo data; leave indicators as-is or cleared on next refresh

    def open_indicator_chart(self, series_id: str):
        """Open a small chart dialog for the given FRED series from 2024-01-01 to today."""
        if not self.fred_api_key:
            self.status_label.setText("FRED API key required for charts")
            self.status_label.setStyleSheet("color: #f44336;")
            return
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)
            start_str = self.chart_start_date.strftime('%Y-%m-%d')
            end_str = datetime.now().strftime('%Y-%m-%d')
            series = fred.get_series(series_id, observation_start=start_str, observation_end=end_str)
            if series is None or series.empty:
                self.status_label.setText(f"No data available for {series_id}")
                self.status_label.setStyleSheet("color: #ff9800;")
                return
            # Prepare dates and values
            dates = [dt.date() for dt in series.index.to_pydatetime()]
            values = series.values.tolist()
            title = f"{series_id} • {start_str} → {end_str}"
            dlg = GraphDialog(title, dates, values, parent=self)
            dlg.exec()
        except ImportError:
            self.status_label.setText("fredapi or pyqtgraph not installed")
            self.status_label.setStyleSheet("color: #f44336;")
        except Exception as e:
            self.status_label.setText(f"Chart error: {str(e)[:60]}…")
            self.status_label.setStyleSheet("color: #f44336;")

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