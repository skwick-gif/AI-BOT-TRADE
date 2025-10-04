"""
Macro indicators tab for economic data
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame, 
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ..widgets.macro_widget import MacroWidget

class MacroTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🌍 Macroeconomic Indicators & Events")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Add the macro widget
        self.macro_widget = MacroWidget()
        layout.addWidget(self.macro_widget)
        
        # Add Event Calendar section
        self.create_event_calendar_section(layout)
    
    def create_event_calendar_section(self, layout):
        """Create event calendar section"""
        calendar_frame = QFrame()
        calendar_frame.setFrameStyle(QFrame.Shape.Box)
        calendar_layout = QVBoxLayout(calendar_frame)
        
        # Title
        title = QLabel("📅 Event Calendar")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        calendar_layout.addWidget(title)
        
        # Create tabs for different event types
        calendar_tabs = QTabWidget()
        calendar_tabs.setMaximumHeight(300)
        
        # Earnings tab
        earnings_widget = self.create_earnings_widget()
        calendar_tabs.addTab(earnings_widget, "📊 Earnings")
        
        # Economic events tab
        economic_widget = self.create_economic_events_widget()
        calendar_tabs.addTab(economic_widget, "🏛️ Economic")
        
        # Dividend dates tab
        dividend_widget = self.create_dividend_widget()
        calendar_tabs.addTab(dividend_widget, "💰 Dividends")
        
        calendar_layout.addWidget(calendar_tabs)
        layout.addWidget(calendar_frame)
    
    def create_earnings_widget(self):
        """Create earnings calendar widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sample earnings data
        earnings_data = [
            ("AAPL", "Apple Inc.", "2024-01-25", "After Market"),
            ("MSFT", "Microsoft Corp.", "2024-01-24", "After Market"),
            ("GOOGL", "Alphabet Inc.", "2024-01-23", "After Market"),
            ("TSLA", "Tesla Inc.", "2024-01-24", "After Market"),
            ("AMZN", "Amazon.com Inc.", "2024-01-26", "After Market")
        ]
        
        earnings_table = QTableWidget()
        earnings_table.setRowCount(len(earnings_data))
        earnings_table.setColumnCount(4)
        earnings_table.setHorizontalHeaderLabels(["Symbol", "Company", "Date", "Time"])
        
        for row, (symbol, company, date, time) in enumerate(earnings_data):
            earnings_table.setItem(row, 0, QTableWidgetItem(symbol))
            earnings_table.setItem(row, 1, QTableWidgetItem(company))
            earnings_table.setItem(row, 2, QTableWidgetItem(date))
            earnings_table.setItem(row, 3, QTableWidgetItem(time))
        
        # Style the table
        earnings_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #404040;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Resize columns
        header = earnings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        
        earnings_table.setColumnWidth(0, 80)
        earnings_table.setColumnWidth(2, 100)
        earnings_table.setColumnWidth(3, 120)
        
        layout.addWidget(earnings_table)
        return widget
    
    def create_economic_events_widget(self):
        """Create economic events widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sample economic events
        economic_data = [
            ("2024-01-15", "Consumer Price Index", "High", "3.4% (Previous)"),
            ("2024-01-17", "Housing Starts", "Medium", "1.35M (Previous)"),
            ("2024-01-25", "GDP Growth Rate", "High", "2.1% (Previous)"),
            ("2024-01-31", "Federal Reserve Meeting", "High", "5.25% (Previous)")
        ]
        
        events_table = QTableWidget()
        events_table.setRowCount(len(economic_data))
        events_table.setColumnCount(4)
        events_table.setHorizontalHeaderLabels(["Date", "Event", "Impact", "Previous"])
        
        for row, (date, event, impact, previous) in enumerate(economic_data):
            events_table.setItem(row, 0, QTableWidgetItem(date))
            events_table.setItem(row, 1, QTableWidgetItem(event))
            
            # Color code impact
            impact_item = QTableWidgetItem(impact)
            if impact == "High":
                impact_item.setBackground(Qt.GlobalColor.red)
            elif impact == "Medium":
                impact_item.setBackground(Qt.GlobalColor.yellow)
            else:
                impact_item.setBackground(Qt.GlobalColor.green)
            events_table.setItem(row, 2, impact_item)
            
            events_table.setItem(row, 3, QTableWidgetItem(previous))
        
        # Style the table
        events_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #404040;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Resize columns
        header = events_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        events_table.setColumnWidth(0, 100)
        events_table.setColumnWidth(2, 80)
        
        layout.addWidget(events_table)
        return widget
    
    def create_dividend_widget(self):
        """Create dividend calendar widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sample dividend data
        dividend_data = [
            ("AAPL", "Apple Inc.", "2024-01-15", "$0.25"),
            ("MSFT", "Microsoft Corp.", "2024-01-18", "$0.75"),
            ("JPM", "JPMorgan Chase", "2024-01-20", "$1.05"),
            ("JNJ", "Johnson & Johnson", "2024-01-22", "$1.19"),
            ("PG", "Procter & Gamble", "2024-01-25", "$0.91")
        ]
        
        dividend_table = QTableWidget()
        dividend_table.setRowCount(len(dividend_data))
        dividend_table.setColumnCount(4)
        dividend_table.setHorizontalHeaderLabels(["Symbol", "Company", "Ex-Date", "Amount"])
        
        for row, (symbol, company, date, amount) in enumerate(dividend_data):
            dividend_table.setItem(row, 0, QTableWidgetItem(symbol))
            dividend_table.setItem(row, 1, QTableWidgetItem(company))
            dividend_table.setItem(row, 2, QTableWidgetItem(date))
            dividend_table.setItem(row, 3, QTableWidgetItem(amount))
        
        # Style the table
        dividend_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #404040;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Resize columns
        header = dividend_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        
        dividend_table.setColumnWidth(0, 80)
        dividend_table.setColumnWidth(2, 100)
        dividend_table.setColumnWidth(3, 80)
        
        layout.addWidget(dividend_table)
        return widget
    
    def refresh_data(self):
        # Refresh the macro widget data
        if hasattr(self, 'macro_widget'):
            self.macro_widget.refresh_data()