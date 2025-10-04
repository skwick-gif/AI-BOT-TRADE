"""
Dashboard tab for the main trading interface
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGridLayout, QScrollArea,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import qtawesome as qta

# Import the macro widget and dashboard widget
from ..widgets.macro_widget import MacroWidget
from ..widgets.dashboard_widget import DashboardWidget

class DashboardTab(QWidget):
    """Main dashboard tab showing overview of trading status"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()
        
        # Setup update timer
        self.setup_timer()
    
    def init_ui(self):
        """Initialize the dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Trading Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Account summary section
        self.create_account_section(scroll_layout)
        
        # Portfolio positions section
        self.create_portfolio_section(scroll_layout)
        
        # Macro indicators section
        try:
            print("DEBUG: About to create macro section...")
            self.create_macro_section(scroll_layout)
            print("DEBUG: Macro section created successfully")
        except Exception as e:
            print(f"DEBUG: Exception in macro section creation: {e}")
            import traceback
            traceback.print_exc()
        
        # Event calendar section
        try:
            print("DEBUG: About to create event calendar...")
            self.create_event_calendar_section(scroll_layout)
            print("DEBUG: Event calendar created successfully")
        except Exception as e:
            print(f"DEBUG: Exception in event calendar creation: {e}")
            import traceback
            traceback.print_exc()
        
        # Market overview section
        self.create_market_section(scroll_layout)
        
        # Recent activity section
        self.create_activity_section(scroll_layout)
        print("DEBUG: Activity section created")
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
    
    def create_account_section(self, layout):
        """Create account summary section"""
        # Account frame
        account_frame = QFrame()
        account_frame.setFrameStyle(QFrame.Shape.Box)
        account_layout = QGridLayout(account_frame)
        
        # Title
        title = QLabel("💰 Account Summary")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        account_layout.addWidget(title, 0, 0, 1, 4)
        
        # Placeholder metrics
        metrics = [
            ("Total Value", "$50,000.00", "#00ff88"),
            ("Available Cash", "$15,000.00", "#88aaff"),
            ("P&L Today", "+$1,250.50", "#00ff88"),
            ("P&L Total", "+$5,750.25", "#00ff88")
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            lbl = QLabel(label + ":")
            val = QLabel(value)
            val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
            
            row = (i // 2) + 1
            col = (i % 2) * 2
            account_layout.addWidget(lbl, row, col)
            account_layout.addWidget(val, row, col + 1)
        
        layout.addWidget(account_frame)
    
    def create_portfolio_section(self, layout):
        """Create portfolio positions section"""
        portfolio_frame = QFrame()
        portfolio_frame.setFrameStyle(QFrame.Shape.Box)
        portfolio_layout = QVBoxLayout(portfolio_frame)
        
        # Title
        title = QLabel("📊 Portfolio Positions")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        portfolio_layout.addWidget(title)
        
        # Portfolio table
        portfolio_table = QTableWidget()
        portfolio_table.setColumnCount(6)
        portfolio_table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Avg Price", "Current Price", "P&L", "P&L %"
        ])
        
        # Sample portfolio data
        sample_positions = [
            ("AAPL", "100", "$150.00", "$155.50", "+$550.00", "+3.67%"),
            ("MSFT", "50", "$300.00", "$295.25", "-$237.50", "-1.58%"),
            ("GOOGL", "25", "$140.00", "$142.75", "+$68.75", "+1.96%"),
            ("TSLA", "30", "$200.00", "$195.80", "-$126.00", "-2.10%"),
        ]
        
        portfolio_table.setRowCount(len(sample_positions))
        for i, (symbol, qty, avg, current, pnl, pnl_pct) in enumerate(sample_positions):
            portfolio_table.setItem(i, 0, QTableWidgetItem(symbol))
            portfolio_table.setItem(i, 1, QTableWidgetItem(qty))
            portfolio_table.setItem(i, 2, QTableWidgetItem(avg))
            portfolio_table.setItem(i, 3, QTableWidgetItem(current))
            
            # Color code P&L
            pnl_item = QTableWidgetItem(pnl)
            pnl_pct_item = QTableWidgetItem(pnl_pct)
            color = "#00ff88" if pnl.startswith("+") else "#ff4444"
            pnl_item.setForeground(color)
            pnl_pct_item.setForeground(color)
            
            portfolio_table.setItem(i, 4, pnl_item)
            portfolio_table.setItem(i, 5, pnl_pct_item)
        
        # Table styling
        portfolio_table.setMaximumHeight(200)
        portfolio_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #404040;
                font-size: 12px;
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
        header = portfolio_table.horizontalHeader()
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        
        portfolio_layout.addWidget(portfolio_table)
        layout.addWidget(portfolio_frame)
    
    def create_market_section(self, layout):
        """Create market overview section"""
        market_frame = QFrame()
        market_frame.setFrameStyle(QFrame.Shape.Box)
        market_layout = QVBoxLayout(market_frame)
        
        # Title
        title = QLabel("📈 Market Overview")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        market_layout.addWidget(title)
        
        # Market indices
        indices_layout = QGridLayout()
        indices = [
            ("S&P 500", "4,750.25", "+1.2%", "#00ff88"),
            ("NASDAQ", "15,250.75", "+0.8%", "#00ff88"),
            ("DOW", "35,500.50", "-0.3%", "#ff6666"),
            ("VIX", "18.75", "-2.1%", "#00ff88")
        ]
        
        for i, (name, price, change, color) in enumerate(indices):
            name_lbl = QLabel(name)
            price_lbl = QLabel(price)
            change_lbl = QLabel(change)
            change_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            indices_layout.addWidget(name_lbl, i, 0)
            indices_layout.addWidget(price_lbl, i, 1)
            indices_layout.addWidget(change_lbl, i, 2)
        
        market_layout.addLayout(indices_layout)
        layout.addWidget(market_frame)
    
    def create_activity_section(self, layout):
        """Create recent activity section"""
        activity_frame = QFrame()
        activity_frame.setFrameStyle(QFrame.Shape.Box)
        activity_layout = QVBoxLayout(activity_frame)
        
        # Title
        title = QLabel("📋 Recent Activity")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        activity_layout.addWidget(title)
        
        # Activity list (placeholder)
        activities = [
            "🟢 Bought 100 shares of AAPL at $175.25",
            "🔴 Sold 50 shares of TSLA at $245.80",
            "🟡 Set stop loss for MSFT at $380.00",
            "🔵 AI Analysis completed for portfolio"
        ]
        
        for activity in activities:
            activity_lbl = QLabel(activity)
            activity_lbl.setStyleSheet("padding: 5px; margin: 2px;")
            activity_layout.addWidget(activity_lbl)
        
        layout.addWidget(activity_frame)
    
    def create_macro_section(self, layout):
        """Create macro indicators section"""
        macro_frame = QFrame()
        macro_frame.setFrameStyle(QFrame.Shape.Box)
        macro_layout = QVBoxLayout(macro_frame)
        
        # Title
        title = QLabel("📈 Macroeconomic Indicators")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        macro_layout.addWidget(title)
        
        # Add the macro widget
        self.macro_widget = MacroWidget()
        macro_layout.addWidget(self.macro_widget)
        
        layout.addWidget(macro_frame)
    
    def create_event_calendar_section(self, layout):
        """Create event calendar section"""
        print("DEBUG: Creating event calendar section...")
        try:
            calendar_frame = QFrame()
            calendar_frame.setFrameStyle(QFrame.Shape.Box)
            calendar_layout = QVBoxLayout(calendar_frame)
            
            # Title
            title = QLabel("📅 Event Calendar")
            title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            calendar_layout.addWidget(title)
            print("DEBUG: Added title to calendar")
            
            # Create tabs for different event types
            calendar_tabs = QTabWidget()
            calendar_tabs.setMaximumHeight(300)
            print("DEBUG: Created tab widget")
            
            # Earnings tab
            earnings_widget = self.create_earnings_widget()
            calendar_tabs.addTab(earnings_widget, "📊 Earnings")
            print("DEBUG: Added earnings tab")
            
            # Economic events tab
            economic_widget = self.create_economic_events_widget()
            calendar_tabs.addTab(economic_widget, "🏛️ Economic")
            print("DEBUG: Added economic tab")
            
            # Dividend dates tab
            dividend_widget = self.create_dividend_widget()
            calendar_tabs.addTab(dividend_widget, "💰 Dividends")
            print("DEBUG: Added dividend tab")
            
            calendar_layout.addWidget(calendar_tabs)
            layout.addWidget(calendar_frame)
            print("DEBUG: Event calendar section completed successfully")
        except Exception as e:
            print(f"DEBUG: Error creating event calendar: {e}")
            import traceback
            traceback.print_exc()
    
    def create_earnings_widget(self):
        """Create earnings calendar widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sample earnings data (in real implementation, this would come from API)
        earnings_data = [
            ("AAPL", "Apple Inc.", "2024-01-25", "After Market"),
            ("MSFT", "Microsoft Corp.", "2024-01-24", "After Market"),
            ("GOOGL", "Alphabet Inc.", "2024-01-23", "After Market"),
            ("TSLA", "Tesla Inc.", "2024-01-24", "After Market"),
            ("AMZN", "Amazon.com Inc.", "2024-01-26", "After Market")
        ]
        
        # Create table-like layout
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
    
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(30000)  # Update every 30 seconds
    
    def refresh_data(self):
        """Refresh dashboard data"""
        # This will be implemented with real data fetching
        print("Refreshing dashboard data...")