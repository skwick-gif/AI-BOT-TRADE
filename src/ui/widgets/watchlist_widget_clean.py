from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QLineEdit, QHeaderView,
    QMenu, QMessageBox, QSplitter, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QAction
from datetime import datetime
import json
import re

# Optional imports with fallbacks
try:
    from core.config_manager import ConfigManager
    from utils.logger import get_logger
except ImportError:
    ConfigManager = None
    get_logger = None

class WatchlistTable(QTableWidget):
    symbol_selected = pyqtSignal(str)
    trade_requested = pyqtSignal(str)
    tools_action = pyqtSignal(str, str)  # action, symbol

    def __init__(self):
        super().__init__()
        self.symbol_data = {}
        self.setup_ui()

    def setup_ui(self):
        headers = [
            "Symbol", "נוסף בתאריך", "Price", "Volume",
            "יום 1", "יום 2", "יום 3", "יום 4", "יום 5", "יום 6", "יום 7",
            "AI Rating", "Price Target", "Stop Loss", "Signal", "Sharpe Ratio", "Tools"
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        header = self.horizontalHeader()
        for i in range(len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(headers)-1, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(len(headers)-1, 80)
        self.setSortingEnabled(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)

    def add_symbol(self, symbol: str):
        if symbol in self.symbol_data:
            return False
        
        row = self.rowCount()
        self.insertRow(row)
        symbol_item = QTableWidgetItem(symbol)
        symbol_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.setItem(row, 0, symbol_item)
        
        added_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        added_item = QTableWidgetItem(added_at)
        added_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 1, added_item)
        
        for col in range(2, self.columnCount() - 1):
            item = QTableWidgetItem("-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, col, item)

        # Add tools column with AI and Chart buttons
        tools_widget = QWidget()
        hl = QHBoxLayout(tools_widget)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        def mk_btn(tip: str, icon_text: str):
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setText(icon_text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setAutoRaise(True)
            btn.setStyleSheet("font-size: 18px; padding: 0 4px;")
            return btn
        
        ai_btn = mk_btn("Ask AI", "🤖")
        ai_btn.clicked.connect(lambda: self.tools_action.emit('ai', symbol))
        hl.addWidget(ai_btn)
        
        chart_btn = mk_btn("Open Chart", "📈")
        chart_btn.clicked.connect(lambda: self.tools_action.emit('chart', symbol))
        hl.addWidget(chart_btn)
        
        self.setCellWidget(row, self.columnCount() - 1, tools_widget)
        
        self.symbol_data[symbol] = {'row': row, 'data': {}}
        return True

    def remove_symbol(self, symbol: str):
        if symbol not in self.symbol_data:
            return False
        
        row = self.symbol_data[symbol]['row']
        self.removeRow(row)
        del self.symbol_data[symbol]
        
        # Update row indices for remaining symbols
        for sym, data in self.symbol_data.items():
            if data['row'] > row:
                data['row'] -= 1
        
        return True

    def show_context_menu(self, position):
        """Show context menu"""
        if self.itemAt(position) is None:
            return
        
        menu = QMenu(self)
        row = self.itemAt(position).row()
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            
            buy_action = QAction(f"Buy {symbol}", self)
            buy_action.triggered.connect(lambda: self.trade_requested.emit(f"BUY {symbol}"))
            menu.addAction(buy_action)
            
            sell_action = QAction(f"Sell {symbol}", self)
            sell_action.triggered.connect(lambda: self.trade_requested.emit(f"SELL {symbol}"))
            menu.addAction(sell_action)
            
            menu.addSeparator()
            
            remove_action = QAction(f"Remove {symbol}", self)
            remove_action.triggered.connect(lambda: self.remove_symbol(symbol))
            menu.addAction(remove_action)
            
            menu.exec(self.mapToGlobal(position))

    def on_cell_double_clicked(self, row: int, column: int):
        """Handle cell double click"""
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            self.symbol_selected.emit(symbol)

    def get_symbols(self) -> list:
        """Get all symbols in the watchlist"""
        return list(self.symbol_data.keys())

    def request_ai_rating(self, symbol, row):
        """Request AI rating for symbol (placeholder)"""
        # This would normally make an AI API call
        # For now, just update the cell with a placeholder
        if row < self.rowCount():
            ai_item = QTableWidgetItem("Analyzing...")
            self.setItem(row, 11, ai_item)  # AI Rating column


class WatchlistDetails(QFrame):
    """Details panel for selected symbol"""
    ask_ai_clicked = pyqtSignal(str)
    chart_clicked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_symbol = None
        self.setup_ui()

    def setup_ui(self):
        """Setup details panel"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMaximumHeight(200)
        
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel("Symbol Details")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        
        details_layout = QHBoxLayout()
        
        # Left column
        left_layout = QVBoxLayout()
        self.symbol_label = QLabel("Symbol: -")
        self.price_label = QLabel("Price: -")
        self.change_label = QLabel("Change: -")
        left_layout.addWidget(self.symbol_label)
        left_layout.addWidget(self.price_label)
        left_layout.addWidget(self.change_label)
        details_layout.addLayout(left_layout)
        
        # Right column
        right_layout = QVBoxLayout()
        self.volume_label = QLabel("Volume: -")
        self.bid_ask_label = QLabel("Bid/Ask: -")
        self.high_low_label = QLabel("High/Low: -")
        right_layout.addWidget(self.volume_label)
        right_layout.addWidget(self.bid_ask_label)
        right_layout.addWidget(self.high_low_label)
        details_layout.addLayout(right_layout)
        
        layout.addLayout(details_layout)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        self.buy_button = QPushButton("Buy")
        self.sell_button = QPushButton("Sell")
        self.buy_button.setEnabled(False)
        self.sell_button.setEnabled(False)
        actions_layout.addWidget(self.buy_button)
        actions_layout.addWidget(self.sell_button)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

    def update_details(self, symbol: str, data: dict):
        """Update details for selected symbol"""
        self.current_symbol = symbol
        self.symbol_label.setText(f"Symbol: {symbol}")
        
        if data:
            self.price_label.setText(f"Price: {data.get('price', '-')}")
            self.change_label.setText(f"Change: {data.get('change', '-')}")
            self.volume_label.setText(f"Volume: {data.get('volume', '-')}")
            self.bid_ask_label.setText(f"Bid/Ask: {data.get('bid', '-')}/{data.get('ask', '-')}")
            self.high_low_label.setText(f"High/Low: {data.get('high', '-')}/{data.get('low', '-')}")
        else:
            self.price_label.setText("Price: -")
            self.change_label.setText("Change: -")
            self.volume_label.setText("Volume: -")
            self.bid_ask_label.setText("Bid/Ask: -")
            self.high_low_label.setText("High/Low: -")
        
        self.buy_button.setEnabled(True)
        self.sell_button.setEnabled(True)


class WatchlistWidget(QWidget):
    """Main watchlist widget"""
    trade_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        # Initialize fallback config and logger
        if ConfigManager is None:
            # Create mock config
            class MockConfig:
                def __init__(self):
                    self.ui = type('obj', (object,), {'update_interval': 5000})()
            self.config = MockConfig()
        else:
            self.config = ConfigManager()
        
        if get_logger is None:
            # Create mock logger
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = get_logger(__name__)
        
        # Initialize data worker as None (will be set by parent if available)
        self.data_worker = None
        
        self.setup_ui()
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(self.config.ui.update_interval)
        
        self.logger.info("Watchlist widget initialized")

    def setup_ui(self):
        """Setup the watchlist UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title and controls
        title_layout = QHBoxLayout()
        
        title = QLabel("Stock Watchlist")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # Add symbol input
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL)")
        self.symbol_input.setMaximumWidth(150)
        
        # Allow only English letters A-Z in the input field
        try:
            from PyQt6.QtGui import QRegularExpressionValidator
            from PyQt6.QtCore import QRegularExpression
            regex = QRegularExpression(r"^[A-Za-z]+$")
            validator = QRegularExpressionValidator(regex)
            self.symbol_input.setValidator(validator)
        except Exception:
            pass
        
        self.symbol_input.returnPressed.connect(self.add_symbol)
        title_layout.addWidget(self.symbol_input)
        
        add_btn = QPushButton("➕ Add")
        add_btn.clicked.connect(self.add_symbol)
        title_layout.addWidget(add_btn)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        title_layout.addWidget(refresh_btn)
        
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.clicked.connect(self.delete_selected)
        title_layout.addWidget(delete_btn)
        
        layout.addLayout(title_layout)
        
        # Create splitter for table and details
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Watchlist table
        table_frame = QFrame()
        table_frame.setFrameStyle(QFrame.Shape.Box)
        table_layout = QVBoxLayout(table_frame)
        
        table_title = QLabel("Monitored Symbols")
        table_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        table_layout.addWidget(table_title)
        
        self.watchlist_table = WatchlistTable()
        self.watchlist_table.symbol_selected.connect(self.on_symbol_selected)
        self.watchlist_table.trade_requested.connect(self.trade_requested.emit)
        self.watchlist_table.tools_action.connect(self._handle_tools_action)
        table_layout.addWidget(self.watchlist_table)
        
        splitter.addWidget(table_frame)
        
        # Details panel
        self.details_panel = WatchlistDetails()
        self.details_panel.ask_ai_clicked.connect(self._handle_ask_ai)
        self.details_panel.chart_clicked.connect(self._handle_open_chart)
        splitter.addWidget(self.details_panel)
        
        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

    def _handle_tools_action(self, action: str, symbol: str):
        """Handle per-row tools icon actions: AI and Chart"""
        if action == 'ai':
            self._handle_ask_ai(symbol)
        elif action == 'chart':
            self._handle_open_chart(symbol)

    def _handle_ask_ai(self, symbol: str):
        """Handle AI request for symbol"""
        if not symbol:
            return
        
        row = self.watchlist_table.symbol_data.get(symbol, {}).get('row')
        if row is None:
            return
        
        try:
            self.watchlist_table.request_ai_rating(symbol, row)
            self.logger.info(f"AI rating requested for {symbol}")
        except Exception as e:
            self.logger.error(f"Error requesting AI rating for {symbol}: {e}")

    def _handle_open_chart(self, symbol: str):
        """Handle chart opening for symbol"""
        self.logger.info(f"Chart requested for {symbol}")
        # TODO: Implement chart opening logic

    def add_symbol(self):
        """Add symbol from input field"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return
        
        if self.watchlist_table.add_symbol(symbol):
            self.symbol_input.clear()
            symbols = self.watchlist_table.get_symbols()
            if self.data_worker and hasattr(self.data_worker, 'start_monitoring'):
                try:
                    self.data_worker.start_monitoring(symbols)
                except Exception as e:
                    self.logger.error(f"Error starting monitoring: {e}")
            self.logger.info(f"Added symbol: {symbol}")
        else:
            QMessageBox.information(self, "Info", f"Symbol {symbol} is already in the watchlist")

    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if self.watchlist_table.remove_symbol(symbol):
            symbols = self.watchlist_table.get_symbols()
            if self.data_worker and hasattr(self.data_worker, 'start_monitoring'):
                try:
                    self.data_worker.start_monitoring(symbols)
                except Exception as e:
                    self.logger.error(f"Error updating monitoring: {e}")
            self.logger.info(f"Removed symbol: {symbol}")

    def on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        if symbol in self.watchlist_table.symbol_data:
            data = self.watchlist_table.symbol_data[symbol].get('data', {})
            self.details_panel.update_details(symbol, data)

    def update_data(self):
        """Update data from data worker"""
        if not self.data_worker or not hasattr(self.data_worker, 'is_running'):
            return
        
        try:
            if self.data_worker.is_running:
                # TODO: Update table with new data from data worker
                pass
        except Exception as e:
            self.logger.error(f"Update data error: {e}")

    def refresh_data(self):
        """Manually refresh watchlist data"""
        symbols = self.watchlist_table.get_symbols()
        if symbols and self.data_worker and hasattr(self.data_worker, 'start_monitoring'):
            try:
                self.data_worker.stop_monitoring()
                self.data_worker.start_monitoring(symbols)
            except Exception as e:
                self.logger.error(f"Error refreshing data: {e}")
        
        self.logger.info("Manual watchlist refresh requested")

    def delete_selected(self):
        """Delete the currently selected ticker from the list"""
        sel_ranges = self.watchlist_table.selectedItems()
        if not sel_ranges:
            return
        
        row = sel_ranges[0].row()
        symbol_item = self.watchlist_table.item(row, 0)
        if not symbol_item:
            return
        
        symbol = symbol_item.text()
        self.remove_symbol(symbol)

    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Watchlist tab activated")
        self.refresh_data()

    def add_symbol_from_scanner(self, symbol: str, switch_to_tab: bool = False):
        """Public method to add a symbol coming from external widgets"""
        if self.watchlist_table.add_symbol(symbol):
            symbols = self.watchlist_table.get_symbols()
            if self.data_worker and hasattr(self.data_worker, 'start_monitoring'):
                try:
                    self.data_worker.start_monitoring(symbols)
                except Exception as e:
                    self.logger.error(f"Error starting monitoring: {e}")
            
            # Select the newly added symbol
            if symbol in self.watchlist_table.symbol_data:
                row = self.watchlist_table.symbol_data[symbol]['row']
                self.watchlist_table.clearSelection()
                self.watchlist_table.selectRow(row)
                data = self.watchlist_table.symbol_data[symbol].get('data', {})
                self.details_panel.update_details(symbol, data)
            
            self.logger.info(f"Added symbol from scanner: {symbol}")
            return True
        return False

    def cleanup(self):
        """Stop timers and threads gracefully"""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if self.data_worker and hasattr(self.data_worker, 'stop_monitoring'):
            try:
                self.data_worker.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping data worker: {e}")

    def closeEvent(self, event):
        """Qt close event override to cleanup resources"""
        self.cleanup()
        super().closeEvent(event)