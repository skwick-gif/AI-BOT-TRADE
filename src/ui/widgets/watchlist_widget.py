"""
Watchlist Widget
For monitoring and managing stock watchlists
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QLineEdit, QComboBox, QHeaderView,
    QMenu, QMessageBox, QInputDialog, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QColor, QAction

from core.config_manager import ConfigManager
from utils.logger import get_logger


class WatchlistDataWorker(QObject):
    """Worker thread for fetching watchlist data"""
    
    data_updated = pyqtSignal(str, dict)  # symbol, data
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.logger = get_logger("WatchlistWorker")
        self.is_running = False
    
    def start_monitoring(self, symbols: list):
        """Start monitoring symbols"""
        self.is_running = True
        self.symbols = symbols
        self.fetch_data()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
    
    def fetch_data(self):
        """Fetch data for all symbols from local bronze parquet files"""
        try:
            from pathlib import Path
            import math
            import pandas as pd
            if not hasattr(self, 'symbols'):
                return
            bronze_dir = Path("data/bronze/daily")
            for symbol in list(self.symbols):
                if not self.is_running:
                    break
                try:
                    fp = bronze_dir / f"{symbol}.parquet"
                    if not fp.exists():
                        # Skip symbols without local parquet data (no mock data)
                        continue
                    df = pd.read_parquet(fp)
                    if df.empty:
                        continue
                    # Ensure sorted by date if present
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.dropna(subset=['date']).sort_values('date')
                    last = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else None
                    price = float(last.get('close', last.get('adj_close', math.nan)))
                    previous_close = float(prev.get('close')) if prev is not None and 'close' in df.columns else price
                    change = price - previous_close
                    change_pct = (change / previous_close * 100) if previous_close else 0.0
                    volume = int(last.get('volume', 0) or 0)
                    bid = price  # placeholder
                    ask = price  # placeholder
                    high = float(last.get('high', price))
                    low = float(last.get('low', price))
                    data = {
                        'symbol': symbol,
                        'price': price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': volume,
                        'bid': bid,
                        'ask': ask,
                        'high': high,
                        'low': low,
                        'previous_close': previous_close
                    }
                    self.data_updated.emit(symbol, data)
                except Exception as ie:
                    self.logger.warning(f"Watchlist read error for {symbol}: {ie}")
        except Exception as e:
            self.error_occurred.emit(str(e))


class WatchlistTable(QTableWidget):
    """Enhanced table widget for watchlist data"""
    
    symbol_selected = pyqtSignal(str)
    trade_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.symbol_data = {}
    
    def setup_ui(self):
        """Setup watchlist table"""
        # Set columns
        self.setColumnCount(9)
        headers = ["Symbol", "Price", "Change", "Change %", "Volume", "Bid", "Ask", "High", "Low"]
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Configure headers
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)   # Symbol
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)   # Price
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)   # Change
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)   # Change %
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)   # Volume
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)   # Bid
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)   # Ask
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)   # High
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch) # Low
        
        # Set column widths
        self.setColumnWidth(0, 80)   # Symbol
        self.setColumnWidth(1, 100)  # Price
        self.setColumnWidth(2, 80)   # Change
        self.setColumnWidth(3, 80)   # Change %
        self.setColumnWidth(4, 100)  # Volume
        self.setColumnWidth(5, 80)   # Bid
        self.setColumnWidth(6, 80)   # Ask
        self.setColumnWidth(7, 80)   # High
        self.setColumnWidth(8, 80)   # Low
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Double click
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to the watchlist"""
        if symbol in self.symbol_data:
            return False
        
        row = self.rowCount()
        self.insertRow(row)
        
        # Symbol column
        symbol_item = QTableWidgetItem(symbol)
        symbol_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.setItem(row, 0, symbol_item)
        
        # Initialize other columns with placeholder data
        for col in range(1, 9):
            item = QTableWidgetItem("-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, col, item)
        
        self.symbol_data[symbol] = {'row': row}
        return True
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol not in self.symbol_data:
            return False
        
        row = self.symbol_data[symbol]['row']
        self.removeRow(row)
        del self.symbol_data[symbol]
        
        # Update row numbers for remaining symbols
        for s, data in self.symbol_data.items():
            if data['row'] > row:
                data['row'] -= 1
        
        return True
    
    def update_symbol_data(self, symbol: str, data: dict):
        """Update data for a specific symbol"""
        if symbol not in self.symbol_data:
            return
        
        row = self.symbol_data[symbol]['row']
        
        # Price
        price_item = self.item(row, 1)
        price_item.setText(f"${data['price']:.2f}")
        
        # Change
        change_item = self.item(row, 2)
        change_item.setText(f"${data['change']:+.2f}")
        if data['change'] >= 0:
            change_item.setForeground(QColor("#4CAF50"))
        else:
            change_item.setForeground(QColor("#f44336"))
        
        # Change %
        change_pct_item = self.item(row, 3)
        change_pct_item.setText(f"{data['change_pct']:+.2f}%")
        if data['change_pct'] >= 0:
            change_pct_item.setForeground(QColor("#4CAF50"))
        else:
            change_pct_item.setForeground(QColor("#f44336"))
        
        # Volume
        volume_item = self.item(row, 4)
        volume_item.setText(f"{data['volume']:,}")
        
        # Bid
        bid_item = self.item(row, 5)
        bid_item.setText(f"${data['bid']:.2f}")
        
        # Ask
        ask_item = self.item(row, 6)
        ask_item.setText(f"${data['ask']:.2f}")
        
        # High
        high_item = self.item(row, 7)
        high_item.setText(f"${data['high']:.2f}")
        
        # Low
        low_item = self.item(row, 8)
        low_item.setText(f"${data['low']:.2f}")
        
        # Store data
        self.symbol_data[symbol]['data'] = data
    
    def show_context_menu(self, position):
        """Show context menu"""
        if self.itemAt(position) is None:
            return
        
        menu = QMenu(self)
        
        # Get selected symbol
        row = self.itemAt(position).row()
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            
            # Menu actions
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


class WatchlistDetails(QFrame):
    """Details panel for selected symbol"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_symbol = None
    
    def setup_ui(self):
        """Setup details panel"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMaximumHeight(200)
        
        layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel("Symbol Details")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        
        # Details layout
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
        button_layout = QHBoxLayout()
        
        self.buy_button = QPushButton("📈 Buy")
        self.buy_button.setEnabled(False)
        self.sell_button = QPushButton("📉 Sell") 
        self.sell_button.setEnabled(False)
        self.chart_button = QPushButton("📊 Chart")
        self.chart_button.setEnabled(False)
        
        button_layout.addWidget(self.buy_button)
        button_layout.addWidget(self.sell_button)
        button_layout.addWidget(self.chart_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def update_details(self, symbol: str, data: dict):
        """Update details for selected symbol"""
        self.current_symbol = symbol
        
        self.title_label.setText(f"{symbol} Details")
        self.symbol_label.setText(f"Symbol: {symbol}")
        self.price_label.setText(f"Price: ${data['price']:.2f}")
        
        change_text = f"Change: ${data['change']:+.2f} ({data['change_pct']:+.2f}%)"
        self.change_label.setText(change_text)
        if data['change'] >= 0:
            self.change_label.setStyleSheet("color: #4CAF50;")
        else:
            self.change_label.setStyleSheet("color: #f44336;")
        
        self.volume_label.setText(f"Volume: {data['volume']:,}")
        self.bid_ask_label.setText(f"Bid/Ask: ${data['bid']:.2f} / ${data['ask']:.2f}")
        self.high_low_label.setText(f"High/Low: ${data['high']:.2f} / ${data['low']:.2f}")
        
        # Enable buttons
        self.buy_button.setEnabled(True)
        self.sell_button.setEnabled(True)
        self.chart_button.setEnabled(True)


class WatchlistWidget(QWidget):
    """Main watchlist widget"""
    
    trade_requested = pyqtSignal(str)  # action and symbol
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("Watchlist")
        self.config = ConfigManager()
        
        # Initialize data worker
        self.data_thread = QThread()
        self.data_worker = WatchlistDataWorker(self.config)
        self.data_worker.moveToThread(self.data_thread)
        self.data_worker.data_updated.connect(self.on_data_updated)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_thread.start()
        
        # Setup UI
        self.setup_ui()
        
        # Load default symbols
        self.load_default_symbols()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(self.config.ui.update_interval)
        
        self.logger.info("Watchlist widget initialized")

    def add_symbol_from_scanner(self, symbol: str, switch_to_tab: bool = False):
        """Public method to add a symbol coming from external widgets (e.g., Scanner).
        Adds the symbol if missing, restarts monitoring to include it, and focuses/selects it.
        """
        try:
            if not symbol:
                return
            sym = symbol.strip().upper()
            if not sym:
                return

            # Try to add (no-op if already present)
            added = self.watchlist_table.add_symbol(sym)

            # Update monitoring universe
            symbols = self.watchlist_table.get_symbols()
            if hasattr(self, 'data_worker'):
                self.data_worker.stop_monitoring()
                if symbols:
                    self.data_worker.start_monitoring(symbols)

            # Focus/select the symbol row if present
            if sym in self.watchlist_table.symbol_data:
                row = self.watchlist_table.symbol_data[sym]['row']
                self.watchlist_table.clearSelection()
                self.watchlist_table.selectRow(row)
                # Update details immediately if we have cached data
                data = self.watchlist_table.symbol_data[sym].get('data') if isinstance(self.watchlist_table.symbol_data.get(sym), dict) else None
                if data:
                    self.details_panel.update_details(sym, data)
                else:
                    # Ensure details title reflects selection even before data arrives
                    self.details_panel.title_label.setText(f"{sym} Details")
                    self.details_panel.symbol_label.setText(f"Symbol: {sym}")

            self.logger.info(f"Symbol {'added' if added else 'already in'} watchlist via Scanner: {sym}")
        except Exception as e:
            self.logger.error(f"Failed to add symbol from scanner: {symbol} | {e}")
    
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
        table_layout.addWidget(self.watchlist_table)
        
        splitter.addWidget(table_frame)
        
        # Details panel
        self.details_panel = WatchlistDetails()
        splitter.addWidget(self.details_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 3)  # Table gets more space
        splitter.setStretchFactor(1, 1)  # Details gets less space
        
        layout.addWidget(splitter)
    
    def load_default_symbols(self):
        """Load default symbols from config"""
        default_symbols = self.config.ui.default_symbols
        for symbol in default_symbols:
            self.watchlist_table.add_symbol(symbol)
        
        # Start monitoring
        if default_symbols:
            self.data_worker.start_monitoring(default_symbols)
    
    def add_symbol(self):
        """Add new symbol to watchlist"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return
        
        if self.watchlist_table.add_symbol(symbol):
            self.symbol_input.clear()
            
            # Update monitoring
            symbols = self.watchlist_table.get_symbols()
            self.data_worker.stop_monitoring()
            self.data_worker.start_monitoring(symbols)
            
            self.logger.info(f"Added symbol to watchlist: {symbol}")
        else:
            QMessageBox.warning(self, "Warning", f"{symbol} is already in the watchlist")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if self.watchlist_table.remove_symbol(symbol):
            # Update monitoring
            symbols = self.watchlist_table.get_symbols()
            self.data_worker.stop_monitoring()
            if symbols:
                self.data_worker.start_monitoring(symbols)
            
            self.logger.info(f"Removed symbol from watchlist: {symbol}")
    
    def on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        if symbol in self.watchlist_table.symbol_data:
            data = self.watchlist_table.symbol_data[symbol].get('data')
            if data:
                self.details_panel.update_details(symbol, data)
        
        self.logger.debug(f"Symbol selected: {symbol}")
    
    def on_data_updated(self, symbol: str, data: dict):
        """Handle data update from worker"""
        self.watchlist_table.update_symbol_data(symbol, data)
        
        # Update details if this symbol is currently selected
        if (hasattr(self.details_panel, 'current_symbol') and 
            self.details_panel.current_symbol == symbol):
            self.details_panel.update_details(symbol, data)
    
    def on_data_error(self, error: str):
        """Handle data error"""
        self.logger.error(f"Watchlist data error: {error}")
    
    def update_data(self):
        """Update watchlist data"""
        try:
            if hasattr(self, 'data_worker') and self.data_worker.is_running:
                # Trigger fetch in worker thread
                QTimer.singleShot(0, self.data_worker.fetch_data)
        except Exception as e:
            self.logger.error(f"Update data error: {e}")
    
    def refresh_data(self):
        """Manually refresh watchlist data"""
        symbols = self.watchlist_table.get_symbols()
        if symbols:
            self.data_worker.stop_monitoring()
            self.data_worker.start_monitoring(symbols)
        
        self.logger.info("Manual watchlist refresh requested")

    def delete_selected(self):
        """Delete the currently selected ticker from the list"""
        sel_ranges = self.watchlist_table.selectedItems()
        if not sel_ranges:
            return
        # Get the row of the first selected item
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
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop update timer
        if hasattr(self, 'update_timer') and isinstance(self.update_timer, QTimer):
            try:
                self.update_timer.stop()
            except Exception:
                pass
        # Stop data worker
        if hasattr(self, 'data_worker'):
            self.data_worker.stop_monitoring()
        
        # Stop worker thread
        if hasattr(self, 'data_thread'):
            self.data_thread.quit()
            self.data_thread.wait()
        
        event.accept()