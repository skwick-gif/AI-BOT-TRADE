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
import os

# Optional imports with fallbacks
try:
    from core.config_manager import ConfigManager
    from utils.logger import get_logger
    from services.ai_service import AIService
except ImportError:
    ConfigManager = None
    get_logger = None
    AIService = None

# ChartDialog will be imported dynamically when needed

class WatchlistTable(QTableWidget):
    symbol_selected = pyqtSignal(str)
    trade_requested = pyqtSignal(str)
    tools_action = pyqtSignal(str, str)  # action, symbol

    def __init__(self):
        super().__init__()
        self.symbol_data = {}
        self.watchlist_file = "data/watchlist.json"
        self.setup_ui()
        self.load_watchlist()

    def setup_ui(self):
        headers = [
            "Symbol", "נוסף בתאריך", "Price", "Volume",
            "יום 1", "יום 2", "יום 3", "יום 4", "יום 5", "יום 6", "יום 7",
            "AI Rating", "AI Prediction", "Stop Loss", "Signal", "Sharpe Ratio", "Tools"
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
            btn.setAutoRaise(False)  # Make button more visible
            btn.setStyleSheet("""
                QToolButton {
                    font-size: 8px; 
                    font-weight: bold;
                    padding: 2px 3px;
                    border: 1px solid #ccc;
                    background-color: #f8f9fa;
                    border-radius: 3px;
                    color: #333;
                    min-width: 18px;
                    min-height: 18px;
                }
                QToolButton:hover {
                    background-color: #e9ecef;
                    border-color: #007bff;
                    color: #007bff;
                }
                QToolButton:pressed {
                    background-color: #d4edda;
                    border-color: #28a745;
                    color: #155724;
                }
            """)
            return btn
        
        ai_btn = mk_btn("Ask AI", "🤖")
        ai_btn.clicked.connect(lambda: self.tools_action.emit('ai', symbol))
        hl.addWidget(ai_btn)
        
        chart_btn = mk_btn("Open Chart", "📊")
        chart_btn.clicked.connect(lambda: self.tools_action.emit('chart', symbol))
        hl.addWidget(chart_btn)
        
        self.setCellWidget(row, self.columnCount() - 1, tools_widget)
        
        self.symbol_data[symbol] = {
            'row': row, 
            'data': {},
            'ai_rating': None,
            'ai_prediction': None,
            'ai_timestamp': None
        }
        self.save_watchlist()  # Auto-save when symbol added
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
        
        self.save_watchlist()  # Auto-save when symbol removed
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
        """Request AI rating for symbol using Perplexity API"""
        if row >= self.rowCount():
            return
            
        # Set loading state
        ai_item = QTableWidgetItem("🤖...")
        self.setItem(row, 11, ai_item)  # AI Rating column
        
        # Start AI rating request in background thread
        from PyQt6.QtCore import QThread, QObject, pyqtSignal
        
        class AIRatingWorker(QObject):
            finished = pyqtSignal(str, int, str, str)  # symbol, row, rating, prediction
            error = pyqtSignal(str, int, str)  # symbol, row, error_msg
            
            def __init__(self, symbol, row, config):
                super().__init__()
                self.symbol = symbol
                self.row = row
                self.config = config
                
            def run(self):
                try:
                    if AIService is None:
                        self.error.emit(self.symbol, self.row, "AI Service not available")
                        return
                        
                    import asyncio
                    
                    async def get_rating_and_prediction():
                        ai_service = AIService(self.config)
                        
                        # Get AI rating (0-10 score)
                        try:
                            rating_score = ai_service.score_symbol_numeric_sync(
                                self.symbol, 
                                timeout=10.0,
                                profile="swing"  # Default to swing trading
                            )
                            rating = f"{rating_score:.1f}/10"
                        except Exception as e:
                            error_msg = str(e)
                            if "SSL" in error_msg or "certificate" in error_msg:
                                rating = "SSL Error"
                            elif "timeout" in error_msg.lower():
                                rating = "Timeout"
                            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                                rating = "Auth Error"
                            else:
                                rating = f"Error: {error_msg[:15]}"
                        
                        # Get AI prediction/recommendation
                        try:
                            result = await ai_service.score_symbol(
                                self.symbol,
                                timeout=15.0
                            )
                            action = result.get('action', 'HOLD')
                            reason = result.get('reason', 'No reason provided')
                            prediction = f"{action}: {reason[:25]}..."
                        except Exception as e:
                            error_msg = str(e)
                            if "SSL" in error_msg or "certificate" in error_msg:
                                prediction = "SSL Error - Check network"
                            elif "timeout" in error_msg.lower():
                                prediction = "Request timed out"
                            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                                prediction = "API key invalid"
                            else:
                                prediction = f"Error: {error_msg[:25]}"
                        
                        return rating, prediction
                    
                    # Run async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        rating, prediction = loop.run_until_complete(get_rating_and_prediction())
                        self.finished.emit(self.symbol, self.row, rating, prediction)
                    finally:
                        loop.close()
                        
                except Exception as e:
                    self.error.emit(self.symbol, self.row, str(e))
        
        # Get config from parent widget
        config = None
        parent = self.parent()
        while parent and not hasattr(parent, 'config'):
            parent = parent.parent()
        if parent and hasattr(parent, 'config'):
            config = parent.config
        else:
            # Fallback config
            if ConfigManager:
                config = ConfigManager()
        
        if config:
            self.ai_worker = AIRatingWorker(symbol, row, config)
            self.ai_thread = QThread()
            self.ai_worker.moveToThread(self.ai_thread)
            
            # Connect signals
            self.ai_worker.finished.connect(self._on_ai_rating_finished)
            self.ai_worker.error.connect(self._on_ai_rating_error)
            self.ai_thread.started.connect(self.ai_worker.run)
            self.ai_thread.finished.connect(self.ai_thread.deleteLater)
            
            # Start thread
            self.ai_thread.start()
    
    def _on_ai_rating_finished(self, symbol, row, rating, prediction):
        """Handle AI rating completion"""
        if row < self.rowCount():
            # Update AI Rating column (11)
            ai_item = QTableWidgetItem(rating)
            self.setItem(row, 11, ai_item)
            
            # Update Price Target column (12) with prediction
            pred_item = QTableWidgetItem(prediction)
            self.setItem(row, 12, pred_item)
            
            # Store in symbol data for persistence
            if symbol in self.symbol_data:
                from datetime import datetime
                timestamp = datetime.now()
                self.symbol_data[symbol].update({
                    'ai_rating': rating,
                    'ai_prediction': prediction,
                    'ai_timestamp': timestamp.isoformat()
                })
                
                # Add timestamp tooltip to AI Rating cell
                time_str = timestamp.strftime("%H:%M:%S")
                ai_item.setToolTip(f"AI Rating updated at {time_str}")
                pred_item.setToolTip(f"AI Prediction updated at {time_str}")
                
                # Update statistics
                parent = self.parent()
                while parent and not hasattr(parent, 'update_ai_stats'):
                    parent = parent.parent()
                if parent and hasattr(parent, 'update_ai_stats'):
                    parent.update_ai_stats()
    
    def _on_ai_rating_error(self, symbol, row, error_msg):
        """Handle AI rating error"""
        if row < self.rowCount():
            ai_item = QTableWidgetItem(f"❌ {error_msg[:10]}")
            self.setItem(row, 11, ai_item)
    
    def save_watchlist(self):
        """Save watchlist to JSON file"""
        try:
            import os
            os.makedirs("data", exist_ok=True)
            
            # Collect current symbols with their add dates
            watchlist_data = {
                "symbols": [],
                "last_updated": datetime.now().isoformat()
            }
            
            for symbol, data in self.symbol_data.items():
                row = data['row']
                # Get the date from the table
                date_item = self.item(row, 1)
                added_date = date_item.text() if date_item else datetime.now().strftime("%Y-%m-%d %H:%M")
                
                watchlist_data["symbols"].append({
                    "symbol": symbol,
                    "added_at": added_date,
                    "ai_rating": data.get('ai_rating'),
                    "ai_prediction": data.get('ai_prediction'),
                    "ai_timestamp": data.get('ai_timestamp')
                })
            
            with open(self.watchlist_file, 'w') as f:
                json.dump(watchlist_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving watchlist: {e}")
    
    def load_watchlist(self):
        """Load watchlist from JSON file"""
        try:
            if not os.path.exists(self.watchlist_file):
                return
                
            with open(self.watchlist_file, 'r') as f:
                watchlist_data = json.load(f)
            
            # Add symbols from saved data
            for symbol_info in watchlist_data.get("symbols", []):
                symbol = symbol_info["symbol"]
                if symbol not in self.symbol_data:
                    # Add symbol without triggering save
                    self._add_symbol_silent(symbol, symbol_info.get("added_at"))
                    
        except Exception as e:
            print(f"Error loading watchlist: {e}")
    
    def _add_symbol_silent(self, symbol: str, added_at: str = None):
        """Add symbol without triggering save (for loading)"""
        if symbol in self.symbol_data:
            return False
        
        row = self.rowCount()
        self.insertRow(row)
        symbol_item = QTableWidgetItem(symbol)
        symbol_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.setItem(row, 0, symbol_item)
        
        if not added_at:
            added_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        added_item = QTableWidgetItem(added_at)
        added_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 1, added_item)
        
        for col in range(2, self.columnCount() - 1):
            item = QTableWidgetItem("-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, col, item)

        # Add tools column (same as regular add_symbol)
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
            btn.setAutoRaise(False)
            btn.setStyleSheet("""
                QToolButton {
                    font-size: 8px; 
                    font-weight: bold;
                    padding: 2px 3px;
                    border: 1px solid #ccc;
                    background-color: #f8f9fa;
                    border-radius: 3px;
                    color: #333;
                    min-width: 18px;
                    min-height: 18px;
                }
                QToolButton:hover {
                    background-color: #e9ecef;
                    border-color: #adb5bd;
                }
            """)
            return btn
        
        # AI button
        ai_btn = mk_btn("Get AI Rating", "🤖")
        ai_btn.clicked.connect(lambda: self.tools_action.emit("ai", symbol))
        hl.addWidget(ai_btn)
        
        # Chart button  
        chart_btn = mk_btn("Open Chart", "📊") 
        chart_btn.clicked.connect(lambda: self.tools_action.emit("chart", symbol))
        hl.addWidget(chart_btn)
        
        self.setCellWidget(row, self.columnCount() - 1, tools_widget)
        
        self.symbol_data[symbol] = {
            'row': row, 
            'data': {},
            'ai_rating': None,
            'ai_prediction': None,
            'ai_timestamp': None
        }
        return True



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
        
        # Initialize AI service
        self.ai_service = None
        if AIService:
            try:
                self.ai_service = AIService(self.config)
            except Exception as e:
                self.logger.warning(f"Could not initialize AI service: {e}")
        
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
        
        # AI rating for all symbols button
        ai_all_btn = QPushButton("🤖 Rate All")
        ai_all_btn.setToolTip("Get AI ratings for all symbols in watchlist")
        ai_all_btn.clicked.connect(self.rate_all_symbols)
        title_layout.addWidget(ai_all_btn)
        
        # Clear AI ratings button
        clear_ai_btn = QPushButton("🧹 Clear AI")
        clear_ai_btn.setToolTip("Clear all AI ratings and predictions")
        clear_ai_btn.clicked.connect(self.clear_all_ai_ratings)
        title_layout.addWidget(clear_ai_btn)
        
        # AI Filter dropdown
        from PyQt6.QtWidgets import QComboBox
        self.ai_filter = QComboBox()
        self.ai_filter.addItems(["All", "BUY", "SELL", "HOLD", "No AI Rating"])
        self.ai_filter.setToolTip("Filter symbols by AI recommendation")
        self.ai_filter.currentTextChanged.connect(self.apply_ai_filter)
        title_layout.addWidget(QLabel("Filter:"))
        title_layout.addWidget(self.ai_filter)
        
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
        
        # No details panel - removed for cleaner UI
        splitter.setSizes([400])  # Full space to table
        layout.addWidget(splitter)


    def _handle_tools_action(self, action: str, symbol: str):
        """Handle per-row tools icon actions: AI and Chart"""
        self.logger.info(f"Tools action requested: {action} for {symbol}")
        if action == 'ai':
            self._handle_ask_ai(symbol)
        elif action == 'chart':
            self._handle_open_chart(symbol)
        else:
            self.logger.warning(f"Unknown tools action: {action}")

    def _handle_ask_ai(self, symbol: str):
        """Handle AI request for symbol"""
        if not symbol:
            return
        
        row = self.watchlist_table.symbol_data.get(symbol, {}).get('row')
        if row is None:
            return
        
        try:
            if AIService is None:
                QMessageBox.warning(self, "AI Service", "AI Service is not available.\nPlease check your configuration.")
                return
            self.watchlist_table.request_ai_rating(symbol, row)
            self.logger.info(f"AI rating requested for {symbol}")
        except Exception as e:
            self.logger.error(f"Error requesting AI rating for {symbol}: {e}")
            QMessageBox.warning(self, "AI Error", f"Failed to get AI rating for {symbol}:\n{str(e)[:100]}")

    def _handle_open_chart(self, symbol: str):
        """Handle chart opening for symbol"""
        self.logger.info(f"Chart requested for {symbol}")
        
        try:
            # Import ChartDialog dynamically to avoid circular imports
            from ui.widgets.scanner_widget import ChartDialog as CD
            self.logger.info(f"Creating ChartDialog for {symbol}")
            # Open chart dialog with timeframe selector (same as scanner)
            dlg = CD(symbol, self)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.show()
            self.logger.info(f"Chart dialog opened successfully for {symbol}")
        except ImportError as e:
            self.logger.error(f"Could not import ChartDialog: {e}")
            QMessageBox.warning(self, "Chart", "Chart functionality is not available.\nScanner module may not be loaded.")
        except Exception as e:
            self.logger.error(f"Error opening chart for {symbol}: {e}")
            QMessageBox.warning(self, "Chart", f"Failed to open chart for {symbol}:\n{e}")

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
        # Details panel removed - no action needed
        pass

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
    
    def rate_all_symbols(self):
        """Request AI ratings for all symbols in the watchlist"""
        symbols = self.watchlist_table.get_symbols()
        if not symbols:
            QMessageBox.information(self, "Info", "No symbols in watchlist to rate")
            return
        
        reply = QMessageBox.question(
            self, 
            "Rate All Symbols", 
            f"Request AI ratings for all {len(symbols)} symbols?\n\nThis may take a few moments and will include small delays between requests.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Use QTimer to space out requests to avoid API rate limits
            self.batch_rating_symbols = symbols.copy()
            self.batch_rating_index = 0
            self.batch_timer = QTimer()
            self.batch_timer.timeout.connect(self._process_next_rating)
            self.batch_timer.start(1000)  # 1 second delay between requests
            
            self.logger.info(f"Starting batch AI rating for {len(symbols)} symbols")
    
    def _process_next_rating(self):
        """Process the next symbol in batch rating"""
        if self.batch_rating_index >= len(self.batch_rating_symbols):
            # All done
            self.batch_timer.stop()
            self.logger.info("Batch AI rating completed")
            return
        
        symbol = self.batch_rating_symbols[self.batch_rating_index]
        symbol_data = self.watchlist_table.symbol_data.get(symbol)
        if symbol_data:
            row = symbol_data['row']
            self.watchlist_table.request_ai_rating(symbol, row)
            self.logger.debug(f"Processing AI rating for {symbol} ({self.batch_rating_index + 1}/{len(self.batch_rating_symbols)})")
        
        self.batch_rating_index += 1
    
    def clear_all_ai_ratings(self):
        """Clear all AI ratings and predictions from the watchlist"""
        symbols = self.watchlist_table.get_symbols()
        if not symbols:
            return
        
        reply = QMessageBox.question(
            self,
            "Clear AI Ratings",
            f"Clear AI ratings for all {len(symbols)} symbols?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for symbol in symbols:
                symbol_data = self.watchlist_table.symbol_data.get(symbol)
                if symbol_data:
                    row = symbol_data['row']
                    # Clear AI Rating column (11)
                    self.watchlist_table.setItem(row, 11, QTableWidgetItem("-"))
                    # Clear AI Prediction column (12)
                    self.watchlist_table.setItem(row, 12, QTableWidgetItem("-"))
                    
                    # Clear from stored data
                    symbol_data.update({
                        'ai_rating': None,
                        'ai_prediction': None,
                        'ai_timestamp': None
                    })
            
            self.logger.info(f"Cleared AI ratings for {len(symbols)} symbols")
    
    def apply_ai_filter(self, filter_text):
        """Apply AI recommendation filter to the watchlist"""
        if filter_text == "All":
            # Show all rows
            for row in range(self.watchlist_table.rowCount()):
                self.watchlist_table.setRowHidden(row, False)
        else:
            # Filter based on AI prediction
            for row in range(self.watchlist_table.rowCount()):
                pred_item = self.watchlist_table.item(row, 12)  # AI Prediction column
                should_hide = True
                
                if pred_item:
                    pred_text = pred_item.text()
                    if filter_text == "No AI Rating":
                        should_hide = pred_text == "-" or pred_text == ""
                    else:
                        should_hide = not pred_text.startswith(filter_text)
                else:
                    should_hide = filter_text != "No AI Rating"
                
                self.watchlist_table.setRowHidden(row, should_hide)
    

    


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
                # Details panel removed - no update needed
            
            self.logger.info(f"Added symbol from scanner: {symbol}")
            return True
        return False

    def cleanup(self):
        """Stop timers and threads gracefully"""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()

        if hasattr(self, 'batch_timer'):
            self.batch_timer.stop()
        if self.data_worker and hasattr(self.data_worker, 'stop_monitoring'):
            try:
                self.data_worker.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping data worker: {e}")

    def closeEvent(self, event):
        """Qt close event override to cleanup resources"""
        self.cleanup()
        super().closeEvent(event)