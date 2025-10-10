from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QLineEdit, QHeaderView,
    QMenu, QMessageBox, QSplitter, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QAction
from datetime import datetime, timedelta
import json
import re
import os
from pathlib import Path

# Optional pandas import
try:
    import pandas as pd
except ImportError:
    pd = None

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
        self.itemChanged.connect(self.on_item_changed)

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
                    font-size: 6px; 
                    font-weight: bold;
                    padding: 1px 2px;
                    border: 1px solid #ccc;
                    background-color: #f8f9fa;
                    border-radius: 2px;
                    color: #333;
                    min-width: 14px;
                    min-height: 14px;
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
        
        # Make date column editable
        self.make_date_editable(row, 1)
        
        # Load initial data
        self.update_symbol_display(symbol)
        
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

    def on_item_changed(self, item):
        """Handle item content change - specifically for date column edits"""
        if item.column() == 1:  # נוסף בתאריך column
            row = item.row()
            symbol_item = self.item(row, 0)
            if symbol_item:
                symbol = symbol_item.text()
                # Only update if the symbol is fully loaded in symbol_data
                if symbol in self.symbol_data:
                    # Update display with new reference date
                    self.update_symbol_display(symbol)
                    # Save the updated watchlist
                    self.save_watchlist()

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
                    
                    ai_service = AIService(self.config)
                    
                    # Use the SAME API call for both rating and prediction to avoid SSL issues
                    import asyncio
                    
                    async def get_both_rating_and_prediction():
                        try:
                            # Single API call for both values
                            result = await ai_service.score_symbol(
                                self.symbol,
                                timeout=15.0
                            )
                            
                            # Extract rating (0-10 score)
                            score = result.get('score', 5.0)
                            rating = f"{score:.1f}/10"
                            
                            # Extract 7-day price target
                            price_target = result.get('price_target', None)
                            
                            if price_target is not None:
                                try:
                                    target = float(price_target)
                                    prediction = f"${target:.2f} (7d)"
                                except (ValueError, TypeError):
                                    prediction = f"${price_target} (7d)"
                            else:
                                prediction = "No target"
                            
                            return rating, prediction
                            
                        except Exception as e:
                            error_msg = str(e)
                            if "SSL" in error_msg or "certificate" in error_msg:
                                error_result = "SSL Error"
                            elif "timeout" in error_msg.lower():
                                error_result = "Timeout"
                            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                                error_result = "Auth Error"
                            elif "400" in error_msg:
                                error_result = "Bad Request"
                            else:
                                error_result = "Error"
                            
                            return error_result, error_result
                    
                    # Run single async call for both values
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        rating, prediction = loop.run_until_complete(get_both_rating_and_prediction())
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
            
            # Proper thread cleanup
            self.ai_worker.finished.connect(self.ai_thread.quit)
            self.ai_worker.error.connect(self.ai_thread.quit)
            self.ai_thread.finished.connect(self.ai_thread.deleteLater)
            self.ai_thread.finished.connect(self.ai_worker.deleteLater)
            
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
                    font-size: 6px; 
                    font-weight: bold;
                    padding: 1px 2px;
                    border: 1px solid #ccc;
                    background-color: #f8f9fa;
                    border-radius: 2px;
                    color: #333;
                    min-width: 14px;
                    min-height: 14px;
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
        
        # Make date column editable
        self.make_date_editable(row, 1)
        
        # Load initial data
        self.update_symbol_display(symbol)
        
        return True

    def load_symbol_data(self, symbol: str, reference_date: str = None):
        """Load price data for a symbol from parquet files"""
        if pd is None:
            return {'error': 'Pandas not available'}
            
        try:
            # If no reference date provided, use today
            if reference_date is None:
                reference_date = datetime.now().strftime("%Y-%m-%d")
            else:
                # Parse the reference date from the table format
                try:
                    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d %H:%M")
                except ValueError:
                    try:
                        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
                    except ValueError:
                        return {'error': f'Invalid date format: {reference_date}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM'}
                reference_date = ref_dt.strftime("%Y-%m-%d")
            
            # Try to load data from parquet
            parquet_path = Path("data/bronze/daily") / f"{symbol}.parquet"
            if not parquet_path.exists():
                return {'error': f'No data file found for {symbol}'}
                
            df = pd.read_parquet(parquet_path)
            if df.empty:
                return {'error': f'No data available for {symbol}'}
                
            # Ensure date column exists and is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            else:
                return {'error': f'No date column in data for {symbol}'}
            
            # Find the reference date or closest date before it
            ref_date = pd.to_datetime(reference_date)
            
            # Get the most recent available data
            df_filtered = df[df['date'] <= ref_date]
            if df_filtered.empty:
                df_filtered = df.tail(1)
            ref_row = df_filtered.iloc[-1]
            ref_price = ref_row.get('close', ref_row.get('adj_close', 0))
            
            # Check if there are data points after the reference date
            has_future_data = not df[df['date'] > ref_date].empty
            
            # Calculate progression: show 7 days starting from reference date forward
            daily_data = {}
            for i in range(1, 8):  # Days 1-7 (reference date + (i-1) days)
                if not has_future_data and i > 1:
                    # If no data after reference date, only show day 1
                    daily_data[f'day_{i}'] = None
                else:
                    target_date = ref_date + timedelta(days=i-1)  # Day 1 = ref_date, Day 2 = ref_date + 1, etc.
                    # Find the closest available date on or before target_date
                    available_data = df[df['date'] <= target_date]
                    if not available_data.empty:
                        # Get the most recent available date
                        closest_row = available_data.iloc[-1]
                        price = closest_row['close'] if 'close' in closest_row.index else closest_row.get('adj_close', 0)
                        daily_data[f'day_{i}'] = price
                    else:
                        daily_data[f'day_{i}'] = None
            
            return {
                'price': ref_price,  # Use reference price as current price
                'volume': ref_row.get('volume', 0),
                'daily_progression': daily_data,
                'reference_date': reference_date
            }
            
        except Exception as e:
            return {'error': f'Error loading data for {symbol}: {str(e)}'}
    
    def update_symbol_display(self, symbol: str):
        """Update the display data for a symbol"""
        if symbol not in self.symbol_data:
            return
            
        row = self.symbol_data[symbol]['row']
        
        # Get reference date from the "נוסף בתאריך" column
        date_item = self.item(row, 1)
        reference_date = date_item.text() if date_item else None
        
        # Load data
        data = self.load_symbol_data(symbol, reference_date)
        if data is None or 'error' in data:
            # Show error message in price column
            error_msg = data.get('error', 'Unknown error') if data else 'No data available'
            error_item = QTableWidgetItem(f"❌ {error_msg}")
            error_item.setForeground(QColor(200, 0, 0))
            self.setItem(row, 2, error_item)
            
            # Clear other data columns
            for col in range(3, self.columnCount() - 1):
                item = QTableWidgetItem("-")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                item.setForeground(QColor(100, 100, 100))
                self.setItem(row, col, item)
            return
            
        # Update Price column (2)
        price_item = QTableWidgetItem(f"${data['price']:.2f}")
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 2, price_item)
        
        # Update Volume column (3)  
        volume_str = f"{data['volume']:,}" if data['volume'] > 0 else "-"
        volume_item = QTableWidgetItem(volume_str)
        volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 3, volume_item)
        
        # Update daily progression columns (4-10 = Days 1-7, where Day 1 is most recent)
        for i in range(1, 8):
            col_index = 3 + i  # Columns 4-10
            day_price = data['daily_progression'].get(f'day_{i}')
            if day_price is not None:
                # Calculate percentage change from reference price
                if data['price'] > 0:
                    pct_change = ((day_price - data['price']) / data['price']) * 100
                    text = f"${day_price:.2f}\n({pct_change:+.1f}%)"
                    if pct_change >= 0:
                        color = QColor(0, 150, 0)
                        font = QFont()
                    else:
                        color = QColor(255, 0, 0)  # Brighter red
                        font = QFont()
                        font.setBold(True)  # Make negative changes bold
                else:
                    text = f"${day_price:.2f}"
                    color = QColor(0, 0, 0)
                    font = QFont()
            else:
                text = "-"
                color = QColor(100, 100, 100)
                font = QFont()
                
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            item.setForeground(color)
            item.setFont(font)
            self.setItem(row, col_index, item)
        
        # Store data for future reference
        self.symbol_data[symbol]['data'] = data

    def make_date_editable(self, row: int, col: int):
        """Make the date column editable for dynamic reference dates"""
        if col == 1:  # נוסף בתאריך column
            item = self.item(row, col)
            if item:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)


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
        
        # Refresh price data from parquet files
        for symbol in symbols:
            self.watchlist_table.update_symbol_display(symbol)
        
        # Also try the old data worker if available
        if symbols and self.data_worker and hasattr(self.data_worker, 'start_monitoring'):
            try:
                self.data_worker.stop_monitoring()
                self.data_worker.start_monitoring(symbols)
            except Exception as e:
                self.logger.error(f"Error refreshing data: {e}")
        
        self.logger.info(f"Manual watchlist refresh completed for {len(symbols)} symbols")

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