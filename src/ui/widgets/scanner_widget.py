"""
Scanner Widget
For scanning and screening stocks based on various criteria
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QFormLayout, QCheckBox, QGroupBox, QHeaderView,
    QProgressBar, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QColor

from core.config_manager import ConfigManager
from utils.logger import get_logger


class ScannerWorker(QObject):
    """Worker thread for stock scanning"""
    
    progress_updated = pyqtSignal(int)
    results_updated = pyqtSignal(list)
    scan_completed = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.logger = get_logger("ScannerWorker")
        self.is_scanning = False
    
    def start_scan(self, criteria: dict):
        """Start scanning with given criteria"""
        try:
            self.is_scanning = True
            self.progress_updated.emit(0)
            
            # TODO: Implement real stock scanning
            # This should connect to real market data and apply filters
            
            import time
            
            steps = [
                "Initializing scanner...",
                "Connecting to data source...",
                "Loading market data...", 
                "Applying filters...",
                "Processing results...",
                "Finalizing scan...",
                "Scan ready - connect data source for real results"
            ]
            
            results = []
            
            for i, step in enumerate(steps):
                if not self.is_scanning:
                    break
                    
                self.status_updated.emit(step)
                progress = int((i + 1) / len(steps) * 100)
                self.progress_updated.emit(progress)
                
                # Simulate work
                time.sleep(0.5)
            
            if self.is_scanning:
                # Return empty results - real implementation would return filtered stocks
                self.results_updated.emit(results)
                self.scan_completed.emit(0)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.is_scanning = False
    
    def stop_scan(self):
        """Stop scanning process"""
        self.is_scanning = False


class ScanCriteriaWidget(QFrame):
    """Widget for setting scan criteria"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup criteria UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Scan Criteria")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Create tabs for different criteria types
        self.criteria_tabs = QTabWidget()
        
        # Price & Volume Tab
        self.create_price_volume_tab()
        self.criteria_tabs.addTab(self.price_volume_widget, "💰 Price & Volume")
        
        # Technical Tab
        self.create_technical_tab()
        self.criteria_tabs.addTab(self.technical_widget, "📈 Technical")
        
        # Fundamental Tab
        self.create_fundamental_tab()
        self.criteria_tabs.addTab(self.fundamental_widget, "📊 Fundamental")
        
        layout.addWidget(self.criteria_tabs)
        
        # Quick scan presets
        self.create_preset_buttons()
        layout.addWidget(self.preset_frame)
    
    def create_price_volume_tab(self):
        """Create price and volume criteria tab"""
        self.price_volume_widget = QWidget()
        layout = QFormLayout(self.price_volume_widget)
        
        # Price range
        self.min_price_spin = QDoubleSpinBox()
        self.min_price_spin.setRange(0, 10000)
        self.min_price_spin.setValue(0)
        self.min_price_spin.setPrefix("$")
        layout.addRow("Min Price:", self.min_price_spin)
        
        self.max_price_spin = QDoubleSpinBox()
        self.max_price_spin.setRange(0, 10000)
        self.max_price_spin.setValue(1000)
        self.max_price_spin.setPrefix("$")
        layout.addRow("Max Price:", self.max_price_spin)
        
        # Volume
        self.min_volume_spin = QSpinBox()
        self.min_volume_spin.setRange(0, 1000000000)
        self.min_volume_spin.setValue(100000)
        self.min_volume_spin.setSuffix(" shares")
        layout.addRow("Min Volume:", self.min_volume_spin)
        
        # Change percentage
        self.min_change_spin = QDoubleSpinBox()
        self.min_change_spin.setRange(-100, 100)
        self.min_change_spin.setValue(-5)
        self.min_change_spin.setSuffix("%")
        layout.addRow("Min Change:", self.min_change_spin)
        
        self.max_change_spin = QDoubleSpinBox()
        self.max_change_spin.setRange(-100, 100)
        self.max_change_spin.setValue(5)
        self.max_change_spin.setSuffix("%")
        layout.addRow("Max Change:", self.max_change_spin)
    
    def create_technical_tab(self):
        """Create technical criteria tab"""
        self.technical_widget = QWidget()
        layout = QFormLayout(self.technical_widget)
        
        # RSI
        self.min_rsi_spin = QSpinBox()
        self.min_rsi_spin.setRange(0, 100)
        self.min_rsi_spin.setValue(30)
        layout.addRow("Min RSI:", self.min_rsi_spin)
        
        self.max_rsi_spin = QSpinBox()
        self.max_rsi_spin.setRange(0, 100)
        self.max_rsi_spin.setValue(70)
        layout.addRow("Max RSI:", self.max_rsi_spin)
        
        # Moving averages
        self.above_sma20 = QCheckBox("Above 20-day SMA")
        layout.addRow("", self.above_sma20)
        
        self.above_sma50 = QCheckBox("Above 50-day SMA")
        layout.addRow("", self.above_sma50)
        
        self.above_sma200 = QCheckBox("Above 200-day SMA")
        layout.addRow("", self.above_sma200)
        
        # Pattern detection
        self.bullish_patterns = QCheckBox("Bullish Patterns")
        layout.addRow("", self.bullish_patterns)
        
        self.bearish_patterns = QCheckBox("Bearish Patterns")
        layout.addRow("", self.bearish_patterns)
    
    def create_fundamental_tab(self):
        """Create fundamental criteria tab"""
        self.fundamental_widget = QWidget()
        layout = QFormLayout(self.fundamental_widget)
        
        # P/E Ratio
        self.min_pe_spin = QDoubleSpinBox()
        self.min_pe_spin.setRange(0, 1000)
        self.min_pe_spin.setValue(5)
        layout.addRow("Min P/E:", self.min_pe_spin)
        
        self.max_pe_spin = QDoubleSpinBox()
        self.max_pe_spin.setRange(0, 1000)
        self.max_pe_spin.setValue(50)
        layout.addRow("Max P/E:", self.max_pe_spin)
        
        # Market Cap
        self.min_market_cap_spin = QDoubleSpinBox()
        self.min_market_cap_spin.setRange(0, 10000)
        self.min_market_cap_spin.setValue(1)
        self.min_market_cap_spin.setSuffix("B")
        layout.addRow("Min Market Cap:", self.min_market_cap_spin)
        
        # Sector
        self.sector_combo = QComboBox()
        self.sector_combo.addItems([
            "Any Sector",
            "Technology",
            "Healthcare",
            "Financial",
            "Consumer Cyclical",
            "Consumer Defensive",
            "Energy",
            "Industrials",
            "Materials",
            "Utilities",
            "Real Estate",
            "Communication Services"
        ])
        layout.addRow("Sector:", self.sector_combo)
    
    def create_preset_buttons(self):
        """Create preset scan buttons"""
        self.preset_frame = QFrame()
        self.preset_frame.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self.preset_frame)
        
        preset_label = QLabel("Quick Scan Presets")
        preset_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(preset_label)
        
        button_layout = QHBoxLayout()
        
        # Preset buttons
        presets = [
            ("🚀 Growth Stocks", self.load_growth_preset),
            ("💎 Value Stocks", self.load_value_preset),
            ("📈 Momentum", self.load_momentum_preset),
            ("🔻 Oversold", self.load_oversold_preset)
        ]
        
        for text, callback in presets:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            button_layout.addWidget(btn)
        
        layout.addLayout(button_layout)
    
    def load_growth_preset(self):
        """Load growth stock preset"""
        self.min_price_spin.setValue(20)
        self.max_price_spin.setValue(1000)
        self.min_volume_spin.setValue(500000)
        self.min_change_spin.setValue(-2)
        self.max_change_spin.setValue(10)
        self.above_sma20.setChecked(True)
        self.above_sma50.setChecked(True)
    
    def load_value_preset(self):
        """Load value stock preset"""
        self.min_price_spin.setValue(5)
        self.max_price_spin.setValue(100)
        self.min_volume_spin.setValue(100000)
        self.min_pe_spin.setValue(5)
        self.max_pe_spin.setValue(15)
    
    def load_momentum_preset(self):
        """Load momentum preset"""
        self.min_change_spin.setValue(3)
        self.max_change_spin.setValue(20)
        self.min_volume_spin.setValue(1000000)
        self.min_rsi_spin.setValue(50)
        self.max_rsi_spin.setValue(80)
    
    def load_oversold_preset(self):
        """Load oversold preset"""
        self.min_rsi_spin.setValue(20)
        self.max_rsi_spin.setValue(35)
        self.min_change_spin.setValue(-10)
        self.max_change_spin.setValue(-2)
    
    def get_criteria(self) -> dict:
        """Get current scan criteria"""
        return {
            'min_price': self.min_price_spin.value(),
            'max_price': self.max_price_spin.value(),
            'min_volume': self.min_volume_spin.value(),
            'min_change': self.min_change_spin.value(),
            'max_change': self.max_change_spin.value(),
            'min_rsi': self.min_rsi_spin.value(),
            'max_rsi': self.max_rsi_spin.value(),
            'min_pe': self.min_pe_spin.value(),
            'max_pe': self.max_pe_spin.value(),
            'min_market_cap': self.min_market_cap_spin.value(),
            'above_sma20': self.above_sma20.isChecked(),
            'above_sma50': self.above_sma50.isChecked(),
            'above_sma200': self.above_sma200.isChecked(),
            'sector': self.sector_combo.currentText()
        }


class ScanResultsTable(QTableWidget):
    """Table for displaying scan results"""
    
    symbol_selected = pyqtSignal(str)
    add_to_watchlist = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup results table"""
        # Set columns
        self.setColumnCount(8)
        headers = ["Symbol", "Price", "Change %", "Volume", "RSI", "P/E", "Market Cap", "Score"]
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Configure headers
        header = self.horizontalHeader()
        for i in range(7):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        
        # Set column widths
        self.setColumnWidth(0, 80)   # Symbol
        self.setColumnWidth(1, 80)   # Price
        self.setColumnWidth(2, 80)   # Change %
        self.setColumnWidth(3, 100)  # Volume
        self.setColumnWidth(4, 60)   # RSI
        self.setColumnWidth(5, 60)   # P/E
        self.setColumnWidth(6, 100)  # Market Cap
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Double click
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)
    
    def update_results(self, results: list):
        """Update table with scan results"""
        self.setRowCount(len(results))
        
        for row, result in enumerate(results):
            # Symbol
            symbol_item = QTableWidgetItem(result['symbol'])
            symbol_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.setItem(row, 0, symbol_item)
            
            # Price
            price_item = QTableWidgetItem(f"${result['price']:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 1, price_item)
            
            # Change %
            change_item = QTableWidgetItem(f"{result['change_pct']:+.2f}%")
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            if result['change_pct'] >= 0:
                change_item.setForeground(QColor("#4CAF50"))
            else:
                change_item.setForeground(QColor("#f44336"))
            self.setItem(row, 2, change_item)
            
            # Volume
            volume_item = QTableWidgetItem(f"{result['volume']:,}")
            volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 3, volume_item)
            
            # RSI
            rsi_item = QTableWidgetItem(f"{result['rsi']:.1f}")
            rsi_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 4, rsi_item)
            
            # P/E
            pe_item = QTableWidgetItem(f"{result['pe_ratio']:.1f}")
            pe_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 5, pe_item)
            
            # Market Cap
            market_cap_item = QTableWidgetItem(f"${result['market_cap']:.1f}B")
            market_cap_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 6, market_cap_item)
            
            # Score
            score_item = QTableWidgetItem(f"{result['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            score_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.setItem(row, 7, score_item)
    
    def show_context_menu(self, position):
        """Show context menu"""
        if self.itemAt(position) is None:
            return
        
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        menu = QMenu(self)
        
        # Get selected symbol
        row = self.itemAt(position).row()
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            
            add_action = QAction(f"Add {symbol} to Watchlist", self)
            add_action.triggered.connect(lambda: self.add_to_watchlist.emit(symbol))
            menu.addAction(add_action)
            
            menu.exec(self.mapToGlobal(position))
    
    def on_cell_double_clicked(self, row: int, column: int):
        """Handle cell double click"""
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            self.symbol_selected.emit(symbol)


class ScannerWidget(QWidget):
    """Main scanner widget"""
    
    add_to_watchlist = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("Scanner")
        self.config = ConfigManager()
        
        # Initialize scanner worker
        self.scanner_thread = QThread()
        self.scanner_worker = ScannerWorker(self.config)
        self.scanner_worker.moveToThread(self.scanner_thread)
        self.scanner_worker.progress_updated.connect(self.update_progress)
        self.scanner_worker.results_updated.connect(self.update_results)
        self.scanner_worker.scan_completed.connect(self.on_scan_completed)
        self.scanner_worker.error_occurred.connect(self.on_scan_error)
        self.scanner_thread.start()
        
        # Setup UI
        self.setup_ui()
        
        self.logger.info("Scanner widget initialized")
    
    def setup_ui(self):
        """Setup the scanner UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title and controls
        title_layout = QHBoxLayout()
        
        title = QLabel("Stock Scanner")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # Scan button
        self.scan_button = QPushButton("🔍 Start Scan")
        self.scan_button.clicked.connect(self.start_scan)
        title_layout.addWidget(self.scan_button)
        
        self.stop_button = QPushButton("⏹️ Stop Scan")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        title_layout.addWidget(self.stop_button)
        
        layout.addLayout(title_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to scan")
        layout.addWidget(self.status_label)
        
        # Create splitter for criteria and results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Scan criteria
        self.criteria_widget = ScanCriteriaWidget()
        self.criteria_widget.setMaximumWidth(400)
        splitter.addWidget(self.criteria_widget)
        
        # Results table
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.Shape.Box)
        results_layout = QVBoxLayout(results_frame)
        
        results_title = QLabel("Scan Results")
        results_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        results_layout.addWidget(results_title)
        
        self.results_table = ScanResultsTable()
        self.results_table.symbol_selected.connect(self.on_symbol_selected)
        self.results_table.add_to_watchlist.connect(self.add_to_watchlist.emit)
        results_layout.addWidget(self.results_table)
        
        splitter.addWidget(results_frame)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 1)  # Criteria
        splitter.setStretchFactor(1, 2)  # Results
        
        layout.addWidget(splitter)
    
    def start_scan(self):
        """Start stock scan"""
        try:
            # Get scan criteria
            criteria = self.criteria_widget.get_criteria()
            
            # Update UI
            self.scan_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Scanning stocks...")
            
            # Clear previous results
            self.results_table.setRowCount(0)
            
            # Start scan
            QTimer.singleShot(100, lambda: self.scanner_worker.start_scan(criteria))
            
            self.logger.info("Stock scan started")
            
        except Exception as e:
            self.logger.error(f"Error starting scan: {e}")
            self.reset_scan_ui()
    
    def stop_scan(self):
        """Stop stock scan"""
        self.scanner_worker.stop_scan()
        self.reset_scan_ui()
        self.status_label.setText("Scan stopped")
        self.logger.info("Stock scan stopped")
    
    def update_progress(self, progress: int):
        """Update scan progress"""
        self.progress_bar.setValue(progress)
    
    def update_results(self, results: list):
        """Update scan results"""
        self.results_table.update_results(results)
        self.status_label.setText(f"Found {len(results)} matching stocks")
    
    def on_scan_completed(self, count: int):
        """Handle scan completion"""
        self.reset_scan_ui()
        self.status_label.setText(f"Scan completed - Connect to data source for real scanning")
        self.logger.info(f"Scan completed - ready for real data integration")
    
    def on_scan_error(self, error: str):
        """Handle scan error"""
        self.reset_scan_ui()
        self.status_label.setText(f"Scan error: {error}")
        self.logger.error(f"Scan error: {error}")
    
    def on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        self.logger.debug(f"Symbol selected from scanner: {symbol}")
    
    def reset_scan_ui(self):
        """Reset scan UI to default state"""
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Scanner tab activated")
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop scan if running
        if self.stop_button.isEnabled():
            self.stop_scan()
        
        # Stop worker thread
        if hasattr(self, 'scanner_thread'):
            self.scanner_thread.quit()
            self.scanner_thread.wait()
        
        event.accept()