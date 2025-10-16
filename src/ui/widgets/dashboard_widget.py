"""
Dashboard Widget
Main overview widget showing key metrics and market data
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QSpinBox, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPalette

from core.config_manager import ConfigManager
from utils.logger import get_logger
from .macro_widget import MacroWidget
from .calendar_widget import CalendarWidget


class MetricCard(QFrame):
    """Individual metric card widget"""
    
    def __init__(self, title: str, value: str = "0", subtitle: str = ""):
        super().__init__()
        self.setup_ui(title, value, subtitle)
    
    def setup_ui(self, title: str, value: str, subtitle: str):
        """Setup metric card UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setFixedHeight(90)  # Reduced from 120 to 90
        self.setMinimumWidth(180)  # Reduced from 200 to 180
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)  # Reduced margins
        layout.setSpacing(2)  # Reduced spacing
        
        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(9)  # Reduced from 10
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_font = QFont()
        value_font.setPointSize(14)  # Reduced from 16
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        layout.addWidget(self.value_label)
        
        # Subtitle
        self.subtitle_label = None
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle_font = QFont()
            subtitle_font.setPointSize(8)
            self.subtitle_label.setFont(subtitle_font)
            layout.addWidget(self.subtitle_label)
    
    def update_value(self, value: str, subtitle: str = ""):
        """Update the metric value"""
        self.value_label.setText(value)
        if subtitle and self.subtitle_label:
            self.subtitle_label.setText(subtitle)


class MarketOverview(QFrame):
    """Market overview section"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup market overview UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMinimumHeight(200)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Market Overview")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Market data grid
        self.create_market_grid()
        layout.addWidget(self.market_grid)
    
    def create_market_grid(self):
        """Create market data grid"""
        self.market_grid = QFrame()
        grid_layout = QGridLayout(self.market_grid)
        
        # Market indices - will be populated with real data
        indices = [
            ("S&P 500", "Loading...", "Loading..."),
            ("NASDAQ", "Loading...", "Loading..."),
            ("DOW", "Loading...", "Loading..."),
            ("VIX", "Loading...", "Loading...")
        ]
        
        for i, (name, value, change) in enumerate(indices):
            # Name
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            grid_layout.addWidget(name_label, i, 0)
            
            # Value
            value_label = QLabel(value)
            grid_layout.addWidget(value_label, i, 1)
            
            # Change
            change_label = QLabel(change)
            if change.startswith("+"):
                change_label.setStyleSheet("color: #4CAF50;")
            else:
                change_label.setStyleSheet("color: #f44336;")
            grid_layout.addWidget(change_label, i, 2)


class DashboardWidget(QWidget):
    """Main dashboard widget"""
    
    # Signals
    refresh_requested = pyqtSignal()
    connection_requested = pyqtSignal()  # New signal for connection request
    
    def __init__(self, ibkr_service=None):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("Dashboard")
        self.config = ConfigManager()
        
        # IBKR service reference
        self.ibkr_service = ibkr_service

        # Initialize auto-refresh state before building UI (used by controls)
        self.auto_refresh_enabled = True
        self.current_interval_ms = self.config.ui.update_interval

        # Setup UI
        self.setup_ui()

        # Setup update timer (auto-refresh)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(self.current_interval_ms)

        self.logger.info("Dashboard widget initialized")
    
    def setup_ui(self):
        """Setup the dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins
        layout.setSpacing(15)  # Reduced spacing
        
        # Title and refresh button
        title_layout = QHBoxLayout()
        
        title = QLabel("Trading Dashboard")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        title_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setToolTip("Manually refresh all data from IBKR")
        refresh_btn.setFixedSize(120, 30)
        refresh_btn.setStyleSheet(
            """
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
            """
        )
        refresh_btn.clicked.connect(self.refresh_data)
        title_layout.addWidget(refresh_btn)

        # Auto-refresh controls: toggle + interval spinbox
        self.auto_refresh_chk = QCheckBox("Auto")
        self.auto_refresh_chk.setChecked(True)
        self.auto_refresh_chk.setToolTip("Toggle automatic refresh on/off")
        self.auto_refresh_chk.stateChanged.connect(self.on_auto_refresh_toggled)
        title_layout.addWidget(self.auto_refresh_chk)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3600)
        self.interval_spin.setValue(max(1, self.current_interval_ms // 1000))
        self.interval_spin.setSuffix(" s")
        self.interval_spin.setToolTip("Change auto-refresh interval (seconds)")
        self.interval_spin.valueChanged.connect(self.on_interval_changed)
        title_layout.addWidget(self.interval_spin)
        
        # Add connection indicator
        self.connection_indicator = QLabel("⚪ Checking...")
        self.connection_indicator.setToolTip("IBKR connection status")
        title_layout.addWidget(self.connection_indicator)
        
        # Quick connect button (only shown when disconnected)
        self.quick_connect_btn = QPushButton("🔌 Connect IBKR")
        self.quick_connect_btn.setToolTip("Quick connect to IBKR")
        self.quick_connect_btn.clicked.connect(self.request_ibkr_connection)
        title_layout.addWidget(self.quick_connect_btn)
        
        layout.addLayout(title_layout)
        
        # Create scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # First row: Account metrics and Recent Activity
        first_row_layout = QHBoxLayout()
        
        # Account metrics (left side)
        self.create_account_metrics()
        first_row_layout.addWidget(self.account_frame, 2)  # 2/3 of the width
        
        # Recent activity (right side)
        self.create_recent_activity()
        first_row_layout.addWidget(self.activity_frame, 1)  # 1/3 of the width
        
        content_layout.addLayout(first_row_layout)
        
        # Live ticker (new)
        try:
            self.create_live_ticker()
            content_layout.addWidget(self.live_frame)
        except Exception as e:
            # Do not break dashboard if live ticker init fails
            self.logger.warning(f"Live ticker unavailable: {e}")

        # Portfolio summary (moved up)
        self.create_portfolio_summary()
        content_layout.addWidget(self.portfolio_frame)

        # Macro + Calendar side-by-side to save space
        mc_row = QHBoxLayout()
        self.macro_widget = MacroWidget()
        self.calendar_widget = CalendarWidget()
        mc_row.addWidget(self.macro_widget, 1)
        mc_row.addWidget(self.calendar_widget, 1)
        content_layout.addLayout(mc_row)

        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
    
    def create_live_ticker(self):
        """Create a lightweight live ticker section with optional sparkline."""
        self.live_frame = QFrame()
        self.live_frame.setFrameStyle(QFrame.Shape.Box)
        self.live_frame.setMinimumHeight(100)

        v = QVBoxLayout(self.live_frame)
        v.setContentsMargins(15, 10, 15, 10)
        v.setSpacing(8)

        title = QLabel("Live Ticker")
        tf = QFont()
        tf.setPointSize(12)
        tf.setBold(True)
        title.setFont(tf)
        v.addWidget(title)

        # Controls row
        ctl = QHBoxLayout()
        ctl.setSpacing(10)

        ctl.addWidget(QLabel("Symbol:"))
        self.live_symbol_edit = QLineEdit()
        default_symbol = "AAPL"
        try:
            ds = self.config.ui.default_symbols or []
            if len(ds) > 0:
                default_symbol = ds[0]
        except Exception:
            pass
        self.live_symbol_edit.setText(default_symbol)
        self.live_symbol_edit.setFixedWidth(100)
        ctl.addWidget(self.live_symbol_edit)

        self.live_status_label = QLabel("⏸️ Idle")
        ctl.addWidget(self.live_status_label)

        self.live_start_btn = QPushButton("▶ Start Live")
        self.live_start_btn.setFixedSize(110, 28)
        self.live_start_btn.clicked.connect(self.toggle_live_stream)
        ctl.addWidget(self.live_start_btn)

        ctl.addStretch()
        v.addLayout(ctl)

        # Last price row
        price_row = QHBoxLayout()
        price_row.addWidget(QLabel("Last Price:"))
        self.live_price_label = QLabel("-")
        pf = QFont()
        pf.setPointSize(14)
        pf.setBold(True)
        self.live_price_label.setFont(pf)
        price_row.addWidget(self.live_price_label)
        price_row.addStretch()
        v.addLayout(price_row)

        # Optional sparkline using pyqtgraph
        self._live_plot = None
        self._live_curve = None
        try:
            import pyqtgraph as pg  # type: ignore
            self._live_plot = pg.PlotWidget()
            self._live_plot.setBackground('k')
            self._live_plot.showGrid(x=False, y=True, alpha=0.2)
            self._live_plot.setMaximumHeight(120)
            self._live_curve = self._live_plot.plot(pen=pg.mkPen('#14a085', width=2))
            v.addWidget(self._live_plot)
        except Exception:
            # pyqtgraph not installed; sparkline omitted
            pass

        # Streaming state
        self._live_prices = []  # keep last N prices
        self._live_stream_thread = None
        self._live_worker = None
        self._live_stop_flag = None

    def create_account_metrics(self):
        """Create account metrics section"""
        self.account_frame = QFrame()
        self.account_frame.setFrameStyle(QFrame.Shape.Box)
        self.account_frame.setFixedHeight(150)  # Fixed height to match activity frame
        
        layout = QVBoxLayout(self.account_frame)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)  # Reduced spacing
        
        # Title
        title = QLabel("Account Overview")
        title_font = QFont()
        title_font.setPointSize(12)  # Smaller title to match activity
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Metrics grid
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)  # Reduced spacing between cards
        
        self.net_liquidation_card = MetricCard("Net Liquidation", "$0.00")
        self.buying_power_card = MetricCard("Buying Power", "$0.00")
        self.day_pnl_card = MetricCard("Day P&L", "$0.00")
        self.unrealized_pnl_card = MetricCard("Unrealized P&L", "$0.00")
        
        metrics_layout.addWidget(self.net_liquidation_card)
        metrics_layout.addWidget(self.buying_power_card)
        metrics_layout.addWidget(self.day_pnl_card)
        metrics_layout.addWidget(self.unrealized_pnl_card)
        metrics_layout.addStretch()
        
        layout.addLayout(metrics_layout)

    # -------- Live stream controls --------
    def toggle_live_stream(self):
        try:
            if getattr(self, '_live_stream_thread', None):
                self.stop_live_stream()
            else:
                self.start_live_stream()
        except Exception as e:
            self.logger.error(f"Live stream toggle error: {e}")

    def start_live_stream(self):
        if not (self.ibkr_service and self.ibkr_service.is_connected()):
            self.live_status_label.setText("❌ Not connected")
            return
        symbol = (self.live_symbol_edit.text() or "AAPL").strip().upper()
        if not symbol:
            return
        # Prepare stop flag and thread
        self._live_stop_flag = [False]
        from PyQt6.QtCore import QObject, pyqtSignal, QThread

        class _LiveWorker(QObject):
            tick = pyqtSignal(dict)
            done = pyqtSignal()
            def __init__(self, svc, sym, stop_flag):
                super().__init__()
                self._svc = svc
                self._sym = sym
                self._stop = stop_flag
            def run(self):
                try:
                    def on_tick(data):
                        try:
                            self.tick.emit(data)
                        except Exception:
                            pass
                    self._svc.stream_live_data(self._sym, 'STK', 'SMART', on_tick=on_tick, stop_flag=self._stop)
                finally:
                    self.done.emit()

        self._live_stream_thread = QThread(self)
        self._live_worker = _LiveWorker(self.ibkr_service, symbol, self._live_stop_flag)
        self._live_worker.moveToThread(self._live_stream_thread)
        self._live_stream_thread.started.connect(self._live_worker.run)
        self._live_worker.tick.connect(self.on_live_tick)
        self._live_worker.done.connect(self.on_live_done)
        self._live_worker.done.connect(self._live_stream_thread.quit)
        self._live_stream_thread.start()
        self.live_status_label.setText(f"🟢 Live: {symbol}")
        self.live_start_btn.setText("⏹ Stop Live")

    def stop_live_stream(self):
        try:
            if self._live_stop_flag is not None:
                self._live_stop_flag[0] = True
        except Exception:
            pass
        # Thread will cleanly exit via done signal
        self.live_status_label.setText("⏸️ Idle")
        self.live_start_btn.setText("▶ Start Live")
        self._live_stream_thread = None
        self._live_worker = None

    def on_live_tick(self, data: dict):
        try:
            price = float(data.get('Price')) if 'Price' in data else float(data.get('price', 'nan'))
            if price != price:  # NaN check
                return
            self.live_price_label.setText(f"{price:.4f}")
            # Update sparkline buffer
            self._live_prices.append(price)
            if len(self._live_prices) > 300:
                self._live_prices = self._live_prices[-300:]
            if self._live_curve is not None:
                try:
                    import numpy as np  # type: ignore
                    x = np.arange(len(self._live_prices))
                    y = np.array(self._live_prices, dtype=float)
                    self._live_curve.setData(x, y)
                except Exception:
                    # If numpy not available, setData with list indices
                    self._live_curve.setData(list(range(len(self._live_prices))), self._live_prices)
        except Exception as e:
            # keep UI resilient
            self.logger.debug(f"Live tick parse error: {e}")

    def on_live_done(self):
        self.live_status_label.setText("⏸️ Idle")
        self.live_start_btn.setText("▶ Start Live")
        self._live_stream_thread = None
        self._live_worker = None
    
    def create_portfolio_summary(self):
        """Create portfolio summary section"""
        self.portfolio_frame = QFrame()
        self.portfolio_frame.setFrameStyle(QFrame.Shape.Box)
        self.portfolio_frame.setMinimumHeight(250)  # Increased height for better visibility
        
        layout = QVBoxLayout(self.portfolio_frame)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)
        
        # Title with refresh button
        title_layout = QHBoxLayout()
        
        title = QLabel("Portfolio Positions")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # Sync button with better size
        sync_btn = QPushButton("🔄 Sync IBKR")
        sync_btn.setFixedSize(120, 30)
        sync_btn.setStyleSheet(
            """
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
            """
        )
        sync_btn.setToolTip("Sync with IBKR positions")
        sync_btn.clicked.connect(self.sync_portfolio_data)  # Specific function
        title_layout.addWidget(sync_btn)
        
        layout.addLayout(title_layout)
        
        # Create table for positions
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)  # Added Market Value column
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Shares", "Avg Cost", "Market Price", "Market Value", "P&L"
        ])
        
        # Set table properties with better height
        self.positions_table.setMinimumHeight(150)  # Increased from 120
        self.positions_table.setMaximumHeight(180)  # Max height to prevent too tall
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.positions_table.verticalHeader().setVisible(False)
        self.positions_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #4a4a4a;
                font-size: 11px;
                border: 1px solid #3d3d3d;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #3a3a3a;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #2f2f2f;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        
        # Auto-resize columns
        header = self.positions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Symbol
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Shares
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Avg Cost
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Market Price
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Market Value
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # P&L
        
        layout.addWidget(self.positions_table)
        
        # Status label with better styling
        self.positions_status = QLabel("Connect to IBKR to view positions")
        self.positions_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.positions_status.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.positions_status)
    
    def create_recent_activity(self):
        """Create recent activity section"""
        self.activity_frame = QFrame()
        self.activity_frame.setFrameStyle(QFrame.Shape.Box)
        self.activity_frame.setFixedHeight(150)  # Fixed height to match account metrics area
        
        layout = QVBoxLayout(self.activity_frame)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # Title with refresh button
        title_layout = QHBoxLayout()
        
        title = QLabel("Recent Activity")
        title_font = QFont()
        title_font.setPointSize(12)  # Smaller title
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # Add refresh button here
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedSize(30, 25)  # Small compact button
        refresh_btn.setToolTip("Refresh activity")
        refresh_btn.clicked.connect(self.refresh_all_data)  # Connect to a method we'll create
        title_layout.addWidget(refresh_btn)
        
        layout.addLayout(title_layout)
        
        # Activity list
        self.activity_label = QLabel("Loading...")
        self.activity_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.activity_label.setWordWrap(True)
        self.activity_label.setStyleSheet("font-size: 11px;")  # Smaller text
        layout.addWidget(self.activity_label)
    
    def update_recent_activity(self):
        """Update recent activity"""
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                # Get account summary for connection verification
                account_summary = self.ibkr_service.get_account_summary()
                positions = self.ibkr_service.get_positions()
                
                # Build activity text
                activity_text = "✅ Connected to IBKR\n"
                activity_text += f"📊 Account Summary: {len(account_summary)} fields\n"
                activity_text += f"💼 Positions: {len([p for p in positions if p.get('position', 0) != 0])} active\n"
                activity_text += f"🔄 Auto-refresh: {self.config.ui.update_interval//1000}s interval\n"
                activity_text += f"🕒 Last update: {self.get_current_time()}"
                
                self.activity_label.setText(activity_text)
                self.activity_label.setStyleSheet("color: #4CAF50;")  # Green for connected
            else:
                activity_text = "❌ Not connected to IBKR\n"
                activity_text += "Use Connection menu to connect\n"
                activity_text += "Ensure TWS is running\n"
                activity_text += f"🕒 Last check: {self.get_current_time()}"
                
                self.activity_label.setText(activity_text)
                self.activity_label.setStyleSheet("color: #f44336;")  # Red for disconnected
                
        except Exception as e:
            self.logger.error(f"Error updating recent activity: {e}")
            self.activity_label.setText(f"❌ Error: {str(e)[:50]}...")
            self.activity_label.setStyleSheet("color: #f44336;")  # Red for error
    
    def get_current_time(self):
        """Get current time formatted"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def update_data(self):
        """Update dashboard data"""
        try:
            # Update connection indicator and button visibility
            if self.ibkr_service and self.ibkr_service.is_connected():
                self.connection_indicator.setText("🟢 Live")
                self.connection_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.quick_connect_btn.hide()  # Hide connect button when connected
            else:
                self.connection_indicator.setText("🔴 Offline")
                self.connection_indicator.setStyleSheet("color: #f44336; font-weight: bold;")
                self.quick_connect_btn.show()  # Show connect button when disconnected
            
            # Update all data sections
            self.update_account_metrics()
            self.update_portfolio_summary()
            self.update_recent_activity()
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
            self.connection_indicator.setText("⚠️ Error")
            self.connection_indicator.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.quick_connect_btn.show()  # Show connect button on error
    
    def request_ibkr_connection(self):
        """Request IBKR connection from main window"""
        self.logger.info("Dashboard requested IBKR connection")
        self.connection_requested.emit()
    
    def update_account_metrics(self):
        """Update account metrics with real data"""
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                # Get real account data from IBKR
                account_summary = self.ibkr_service.get_account_summary()
                
                # Update Net Liquidation Value
                net_liq = account_summary.get('NetLiquidation', {})
                if net_liq:
                    net_liq_value = f"${float(net_liq.get('value', 0)):,.2f}"
                    currency = net_liq.get('currency', 'USD')
                    self.net_liquidation_card.update_value(net_liq_value, currency)
                
                # Update Buying Power
                buying_power = account_summary.get('BuyingPower', {})
                if buying_power:
                    bp_value = f"${float(buying_power.get('value', 0)):,.2f}"
                    currency = buying_power.get('currency', 'USD')
                    self.buying_power_card.update_value(bp_value, currency)
                else:
                    # Try alternative field names
                    excess_liquidity = account_summary.get('ExcessLiquidity', {})
                    if excess_liquidity:
                        bp_value = f"${float(excess_liquidity.get('value', 0)):,.2f}"
                        currency = excess_liquidity.get('currency', 'USD')
                        self.buying_power_card.update_value(bp_value, f"{currency} (Excess)")
                
                # Update Day P&L
                day_pnl = account_summary.get('DayTradesRemaining', {})
                realized_pnl = account_summary.get('RealizedPnL', {})
                if realized_pnl:
                    pnl_value = float(realized_pnl.get('value', 0))
                    pnl_formatted = f"${abs(pnl_value):,.2f}"
                    if pnl_value >= 0:
                        pnl_formatted = f"+{pnl_formatted}"
                    else:
                        pnl_formatted = f"-{pnl_formatted}"
                    self.day_pnl_card.update_value(pnl_formatted, "Realized")
                
                # Update Unrealized P&L
                unrealized_pnl = account_summary.get('UnrealizedPnL', {})
                if unrealized_pnl:
                    pnl_value = float(unrealized_pnl.get('value', 0))
                    pnl_formatted = f"${abs(pnl_value):,.2f}"
                    if pnl_value >= 0:
                        pnl_formatted = f"+{pnl_formatted}"
                    else:
                        pnl_formatted = f"-{pnl_formatted}"
                    self.unrealized_pnl_card.update_value(pnl_formatted, "Unrealized")
                
                self.logger.debug(f"Updated account metrics with {len(account_summary)} IBKR values")
                
            else:
                # Show disconnected state
                self.net_liquidation_card.update_value("Connect to IBKR", "to view data")
                self.buying_power_card.update_value("Connect to IBKR", "to view data")
                self.day_pnl_card.update_value("Connect to IBKR", "to view data")
                self.unrealized_pnl_card.update_value("Connect to IBKR", "to view data")
                
        except Exception as e:
            self.logger.error(f"Error updating account metrics: {e}")
            # Show error state
            self.net_liquidation_card.update_value("Error", "Check connection")
            self.buying_power_card.update_value("Error", "Check connection")
            self.day_pnl_card.update_value("Error", "Check connection")
            self.unrealized_pnl_card.update_value("Error", "Check connection")
    
    def update_portfolio_summary(self):
        """Update portfolio summary with real IBKR positions"""
        try:
            self.logger.debug("Starting portfolio summary update")
            
            if not self.ibkr_service:
                # No service available - update UI but avoid logging repeatedly on every timer tick
                self.positions_table.setRowCount(0)
                self.positions_status.setText("❌ IBKR service not available - Connect to view real positions")
                self.positions_status.setStyleSheet("color: #f44336; font-weight: bold;")
                return

            if not self.ibkr_service.is_connected():
                # Service exists but not connected - show disconnected UI state without spamming logs
                self.positions_table.setRowCount(0)
                self.positions_status.setText("❌ Connect to IBKR to view real positions")
                self.positions_status.setStyleSheet("color: #f44336; font-weight: bold;")
                return
            
            # Get real portfolio data from IBKR
            self.logger.info("Fetching portfolio from IBKR...")
            positions = self.ibkr_service.get_portfolio()  # Changed from get_positions to get_portfolio
            self.logger.info(f"Received {len(positions)} total portfolio items from IBKR")
            
            # Filter out zero positions
            active_positions = [pos for pos in positions if abs(pos.get('position', 0)) > 0]
            self.logger.info(f"Found {len(active_positions)} active positions")
            
            # Update table
            self.positions_table.setRowCount(len(active_positions))
            
            if active_positions:
                self.positions_status.setText(f"📊 {len(active_positions)} active positions • Last updated: {self.get_current_time()}")
                self.positions_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                for row, pos in enumerate(active_positions):
                    self.logger.debug(f"Processing position {row}: {pos.get('symbol', 'N/A')}")
                    
                    # Symbol
                    symbol_item = QTableWidgetItem(str(pos.get('symbol', 'N/A')))
                    symbol_item.setFlags(symbol_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    self.positions_table.setItem(row, 0, symbol_item)
                    
                    # Shares
                    position_value = pos.get('position', 0)
                    shares_item = QTableWidgetItem(f"{position_value:,.0f}")
                    shares_item.setFlags(shares_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    # Color long/short positions differently
                    if position_value > 0:
                        shares_item.setData(Qt.ItemDataRole.ForegroundRole, "#2e7d32")  # Green for long
                    else:
                        shares_item.setData(Qt.ItemDataRole.ForegroundRole, "#d32f2f")  # Red for short
                    self.positions_table.setItem(row, 1, shares_item)
                    
                    # Average Cost
                    avg_cost = pos.get('average_cost', 0)
                    avg_cost_item = QTableWidgetItem(f"${avg_cost:.2f}")
                    avg_cost_item.setFlags(avg_cost_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    self.positions_table.setItem(row, 2, avg_cost_item)
                    
                    # Market Price
                    market_price = pos.get('market_price', 0)
                    market_price_item = QTableWidgetItem(f"${market_price:.2f}")
                    market_price_item.setFlags(market_price_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    self.positions_table.setItem(row, 3, market_price_item)
                    
                    # Market Value
                    market_value = pos.get('market_value', 0)
                    market_value_item = QTableWidgetItem(f"${market_value:,.2f}")
                    market_value_item.setFlags(market_value_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    self.positions_table.setItem(row, 4, market_value_item)
                    
                    # P&L
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                    pnl_text = f"${abs(unrealized_pnl):,.2f}"
                    if unrealized_pnl > 0:
                        pnl_text = f"+{pnl_text}"
                    elif unrealized_pnl < 0:
                        pnl_text = f"-{pnl_text}"
                    
                    pnl_item = QTableWidgetItem(pnl_text)
                    pnl_item.setFlags(pnl_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    
                    # Color code P&L
                    if unrealized_pnl > 0:
                        pnl_item.setData(Qt.ItemDataRole.ForegroundRole, "#2e7d32")  # Green
                    elif unrealized_pnl < 0:
                        pnl_item.setData(Qt.ItemDataRole.ForegroundRole, "#d32f2f")  # Red
                    
                    self.positions_table.setItem(row, 5, pnl_item)
                
                self.logger.info(f"Successfully updated portfolio table with {len(active_positions)} positions")
                
            else:
                self.positions_status.setText("📭 No active positions found • Connected to IBKR")
                self.positions_status.setStyleSheet("color: #666; font-style: italic;")
                self.logger.info("No active positions found in IBKR account")
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio summary: {e}")
            # Show error state
            self.positions_table.setRowCount(0)
            self.positions_status.setText(f"⚠️ Error loading positions: {str(e)[:40]}...")
            self.positions_status.setStyleSheet("color: #ff9800; font-weight: bold;")
    
    def sync_portfolio_data(self):
        """Specific function to sync portfolio data"""
        self.logger.info("Manual portfolio sync requested")
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                self.positions_status.setText("🔄 Syncing with IBKR...")
                self.positions_status.setStyleSheet("color: #2196F3; font-weight: bold;")
                
                # Force update of portfolio data
                self.update_portfolio_summary()
                
                self.logger.info("Portfolio sync completed")
            else:
                self.positions_status.setText("❌ Not connected to IBKR")
                self.positions_status.setStyleSheet("color: #f44336; font-weight: bold;")
                
        except Exception as e:
            self.logger.error(f"Error during portfolio sync: {e}")
            self.positions_status.setText(f"⚠️ Sync error: {str(e)[:30]}...")
            self.positions_status.setStyleSheet("color: #ff9800; font-weight: bold;")
    
    def refresh_all_data(self):
        """Refresh all data - called from activity area refresh button"""
        self.refresh_data()
    
    def set_ibkr_service(self, ibkr_service):
        """Set the IBKR service reference"""
        self.ibkr_service = ibkr_service
        self.logger.info("IBKR service set for dashboard")
        # Update data immediately
        self.update_data()
        # Enable/disable live controls
        try:
            connected = bool(self.ibkr_service and self.ibkr_service.is_connected())
            self.live_start_btn.setEnabled(connected)
        except Exception:
            pass
    
    def refresh_data(self):
        """Manually refresh all data"""
        self.logger.info("Manual refresh requested")
        self.refresh_requested.emit()

    def on_auto_refresh_toggled(self, state):
        self.auto_refresh_enabled = state == Qt.CheckState.Checked
        if self.auto_refresh_enabled:
            self.update_timer.start(self.current_interval_ms)
            self.logger.info(f"Auto-refresh enabled at {self.current_interval_ms//1000}s")
        else:
            self.update_timer.stop()
            self.logger.info("Auto-refresh disabled")

    def on_interval_changed(self, seconds: int):
        self.current_interval_ms = max(1, int(seconds)) * 1000
        if self.auto_refresh_enabled:
            self.update_timer.start(self.current_interval_ms)
        self.logger.info(f"Auto-refresh interval set to {seconds}s")
        # Optional: trigger an immediate refresh so the user sees the new cadence reflected
        self.update_data()
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Dashboard tab activated")
        self.update_data()