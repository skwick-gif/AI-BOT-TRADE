"""
Main window for the AI Trading Bot PyQt6 application
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QStatusBar, QMenuBar, QToolBar,
                             QLabel, QPushButton, QSplitter, QDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import os
from PyQt6.QtGui import QAction, QFont
import qtawesome as qta

from .tabs.dashboard_tab import DashboardTab
from .tabs.trading_tab import TradingTab
from .tabs.macro_tab import MacroTab
from .tabs.portfolio_tab import PortfolioTab
from .tabs.chat_tab import ChatTab
from .dialogs.api_keys_dialog import APIKeysDialog
from core.config import validate_api_keys

class MainWindow(QMainWindow):
    """Main application window"""
    
    # Signals
    status_message = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("AI Trading Bot - PyQt6")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set window icon
        self.setWindowIcon(qta.icon('fa5s.chart-line', color='#00ff88'))
        
        # Initialize UI
        self.init_ui()
        
        # Setup status updates
        self.setup_status_timer()
        
        # Check API keys
        self.check_api_keys()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create tab widget
        self.create_tabs()
        main_layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.create_status_bar()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Settings action
        settings_action = QAction(qta.icon('fa5s.cog'), '&Settings', self)
        settings_action.setShortcut('Ctrl+S')
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction(qta.icon('fa5s.times'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Trading menu
        trading_menu = menubar.addMenu('&Trading')
        
        # Connect to IBKR
        connect_action = QAction(qta.icon('fa5s.link'), '&Connect to IBKR', self)
        # Use a wrapper to mark this as a user-initiated (forced) connect
        connect_action.triggered.connect(lambda checked=False: self.connect_ibkr(force=True))
        trading_menu.addAction(connect_action)
        
        # Disconnect from IBKR
        disconnect_action = QAction(qta.icon('fa5s.unlink'), '&Disconnect from IBKR', self)
        disconnect_action.triggered.connect(self.disconnect_ibkr)
        trading_menu.addAction(disconnect_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About action
        about_action = QAction(qta.icon('fa5s.info'), '&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Connection status
        self.connection_status = QLabel("🔴 Disconnected")
        self.connection_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        toolbar.addWidget(self.connection_status)
        
        toolbar.addSeparator()
        
        # Quick connect button
        connect_btn = QPushButton("Connect IBKR")
        connect_btn.setIcon(qta.icon('fa5s.link'))
        connect_btn.clicked.connect(lambda: self.connect_ibkr(force=True))
        toolbar.addWidget(connect_btn)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(qta.icon('fa5s.sync'))
        refresh_btn.clicked.connect(self.refresh_data)
        toolbar.addWidget(refresh_btn)
    
    def create_tabs(self):
        """Create the tab widget with all tabs"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self.dashboard_tab = DashboardTab(self.config)
        self.trading_tab = TradingTab(self.config)
        self.portfolio_tab = PortfolioTab(self.config)
        self.macro_tab = MacroTab(self.config)
        self.chat_tab = ChatTab(self.config)
        
        # Add tabs
        self.tab_widget.addTab(self.dashboard_tab, 
                              qta.icon('fa5s.tachometer-alt'), "Dashboard")
        self.tab_widget.addTab(self.trading_tab, 
                              qta.icon('fa5s.chart-line'), "Trading")
        self.tab_widget.addTab(self.portfolio_tab, 
                              qta.icon('fa5s.briefcase'), "Portfolio")
        self.tab_widget.addTab(self.macro_tab, 
                              qta.icon('fa5s.globe'), "Macro")
        self.tab_widget.addTab(self.chat_tab, 
                              qta.icon('fa5s.comments'), "AI Chat")
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)
        
        # Connect status message signal
        self.status_message.connect(self.status_bar.showMessage)
        
        self.status_bar.showMessage("Ready")
    
    def setup_status_timer(self):
        """Setup timer for status updates"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)  # Update every second
    
    def update_status(self):
        """Update status information"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(f"Time: {current_time}")
    
    def check_api_keys(self):
        """Check if API keys are configured"""
        missing_keys = validate_api_keys(self.config)
        if missing_keys:
            self.status_message.emit(f"Missing API keys: {', '.join(missing_keys)}")
        else:
            self.status_message.emit("All API keys configured")
    
    def connect_ibkr(self, force: bool = False):
        """Connect to IBKR. Must be forced (user action) or enabled via AUTO_CONNECT_IBKR env var."""
        # Disallow programmatic auto-connect unless forced or env var enabled
        auto_flag = os.getenv('AUTO_CONNECT_IBKR', '0').lower()
        if not force and auto_flag not in ('1', 'true', 'yes'):
            # Notify user via status message but do not change UI state
            try:
                self.status_message.emit("Auto-connect to IBKR is disabled. Use Connect IBKR to connect.")
            except Exception:
                pass
            return

        # This will be implemented with the IBKR connection logic
        self.status_message.emit("Connecting to IBKR...")
        self.connection_status.setText("🟡 Connecting...")
        self.connection_status.setStyleSheet("color: #ffaa00; font-weight: bold;")

        # Simulate connection (replace with actual IBKR connection)
        QTimer.singleShot(2000, self.on_ibkr_connected)
    
    def on_ibkr_connected(self):
        """Handle IBKR connection success"""
        self.connection_status.setText("🟢 Connected")
        self.connection_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.status_message.emit("IBKR connected successfully")
    
    def disconnect_ibkr(self):
        """Disconnect from IBKR"""
        self.connection_status.setText("🔴 Disconnected")
        self.connection_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.status_message.emit("IBKR disconnected")
    
    def refresh_data(self):
        """Refresh all data"""
        self.status_message.emit("Refreshing data...")
        # Trigger refresh on all tabs
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'refresh_data'):
                tab.refresh_data()
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = APIKeysDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.status_message.emit("API keys updated successfully - Reloading configuration...")
            # Reload environment variables without restart
            import os
            from pathlib import Path
            
            # Reload .env file
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            value = value.strip('"\'')
                            os.environ[key] = value
            
            # Update chat tab with new API key if needed
            chat_tab = None
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if hasattr(tab, 'perplexity_api_key'):
                    tab.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
                    chat_tab = tab
                    break
            
            # Refresh tabs to use new keys
            self.refresh_data()
            
            if chat_tab and os.getenv("PERPLEXITY_API_KEY"):
                self.status_message.emit("✅ Configuration updated - AI Chat ready!")
            else:
                self.status_message.emit("⚠️ Configuration updated - Some features may need API keys")
        else:
            self.status_message.emit("Settings cancelled")
    
    def show_about(self):
        """Show about dialog"""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(self, "About AI Trading Bot", 
                         "AI Trading Bot v2.0\\nPyQt6 Version\\n\\nProfessional trading interface with AI assistance")