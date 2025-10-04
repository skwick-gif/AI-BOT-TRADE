"""
Main Application Window
Based on the provided PyQt6 examples with enhanced functionality
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QMenuBar, QMessageBox,
    QToolBar, QLabel, QPushButton, QDialog, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QAction, QIcon, QFont
import asyncio

from ui.widgets.dashboard_widget import DashboardWidget
from ui.widgets.ai_trading_widget import AiTradingWidget
from ui.widgets.chat_widget import ChatWidget
from ui.widgets.ml_widget import MLWidget
from ui.widgets.watchlist_widget import WatchlistWidget
from ui.widgets.scanner_widget import ScannerWidget
from ui.dialogs.api_keys_dialog import APIKeysDialog
from ui.themes.theme_manager import ThemeManager
from core.config_manager import ConfigManager
from services.ibkr_service import IBKRService
from services.ai_service import AIService
from utils.logger import get_logger
from pathlib import Path


class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""
    
    # Signals
    connection_status_changed = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger
        self.logger = get_logger("MainWindow")
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # Initialize services
        self.ibkr_service = None
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Setup UI
        self.setup_ui()
        
        self.logger.info("Main window initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("AI Trading Bot")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_tabs()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.create_status_bar()
        
        # Setup connections after creating tabs
        self.setup_connections()
        
        # Apply theme
        self.apply_theme()
        
        # Setup status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(self.config.ui.update_interval)
        
    # Auto-connect disabled by user request
    
    def create_tabs(self):
        """Create all tabs"""
        
        # Dashboard Tab
        self.dashboard_widget = DashboardWidget(None)  # Start with None, will be set when connected
        self.tab_widget.addTab(self.dashboard_widget, "📊 Dashboard")
        # AI Trading Tab (replaces Portfolio)
        self.ai_trading_widget = AiTradingWidget()
        # Provide AI service instance for scoring
        try:
            self.ai_trading_widget.set_ai_service(AIService(self.config))
        except Exception as e:
            self.logger.warning(f"Failed to initialize AIService: {e}")
        self.tab_widget.addTab(self.ai_trading_widget, "⚡ AI Trading")
        
        # Chat Tab (AI Agent)
        self.chat_widget = ChatWidget()
        self.tab_widget.addTab(self.chat_widget, "🤖 AI Agent")
        
        # ML Tab
        self.ml_widget = MLWidget()
        self.tab_widget.addTab(self.ml_widget, "🧠 ML Training")
        
        # Watchlist Tab
        self.watchlist_widget = WatchlistWidget()
        self.tab_widget.addTab(self.watchlist_widget, "👁️ Watchlist")
        
        # Scanner Tab
        self.scanner_widget = ScannerWidget()
        self.tab_widget.addTab(self.scanner_widget, "🔍 Scanner")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Connection Menu
        connection_menu = menubar.addMenu("Connection")
        
        self.connect_action = QAction("Connect to IBKR", self)
        self.connect_action.triggered.connect(self.toggle_ibkr_connection)
        connection_menu.addAction(self.connect_action)
        
        connection_menu.addSeparator()
        
        api_keys_action = QAction("Configure API Keys", self)
        api_keys_action.triggered.connect(self.show_api_keys_dialog)
        connection_menu.addAction(api_keys_action)
        
        # View Menu
        view_menu = menubar.addMenu("View")
        
        self.theme_action = QAction("Toggle Theme", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        # Help Menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # AI Chat Guide (opens Markdown guide in a dialog)
        ai_guide_action = QAction("AI Chat Guide", self)
        ai_guide_action.triggered.connect(self.show_ai_chat_guide)
        help_menu.addAction(ai_guide_action)
    
    def create_toolbar(self):
        """Create toolbar"""
        # Note: Toolbar buttons removed per user request
        # All functionality is available through the menu bar
        pass
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connection status
        self.connection_status_label = QLabel("Disconnected")
        self.status_bar.addWidget(self.connection_status_label)
        
        # Time label
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)
        
        # Update time
        self.update_time()
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Connect connection status signal
        self.connection_status_changed.connect(self.on_connection_status_changed)
        
        # Connect dashboard signals
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.connection_requested.connect(self.toggle_ibkr_connection)
        # Connect AI Trading signals
        if hasattr(self, 'ai_trading_widget'):
            self.ai_trading_widget.connect_ibkr_requested.connect(self.toggle_ibkr_connection)
    
    def apply_theme(self):
        """Apply current theme"""
        self.theme_manager.apply_theme(self, self.config.ui.theme)
    
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        current_theme = self.config.ui.theme
        new_theme = "light" if current_theme == "dark" else "dark"
        self.config.ui.theme = new_theme
        self.apply_theme()
        self.logger.info(f"Theme changed to: {new_theme}")
    
    def toggle_ibkr_connection(self):
        """Toggle IBKR connection"""
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                self.disconnect_ibkr()
            else:
                self.connect_ibkr()
        except Exception as e:
            self.logger.error(f"Error toggling IBKR connection: {e}")
            QMessageBox.critical(self, "Connection Error", str(e))
    
    def auto_connect_ibkr(self):
        """Auto-connect is disabled"""
        self.logger.info("Auto-connect is disabled")

    def connect_ibkr(self):
        """Connect to IBKR on the main thread (ib_insync prefers main Qt loop)."""
        try:
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(False)
                self.connect_action.setText("Connecting…")

            # Ensure service exists (create on main thread)
            if not self.ibkr_service:
                from services.ibkr_service import IBKRService
                self.ibkr_service = IBKRService(self.config.ibkr)

            success = self.ibkr_service.connect()
            if success:
                # Wire services into widgets
                if hasattr(self, 'dashboard_widget'):
                    self.dashboard_widget.set_ibkr_service(self.ibkr_service)
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_status(True)
                self.connection_status_changed.emit(True)
                self.logger.info("Connected to IBKR successfully")
                if hasattr(self, 'connect_action'):
                    self.connect_action.setText("Disconnect from IBKR")
            else:
                detail = "Failed to connect to IBKR"
                try:
                    if self.ibkr_service and getattr(self.ibkr_service, 'last_error', None):
                        detail = self.ibkr_service.last_error
                except Exception:
                    pass
                self.logger.error(f"IBKR connection failed: {detail}")
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to IBKR.\n{detail}")
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_status(False)
        except Exception as e:
            self.logger.error(f"IBKR connection error: {e}")
            QMessageBox.critical(self, "Connection Error", f"IBKR connect error:\n{e}")
        finally:
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(True)

    def disconnect_ibkr(self):
        """Disconnect from IBKR"""
        try:
            if self.ibkr_service:
                self.ibkr_service.disconnect()
                self.connection_status_changed.emit(False)
                self.logger.info("Disconnected from IBKR")
                # Clear service in widgets
                if hasattr(self, 'dashboard_widget'):
                    self.dashboard_widget.set_ibkr_service(None)
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_status(False)
        except Exception as e:
            self.logger.error(f"Error disconnecting from IBKR: {e}")
    
    def on_connection_status_changed(self, connected: bool):
        """Handle connection status change"""
        if connected:
            self.connection_status_label.setText("🟢 Connected to IBKR")
            self.connect_action.setText("Disconnect from IBKR")
            # Update dashboard when connected
            if hasattr(self, 'dashboard_widget'):
                self.dashboard_widget.set_ibkr_service(self.ibkr_service)
            if hasattr(self, 'ai_trading_widget'):
                self.ai_trading_widget.set_ibkr_status(True)
        else:
            self.connection_status_label.setText("🔴 Disconnected")
            self.connect_action.setText("Connect to IBKR")
            # Update dashboard when disconnected
            if hasattr(self, 'dashboard_widget'):
                self.dashboard_widget.set_ibkr_service(None)
            if hasattr(self, 'ai_trading_widget'):
                self.ai_trading_widget.set_ibkr_status(False)
    
    def on_tab_changed(self, index):
        """Handle tab change"""
        tab_name = self.tab_widget.tabText(index)
        self.logger.debug(f"Switched to tab: {tab_name}")
        
        # Update specific widgets when they become active
        current_widget = self.tab_widget.currentWidget()
        if hasattr(current_widget, 'on_tab_activated'):
            current_widget.on_tab_activated()
    
    def update_status(self):
        """Update status information"""
        self.update_time()
        
        # Update connection status if service exists
        if self.ibkr_service:
            is_connected = self.ibkr_service.is_connected()
            current_status = "Connected" in self.connection_status_label.text()
            if is_connected != current_status:
                self.connection_status_changed.emit(is_connected)
    
    def update_time(self):
        """Update time display"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About AI Trading Bot",
            "AI Trading Bot v2.0\\n\\n"
            "A professional trading application with AI integration\\n"
            "Built with PyQt6 and Interactive Brokers API\\n\\n"
            "Features:\\n"
            "• Real-time market data\\n"
            "• AI-powered trading assistant\\n"
            "• Machine learning models\\n"
            "• Advanced portfolio management\\n"
            "• Market scanning and watchlists"
        )

    def show_ai_chat_guide(self):
        """Open the AI Chat guide (Markdown) in a right-to-left styled dialog"""
        try:
            # Resolve docs path relative to project root
            project_root = Path(__file__).resolve().parents[3]
            guide_path = project_root / "docs" / "help" / "ai_chat_guide.md"
            if not guide_path.exists():
                QMessageBox.warning(self, "Guide Not Found", f"Could not find guide at:\n{guide_path}")
                return

            text = guide_path.read_text(encoding="utf-8")

            # Try to render Markdown to HTML if 'markdown' is available
            html = None
            try:
                import markdown  # type: ignore
                html_body = markdown.markdown(text, extensions=["extra", "sane_lists", "smarty", "tables", "fenced_code"])
                html = f"""
                <html>
                <head>
                    <meta charset='utf-8'>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; direction: rtl; text-align: right; }}
                        h1, h2, h3 {{ color: #2e7d32; }}
                        code, pre {{ background: #1f1f1f; color: #e0e0e0; padding: 4px 6px; border-radius: 4px; }}
                        blockquote {{ border-right: 4px solid #4CAF50; margin: 0; padding-right: 10px; color: #666; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #444; padding: 6px; }}
                    </style>
                </head>
                <body>{html_body}</body>
                </html>
                """
            except Exception:
                pass

            dlg = QDialog(self)
            dlg.setWindowTitle("AI Chat Guide")
            dlg.resize(820, 600)
            dlg.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

            view = QTextBrowser(dlg)
            view.setOpenExternalLinks(True)
            view.setStyleSheet("QTextBrowser { background: #111; color: #ddd; padding: 12px; }")

            if html:
                view.setHtml(html)
            else:
                # Fallback to plain text (will still be RTL layout)
                view.setPlainText(text)

            layout = QVBoxLayout(dlg)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(view)
            dlg.setLayout(layout)
            dlg.exec()

        except Exception as e:
            self.logger.error(f"Error opening AI Chat guide: {e}")
            QMessageBox.critical(self, "Guide Error", f"Failed to open guide:\n{e}")
    
    def show_api_keys_dialog(self):
        """Show API keys configuration dialog"""
        try:
            dialog = APIKeysDialog(self)
            result = dialog.exec()
            
            if result == dialog.DialogCode.Accepted:
                # Reload configuration after API keys are updated
                self.config = ConfigManager()
                self.logger.info("API keys configuration updated")
                QMessageBox.information(
                    self,
                    "Configuration Updated",
                    "API keys have been updated successfully.\\n"
                    "Some changes may require restarting the application."
                )
        except Exception as e:
            self.logger.error(f"Error showing API keys dialog: {e}")
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to open API keys configuration:\\n{e}"
            )
    
    def closeEvent(self, event):
        """Handle application close"""
        try:
            # Disconnect from IBKR if connected
            if self.ibkr_service and self.ibkr_service.is_connected():
                self.disconnect_ibkr()
            
            # Stop timers
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            # Ensure background thread is stopped
            if hasattr(self, '_ibkr_thread') and self._ibkr_thread.isRunning():
                self._ibkr_thread.quit()
                self._ibkr_thread.wait()
            
            self.logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            self.logger.error(f"Error during application close: {e}")
            event.accept()

# Worker class kept minimal and at module scope
class IBKRConnectWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self, service_factory):
        super().__init__()
        self._service_factory = service_factory
        self.service = None

    def run(self):
        try:
            # Ensure this QThread has an asyncio event loop for ib_insync
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if not self.service:
                self.service = self._service_factory()
            success = self.service.connect()
            if success:
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, "Failed to connect to IBKR")
        except Exception as e:
            self.finished.emit(False, str(e))