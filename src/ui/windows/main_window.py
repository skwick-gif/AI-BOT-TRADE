"""
Main Application Window
Based on the provided PyQt6 examples with enhanced functionality
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QMenuBar, QMessageBox,
    QToolBar, QLabel, QPushButton, QDialog, QTextBrowser, QTextEdit
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
from ui.widgets.options_widget import OptionsWidget
from ui.dialogs.api_keys_dialog import APIKeysDialog
from ui.dialogs.data_update_dialog import DataUpdateDialog
from ui.themes.theme_manager import ThemeManager
from core.config_manager import ConfigManager
from services.ibkr_service import IBKRService
from services.data_update_service import DataUpdateService
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
        # Make all tabs a uniform, reduced width so tab rectangles are shorter
        # Adjust TAB_WIDTH to taste (in pixels)
        TAB_WIDTH = 110
        self.tab_widget.setStyleSheet(
            f"QTabBar::tab {{ min-width: {TAB_WIDTH}px; max-width: {TAB_WIDTH}px; }}"
        )
        main_layout.addWidget(self.tab_widget)
        
        # Start daily data update scheduler (create service first so tabs can reuse it)
        try:
            self.data_update_service = DataUpdateService()
            # default schedule 01:30 local; dialog can change
            from datetime import time as dtime
            self.data_update_service.set_scheduled_time(dtime(hour=1, minute=30))
            self.data_update_service.start()
        except Exception as e:
            self.logger.warning(f"Failed to start DataUpdateService: {e}")

        # Create tabs (pass services where needed)
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
        
        # (DataUpdateService already created above before tabs)

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
        # DATA Tab using dedicated DataWidget
        from ui.widgets.data_widget import DataWidget
        # Pass the centralized data_update_service into the DataWidget
        self.data_tab = DataWidget(service=getattr(self, 'data_update_service', None))
        self.tab_widget.addTab(self.data_tab, "DATA")
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
        
        # OPTIONS Tab (with MAIN and BANK sub-tabs)
        self.options_widget = OptionsWidget()
        self.tab_widget.addTab(self.options_widget, "OPTIONS")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")

        edit_prompts_action = QAction("Edit Prompts...", self)
        edit_prompts_action.triggered.connect(self.show_prompts_editor)
        file_menu.addAction(edit_prompts_action)
        file_menu.addSeparator()
        
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
        about_action.triggered.connect(self.show_dashboard_about)
        help_menu.addAction(about_action)

        # AI Chat Guide (opens Markdown guide in a dialog)
        ai_guide_action = QAction("AI Chat Guide", self)
        ai_guide_action.triggered.connect(self.show_ai_chat_guide)
        help_menu.addAction(ai_guide_action)

        # AI Trading Guide
        ai_trading_action = QAction("AI Trading Guide", self)
        ai_trading_action.triggered.connect(self.show_ai_trading_guide)
        help_menu.addAction(ai_trading_action)

        # Watchlist Guide
        watchlist_guide_action = QAction("Watchlist Guide", self)
        watchlist_guide_action.triggered.connect(self.show_watchlist_guide)
        help_menu.addAction(watchlist_guide_action)

        # ML Training & Scanner Guide
        ml_scanner_guide_action = QAction("ML Training & Scanner Guide", self)
        ml_scanner_guide_action.triggered.connect(self.show_ml_scanner_guide)
        help_menu.addAction(ml_scanner_guide_action)

        # Data menu for Daily Update dialog
        data_menu = menubar.addMenu("Data")
        daily_update_action = QAction("Daily Update…", self)
        daily_update_action.triggered.connect(self.show_daily_update_dialog)
        data_menu.addAction(daily_update_action)

    def show_prompts_editor(self):
        """Open a minimal dialog to view/edit/save config/prompts.json"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            prompts_path = project_root / "config" / "prompts.json"

            # Ensure parent exists
            if not prompts_path.parent.exists():
                prompts_path.parent.mkdir(parents=True, exist_ok=True)

            if prompts_path.exists():
                try:
                    text = prompts_path.read_text(encoding="utf-8")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to read prompts file:\n{e}")
                    return
            else:
                # Start with an empty JSON object
                text = "{}"

            dlg = QDialog(self)
            dlg.setWindowTitle("Edit Prompts (config/prompts.json)")
            dlg.resize(900, 600)

            layout = QVBoxLayout(dlg)

            editor = QTextEdit(dlg)
            editor.setPlainText(text)
            font = editor.font()
            font.setFamily("Consolas")
            font.setPointSize(10)
            editor.setFont(font)
            layout.addWidget(editor)

            btn_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            reload_btn = QPushButton("Reload")
            close_btn = QPushButton("Close")
            btn_layout.addWidget(save_btn)
            btn_layout.addWidget(reload_btn)
            btn_layout.addStretch()
            btn_layout.addWidget(close_btn)
            layout.addLayout(btn_layout)

            def do_save():
                raw = editor.toPlainText()
                # Validate JSON
                try:
                    import json as _json
                    _json.loads(raw)
                except Exception as e:
                    QMessageBox.warning(dlg, "Invalid JSON", f"Cannot save: JSON is invalid:\n{e}")
                    return
                try:
                    prompts_path.write_text(raw, encoding="utf-8")
                    QMessageBox.information(dlg, "Saved", "prompts.json saved successfully.")
                except Exception as e:
                    QMessageBox.critical(dlg, "Save Error", f"Failed to save prompts file:\n{e}")

            def do_reload():
                try:
                    if prompts_path.exists():
                        editor.setPlainText(prompts_path.read_text(encoding="utf-8"))
                    else:
                        editor.setPlainText("{}")
                except Exception as e:
                    QMessageBox.critical(dlg, "Reload Error", f"Failed to reload prompts file:\n{e}")

            save_btn.clicked.connect(do_save)
            reload_btn.clicked.connect(do_reload)
            close_btn.clicked.connect(dlg.accept)

            dlg.exec()

        except Exception as e:
            self.logger.error(f"Error opening Prompts editor: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open Prompts editor:\n{e}")
    
    def create_toolbar(self):
        """Create toolbar"""
        # Note: Toolbar buttons removed per user request
        # All functionality is available through the menu bar
        pass

    def closeEvent(self, event):
        """Ensure background services and threads are stopped on close"""
        try:
            # Stop data update scheduler if running
            if hasattr(self, 'data_update_service') and self.data_update_service:
                try:
                    self.data_update_service.stop()
                except Exception:
                    pass
            # Cleanup watchlist widget threads/timers
            if hasattr(self, 'watchlist_widget') and self.watchlist_widget:
                try:
                    self.watchlist_widget.cleanup()
                except Exception:
                    pass
            # Optionally cleanup scanner if it has threads
            if hasattr(self, 'scanner_widget') and hasattr(self.scanner_widget, 'cleanup'):
                try:
                    self.scanner_widget.cleanup()
                except Exception:
                    pass
        finally:
            super().closeEvent(event)
    
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
        # Connect Scanner -> Watchlist bridge
        if hasattr(self, 'scanner_widget') and hasattr(self, 'watchlist_widget'):
            try:
                self.scanner_widget.add_to_watchlist.connect(self.on_add_from_scanner)
            except Exception as e:
                self.logger.warning(f"Failed to wire Scanner->Watchlist: {e}")
    
    def apply_theme(self):
        """Apply current theme"""
        self.theme_manager.apply_theme(self, self.config.ui.theme)

    def show_daily_update_dialog(self):
        try:
            dlg = DataUpdateDialog(self, getattr(self, 'data_update_service', None))
            dlg.setModal(False)
            dlg.show()
        except Exception as e:
            QMessageBox.critical(self, "Daily Update", f"Failed to open dialog: {e}")
    
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
        """Attempt to auto-connect to IBKR in a background thread (non-blocking)."""
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                return

            # Check if another connection attempt is already in progress
            if hasattr(self, '_ibkr_thread') and self._ibkr_thread.isRunning():
                self.logger.info("IBKR connection attempt already in progress")
                return

            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(False)
                self.connect_action.setText("Connecting…")

            # Create worker in a QThread to keep UI responsive
            self._ibkr_thread = QThread(self)
            self._ibkr_worker = IBKRConnectWorker(lambda: IBKRService(self.config.ibkr))
            self._ibkr_worker.moveToThread(self._ibkr_thread)

            # Wire signals
            self._ibkr_thread.started.connect(self._ibkr_worker.run)
            self._ibkr_worker.finished.connect(self.on_auto_connect_finished)
            # Ensure thread stops after work
            self._ibkr_worker.finished.connect(self._ibkr_thread.quit)

            self._ibkr_thread.start()
        except Exception as e:
            self.logger.error(f"Auto-connect setup failed: {e}")
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(True)
                self.connect_action.setText("Connect to IBKR")

    def on_auto_connect_finished(self, success: bool, error: str):
        """Handle completion of auto-connect worker."""
        try:
            # Adopt the service instance from worker if connected
            if success and hasattr(self, '_ibkr_worker'):
                self.ibkr_service = getattr(self._ibkr_worker, 'service', None)
                # Wire services into widgets
                if hasattr(self, 'dashboard_widget'):
                    self.dashboard_widget.set_ibkr_service(self.ibkr_service)
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_service(self.ibkr_service)
                    self.ai_trading_widget.set_ibkr_status(True)
                self.connection_status_changed.emit(True)
                self.logger.info("Auto-connected to IBKR successfully")
                if hasattr(self, 'connect_action'):
                    self.connect_action.setText("Disconnect from IBKR")
            else:
                # Report error in status and optionally message box
                detail = error or "Failed to connect to IBKR"
                try:
                    if hasattr(self, '_ibkr_worker') and getattr(self._ibkr_worker, 'service', None):
                        svc = getattr(self._ibkr_worker, 'service')
                        if getattr(svc, 'last_error', None):
                            detail = svc.last_error
                        ports = getattr(svc, 'last_ports_tried', None)
                        if ports:
                            detail = f"Ports tried: {ports}. Last error: {detail}"
                except Exception:
                    pass
                self.logger.error(f"Auto-connect failed: {detail}")
                self.connection_status_changed.emit(False)
                # Keep it unobtrusive at startup: show in status bar, not a blocking dialog
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(f"IBKR auto-connect failed: {detail}", 10000)
                # Also show a small MessageBox so it's visible on startup
                try:
                    QMessageBox.warning(self, "IBKR Auto-Connect Failed", detail)
                except Exception:
                    pass
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_status(False)
            
        finally:
            # Cleanup thread/worker
            try:
                if hasattr(self, '_ibkr_worker'):
                    self._ibkr_worker.deleteLater()
                    del self._ibkr_worker
            except Exception:
                pass
            try:
                if hasattr(self, '_ibkr_thread'):
                    # If still running, let the quit signal stop it; wait a bit
                    if self._ibkr_thread.isRunning():
                        self._ibkr_thread.quit()
                        self._ibkr_thread.wait(2000)
                    del self._ibkr_thread
            except Exception:
                pass
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(True)
                if not (self.ibkr_service and self.ibkr_service.is_connected()):
                    self.connect_action.setText("Connect to IBKR")

    def on_manual_connect_finished(self, success: bool, error: str):
        """Handle completion of manual connect worker."""
        try:
            # Adopt the service instance from worker if connected
            if success and hasattr(self, '_ibkr_worker'):
                self.ibkr_service = getattr(self._ibkr_worker, 'service', None)
                # Wire services into widgets
                if hasattr(self, 'dashboard_widget'):
                    self.dashboard_widget.set_ibkr_service(self.ibkr_service)
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_service(self.ibkr_service)
                    self.ai_trading_widget.set_ibkr_status(True)
                self.connection_status_changed.emit(True)
                self.logger.info("Connected to IBKR successfully")
                if hasattr(self, 'connect_action'):
                    self.connect_action.setText("Disconnect from IBKR")
            else:
                # Report error with user-friendly message box
                detail = error or "Failed to connect to IBKR"
                try:
                    if hasattr(self, '_ibkr_worker') and getattr(self._ibkr_worker, 'service', None):
                        svc = getattr(self._ibkr_worker, 'service')
                        if getattr(svc, 'last_error', None):
                            detail = svc.last_error
                        ports = getattr(svc, 'last_ports_tried', None)
                        if ports:
                            detail = f"Ports tried: {ports}. Last error: {detail}"
                except Exception:
                    pass
                self.logger.error(f"Manual connect failed: {detail}")
                self.connection_status_changed.emit(False)
                # Show blocking error dialog for manual connection attempts
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to IBKR.\n{detail}")
                if hasattr(self, 'ai_trading_widget'):
                    self.ai_trading_widget.set_ibkr_status(False)

        finally:
            # Cleanup thread/worker
            try:
                if hasattr(self, '_ibkr_worker'):
                    self._ibkr_worker.deleteLater()
                    del self._ibkr_worker
            except Exception:
                pass
            try:
                if hasattr(self, '_ibkr_thread'):
                    # If still running, let the quit signal stop it; wait a bit
                    if self._ibkr_thread.isRunning():
                        self._ibkr_thread.quit()
                        self._ibkr_thread.wait(2000)
                    del self._ibkr_thread
            except Exception:
                pass
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(True)
                if not (self.ibkr_service and self.ibkr_service.is_connected()):
                    self.connect_action.setText("Connect to IBKR")

    def connect_ibkr(self):
        """Connect to IBKR in a background thread (non-blocking)."""
        try:
            if self.ibkr_service and self.ibkr_service.is_connected():
                return

            # Check if another connection attempt is already in progress
            if hasattr(self, '_ibkr_thread') and self._ibkr_thread.isRunning():
                self.logger.info("IBKR connection attempt already in progress")
                return

            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(False)
                self.connect_action.setText("Connecting…")

            # Create worker in a QThread to keep UI responsive
            self._ibkr_thread = QThread(self)
            self._ibkr_worker = IBKRConnectWorker(lambda: IBKRService(self.config.ibkr))
            self._ibkr_worker.moveToThread(self._ibkr_thread)

            # Wire signals
            self._ibkr_thread.started.connect(self._ibkr_worker.run)
            self._ibkr_worker.finished.connect(self.on_manual_connect_finished)
            # Ensure thread stops after work
            self._ibkr_worker.finished.connect(self._ibkr_thread.quit)

            self._ibkr_thread.start()
        except Exception as e:
            self.logger.error(f"Manual connect setup failed: {e}")
            if hasattr(self, 'connect_action'):
                self.connect_action.setEnabled(True)
                self.connect_action.setText("Connect to IBKR")

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
                    self.ai_trading_widget.set_ibkr_service(None)
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

    def on_add_from_scanner(self, symbol: str):
        """Handle 'Add to Watchlist' requests coming from the Scanner tab."""
        try:
            if not symbol:
                return
            # Switch to Watchlist tab
            if hasattr(self, 'watchlist_widget'):
                idx = self.tab_widget.indexOf(self.watchlist_widget)
                if idx != -1:
                    self.tab_widget.setCurrentIndex(idx)
                # Add/focus symbol in watchlist
                self.watchlist_widget.add_symbol_from_scanner(symbol, switch_to_tab=True)
        except Exception as e:
            self.logger.error(f"Failed to handle add_from_scanner for {symbol}: {e}")
    
    def show_dashboard_about(self):
        """Open the Dashboard ABOUT markdown in a dialog."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            md_path = project_root / "docs" / "DASHBOARD_ABOUT.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
            else:
                text = "Dashboard ABOUT file not found."
        except Exception as e:
            text = f"Error loading ABOUT: {e}"

        dlg = QDialog(self)
        dlg.setWindowTitle("About • Dashboard")
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setReadOnly(True)
        browser.setOpenExternalLinks(True)
        # Minimal markdown display - QTextBrowser supports basics
        browser.setMarkdown(text)
        layout.addWidget(browser)
        dlg.resize(840, 600)
        dlg.exec()

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

    def show_ai_trading_guide(self):
        """Open the AI Trading guide (Markdown) in a dialog."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            guide_path = project_root / "docs" / "AI_TRADING_ABOUT.md"
            if not guide_path.exists():
                QMessageBox.warning(self, "Guide Not Found", f"Could not find guide at:\n{guide_path}")
                return
            text = guide_path.read_text(encoding="utf-8")
            dlg = QDialog(self)
            dlg.setWindowTitle("AI Trading Guide")
            dlg.resize(840, 600)
            browser = QTextBrowser(dlg)
            browser.setReadOnly(True)
            browser.setOpenExternalLinks(True)
            try:
                import markdown  # type: ignore
                html_body = markdown.markdown(text, extensions=["extra", "sane_lists", "tables", "fenced_code"])
                browser.setHtml(html_body)
            except Exception:
                browser.setMarkdown(text)
            layout = QVBoxLayout(dlg)
            layout.addWidget(browser)
            dlg.setLayout(layout)
            dlg.exec()
        except Exception as e:
            self.logger.error(f"Error opening AI Trading guide: {e}")
            QMessageBox.critical(self, "Guide Error", f"Failed to open AI Trading guide:\n{e}")

    def show_watchlist_guide(self):
        """Open the Watchlist guide (Markdown) in a right-to-left styled dialog"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            guide_path = project_root / "docs" / "WATCHLIST_GUIDE.md"
            if not guide_path.exists():
                QMessageBox.warning(self, "Guide Not Found", f"Could not find guide at:\n{guide_path}")
                return
            text = guide_path.read_text(encoding="utf-8")
            dlg = QDialog(self)
            dlg.setWindowTitle("מדריך רשימת מעקב - Watchlist Guide")
            dlg.resize(1000, 700)  # Larger for Hebrew content
            browser = QTextBrowser(dlg)
            browser.setReadOnly(True)
            browser.setOpenExternalLinks(True)
            
            # Set RTL styling for Hebrew content
            browser.setStyleSheet("""
                QTextBrowser {
                    direction: rtl;
                    text-align: right;
                    font-family: 'Arial', 'Segoe UI', sans-serif;
                    font-size: 11pt;
                    line-height: 1.4;
                }
            """)
            
            try:
                import markdown  # type: ignore
                html_body = markdown.markdown(text, extensions=["extra", "sane_lists", "tables", "fenced_code"])
                # Add RTL styling to HTML
                styled_html = f"""
                <html dir="rtl">
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ 
                            direction: rtl; 
                            text-align: right; 
                            font-family: 'Arial', 'Segoe UI', sans-serif;
                            line-height: 1.6;
                            margin: 20px;
                        }}
                        h1, h2, h3, h4, h5, h6 {{ 
                            text-align: right; 
                            color: #2c3e50; 
                        }}
                        code {{ 
                            background-color: #f8f9fa; 
                            padding: 2px 4px; 
                            border-radius: 3px; 
                        }}
                        pre {{ 
                            background-color: #f8f9fa; 
                            padding: 10px; 
                            border-radius: 5px; 
                            overflow-x: auto; 
                        }}
                        table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                        }}
                        th, td {{ 
                            border: 1px solid #ddd; 
                            padding: 8px; 
                            text-align: right; 
                        }}
                        th {{ 
                            background-color: #f2f2f2; 
                        }}
                    </style>
                </head>
                <body>{html_body}</body>
                </html>
                """
                browser.setHtml(styled_html)
            except Exception:
                browser.setMarkdown(text)
            
            layout = QVBoxLayout(dlg)
            layout.addWidget(browser)
            dlg.setLayout(layout)
            dlg.exec()
        except Exception as e:
            self.logger.error(f"Error opening Watchlist guide: {e}")
            QMessageBox.critical(self, "Guide Error", f"Failed to open Watchlist guide:\n{e}")

    def show_ml_scanner_guide(self):
        """Open the ML Training & Scanner guide (Markdown) in a dialog"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            guide_path = project_root / "docs" / "help" / "ml_training_and_scan.md"
            if not guide_path.exists():
                QMessageBox.warning(self, "Guide Not Found", f"Could not find guide at:\n{guide_path}")
                return
            text = guide_path.read_text(encoding="utf-8")
            dlg = QDialog(self)
            dlg.setWindowTitle("ML Training & Scanner Guide")
            dlg.resize(840, 600)
            browser = QTextBrowser(dlg)
            browser.setReadOnly(True)
            browser.setOpenExternalLinks(True)
            try:
                import markdown  # type: ignore
                html_body = markdown.markdown(text, extensions=["extra", "sane_lists", "tables", "fenced_code"])
                browser.setHtml(html_body)
            except Exception:
                browser.setMarkdown(text)
            layout = QVBoxLayout(dlg)
            layout.addWidget(browser)
            dlg.setLayout(layout)
            dlg.exec()
        except Exception as e:
            self.logger.error(f"Error opening ML Scanner guide: {e}")
            QMessageBox.critical(self, "Guide Error", f"Failed to open ML Scanner guide:\n{e}")
    
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