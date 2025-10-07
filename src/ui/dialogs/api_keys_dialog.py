"""
API Keys Configuration Dialog
Allows users to edit API keys from the UI
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QTabWidget, QWidget,
    QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont
import os
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, QThread
import socket
import requests

from core.config_manager import ConfigManager, IBKRConfig
from utils.logger import get_logger
from services.ibkr_service import IBKRService


class APIKeysDialog(QDialog):
    """Dialog for editing API keys"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("APIKeysDialog")
        self.config = ConfigManager()
        self.env_file = Path(__file__).parent.parent.parent.parent / ".env"
        
        self.setup_ui()
        self.load_current_values()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("API Keys Configuration")
        self.setFixedSize(500, 600)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("API Keys Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create tabs for different API categories
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Essential APIs Tab
        essential_tab = self.create_essential_tab()
        tab_widget.addTab(essential_tab, "Essential APIs")
        
        # Optional APIs Tab
        optional_tab = self.create_optional_tab()
        tab_widget.addTab(optional_tab, "Optional APIs")
        
        # IBKR Settings Tab
        ibkr_tab = self.create_ibkr_tab()
        tab_widget.addTab(ibkr_tab, "IBKR Settings")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_keys)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def create_essential_tab(self):
        """Create essential APIs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Essential APIs group
        essential_group = QGroupBox("Essential for AI Trading")
        essential_layout = QFormLayout(essential_group)
        
        # Perplexity API
        self.perplexity_key = QLineEdit()
        self.perplexity_key.setPlaceholderText("Enter your Perplexity API key")
        self.perplexity_key.setEchoMode(QLineEdit.EchoMode.Password)
        essential_layout.addRow("Perplexity API Key:", self.perplexity_key)
        
        # Show/Hide button for Perplexity
        perplexity_toggle = QPushButton("Show")
        perplexity_toggle.clicked.connect(lambda: self.toggle_password_visibility(self.perplexity_key, perplexity_toggle))
        essential_layout.addRow("", perplexity_toggle)
        
        layout.addWidget(essential_group)
        
        # Info label
        info_label = QLabel(
            "Perplexity API Key is required for AI functionality.\n"
            "Get your key at: https://www.perplexity.ai/"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget
    
    def create_optional_tab(self):
        """Create optional APIs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Market Data APIs
        market_group = QGroupBox("Market Data APIs")
        market_layout = QFormLayout(market_group)
        
        self.fmp_key = QLineEdit()
        self.fmp_key.setPlaceholderText("Financial Modeling Prep API key")
        self.fmp_key.setEchoMode(QLineEdit.EchoMode.Password)
        # Add small test button to verify FMP key/reachability
        fmp_row = QWidget()
        fr_layout = QHBoxLayout(fmp_row)
        fr_layout.setContentsMargins(0,0,0,0)
        fr_layout.addWidget(self.fmp_key)
        self.fmp_test_btn = QPushButton("Test")
        self.fmp_test_btn.setFixedWidth(60)
        fr_layout.addWidget(self.fmp_test_btn)
        # Inline status icon (empty until test runs)
        self.fmp_status = QLabel("")
        self.fmp_status.setFixedWidth(24)
        self.fmp_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fr_layout.addWidget(self.fmp_status)
        market_layout.addRow("FMP API Key:", fmp_row)
        
        self.quantiq_key = QLineEdit()
        self.quantiq_key.setPlaceholderText("QuantiQ API key")
        self.quantiq_key.setEchoMode(QLineEdit.EchoMode.Password)
        # Add small Test button for QuantiQ
        quant_row = QWidget()
        ql = QHBoxLayout(quant_row)
        ql.setContentsMargins(0,0,0,0)
        ql.addWidget(self.quantiq_key)
        self.quantiq_test_btn = QPushButton("Test")
        self.quantiq_test_btn.setFixedWidth(60)
        ql.addWidget(self.quantiq_test_btn)
        self.quantiq_status = QLabel("")
        self.quantiq_status.setFixedWidth(24)
        self.quantiq_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ql.addWidget(self.quantiq_status)
        market_layout.addRow("QuantiQ API Key:", quant_row)
        
        # FRED API with Test button
        self.fred_key = QLineEdit()
        self.fred_key.setPlaceholderText("FRED API key")
        self.fred_key.setEchoMode(QLineEdit.EchoMode.Password)
        fred_row = QWidget()
        fred_l = QHBoxLayout(fred_row)
        fred_l.setContentsMargins(0,0,0,0)
        fred_l.addWidget(self.fred_key)
        self.fred_test_btn = QPushButton("Test")
        self.fred_test_btn.setFixedWidth(60)
        fred_l.addWidget(self.fred_test_btn)
        self.fred_status = QLabel("")
        self.fred_status.setFixedWidth(24)
        self.fred_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fred_l.addWidget(self.fred_status)
        market_layout.addRow("FRED API Key:", fred_row)

        layout.addWidget(market_group)

        # Social Media APIs
        social_group = QGroupBox("Social Media APIs")
        social_layout = QFormLayout(social_group)

        self.reddit_client_id = QLineEdit()
        self.reddit_client_id.setPlaceholderText("Reddit Client ID")
        # Add Test button for Reddit client credentials
        reddit_row = QWidget()
        rlayout = QHBoxLayout(reddit_row)
        rlayout.setContentsMargins(0,0,0,0)
        rlayout.addWidget(self.reddit_client_id)
        self.reddit_test_btn = QPushButton("Test")
        self.reddit_test_btn.setFixedWidth(60)
        rlayout.addWidget(self.reddit_test_btn)
        self.reddit_status = QLabel("")
        self.reddit_status.setFixedWidth(24)
        self.reddit_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rlayout.addWidget(self.reddit_status)
        social_layout.addRow("Reddit Client ID:", reddit_row)

        self.reddit_secret = QLineEdit()
        self.reddit_secret.setPlaceholderText("Reddit Client Secret")
        self.reddit_secret.setEchoMode(QLineEdit.EchoMode.Password)
        social_layout.addRow("Reddit Secret:", self.reddit_secret)

        self.reddit_user_agent = QLineEdit()
        self.reddit_user_agent.setPlaceholderText("Reddit User Agent")
        social_layout.addRow("Reddit User Agent:", self.reddit_user_agent)

        layout.addWidget(social_group)
        
        # News API
        news_group = QGroupBox("News API")
        news_layout = QFormLayout(news_group)
        
        self.news_key = QLineEdit()
        self.news_key.setPlaceholderText("NewsAPI key")
        self.news_key.setEchoMode(QLineEdit.EchoMode.Password)
        # Add Test button for NewsAPI
        news_row = QWidget()
        nr_layout = QHBoxLayout(news_row)
        nr_layout.setContentsMargins(0,0,0,0)
        nr_layout.addWidget(self.news_key)
        self.news_test_btn = QPushButton("Test")
        self.news_test_btn.setFixedWidth(60)
        nr_layout.addWidget(self.news_test_btn)
        self.news_status = QLabel("")
        self.news_status.setFixedWidth(24)
        self.news_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nr_layout.addWidget(self.news_status)
        news_layout.addRow("NewsAPI Key:", news_row)
        
        layout.addWidget(news_group)
        
        layout.addStretch()
        return widget
    
    def create_ibkr_tab(self):
        """Create IBKR settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # IBKR Connection Settings
        ibkr_group = QGroupBox("IBKR Connection Settings")
        ibkr_layout = QFormLayout(ibkr_group)
        
        self.ibkr_host = QLineEdit()
        self.ibkr_host.setPlaceholderText("127.0.0.1")
        ibkr_layout.addRow("Host:", self.ibkr_host)
        
        self.ibkr_port = QLineEdit()
        self.ibkr_port.setPlaceholderText("4001 (IB Gateway), 7497 (TWS paper), 7496 (TWS live)")
        ibkr_layout.addRow("Port:", self.ibkr_port)
        
        self.ibkr_client_id = QLineEdit()
        self.ibkr_client_id.setPlaceholderText("1")
        # IBKR test button (socket connect)
        ibkr_row = QWidget()
        ibr = QHBoxLayout(ibkr_row)
        ibr.setContentsMargins(0,0,0,0)
        ibr.addWidget(self.ibkr_client_id)
        self.ibkr_test_btn = QPushButton("Test")
        self.ibkr_test_btn.setFixedWidth(60)
        ibr.addWidget(self.ibkr_test_btn)
        # IBKR inline status
        self.ibkr_status = QLabel("")
        self.ibkr_status.setFixedWidth(24)
        self.ibkr_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ibr.addWidget(self.ibkr_status)
        ibkr_layout.addRow("Client ID:", ibkr_row)
        
        layout.addWidget(ibkr_group)
        
        # Info label
        info_label = QLabel(
            "IBKR Settings:\n"
            "• Host: Usually 127.0.0.1 (localhost)\n"
            "• Port: 4001 for IB Gateway, 7497 for TWS paper trading, 7496 for TWS live trading\n"
            "• Client ID: Unique identifier (1-999)\n"
            "• Make sure TWS is running with API enabled"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget
    
    def toggle_password_visibility(self, line_edit, button):
        """Toggle password visibility for a line edit"""
        if line_edit.echoMode() == QLineEdit.EchoMode.Password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            button.setText("Hide")
        else:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            button.setText("Show")
    
    def load_current_values(self):
        """Load current values from .env file"""
        try:
            print(f"DEBUG: Loading from {self.env_file}")
            if not self.env_file.exists():
                print("DEBUG: .env file does not exist")
                return
            
            env_vars = {}
            with open(self.env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"DEBUG: File content length: {len(content)}")
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        env_vars[key] = value
                        print(f"DEBUG: Loaded {key} = {value[:10]}...")
            
            print(f"DEBUG: Total variables loaded: {len(env_vars)}")
            
            # Load values into fields
            self.perplexity_key.setText(env_vars.get('PERPLEXITY_API_KEY', ''))
            self.fmp_key.setText(env_vars.get('FMP_API_KEY', ''))
            self.quantiq_key.setText(env_vars.get('QUANTIQ_API', ''))
            self.fred_key.setText(env_vars.get('FRED_API_KEY', ''))
            self.reddit_client_id.setText(env_vars.get('REDDIT_CLIENT_ID', ''))
            self.reddit_secret.setText(env_vars.get('REDDIT_CLIENT_SECRET', ''))
            self.reddit_user_agent.setText(env_vars.get('REDDIT_USER_AGENT', 'StockAnalysisBot/1.0'))
            self.news_key.setText(env_vars.get('NEWSAPI_KEY', ''))
            self.ibkr_host.setText(env_vars.get('IBKR_HOST', '127.0.0.1'))
            self.ibkr_port.setText(env_vars.get('IBKR_PORT', '4001'))  # IB Gateway default
            self.ibkr_client_id.setText(env_vars.get('IBKR_CLIENT_ID', '1'))
        
            # Wire test buttons after values are loaded
            self.fmp_test_btn.clicked.connect(lambda: self._start_test('fmp'))
            self.news_test_btn.clicked.connect(lambda: self._start_test('news'))
            self.ibkr_test_btn.clicked.connect(lambda: self._start_test('ibkr'))
            self.quantiq_test_btn.clicked.connect(lambda: self._start_test('quantiq'))
            self.reddit_test_btn.clicked.connect(lambda: self._start_test('reddit'))
            # FRED uses the same News test button group (optional)
            try:
                self.fred_test_btn.clicked.connect(lambda: self._start_test('fred'))
            except Exception:
                pass

        except Exception as e:
            print(f"DEBUG: Exception details: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error loading current values: {e}")

    def _start_test(self, target: str):
        """Start a non-blocking connectivity test for a target service."""
        try:
            # For IBKR perform a full connect attempt (ib_insync) in a worker
            if target == 'ibkr':
                class _IBKRFullTester(QObject):
                    result = pyqtSignal(bool, str)
                    finished = pyqtSignal()

                    def __init__(self, host: str, port: int, client_id: int):
                        super().__init__()
                        self.host = host
                        self.port = port
                        self.client_id = client_id
                        self.service = None

                    def run(self):
                        try:
                            cfg = IBKRConfig(host=self.host, port=self.port, client_id=self.client_id)
                            svc = IBKRService(cfg)
                            # Attempt connect (this method handles preflight port checks)
                            ok = False
                            try:
                                ok = svc.connect()
                            except Exception as e:
                                self.result.emit(False, f"IBKR connect attempt exception: {e}")
                                ok = False
                            if ok:
                                # Disconnect cleanly after test
                                try:
                                    svc.disconnect()
                                except Exception:
                                    pass
                                self.result.emit(True, f"Connected to IBKR at {self.host}:{self.port}")
                            else:
                                err = getattr(svc, 'last_error', 'Failed to connect to IBKR')
                                self.result.emit(False, f"IBKR connect failed: {err}")
                        finally:
                            self.finished.emit()

                self.test_thread = QThread()
                tester = _IBKRFullTester(self.ibkr_host.text(), int(self.ibkr_port.text() or 0), int(self.ibkr_client_id.text() or 0))
                tester.moveToThread(self.test_thread)
                self.test_thread.started.connect(tester.run)
                tester.finished.connect(self.test_thread.quit)
                tester.finished.connect(tester.deleteLater)
                self.test_thread.finished.connect(self.test_thread.deleteLater)
                # Show running indicator
                try:
                    self.ibkr_status.setText('⏳')
                    self.ibkr_status.setStyleSheet('color: #999')
                except Exception:
                    pass
                tester.result.connect(lambda ok, msg: self._on_test_result('ibkr', ok, msg))
                self.test_thread.start()
            else:
                self.test_thread = QThread()
                self.tester = _ConnectivityTester(target,
                                                  host=self.ibkr_host.text(),
                                                  port=int(self.ibkr_port.text() or 0),
                                                  fmp_key=self.fmp_key.text(),
                                                  news_key=self.news_key.text())
                self.tester.moveToThread(self.test_thread)
                self.test_thread.started.connect(self.tester.run)
                self.tester.finished.connect(self.test_thread.quit)
                self.tester.finished.connect(self.tester.deleteLater)
                self.test_thread.finished.connect(self.test_thread.deleteLater)
                # Set running indicator for the target if label exists
                lbl = getattr(self, f"{target}_status", None)
                if lbl is not None:
                    try:
                        lbl.setText('⏳')
                        lbl.setStyleSheet('color: #999')
                    except Exception:
                        pass

                self.tester.result.connect(lambda ok, msg, t=target: self._on_test_result(t, ok, msg))
                self.test_thread.start()
        except Exception as e:
            QMessageBox.warning(self, "Test Error", f"Failed to start test: {e}")


    def _on_test_result(self, service: str, ok: bool, msg: str):
        """Update inline status label for a service test result."""
        lbl = getattr(self, f"{service}_status", None)
        try:
            if lbl is None:
                # Fallback to a message box if no inline label exists
                if ok:
                    QMessageBox.information(self, f"{service.upper()} Test", msg)
                else:
                    QMessageBox.warning(self, f"{service.upper()} Test", msg)
                return

            if ok:
                lbl.setText('✓')
                lbl.setStyleSheet('color: green; font-weight: bold')
            else:
                lbl.setText('✖')
                lbl.setStyleSheet('color: red; font-weight: bold')
            lbl.setToolTip(msg)
        except Exception as e:
            # If updating label fails, still show a message to the user
            try:
                if ok:
                    QMessageBox.information(self, f"{service.upper()} Test", msg)
                else:
                    QMessageBox.warning(self, f"{service.upper()} Test", msg)
            except Exception:
                pass


    def save_keys(self):
        """Save API keys to .env file"""
        try:
            # Prepare the .env content
            env_content = [
                "# AI APIs - הכנס כאן את המפתחות שלך",
                f'PERPLEXITY_API_KEY="{self.perplexity_key.text()}"',
                f'OPENAI_API_KEY="your_openai_api_key_here"',
                f'QUANTIQ_API="{self.quantiq_key.text()}"',
                "",
                "# Interactive Brokers Configuration",
                f'IBKR_HOST="{self.ibkr_host.text()}"',
                f'IBKR_PORT="{self.ibkr_port.text()}"',
                f'IBKR_CLIENT_ID="{self.ibkr_client_id.text()}"',
                "",
                "# Optional APIs - אלה לא חובה לתפעול בסיסי",
                f'FMP_API_KEY="{self.fmp_key.text()}"',
                f'FRED_API_KEY="{self.fred_key.text()}"',
                f'REDDIT_CLIENT_ID="{self.reddit_client_id.text()}"',
                f'REDDIT_CLIENT_SECRET="{self.reddit_secret.text()}"',
                f'REDDIT_USER_AGENT="{self.reddit_user_agent.text()}"',
                f'NEWSAPI_KEY="{self.news_key.text()}"'
            ]
            
            # Write to .env file
            with open(self.env_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(env_content))
            
            QMessageBox.information(
                self,
                "Success",
                "API keys saved successfully!\n\nConfiguration will be reloaded automatically."
            )
            
            self.accept()
            
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save API keys:\n{e}"
            )


class _ConnectivityTester(QObject):
    """Background tester to run quick connectivity checks."""
    result = pyqtSignal(bool, str)
    finished = pyqtSignal()

    def __init__(self, target: str, host: str = '127.0.0.1', port: int = 0, fmp_key: str = '', news_key: str = ''):
        super().__init__()
        self.target = target
        self.host = host
        self.port = port
        self.fmp_key = fmp_key
        self.news_key = news_key

    def run(self):
        try:
            if self.target == 'ibkr':
                try:
                    with socket.create_connection((self.host, self.port), timeout=3):
                        self.result.emit(True, f"Connected to IBKR at {self.host}:{self.port}")
                except Exception as e:
                    self.result.emit(False, f"IBKR connection failed: {e}")
            elif self.target == 'fmp':
                # Basic reachability check for Financial Modeling Prep
                url = "https://financialmodelingprep.com/api/v3/profile/AAPL"
                try:
                    r = requests.get(url, timeout=4)
                    if r.status_code == 200:
                        self.result.emit(True, "FMP reachable (public endpoint returned 200).")
                    else:
                        self.result.emit(False, f"FMP endpoint returned {r.status_code}")
                except Exception as e:
                    self.result.emit(False, f"FMP check failed: {e}")
            elif self.target == 'news':
                # NewsAPI simple test (requires key) using top-headlines endpoint
                try:
                    headers = {'X-Api-Key': self.news_key} if self.news_key else {}
                    r = requests.get('https://newsapi.org/v2/top-headlines?country=us&pageSize=1', headers=headers, timeout=4)
                    if r.status_code == 200:
                        self.result.emit(True, "NewsAPI reachable and key accepted (200).")
                    elif r.status_code == 401:
                        self.result.emit(False, "NewsAPI unauthorized (invalid key).")
                    else:
                        self.result.emit(False, f"NewsAPI returned {r.status_code}")
                except Exception as e:
                    self.result.emit(False, f"NewsAPI check failed: {e}")
            else:
                # Generic domain reachability
                try:
                    r = requests.get('https://www.google.com', timeout=3)
                    if r.status_code == 200:
                        self.result.emit(True, "Network OK (google.com reachable)")
                    else:
                        self.result.emit(False, f"Network test returned {r.status_code}")
                except Exception as e:
                    self.result.emit(False, f"Network test failed: {e}")
            # Additional service checks
            if self.target == 'quantiq':
                try:
                    # Simple reachability to quantiq base (placeholder public URL)
                    r = requests.get('https://api.quantiq.io/v1/ping', timeout=4)
                    if r.status_code == 200:
                        self.result.emit(True, 'QuantiQ reachable (ping OK).')
                    else:
                        self.result.emit(False, f'QuantiQ returned {r.status_code}')
                except Exception as e:
                    self.result.emit(False, f'QuantiQ check failed: {e}')
            elif self.target == 'fred':
                try:
                    # FRED requires api_key param; use series endpoint with a common series
                    key = getattr(self, 'news_key', '') or ''
                    # Try a public series endpoint to check service availability (doesn't require key for status 200 but may return 403)
                    url = 'https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=' + (key or '')
                    r = requests.get(url, timeout=4)
                    if r.status_code == 200:
                        self.result.emit(True, 'FRED reachable (200).')
                    elif r.status_code == 403 or r.status_code == 401:
                        self.result.emit(False, 'FRED unauthorized (invalid key).')
                    else:
                        self.result.emit(False, f'FRED returned {r.status_code}')
                except Exception as e:
                    self.result.emit(False, f'FRED check failed: {e}')
            elif self.target == 'reddit':
                try:
                    # Basic OAuth check: POST to access_token with client credentials
                    cid = getattr(self, 'fmp_key', '') or ''
                    secret = getattr(self, 'news_key', '') or ''
                    # Use values passed in as properties (the dialog supplies keys appropriately)
                    auth = None
                    headers = {'User-Agent': 'AI-BOT-TRADE/1.0'}
                    if cid and secret:
                        try:
                            r = requests.post('https://www.reddit.com/api/v1/access_token', auth=(cid, secret), data={'grant_type':'client_credentials'}, headers=headers, timeout=6)
                            if r.status_code == 200:
                                self.result.emit(True, 'Reddit credentials accepted (access token retrieved).')
                            elif r.status_code == 401:
                                self.result.emit(False, 'Reddit unauthorized (invalid client id/secret).')
                            else:
                                self.result.emit(False, f'Reddit returned {r.status_code}')
                        except Exception as e:
                            self.result.emit(False, f'Reddit auth check failed: {e}')
                    else:
                        # Fallback: try anonymous public endpoint reachability
                        r = requests.get('https://www.reddit.com/.json', headers=headers, timeout=4)
                        if r.status_code == 200:
                            self.result.emit(True, 'Reddit reachable (public).')
                        else:
                            self.result.emit(False, f'Reddit returned {r.status_code}')
                except Exception as e:
                    self.result.emit(False, f'Reddit check failed: {e}')
        finally:
            self.finished.emit()