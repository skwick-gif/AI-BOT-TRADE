"""
API Keys Configuration Dialog
Allows users to edit API keys from the UI
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QTabWidget, QWidget,
    QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import os
from pathlib import Path

from core.config_manager import ConfigManager
from utils.logger import get_logger


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
        market_layout.addRow("FMP API Key:", self.fmp_key)
        
        self.quantiq_key = QLineEdit()
        self.quantiq_key.setPlaceholderText("QuantiQ API key")
        self.quantiq_key.setEchoMode(QLineEdit.EchoMode.Password)
        market_layout.addRow("QuantiQ API Key:", self.quantiq_key)
        
        self.fred_key = QLineEdit()
        self.fred_key.setPlaceholderText("FRED API key")
        self.fred_key.setEchoMode(QLineEdit.EchoMode.Password)
        market_layout.addRow("FRED API Key:", self.fred_key)
        
        layout.addWidget(market_group)
        
        # Social Media APIs
        social_group = QGroupBox("Social Media APIs")
        social_layout = QFormLayout(social_group)
        
        self.reddit_client_id = QLineEdit()
        self.reddit_client_id.setPlaceholderText("Reddit Client ID")
        social_layout.addRow("Reddit Client ID:", self.reddit_client_id)
        
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
        news_layout.addRow("NewsAPI Key:", self.news_key)
        
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
        self.ibkr_port.setPlaceholderText("7497 (paper) or 7496 (live)")
        ibkr_layout.addRow("Port:", self.ibkr_port)
        
        self.ibkr_client_id = QLineEdit()
        self.ibkr_client_id.setPlaceholderText("1")
        ibkr_layout.addRow("Client ID:", self.ibkr_client_id)
        
        layout.addWidget(ibkr_group)
        
        # Info label
        info_label = QLabel(
            "IBKR Settings:\n"
            "• Host: Usually 127.0.0.1 (localhost)\n"
            "• Port: 7497 for paper trading, 7496 for live trading\n"
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
            self.ibkr_port.setText(env_vars.get('IBKR_PORT', '7497'))
            self.ibkr_client_id.setText(env_vars.get('IBKR_CLIENT_ID', '1'))
            
            print("DEBUG: All fields populated successfully")
            
        except Exception as e:
            print(f"DEBUG: Exception details: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error loading current values: {e}")
    
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