"""
AI Chat tab for AI assistant
"""

import os
import requests
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
                             QLineEdit, QPushButton, QScrollArea, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from datetime import datetime

class ChatThread(QThread):
    """Thread for handling AI chat requests"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, message, api_key):
        super().__init__()
        self.message = message
        self.api_key = api_key
    
    def run(self):
        """Send message to AI and get response"""
        try:
            if not self.api_key:
                self.error_occurred.emit("Perplexity API key not configured")
                return
            
            # Perplexity API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-small-chat",
                "messages": [
                    {"role": "system", "content": "You are a financial AI assistant. Help users with stock analysis, trading advice, and market insights. Keep responses concise and actionable."},
                    {"role": "user", "content": self.message}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data['choices'][0]['message']['content']
                
                # Add citations if available
                if 'citations' in data and data['citations']:
                    ai_response += "\n\n📚 Sources:"
                    for i, citation in enumerate(data['citations'][:3], 1):
                        ai_response += f"\n{i}. {citation}"
                
                self.response_ready.emit(ai_response)
            else:
                self.error_occurred.emit(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")

class ChatMessage(QFrame):
    """Individual chat message widget"""
    
    def __init__(self, message, is_user=True):
        super().__init__()
        self.setup_ui(message, is_user)
    
    def setup_ui(self, message, is_user):
        """Setup the message UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Message content
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Styling based on sender
        if is_user:
            message_label.setStyleSheet("""
                QLabel {
                    background-color: #0d7377;
                    color: white;
                    padding: 10px;
                    border-radius: 10px;
                    margin-left: 50px;
                }
            """)
            layout.addStretch()
            layout.addWidget(message_label)
        else:
            message_label.setStyleSheet("""
                QLabel {
                    background-color: #404040;
                    color: white;
                    padding: 10px;
                    border-radius: 10px;
                    margin-right: 50px;
                }
            """)
            layout.addWidget(message_label)
            layout.addStretch()

class ChatTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.chat_history = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the chat UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("� Perplexity AI Trading Assistant")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ffffff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()  # Push messages to bottom
        
        self.chat_scroll.setWidget(self.chat_widget)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.chat_scroll)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask me about stocks, market trends, real-time data analysis...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #0d7377;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(80, 40)
        self.send_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #0a5b5f;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        
        # Quick actions row
        actions_layout = QHBoxLayout()
        
        quick_actions = [
            ("📈 Market Summary", "Give me a summary of today's market performance"),
            ("🔥 Top Movers", "What are the top gaining and losing stocks today?"),
            ("📰 Market News", "What are the most important market news today?"),
            ("💡 Trade Ideas", "Give me some trade ideas based on current market conditions")
        ]
        
        for text, prompt in quick_actions:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #606060;
                    border-radius: 5px;
                    padding: 5px 10px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #0d7377;
                }
            """)
            btn.clicked.connect(lambda checked, p=prompt: self.send_quick_message(p))
            actions_layout.addWidget(btn)
        
        layout.addLayout(input_layout)
        layout.addLayout(actions_layout)
        
        # Add welcome message
        self.add_message("👋 Hello! I'm your Perplexity AI trading assistant with real-time market access. I can help you with:\n\n" +
                        "📈 Real-time stock analysis and prices\n" +
                        "📊 Current market trends and news\n" +
                        "💡 Data-driven trading strategies\n" +
                        "🔍 Live market research and insights\n" +
                        "📰 Latest financial news and events\n\n" +
                        "Ask me anything about current markets!", False)
    
    def add_message(self, message, is_user=True):
        """Add a message to the chat"""
        # Remove the stretch before adding new message
        if self.chat_layout.count() > 0:
            self.chat_layout.removeItem(self.chat_layout.itemAt(self.chat_layout.count() - 1))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        message_with_time = f"[{timestamp}] {message}"
        
        # Create message widget
        message_widget = ChatMessage(message_with_time, is_user)
        self.chat_layout.addWidget(message_widget)
        
        # Add stretch to push messages to bottom
        self.chat_layout.addStretch()
        
        # Scroll to bottom
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )
        
        # Store in history
        self.chat_history.append({
            "role": "user" if is_user else "assistant",
            "content": message,
            "timestamp": timestamp
        })
    
    def send_message(self):
        """Send user message and get AI response"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Disable input while processing
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.message_input.clear()
        
        # Add user message
        self.add_message(message, True)
        
        # Check if Perplexity API key is configured
        if not self.perplexity_api_key:
            self.add_message("❌ Perplexity API key not configured. Please set PERPLEXITY_API_KEY in your .env file.", False)
            self.enable_input()
            return
        
        # Add thinking message
        thinking_message = ChatMessage("🤔 Thinking...", False)
        if self.chat_layout.count() > 0:
            self.chat_layout.removeItem(self.chat_layout.itemAt(self.chat_layout.count() - 1))
        self.chat_layout.addWidget(thinking_message)
        self.chat_layout.addStretch()
        
        # Start AI request thread
        self.chat_thread = ChatThread(message, self.perplexity_api_key)
        self.chat_thread.response_ready.connect(self.on_response_received)
        self.chat_thread.error_occurred.connect(self.on_error_occurred)
        self.chat_thread.start()
    
    def on_response_received(self, response):
        """Handle AI response"""
        # Remove thinking message
        if self.chat_layout.count() > 0:
            thinking_widget = self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()
            if thinking_widget:
                thinking_widget.setParent(None)
        
        # Add AI response
        self.add_message(response, False)
        self.enable_input()
    
    def on_error_occurred(self, error):
        """Handle AI request error"""
        # Remove thinking message
        if self.chat_layout.count() > 0:
            thinking_widget = self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()
            if thinking_widget:
                thinking_widget.setParent(None)
        
        # Add error message
        self.add_message(f"❌ Error: {error}", False)
        self.enable_input()
    
    def send_quick_message(self, message):
        """Send a predefined quick message"""
        self.message_input.setText(message)
        self.send_message()
    
    def enable_input(self):
        """Re-enable input after processing"""
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.message_input.setFocus()
    
    def refresh_data(self):
        """Refresh chat data (placeholder)"""
        pass