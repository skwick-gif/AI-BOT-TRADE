"""
Chat Widget for AI Agent Communication
Based on the provided PyQt6 chat example with message bubbles
Enhanced with special trading commands support
"""

import asyncio
import time
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QScrollArea, QFrame, QLabel, QSizePolicy,
    QMessageBox, QSpinBox, QComboBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QTextCursor, QPalette

from core.config_manager import ConfigManager
from services.ai_service import AIService
from services.ibkr_service import IBKRService
from utils.logger import get_logger
from pathlib import Path
import re


class MessageBubble(QFrame):
    """Individual message bubble widget"""
    
    def __init__(self, message: str, is_user: bool = True, timestamp: str = ""):
        super().__init__()
        self.is_user = is_user
        self.setup_ui(message, timestamp)
    
    def setup_ui(self, message: str, timestamp: str):
        """Setup message bubble UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        
        # Set different styles for user vs AI messages
        if self.is_user:
            self.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                                stop: 0 #4CAF50, stop: 1 #45a049);
                    border: 2px solid #4CAF50;
                    border-radius: 15px;
                    margin: 5px;
                    padding: 10px;
                }
                QLabel {
                    background: transparent;
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                                stop: 0 #3d3d3d, stop: 1 #2b2b2b);
                    border: 2px solid #3d3d3d;
                    border-radius: 15px;
                    margin: 5px;
                    padding: 10px;
                }
                QLabel {
                    background: transparent;
                    color: white;
                }
            """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Message content
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Arial", 10))
        layout.addWidget(message_label)
        
        # Timestamp
        if timestamp:
            time_label = QLabel(timestamp)
            time_label.setFont(QFont("Arial", 8))
            time_label.setAlignment(Qt.AlignmentFlag.AlignRight if self.is_user else Qt.AlignmentFlag.AlignLeft)
            time_label.setStyleSheet("color: #cccccc;")
            layout.addWidget(time_label)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)


class MessageContainer(QWidget):
    """Container for message bubbles with proper alignment"""
    
    def __init__(self, message: str, is_user: bool = True, timestamp: str = ""):
        super().__init__()
        self.setup_ui(message, is_user, timestamp)
    
    def setup_ui(self, message: str, is_user: bool, timestamp: str):
        """Setup message container"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 5, 20, 5)
        
        # Create message bubble
        bubble = MessageBubble(message, is_user, timestamp)
        bubble.setMaximumWidth(400)  # Limit bubble width
        
        if is_user:
            # User messages on the right
            layout.addStretch()
            layout.addWidget(bubble)
        else:
            # AI messages on the left
            layout.addWidget(bubble)
            layout.addStretch()


class AIWorker(QObject):
    """Worker thread for AI processing"""
    
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    api_test_result = pyqtSignal(bool, str)
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.ai_service = None
        self.ibkr_service = None
        self.logger = get_logger("AIWorker")
        self.loop = None  # Will initialize on first call in worker thread
        
        # Import trading helpers
        from utils.trading_helpers import (
            get_portfolio_positions, get_current_price, get_technicals,
            get_company_name, get_financials, get_stock_recommendation,
            get_small_cap_stocks, fetch_reddit_posts, fetch_news,
            get_government_official_trades, get_polymarket_odds,
            update_portfolio
        )
        
        # Import IBKR trading helpers
        from utils.ibkr_trading_helpers import (
            booktrade_ibkr, create_bracket_order, cancel_order,
            get_open_orders, sync_portfolio_with_ibkr
        )
        
        self.trading_helpers = {
            'get_portfolio_positions': get_portfolio_positions,
            'get_current_price': get_current_price,
            'get_technicals': get_technicals,
            'get_company_name': get_company_name,
            'get_financials': get_financials,
            'get_stock_recommendation': get_stock_recommendation,
            'get_small_cap_stocks': get_small_cap_stocks,
            'fetch_reddit_posts': fetch_reddit_posts,
            'fetch_news': fetch_news,
            'get_government_official_trades': get_government_official_trades,
            'get_polymarket_odds': get_polymarket_odds,
            'update_portfolio': update_portfolio,
            'booktrade_ibkr': booktrade_ibkr,
            'create_bracket_order': create_bracket_order,
            'cancel_order': cancel_order,
            'get_open_orders': get_open_orders,
            'sync_portfolio_with_ibkr': sync_portfolio_with_ibkr
        }
    
    def process_message(self, message: str):
        """Process user message and generate AI response"""
        try:
            # Ensure a persistent event loop bound to this worker thread
            if not self.loop:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            # Check for special commands first
            response = self.handle_special_commands(message)
            
            if response:
                self.response_ready.emit(response)
            else:
                # Use AI service for general queries
                # Use persistent loop for lower latency
                try:
                    # Initialize AI service if needed
                    if not self.ai_service:
                        self.ai_service = AIService(self.config)
                    
                    # Get portfolio context
                    context = self.get_trading_context()
                    
                    # Get AI response
                    response = self.loop.run_until_complete(
                        self.ai_service.get_ai_response(message, context)
                    )
                    
                    self.response_ready.emit(response)
                    
                finally:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.error_occurred.emit(f"Error processing your message: {e}")

    def test_api(self):
        """Perform a lightweight Perplexity API connectivity test"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if not self.ai_service:
                    self.ai_service = AIService(self.config)
                # Simple prompt to validate response
                result = loop.run_until_complete(
                    self.ai_service.get_ai_response("Ping", {})
                )
                # If the service returns a friendly error about missing key, surface it
                if isinstance(result, str) and result.lower().startswith("perplexity api key not configured"):
                    self.api_test_result.emit(False, "Perplexity API key not configured.")
                    self.error_occurred.emit("Perplexity API key not configured. Please set PERPLEXITY_API_KEY in your .env and restart the app.")
                else:
                    # Success
                    self.api_test_result.emit(True, "Perplexity API is reachable.")
                    self.response_ready.emit("✅ Perplexity API is reachable.")
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"API test failed: {e}")
            self.api_test_result.emit(False, f"API test failed: {e}")
            self.error_occurred.emit(f"API test failed: {e}")
    
    def handle_special_commands(self, message: str) -> Optional[str]:
        """Handle special trading commands"""
        message_lower = message.lower()
        
        try:
            # Get technicals command
            if "get technicals" in message_lower:
                return self.handle_technicals_command(message)
            
            # Analyze command
            elif "analyze" in message_lower:
                return self.handle_analyze_command(message)
            
            # Find stocks command
            elif "find stocks" in message_lower:
                return self.handle_find_stocks_command()
            
            # Portfolio status command
            elif any(phrase in message_lower for phrase in ["portfolio status", "my portfolio", "current positions"]):
                return self.handle_portfolio_status_command()
            
            # Government trades command
            elif "government trades" in message_lower or "house and senate" in message_lower:
                return self.handle_government_trades_command(message)
            
            # Buy command with IBKR execution
            elif "buy" in message_lower and any(word in message_lower for word in ["shares", "stock"]):
                return self.handle_buy_command(message)
            
            # Sell command with IBKR execution
            elif "sell" in message_lower and any(word in message_lower for word in ["shares", "stock", "position"]):
                return self.handle_sell_command(message)
            
            # Open orders command
            elif "open orders" in message_lower or "my orders" in message_lower:
                return self.handle_open_orders_command()
            
            # Sync portfolio command
            elif "sync portfolio" in message_lower or "sync positions" in message_lower:
                return self.handle_sync_portfolio_command()
            
            # Cancel order command
            elif "cancel order" in message_lower:
                return self.handle_cancel_order_command(message)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling special command: {e}")
            return f"Error processing command: {e}"
    
    def handle_technicals_command(self, message: str) -> str:
        """Handle 'get technicals [TICKER]' command"""
        try:
            # Extract ticker from message
            words = message.split()
            ticker = None
            for word in words:
                if word.isupper() and len(word) <= 5:
                    ticker = word
                    break
            
            if not ticker:
                # Try to get last word as ticker
                ticker = words[-1].upper()
            
            if not ticker:
                return "Please specify a ticker symbol. Example: 'get technicals AAPL'"
            
            # Get technical data
            technicals_df = self.trading_helpers['get_technicals'](ticker)
            
            if technicals_df.empty:
                return f"Could not retrieve technical data for {ticker}. Please check the ticker symbol."
            
            # Extract latest technical indicators
            latest = technicals_df.iloc[-1]
            price = latest['Close']
            
            response = f"**Technical Analysis for {ticker}**\n\n"
            response += f"Current Price: ${price:.2f}\n"
            
            if 'SMA_20' in latest:
                sma_20 = latest['SMA_20']
                response += f"20-day SMA: ${sma_20:.2f} ({'Above' if price > sma_20 else 'Below'})\n"
            
            if 'SMA_50' in latest:
                sma_50 = latest['SMA_50']
                response += f"50-day SMA: ${sma_50:.2f} ({'Above' if price > sma_50 else 'Below'})\n"
            
            if 'RSI_14' in latest:
                rsi = latest['RSI_14']
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                response += f"RSI (14): {rsi:.2f} ({rsi_signal})\n"
            
            if 'MACD_12_26_9' in latest:
                macd = latest['MACD_12_26_9']
                macd_signal = latest.get('MACDs_12_26_9', 0)
                macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                response += f"MACD: {macd:.4f} ({macd_trend})\n"
            
            return response
            
        except Exception as e:
            return f"Error getting technical data: {e}"
    
    def handle_analyze_command(self, message: str) -> str:
        """Handle 'analyze [TICKER]' command"""
        try:
            # Extract ticker
            words = message.split()
            ticker = words[-1].upper()
            
            if len(ticker) > 5:
                return "Please provide a valid ticker symbol. Example: 'analyze AAPL'"
            
            # Get financial data
            financials = self.trading_helpers['get_financials'](ticker)
            
            if not financials:
                return f"Could not retrieve financial data for {ticker}"
            
            # Get recommendation
            recommendation = self.trading_helpers['get_stock_recommendation'](ticker, financials)
            
            return recommendation
            
        except Exception as e:
            return f"Error analyzing {ticker}: {e}"
    
    def handle_find_stocks_command(self) -> str:
        """Handle 'find stocks' command"""
        try:
            tickers = self.trading_helpers['get_small_cap_stocks']()
            
            if tickers:
                response = "Here are some interesting stocks to consider:\n\n"
                for ticker in tickers:
                    price = self.trading_helpers['get_current_price'](ticker)
                    company = self.trading_helpers['get_company_name'](ticker)
                    if price:
                        response += f"• **{ticker}** ({company}): ${price:.2f}\n"
                    else:
                        response += f"• **{ticker}** ({company})\n"
                
                response += "\nWould you like me to analyze any of these stocks? Just say 'analyze [TICKER]'"
                return response
            else:
                return "Sorry, I couldn't fetch any stock suggestions at the moment."
                
        except Exception as e:
            return f"Error finding stocks: {e}"
    
    def handle_portfolio_status_command(self) -> str:
        """Handle portfolio status commands"""
        try:
            positions = self.trading_helpers['get_portfolio_positions']()
            
            if not positions:
                return "Your portfolio is currently empty. Start by analyzing some stocks!"
            
            response = "**Your Current Portfolio:**\n\n"
            total_value = 0
            
            for ticker, shares in positions.items():
                current_price = self.trading_helpers['get_current_price'](ticker)
                if current_price:
                    position_value = shares * current_price
                    total_value += position_value
                    response += f"• **{ticker}**: {shares} shares @ ${current_price:.2f} = ${position_value:.2f}\n"
                else:
                    response += f"• **{ticker}**: {shares} shares (price unavailable)\n"
            
            if total_value > 0:
                response += f"\n**Total Portfolio Value: ${total_value:.2f}**"
            
            return response
            
        except Exception as e:
            return f"Error getting portfolio status: {e}"
    
    def handle_government_trades_command(self, message: str) -> str:
        """Handle government trades command"""
        try:
            # Extract ticker
            words = message.split()
            ticker = None
            for word in words:
                if word.isupper() and len(word) <= 5:
                    ticker = word
                    break
            
            if not ticker:
                ticker = words[-1].upper()
            
            if not ticker or len(ticker) > 5:
                return "Please specify a ticker symbol. Example: 'check government trades AAPL'"
            
            trades = self.trading_helpers['get_government_official_trades'](ticker)
            return f"Government official trades for {ticker}:\n\n{trades}"
            
        except Exception as e:
            return f"Error checking government trades: {e}"
    
    def handle_sell_command(self, message: str) -> str:
        """Handle sell commands with IBKR execution"""
        try:
            # Extract ticker and optionally shares
            words = message.split()
            ticker = None
            shares_to_sell = None
            
            for i, word in enumerate(words):
                if word.isupper() and len(word) <= 5:
                    ticker = word
                    # Check if next word is a number (shares)
                    if i + 1 < len(words) and words[i + 1].isdigit():
                        shares_to_sell = int(words[i + 1])
                    break
            
            if not ticker:
                return "Please specify which stock to sell. Example: 'sell AAPL' or 'sell 100 AAPL'"
            
            positions = self.trading_helpers['get_portfolio_positions']()
            
            if ticker not in positions:
                return f"You don't currently own any shares of {ticker}."
            
            available_shares = positions[ticker]
            
            # If no specific shares mentioned, sell all
            if shares_to_sell is None:
                shares_to_sell = available_shares
            elif shares_to_sell > available_shares:
                return f"You only own {available_shares} shares of {ticker}, cannot sell {shares_to_sell}."
            
            # Execute sell order through IBKR
            result = self.trading_helpers['booktrade_ibkr'](
                ticker=ticker,
                shares=shares_to_sell,
                action="SELL",
                order_type="MKT"
            )
            
            if result.success:
                return f"✅ {result.message}"
            else:
                return f"❌ Failed to sell {ticker}: {result.message}"
                
        except Exception as e:
            return f"Error processing sell command: {e}"
    
    def handle_buy_command(self, message: str) -> str:
        """Handle buy commands with IBKR execution"""
        try:
            # Extract ticker and shares
            words = message.split()
            ticker = None
            shares_to_buy = None
            
            for i, word in enumerate(words):
                if word.isdigit():
                    shares_to_buy = int(word)
                elif word.isupper() and len(word) <= 5:
                    ticker = word
            
            if not ticker:
                return "Please specify which stock to buy. Example: 'buy 100 AAPL'"
            
            if not shares_to_buy:
                return "Please specify how many shares to buy. Example: 'buy 100 AAPL'"
            
            # Get current price for reference
            current_price = self.trading_helpers['get_current_price'](ticker)
            if current_price:
                total_cost = shares_to_buy * current_price
                response = f"Attempting to buy {shares_to_buy} shares of {ticker} at ~${current_price:.2f} (Total: ~${total_cost:.2f})\n\n"
            else:
                response = f"Attempting to buy {shares_to_buy} shares of {ticker}\n\n"
            
            # Execute buy order through IBKR
            result = self.trading_helpers['booktrade_ibkr'](
                ticker=ticker,
                shares=shares_to_buy,
                action="BUY",
                order_type="MKT"
            )
            
            if result.success:
                response += f"✅ {result.message}"
            else:
                response += f"❌ Failed to buy {ticker}: {result.message}"
            
            return response
                
        except Exception as e:
            return f"Error processing buy command: {e}"
    
    def handle_open_orders_command(self) -> str:
        """Handle open orders command"""
        try:
            orders = self.trading_helpers['get_open_orders']()
            
            if not orders:
                return "You have no open orders."
            
            response = f"**Your Open Orders ({len(orders)}):**\n\n"
            
            for order in orders:
                response += f"• **Order #{order['order_id']}**: {order['action']} {order['quantity']} shares of {order['symbol']}\n"
                response += f"  Type: {order['order_type']} | Status: {order['status']}\n"
                if order.get('limit_price'):
                    response += f"  Limit Price: ${order['limit_price']:.2f}\n"
                if order.get('stop_price'):
                    response += f"  Stop Price: ${order['stop_price']:.2f}\n"
                response += f"  Filled: {order['filled']} | Remaining: {order['remaining']}\n\n"
            
            response += "To cancel an order, say 'cancel order [ORDER_ID]'"
            return response
            
        except Exception as e:
            return f"Error getting open orders: {e}"
    
    def handle_sync_portfolio_command(self) -> str:
        """Handle sync portfolio command"""
        try:
            result = self.trading_helpers['sync_portfolio_with_ibkr']()
            
            if result['success']:
                response = f"**Portfolio Sync Results:**\n\n"
                response += f"IBKR Positions: {result['ibkr_positions']}\n"
                response += f"Local Positions: {result['local_positions']}\n"
                response += f"Discrepancies: {len(result['discrepancies'])}\n\n"
                
                if result['discrepancies']:
                    response += "**Discrepancies Found:**\n"
                    for disc in result['discrepancies']:
                        response += f"• {disc['symbol']}: IBKR={disc['ibkr_shares']}, Local={disc['local_shares']} (Diff: {disc['difference']})\n"
                else:
                    response += "✅ All positions are synchronized!"
                
                return response
            else:
                return f"❌ Sync failed: {result['message']}"
                
        except Exception as e:
            return f"Error syncing portfolio: {e}"
    
    def handle_cancel_order_command(self, message: str) -> str:
        """Handle cancel order command"""
        try:
            # Extract order ID
            words = message.split()
            order_id = None
            
            for word in words:
                if word.isdigit():
                    order_id = int(word)
                    break
            
            if not order_id:
                return "Please specify an order ID to cancel. Example: 'cancel order 12345'"
            
            result = self.trading_helpers['cancel_order'](order_id)
            
            if result.success:
                return f"✅ {result.message}"
            else:
                return f"❌ {result.message}"
                
        except Exception as e:
            return f"Error canceling order: {e}"
    
    def get_trading_context(self) -> Dict[str, Any]:
        """Get current trading context for AI"""
        context = {}
        
        try:
            # Get portfolio positions
            positions = self.trading_helpers['get_portfolio_positions']()
            if positions:
                # To reduce latency, avoid network price fetch here.
                # Provide a lightweight snapshot of up to 5 holdings.
                top = []
                for i, (ticker, shares) in enumerate(positions.items()):
                    if i >= 5:
                        break
                    top.append({'symbol': ticker, 'shares': shares})
                context['portfolio'] = top
                context['portfolio_count'] = len(positions)
            
            # Add market timestamp
            context['timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error getting trading context: {e}")
        
        return context


class ChatWidget(QWidget):
    """Main chat widget for AI agent interaction"""
    # Ensure heavy work runs off the GUI thread by emitting to the worker in its thread
    message_submitted = pyqtSignal(str)
    api_test_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("ChatWidget")
        self.config = ConfigManager()
        
        # Initialize AI worker
        self.ai_thread = QThread()
        self.ai_worker = AIWorker(self.config)
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_worker.response_ready.connect(self.add_ai_message)
        self.ai_worker.error_occurred.connect(self.add_error_message)
        # Listen to API test results and update a visible status indicator
        self.ai_worker.api_test_result.connect(self.on_api_test_result)
        # Route message submissions to the worker thread
        self.message_submitted.connect(self.ai_worker.process_message)
        # Route API test requests to the worker thread
        self.api_test_requested.connect(self.ai_worker.test_api)
        self.ai_thread.start()
        
        # Setup UI
        self.setup_ui()
        # Internal flags
        self.testing_api = False
        
        # Add welcome message
        self.add_ai_message("Hello! I'm your AI trading assistant. Please configure your API keys and connect to your trading platform to get started with personalized trading insights.")
        
        self.logger.info("Chat widget initialized")
    
    def setup_ui(self):
        """Setup the chat UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("AI Trading Assistant")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create scrollable chat area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Chat content widget
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(5)
        self.chat_layout.addStretch()  # Push messages to bottom initially
        
        self.scroll_area.setWidget(self.chat_content)
        layout.addWidget(self.scroll_area)
        
        # Input area
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.Shape.Box)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)
        
        # Message input
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setFont(QFont("Arial", 10))
        self.message_input.setMinimumHeight(40)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setMinimumSize(80, 40)
        self.send_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        # Test API button
        self.test_api_button = QPushButton("Test API")
        self.test_api_button.setMinimumSize(90, 40)
        self.test_api_button.setFont(QFont("Arial", 10))
        self.test_api_button.setToolTip("Test Perplexity API connectivity")
        self.test_api_button.clicked.connect(self.test_api)
        input_layout.addWidget(self.test_api_button)
        
        # API status label (shows model and connectivity state)
        self.api_status_label = QLabel("API: Unknown")
        self.api_status_label.setMinimumHeight(40)
        self.api_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.api_status_label.setStyleSheet("color: #888;")
        input_layout.addWidget(self.api_status_label)

        # Model selector (quick switcher)
        self.model_selector = QComboBox()
        self.model_selector.setMinimumHeight(40)
        self.model_selector.setToolTip("Select Perplexity model")
        # Common models list; you can adjust according to your account
        models = [
            "reasoning-pro",
            "sonar",
            "sonar-pro",
            "sonar-small",
        ]
        self.model_selector.addItems(models)
        # Set current selection from config if present
        current_model = (self.config.perplexity.model or "reasoning-pro").strip()
        if current_model in models:
            self.model_selector.setCurrentText(current_model)
        else:
            self.model_selector.setCurrentText("reasoning-pro")
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        input_layout.addWidget(self.model_selector)
        
        layout.addWidget(input_frame)
        
        # Quick actions
        self.create_quick_actions()
        layout.addWidget(self.quick_actions_frame)
    
    def create_quick_actions(self):
        """Create quick action buttons"""
        self.quick_actions_frame = QFrame()
        self.quick_actions_frame.setFrameStyle(QFrame.Shape.Box)
        actions_layout = QHBoxLayout(self.quick_actions_frame)
        actions_layout.setContentsMargins(10, 5, 10, 5)
        
        # Quick action buttons
        actions = [
            ("📊 Portfolio Status", "What is my current portfolio status?"),
            ("📈 Market Update", "What's the current market situation?"),
            ("🎯 Trading Help", "I need help with trading decisions"),
            ("⚠️ Risk Analysis", "Please analyze my portfolio risk")
        ]
        
        for text, message in actions:
            btn = QPushButton(text)
            btn.setFont(QFont("Arial", 9))
            btn.clicked.connect(lambda checked, msg=message: self.send_quick_message(msg))
            actions_layout.addWidget(btn)
    
    def add_message_to_chat(self, message: str, is_user: bool = True):
        """Add a message to the chat"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M")
        
        # Create message container
        message_container = MessageContainer(message, is_user, timestamp)
        
        # Insert before the stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_container)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def add_user_message(self, message: str):
        """Add user message to chat"""
        self.add_message_to_chat(message, is_user=True)
        self.logger.debug(f"User message added: {message[:50]}...")
    
    def add_ai_message(self, message: str):
        """Add AI response to chat"""
        self.add_message_to_chat(message, is_user=False)
        self.logger.debug(f"AI message added: {message[:50]}...")
        
        # Re-enable send button
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        # Re-enable Test API button if we were testing
        if getattr(self, 'testing_api', False):
            self.testing_api = False
            self.test_api_button.setEnabled(True)
            self.test_api_button.setText("Test API")
    
    def add_error_message(self, error: str):
        """Add error message to chat"""
        error_msg = f"❌ Error: {error}"
        self.add_message_to_chat(error_msg, is_user=False)
        
        # Re-enable send button
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        # Re-enable Test API button if we were testing
        if getattr(self, 'testing_api', False):
            self.testing_api = False
            self.test_api_button.setEnabled(True)
            self.test_api_button.setText("Test API")
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_message(self):
        """Send user message and get AI response"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Add user message to chat
        self.add_user_message(message)
        
        # Clear input
        self.message_input.clear()
        
        # Disable send button while processing
        self.send_button.setEnabled(False)
        self.send_button.setText("Thinking...")
        
        # Process with AI worker in its own thread (queued connection)
        self.message_submitted.emit(message)
        
        self.logger.info(f"Message sent: {message[:50]}...")

    def test_api(self):
        """Trigger a Perplexity API connectivity test via the worker thread"""
        # Add info message to chat
        self.add_message_to_chat("🔌 Testing Perplexity API connectivity...", is_user=False)
        # Disable button while testing and update status label
        self.testing_api = True
        if hasattr(self, 'test_api_button'):
            self.test_api_button.setEnabled(False)
            self.test_api_button.setText("Testing…")
        if hasattr(self, 'api_status_label'):
            self.api_status_label.setText(f"API: Testing ({self.config.perplexity.model})…")
            self.api_status_label.setStyleSheet("color: #2196F3;")
        # Invoke worker method in its own thread
        self.api_test_requested.emit()

    def on_api_test_result(self, ok: bool, message: str):
        """Handle API test result and update status label"""
        self.testing_api = False
        if hasattr(self, 'test_api_button'):
            self.test_api_button.setEnabled(True)
            self.test_api_button.setText("Test API")
        if ok:
            self.api_status_label.setText(f"API: OK ({self.config.perplexity.model})")
            self.api_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.api_status_label.setText(f"API: Error — {message}")
            self.api_status_label.setStyleSheet("color: #f44336; font-weight: bold;")

    def on_model_changed(self, model: str):
        """Update model in runtime config and persist to .env"""
        try:
            model = (model or "").strip()
            if not model:
                return
            # Update runtime config
            self.config.perplexity.model = model
            # Reflect immediately in status label if it shows a model name
            if self.api_status_label.text().startswith("API:") and "(" in self.api_status_label.text():
                base = self.api_status_label.text().split("(")[0].strip()
                self.api_status_label.setText(f"{base}({model})")
            # Persist to .env
            self._persist_env_var("PERPLEXITY_MODEL", model)
            self.logger.info(f"Perplexity model changed to: {model}")
        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")

    def _persist_env_var(self, key: str, value: str):
        """Persist a single key=value in project .env (update or add)."""
        try:
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            lines = []
            if env_path.exists():
                content = env_path.read_text(encoding="utf-8").splitlines()
                pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
                updated = False
                for line in content:
                    if pattern.match(line):
                        lines.append(f"{key}={value}")
                        updated = True
                    else:
                        lines.append(line)
                if not updated:
                    lines.append(f"{key}={value}")
            else:
                lines = [f"{key}={value}"]
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            # Non-fatal; UI already updated runtime config
            self.logger.error(f"Failed to persist {key} to .env: {e}")
    
    def send_quick_message(self, message: str):
        """Send a quick action message"""
        self.message_input.setText(message)
        self.send_message()
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Chat tab activated")
        self.message_input.setFocus()
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop AI worker thread safely
        try:
            if hasattr(self, 'ai_thread') and self.ai_thread.isRunning():
                self.ai_thread.quit()
                self.ai_thread.wait()
        except Exception:
            pass
        event.accept()