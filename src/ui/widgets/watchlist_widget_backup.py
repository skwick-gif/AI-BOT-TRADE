from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QLineEdit, QHeaderView,
    QMenu, QMessageBox, QSplitter, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QAction
from datetime import datetime
import json
import re

# Optional imports with fallbacks
try:
    from core.config_manager import ConfigManager
    from utils.logger import get_logger
except ImportError:
    ConfigManager = None
    get_logger = None

class WatchlistTable(QTableWidget):
    symbol_selected = pyqtSignal(str)
    trade_requested = pyqtSignal(str)
    tools_action = pyqtSignal(str, str)  # action, symbol

    def __init__(self):
        super().__init__()
        self.symbol_data = {}
        self.setup_ui()

    def setup_ui(self):
        headers = [
            "Symbol", "נוסף בתאריך", "Price", "Volume",
            "יום 1", "יום 2", "יום 3", "יום 4", "יום 5", "יום 6", "יום 7",
            "AI Rating", "Price Target", "Stop Loss", "Signal", "Sharpe Ratio", "Tools"
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

    def add_symbol(self, symbol: str):
        row = self.rowCount()
        self.insertRow(row)
        symbol_item = QTableWidgetItem(symbol)
        symbol_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.setItem(row, 0, symbol_item)
        added_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        added_item = QTableWidgetItem(added_at)
        added_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 1, added_item)
        for col in range(2, self.columnCount()):
            item = QTableWidgetItem("-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, col, item)
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
            btn.setAutoRaise(True)
            btn.setStyleSheet("font-size: 18px; padding: 0 4px;")
            return btn
        btn_ai = mk_btn("Ask AI for this symbol", "🤖")
        btn_chart = mk_btn("Open chart for this symbol", "📈")
        btn_ai.clicked.connect(lambda _, s=symbol: self.tools_action.emit('ai', s))
        btn_chart.clicked.connect(lambda _, s=symbol: self.tools_action.emit('chart', s))
        hl.addWidget(btn_ai)
        hl.addWidget(btn_chart)
        hl.addStretch(1)
        self.setCellWidget(row, self.columnCount() - 1, tools_widget)
        self.symbol_data[symbol] = {'row': row, 'added_at': added_at, 'prices': [], 'last_date': None}
        try:
            self.prefill_day_prices(symbol)
        except Exception:
            pass
        try:
            prices = self.symbol_data.get(symbol, {}).get('prices', [])
            price_text = self.item(row, 2).text() if self.item(row, 2) else "-"
            if (not prices) and (price_text == "-"):
                QMessageBox.information(self, "No Data", f"No local data found for {symbol}. The row was added and will auto-fill when data becomes available.")
        except Exception:
            pass
        return True

    def open_chart_for_symbol(self, symbol):
        self.symbol_selected.emit(symbol)

    def prefill_day_prices(self, symbol: str):
        from pathlib import Path
        import pandas as pd
        bronze_dir = Path("data/bronze/daily")
        fp = bronze_dir / f"{symbol}.parquet"
        if not fp.exists():
            return
        df = pd.read_parquet(fp)
        if df.empty:
            return
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date')
        else:
            return
        added_at_str = self.symbol_data[symbol]['added_at']
        added_dt = datetime.strptime(added_at_str, "%Y-%m-%d %H:%M")
        start_date = added_dt.date()
        row_idx = self.symbol_data[symbol]['row']
        day_row = df[df['date'].dt.date == start_date]
        if not day_row.empty:
            price_val = float(day_row.iloc[0].get('close', day_row.iloc[0].get('adj_close', day_row.iloc[0].get('price', 0))) )
            self.item(row_idx, 2).setText(f"{price_val:.2f}")
        else:
            self.item(row_idx, 2).setText("-")
        prices = []
        prev_price = None
        for i in range(7):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            cell = self.item(row_idx, 4 + i)
            if not next_row.empty:
                pval = float(next_row.iloc[0].get('close', next_row.iloc[0].get('adj_close', next_row.iloc[0].get('price', 0))) )
                prices.append(pval)
                cell.setText(f"{pval:.2f}")
                if prev_price is not None:
                    if pval > prev_price:
                        cell.setBackground(QColor(200, 255, 200))
                    elif pval < prev_price:
                        cell.setBackground(QColor(255, 200, 200))
                prev_price = pval
            else:
                prices.append(None)
                cell.setText("-")
                cell.setBackground(QColor(255, 255, 255))
        self.symbol_data[symbol]['prices'] = [p for p in prices if p is not None]
        last_dt = None
        for i in reversed(range(7)):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            if not next_row.empty:
                last_dt = next_row.iloc[0]['date']
                break
        self.symbol_data[symbol]['last_date'] = last_dt

    def request_ai_rating(self, symbol, row):
        import os
        import requests
        api_key = os.getenv('PERPLEXITY_API_KEY', '')
        model = os.getenv('PERPLEXITY_MODEL', 'sonar-small-chat')
        if not api_key:
            if self.item(row, 11):
                self.item(row, 11).setToolTip("Perplexity API key not configured. Set PERPLEXITY_API_KEY in .env")
            return
        price = self.item(row, 2).text()
        prices = [self.item(row, 4 + i).text() for i in range(7)]
        signal = self.item(row, 14).text() if self.item(row, 14) else '-'
        price_target_existing = self.item(row, 12).text() if self.item(row, 12) else ''
        stop_loss = self.item(row, 13).text() if self.item(row, 13) else '-'
        sharpe = self.item(row, 15).text() if self.item(row, 15) else '-'
        system_msg = (
            "You are an AI trading assistant. Return a compact JSON object only. "
            "Schema: {rating: string in [\"Strong Buy\", \"Buy\", \"Hold\", \"Sell\", \"Strong Sell\"], "
            "price_target: number or empty string, explanation: string}."
        )
        user_msg = (
            f"Analyze stock {symbol}. Current price: {price}. 7-day prices: {prices}. "
            f"Signal: {signal}. Existing price target: {price_target_existing}. Stop Loss: {stop_loss}. Sharpe: {sharpe}. "
            "Return JSON per schema."
        )
        try:
            resp = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': system_msg},
                        {'role': 'user', 'content': user_msg}
                    ],
                    'temperature': 0.2
                },
                timeout=20
            )
            data = resp.json()
            try:
                reply = data['choices'][0]['message']['content']
            except Exception:
                reply = json.dumps({'rating': '-', 'price_target': '', 'explanation': 'No content'})
            parsed = None
            try:
                parsed = json.loads(reply)
            except Exception:
                m = re.search(r"\{.*\}", reply, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = {'rating': '-', 'price_target': '', 'explanation': reply[:200]}
                else:
                    parsed = {'rating': '-', 'price_target': '', 'explanation': reply[:200]}
            rating = parsed.get('rating', '-')
            price_target_resp = parsed.get('price_target', '')
            explanation = parsed.get('explanation', '')
        except Exception as e:
            rating = '-'
            price_target_resp = ''
            explanation = f"AI error: {e}"
        self.setItem(row, 11, QTableWidgetItem(str(rating)))
        self.setItem(row, 12, QTableWidgetItem(str(price_target_resp)))
        if self.item(row, 11):
            self.item(row, 11).setToolTip(explanation)

class WatchlistDetails(QFrame):
    ask_ai_clicked = pyqtSignal(str)
    chart_clicked = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_symbol = None
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMaximumHeight(200)
        layout = QVBoxLayout(self)
        self.title_label = QLabel("Symbol Details")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        details_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        self.symbol_label = QLabel("Symbol: -")
        self.price_label = QLabel("Price: -")
        self.change_label = QLabel("Change: -")
        left_layout.addWidget(self.symbol_label)
        left_layout.addWidget(self.price_label)
        left_layout.addWidget(self.change_label)
        details_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        self.volume_label = QLabel("Volume: -")
        self.bid_ask_label = QLabel("Bid/Ask: -")
        self.high_low_label = QLabel("High/Low: -")
        right_layout.addWidget(self.volume_label)
        right_layout.addWidget(self.bid_ask_label)
        right_layout.addWidget(self.high_low_label)
        details_layout.addLayout(right_layout)
        layout.addLayout(details_layout)
        actions_layout = QHBoxLayout()
        self.buy_button = QPushButton("Buy")
        self.sell_button = QPushButton("Sell")
        self.buy_button.setEnabled(False)
        self.sell_button.setEnabled(False)
        actions_layout.addWidget(self.buy_button)
        actions_layout.addWidget(self.sell_button)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

    def update_details(self, symbol: str, data: dict):
        self.current_symbol = symbol
        self.title_label.setText(f"{symbol} Details")
        self.symbol_label.setText(f"Symbol: {symbol}")
        self.price_label.setText(f"Price: ${data['price']:.2f}")
        change_text = f"Change: ${data['change']:+.2f} ({data['change_pct']:+.2f}%)"
        self.change_label.setText(change_text)
        if data['change'] >= 0:
            self.change_label.setStyleSheet("color: #4CAF50;")
        else:
            self.change_label.setStyleSheet("color: #f44336;")
        self.volume_label.setText(f"Volume: {data['volume']:,}")
        self.bid_ask_label.setText(f"Bid/Ask: ${data['bid']:.2f} / ${data['ask']:.2f}")
        self.high_low_label.setText(f"High/Low: ${data['high']:.2f} / ${data['low']:.2f}")
        self.buy_button.setEnabled(True)
        self.sell_button.setEnabled(True)

class WatchlistWidget(QWidget):
    trade_requested = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setup_ui()
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        title_layout = QHBoxLayout()
        title = QLabel("Stock Watchlist")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        title_layout.addStretch()
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL)")
        self.symbol_input.setMaximumWidth(150)
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
        layout.addLayout(title_layout)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.watchlist_table = WatchlistTable()
        self.watchlist_table.symbol_selected.connect(self.on_symbol_selected)
        self.watchlist_table.trade_requested.connect(self.trade_requested.emit)
        self.watchlist_table.tools_action.connect(self._handle_tools_action)
        table_frame = QFrame()
        table_frame.setFrameStyle(QFrame.Shape.Box)
        table_layout = QVBoxLayout(table_frame)
    # ...existing code...

class WatchlistTable(QTableWidget):
    symbol_selected = pyqtSignal(str)
    trade_requested = pyqtSignal(str)
    tools_action = pyqtSignal(str, str)  # action, symbol

    def __init__(self):
        super().__init__()
        self.symbol_data = {}
        self.setup_ui()

    def setup_ui(self):
        headers = [
            "Symbol", "נוסף בתאריך", "Price", "Volume",
            "יום 1", "יום 2", "יום 3", "יום 4", "יום 5", "יום 6", "יום 7",
            "AI Rating", "Price Target", "Stop Loss", "Signal", "Sharpe Ratio", "Tools"
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

    def add_symbol(self, symbol: str):
        row = self.rowCount()
        self.insertRow(row)
        # Symbol column
        symbol_item = QTableWidgetItem(symbol)
        symbol_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.setItem(row, 0, symbol_item)
        # Added At column
        added_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        added_item = QTableWidgetItem(added_at)
        added_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setItem(row, 1, added_item)
        # Initialize other columns with placeholder data
        for col in range(2, self.columnCount()):
            item = QTableWidgetItem("-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, col, item)
        # Tools cell with AI and Chart icons
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
            btn.setAutoRaise(True)
            btn.setStyleSheet("font-size: 18px; padding: 0 4px;")
            return btn
        btn_ai = mk_btn("Ask AI for this symbol", "🤖")
        btn_chart = mk_btn("Open chart for this symbol", "📈")
        btn_ai.clicked.connect(lambda _, s=symbol: self.tools_action.emit('ai', s))
        btn_chart.clicked.connect(lambda _, s=symbol: self.tools_action.emit('chart', s))
        if df.empty:
            return
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date')
        else:
            return
        added_at_str = self.symbol_data[symbol]['added_at']
        added_dt = datetime.strptime(added_at_str, "%Y-%m-%d %H:%M")
        start_date = added_dt.date()
        row_idx = self.symbol_data[symbol]['row']
        day_row = df[df['date'].dt.date == start_date]
        if not day_row.empty:
            price_val = float(day_row.iloc[0].get('close', day_row.iloc[0].get('adj_close', day_row.iloc[0].get('price', 0))) )
            self.item(row_idx, 2).setText(f"{price_val:.2f}")
        else:
            self.item(row_idx, 2).setText("-")
        prices = []
        prev_price = None
        for i in range(7):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            cell = self.item(row_idx, 4 + i)
            if not next_row.empty:
                pval = float(next_row.iloc[0].get('close', next_row.iloc[0].get('adj_close', next_row.iloc[0].get('price', 0))) )
                prices.append(pval)
                cell.setText(f"{pval:.2f}")
                if prev_price is not None:
                    if pval > prev_price:
                        cell.setBackground(QColor(200, 255, 200))
                    elif pval < prev_price:
                        cell.setBackground(QColor(255, 200, 200))
                prev_price = pval
            else:
                prices.append(None)
                cell.setText("-")
                cell.setBackground(QColor(255, 255, 255))
        self.symbol_data[symbol]['prices'] = [p for p in prices if p is not None]
        last_dt = None
        for i in reversed(range(7)):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            if not next_row.empty:
                last_dt = next_row.iloc[0]['date']
                break
        self.symbol_data[symbol]['last_date'] = last_dt
        if df.empty:
            return
        # normalize date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date')
        else:
            return
        added_at_str = self.symbol_data[symbol]['added_at']
        added_dt = datetime.strptime(added_at_str, "%Y-%m-%d %H:%M")
        start_date = added_dt.date()
        # מצא את השורה של יום ההוספה
        day_row = df[df['date'].dt.date == start_date]
        row_idx = self.symbol_data[symbol]['row']
        # מחיר ליום ההוספה
        if not day_row.empty:
            price_val = float(day_row.iloc[0].get('close', day_row.iloc[0].get('adj_close', day_row.iloc[0].get('price', 0))))
            self.item(row_idx, 2).setText(f"{price_val:.2f}")
        else:
            self.item(row_idx, 2).setText("-")
        # מחירים לימים אחרי יום ההוספה (Day 1-7)
        prices = []
        for i in range(7):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            if not next_row.empty:
                pval = float(next_row.iloc[0].get('close', next_row.iloc[0].get('adj_close', next_row.iloc[0].get('price', 0))))
                prices.append(pval)
                self.item(row_idx, 4 + i).setText(f"{pval:.2f}")
            else:
                prices.append(None)
                self.item(row_idx, 4 + i).setText("-")
        self.symbol_data[symbol]['prices'] = [p for p in prices if p is not None]
    
    def calc_sharpe(self, prices):
        if len(prices) < 2:
            return '-'
        import numpy as np
        returns = np.diff(prices) / prices[:-1]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        rf = 0.0  # risk-free rate
        if std_ret == 0:
            return '-'
        return round((mean_ret - rf) / std_ret, 2)

    def calc_sortino(self, prices):
        if len(prices) < 2:
            return '-'
        import numpy as np
        returns = np.diff(prices) / prices[:-1]
        mean_ret = np.mean(returns)
        rf = 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return '-'
        std_down = np.std(downside)
        if std_down == 0:
            return '-'
        return round((mean_ret - rf) / std_down, 2)

    def calc_calmar(self, prices):
        if len(prices) < 2:
            return '-'
        import numpy as np
        returns = np.diff(prices) / prices[:-1]
        mean_ret = np.mean(returns)
        max_drawdown = self.max_drawdown(prices)
        if max_drawdown == 0:
            return '-'
        return round(mean_ret / max_drawdown, 2)

    def max_drawdown(self, prices):
        import numpy as np
        arr = np.array(prices)
        max_dd = 0
        peak = arr[0]
        for p in arr:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def show_context_menu(self, position):
        """Show context menu"""
        if self.itemAt(position) is None:
            return
        
        menu = QMenu(self)
        
        # Get selected symbol
        row = self.itemAt(position).row()
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            
            # Menu actions
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
    
    def get_symbols(self) -> list:
        """Get all symbols in the watchlist"""
        return list(self.symbol_data.keys())
    
    def prefill_day_prices(self, symbol: str):
        from pathlib import Path
        import pandas as pd
        bronze_dir = Path("data/bronze/daily")
        fp = bronze_dir / f"{symbol}.parquet"
        if not fp.exists():
            return
        df = pd.read_parquet(fp)
        if df.empty:
            return
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date')
        else:
            return
        added_at_str = self.symbol_data[symbol]['added_at']
        added_dt = datetime.strptime(added_at_str, "%Y-%m-%d %H:%M")
        start_date = added_dt.date()
        row_idx = self.symbol_data[symbol]['row']
        day_row = df[df['date'].dt.date == start_date]
        if not day_row.empty:
            price_val = float(day_row.iloc[0].get('close', day_row.iloc[0].get('adj_close', day_row.iloc[0].get('price', 0))) )
            self.item(row_idx, 2).setText(f"{price_val:.2f}")
        else:
            self.item(row_idx, 2).setText("-")
        prices = []
        prev_price = None
        for i in range(7):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            cell = self.item(row_idx, 4 + i)
            if not next_row.empty:
                pval = float(next_row.iloc[0].get('close', next_row.iloc[0].get('adj_close', next_row.iloc[0].get('price', 0))) )
                prices.append(pval)
                cell.setText(f"{pval:.2f}")
                if prev_price is not None:
                    if pval > prev_price:
                        cell.setBackground(QColor(200, 255, 200))
                    elif pval < prev_price:
                        cell.setBackground(QColor(255, 200, 200))
                prev_price = pval
            else:
                prices.append(None)
                cell.setText("-")
                cell.setBackground(QColor(255, 255, 255))
        self.symbol_data[symbol]['prices'] = [p for p in prices if p is not None]
        last_dt = None
        for i in reversed(range(7)):
            next_date = start_date + pd.Timedelta(days=i)
            next_row = df[df['date'].dt.date == next_date]
            if not next_row.empty:
                last_dt = next_row.iloc[0]['date']
                break
        self.symbol_data[symbol]['last_date'] = last_dt


class WatchlistDetails(QFrame):
    """Details panel for selected symbol"""
    ask_ai_clicked = pyqtSignal(str)
    chart_clicked = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_symbol = None
    
    def setup_ui(self):
        """Setup details panel"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMaximumHeight(200)
        
        layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel("Symbol Details")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        
        # Details layout
        details_layout = QHBoxLayout()
        
        # Left column
        left_layout = QVBoxLayout()
        self.symbol_label = QLabel("Symbol: -")
        self.price_label = QLabel("Price: -")
        self.change_label = QLabel("Change: -")
        
        left_layout.addWidget(self.symbol_label)
        left_layout.addWidget(self.price_label)
        left_layout.addWidget(self.change_label)
        details_layout.addLayout(left_layout)
        
        # Right column
        right_layout = QVBoxLayout()
        self.volume_label = QLabel("Volume: -")
        self.bid_ask_label = QLabel("Bid/Ask: -")
        self.high_low_label = QLabel("High/Low: -")
        
        right_layout.addWidget(self.volume_label)
        right_layout.addWidget(self.bid_ask_label)
        right_layout.addWidget(self.high_low_label)
        details_layout.addLayout(right_layout)
        
        layout.addLayout(details_layout)

        # Action buttons row (Buy/Sell)
        actions_layout = QHBoxLayout()
        self.buy_button = QPushButton("Buy")
        self.sell_button = QPushButton("Sell")
        self.buy_button.setEnabled(False)
        self.sell_button.setEnabled(False)
        actions_layout.addWidget(self.buy_button)
        actions_layout.addWidget(self.sell_button)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        # Icons row (Ask AI + Chart)
        # Per user request, icons should be per-row in the table; hide details-panel icons to avoid duplication
        # (If needed later, these can be re-enabled.)

    # (removed duplicated UI block)
    
    def update_details(self, symbol: str, data: dict):
        """Update details for selected symbol"""
        self.current_symbol = symbol
        
        self.title_label.setText(f"{symbol} Details")
        self.symbol_label.setText(f"Symbol: {symbol}")
        self.price_label.setText(f"Price: ${data['price']:.2f}")
        
        change_text = f"Change: ${data['change']:+.2f} ({data['change_pct']:+.2f}%)"
        self.change_label.setText(change_text)
        if data['change'] >= 0:
            self.change_label.setStyleSheet("color: #4CAF50;")
        else:
            self.change_label.setStyleSheet("color: #f44336;")
        
        self.volume_label.setText(f"Volume: {data['volume']:,}")
        self.bid_ask_label.setText(f"Bid/Ask: ${data['bid']:.2f} / ${data['ask']:.2f}")
        self.high_low_label.setText(f"High/Low: ${data['high']:.2f} / ${data['low']:.2f}")
        
        # Enable buttons when details are updated
        self.buy_button.setEnabled(True)
        self.sell_button.setEnabled(True)


class WatchlistWidget(QWidget):
    def _handle_ask_ai(self, symbol: str):
        # Ensure status label exists
        if not hasattr(self.details_panel, 'ai_status_label') or self.details_panel.ai_status_label is None:
            from PyQt6.QtWidgets import QLabel
            self.details_panel.ai_status_label = QLabel("")
            self.details_panel.ai_status_label.setStyleSheet("color: #888; font-size: 11px;")
            # Append to bottom of details panel
            if self.details_panel.layout() is not None:
                self.details_panel.layout().addWidget(self.details_panel.ai_status_label)
        if not symbol:
            self.details_panel.ai_status_label.setText("לא נבחרה מניה לשליחת שאילתא ל-AI.")
            return
        row = self.watchlist_table.symbol_data.get(symbol, {}).get('row')
        if row is None:
            self.details_panel.ai_status_label.setText("המניה לא קיימת בטבלה.")
            return
        self.details_panel.ai_status_label.setText("שולח שאילתא ל-AI...")
        try:
            self.watchlist_table.request_ai_rating(symbol, row)
            self.details_panel.ai_status_label.setText("הבקשה נשלחה ל-AI. המתן לתשובה...")
        except Exception as e:
            self.details_panel.ai_status_label.setText(f"שגיאה בשליחת בקשה ל-AI: {e}")

    def _handle_open_chart(self, symbol: str):
        # Ensure status label exists
        if not hasattr(self.details_panel, 'ai_status_label') or self.details_panel.ai_status_label is None:
            from PyQt6.QtWidgets import QLabel
            self.details_panel.ai_status_label = QLabel("")
            self.details_panel.ai_status_label.setStyleSheet("color: #888; font-size: 11px;")
            if self.details_panel.layout() is not None:
                self.details_panel.layout().addWidget(self.details_panel.ai_status_label)
        if not symbol:
            self.details_panel.ai_status_label.setText("לא נבחרה מניה להצגת גרף.")
            return
        self.details_panel.ai_status_label.setText(f"פותח גרף עבור {symbol}...")
    def on_ask_ai_clicked(self):
        symbol = getattr(self, 'current_symbol', None)
        row = self.symbol_data.get(symbol, {}).get('row') if symbol else None
        if symbol and row is not None:
            self.ai_status_label.setText("שולח שאילתא ל-AI...")
            try:
                self.request_ai_rating(symbol, row)
                self.ai_status_label.setText("הבקשה נשלחה ל-AI. המתן לתשובה...")
            except Exception as e:
                self.ai_status_label.setText(f"שגיאה בשליחת בקשה ל-AI: {e}")
        else:
            self.ai_status_label.setText("לא נבחרה מניה לשליחת שאילתא ל-AI.")

    def on_chart_clicked(self):
        symbol = getattr(self, 'current_symbol', None)
        if symbol:
            # כאן יש להפעיל את הפונקציה לפתיחת גרף עבור הסימבול
            self.ai_status_label.setText(f"פותח גרף עבור {symbol}...")
            # self.open_chart.emit(symbol) # אם יש סיגנל כזה
        else:
            self.ai_status_label.setText("לא נבחרה מניה להצגת גרף.")
    """Main watchlist widget"""
    
    trade_requested = pyqtSignal(str)  # action and symbol
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config with fallbacks
        if get_logger:
            self.logger = get_logger("Watchlist")
        else:
            self.logger = None
            
        if ConfigManager:
            self.config = ConfigManager()
        else:
            # Create a mock config object
            class MockConfig:
                class UI:
                    default_symbols = []
                    update_interval = 5000
                ui = UI()
            self.config = MockConfig()
        
        # Initialize data worker if available
        self.data_thread = QThread() if QThread else None
        self.data_worker = None  # Skip data worker initialization for now
        
        # Setup UI
        self.setup_ui()
        
        # Load default symbols (deferred to avoid blocking UI during startup)
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, self.load_default_symbols)
        except Exception:
            # Fallback to direct call if QTimer is not available
            self.load_default_symbols()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(self.config.ui.update_interval)
        
        self.logger.info("Watchlist widget initialized")

        # Ensure cleanup on app exit
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self.cleanup)
        except Exception:
            pass

    def cleanup(self):
        """Stop timers and threads gracefully"""
        try:
            if hasattr(self, 'update_timer') and self.update_timer is not None:
                self.update_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'data_worker'):
                self.data_worker.stop_monitoring()
            if hasattr(self, 'data_thread') and self.data_thread is not None:
                self.data_thread.quit()
                self.data_thread.wait(2000)
        except Exception:
            pass

    def closeEvent(self, event):
        """Qt close event override to cleanup resources"""
        self.cleanup()
        super().closeEvent(event)

    def add_symbol_from_scanner(self, symbol: str, switch_to_tab: bool = False):
        """Public method to add a symbol coming from external widgets (e.g., Scanner).
        Adds the symbol if missing, restarts monitoring to include it, and focuses/selects it.
        """
        try:
            if not symbol:
                return
            sym = symbol.strip().upper()
            if not sym:
                return

            # Try to add (no-op if already present)
            added = self.watchlist_table.add_symbol(sym)

            # Update monitoring universe
            symbols = self.watchlist_table.get_symbols()
            if hasattr(self, 'data_worker'):
                self.data_worker.stop_monitoring()
                if symbols:
                    self.data_worker.start_monitoring(symbols)

            # Focus/select the symbol row if present
            if sym in self.watchlist_table.symbol_data:
                row = self.watchlist_table.symbol_data[sym]['row']
                self.watchlist_table.clearSelection()
                self.watchlist_table.selectRow(row)
                # Update details immediately if we have cached data
                data = self.watchlist_table.symbol_data[sym].get('data') if isinstance(self.watchlist_table.symbol_data.get(sym), dict) else None
                if data:
                    self.details_panel.update_details(sym, data)
                else:
                    # Ensure details title reflects selection even before data arrives
                    self.details_panel.title_label.setText(f"{sym} Details")
                    self.details_panel.symbol_label.setText(f"Symbol: {sym}")

            self.logger.info(f"Symbol {'added' if added else 'already in'} watchlist via Scanner: {sym}")
        except Exception as e:
            self.logger.error(f"Failed to add symbol from scanner: {symbol} | {e}")
    
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
        
    def _handle_tools_action(self, action: str, symbol: str):
        """Handle per-row tools icon actions: AI and Chart"""
        if action == 'ai':
            # Show status in details panel and trigger AI query for symbol
            if not hasattr(self.details_panel, 'ai_status_label') or self.details_panel.ai_status_label is None:
                from PyQt6.QtWidgets import QLabel
                self.details_panel.ai_status_label = QLabel("")
                self.details_panel.ai_status_label.setStyleSheet("color: #888; font-size: 11px;")
                if self.details_panel.layout() is not None:
                    self.details_panel.layout().addWidget(self.details_panel.ai_status_label)
            self.details_panel.ai_status_label.setText(f"שולח שאילתא ל-AI עבור {symbol}...")
            row = self.watchlist_table.symbol_data.get(symbol, {}).get('row')
            if row is not None:
                try:
                    self.watchlist_table.request_ai_rating(symbol, row)
                    self.details_panel.ai_status_label.setText(f"הבקשה נשלחה ל-AI עבור {symbol}. המתן לתשובה...")
                except Exception as e:
                    self.details_panel.ai_status_label.setText(f"שגיאה בשליחת בקשה ל-AI: {e}")
            else:
                self.details_panel.ai_status_label.setText(f"{symbol} לא נמצא בטבלה.")
        elif action == 'chart':
            # Show status and open chart for symbol
            if not hasattr(self.details_panel, 'ai_status_label') or self.details_panel.ai_status_label is None:
                from PyQt6.QtWidgets import QLabel
                self.details_panel.ai_status_label = QLabel("")
                self.details_panel.ai_status_label.setStyleSheet("color: #888; font-size: 11px;")
                if self.details_panel.layout() is not None:
                    self.details_panel.layout().addWidget(self.details_panel.ai_status_label)
            self.details_panel.ai_status_label.setText(f"פותח גרף עבור {symbol}...")
            # You can replace this with a chart dialog call if available
            self.on_symbol_selected(symbol)
    
    def load_default_symbols(self):
        """Load default symbols from config"""
        default_symbols = self.config.ui.default_symbols
        for symbol in default_symbols:
            self.watchlist_table.add_symbol(symbol)
        
        # Start monitoring
        if default_symbols:
            self.data_worker.start_monitoring(default_symbols)
    
    def add_symbol(self):
        """Add new symbol to watchlist"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return
        # Enforce English-only before attempting to add
        import re
        if not re.fullmatch(r"[A-Z]+", symbol):
            QMessageBox.warning(self, "Invalid symbol", "Please enter English letters only (A-Z)")
            return
        
        if self.watchlist_table.add_symbol(symbol):
            self.symbol_input.clear()
            
            # Update monitoring
            symbols = self.watchlist_table.get_symbols()
            self.data_worker.stop_monitoring()
            self.data_worker.start_monitoring(symbols)
            
            self.logger.info(f"Added symbol to watchlist: {symbol}")
        else:
            QMessageBox.warning(self, "Warning", f"{symbol} is already in the watchlist")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if self.watchlist_table.remove_symbol(symbol):
            # Update monitoring
            symbols = self.watchlist_table.get_symbols()
            self.data_worker.stop_monitoring()
            if symbols:
                self.data_worker.start_monitoring(symbols)
            
            self.logger.info(f"Removed symbol from watchlist: {symbol}")
    
    def on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        if symbol in self.watchlist_table.symbol_data:
            data = self.watchlist_table.symbol_data[symbol].get('data')
            if data:
                self.details_panel.update_details(symbol, data)
        
        self.logger.debug(f"Symbol selected: {symbol}")
    
    def on_data_updated(self, symbol: str, data: dict):
        """Handle data update from worker"""
        self.watchlist_table.update_symbol_data(symbol, data)
        
        # Update details if this symbol is currently selected
        if (hasattr(self.details_panel, 'current_symbol') and 
            self.details_panel.current_symbol == symbol):
            self.details_panel.update_details(symbol, data)
    
    def on_data_error(self, error: str):
        """Handle data error"""
        self.logger.error(f"Watchlist data error: {error}")
    
    def update_data(self):
        """Update watchlist data"""
        try:
            if hasattr(self, 'data_worker') and self.data_worker.is_running:
                # Trigger fetch in worker thread
                QTimer.singleShot(0, self.data_worker.fetch_data)
        except Exception as e:
            self.logger.error(f"Update data error: {e}")
    
    def refresh_data(self):
        """Manually refresh watchlist data"""
        symbols = self.watchlist_table.get_symbols()
        if symbols:
            self.data_worker.stop_monitoring()
            self.data_worker.start_monitoring(symbols)
        
        self.logger.info("Manual watchlist refresh requested")

    def delete_selected(self):
        """Delete the currently selected ticker from the list"""
        sel_ranges = self.watchlist_table.selectedItems()
        if not sel_ranges:
            return
        # Get the row of the first selected item
        row = sel_ranges[0].row()
        symbol_item = self.watchlist_table.item(row, 0)
        if not symbol_item:
            return
        symbol = symbol_item.text()
        self.remove_symbol(symbol)
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Watchlist tab activated")
        self.refresh_data()
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop update timer
        if hasattr(self, 'update_timer') and isinstance(self.update_timer, QTimer):
            try:
                self.update_timer.stop()
            except Exception:
                pass
        # Stop data worker
        if hasattr(self, 'data_worker'):
            self.data_worker.stop_monitoring()
        
        # Stop worker thread
        if hasattr(self, 'data_thread'):
            self.data_thread.quit()
            self.data_thread.wait()
        
        event.accept()