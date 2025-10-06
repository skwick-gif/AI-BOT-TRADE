"""
Calendar Widget
Simple calendar display with a placeholder for upcoming events.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QCalendarWidget, QPushButton,
    QListWidget, QListWidgetItem, QDialog, QTextEdit, QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from utils.logger import get_logger
from services.finnhub_client import (
    fetch_earnings_calendar,
    fetch_dividends,
    fetch_economic_calendar,
    fetch_ipo_calendar,
)
from utils.trading_helpers import get_portfolio_positions
from services.sentiment_service import aggregate_symbol_sentiment
from services.earnings_ml import predict_half_day_direction, train_symbol_model, load_saved_model


class _CalendarFetchThread(QThread):
    """Worker thread to fetch events from Finnhub to keep UI responsive."""
    data_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        start_date: str,
        end_date: str,
        symbols: list[str],
        include_earnings: bool = True,
        include_dividends: bool = True,
        include_economic: bool = True,
        include_ipo: bool = True,
    ):
        super().__init__()
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.include_earnings = include_earnings
        self.include_dividends = include_dividends
        self.include_economic = include_economic
        self.include_ipo = include_ipo

    def run(self):
        try:
            events = []
            # Earnings
            if self.include_earnings:
                events.extend(fetch_earnings_calendar(self.api_key, self.start_date, self.end_date, self.symbols))
            # Dividends
            if self.include_dividends:
                events.extend(fetch_dividends(self.api_key, self.start_date, self.end_date, self.symbols))
            # Economic (may be empty if not in plan)
            if self.include_economic:
                events.extend(fetch_economic_calendar(self.api_key, self.start_date, self.end_date))
            # IPOs (US only)
            if self.include_ipo:
                events.extend(fetch_ipo_calendar(self.api_key, self.start_date, self.end_date, us_only=True))
            self.data_ready.emit(events)
        except Exception as e:
            self.error.emit(str(e))


class CalendarWidget(QFrame):
    """Calendar section for the dashboard"""

    def __init__(self):
        super().__init__()
        self.logger = get_logger("Calendar")
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Calendar")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        title_layout.addStretch()
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setFixedSize(120, 30)
        self.refresh_btn.setStyleSheet(
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
        self.refresh_btn.clicked.connect(self.refresh_events)
        title_layout.addWidget(self.refresh_btn)
        layout.addLayout(title_layout)

        # Filter controls row
        controls = QHBoxLayout()
        self.chk_earnings = QCheckBox("Earnings")
        self.chk_earnings.setChecked(True)
        self.chk_dividends = QCheckBox("Dividends")
        self.chk_dividends.setChecked(True)
        self.chk_economic = QCheckBox("Economic")
        self.chk_economic.setChecked(True)
        self.chk_ipo = QCheckBox("IPO")
        self.chk_ipo.setChecked(True)
        for chk in (self.chk_earnings, self.chk_dividends, self.chk_economic, self.chk_ipo):
            chk.stateChanged.connect(self.refresh_events)
            controls.addWidget(chk)

        controls.addStretch()
        # Sentiment/model settings
        controls.addWidget(QLabel("Sentiment days:"))
        self.sentiment_days_spin = QSpinBox()
        self.sentiment_days_spin.setRange(1, 14)
        self.sentiment_days_spin.setValue(3)
        controls.addWidget(self.sentiment_days_spin)

        controls.addWidget(QLabel("Blend:"))
        self.blend_spin = QDoubleSpinBox()
        self.blend_spin.setRange(0.0, 1.0)
        self.blend_spin.setSingleStep(0.05)
        self.blend_spin.setDecimals(2)
        self.blend_spin.setValue(0.20)
        controls.addWidget(self.blend_spin)

        layout.addLayout(controls)

        # Content
        content_layout = QHBoxLayout()

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        self.calendar.selectionChanged.connect(self.on_date_changed)
        content_layout.addWidget(self.calendar, 2)

        # Interactive events list
        self.events_list = QListWidget()
        self.events_list.setAlternatingRowColors(True)
        self.events_list.itemDoubleClicked.connect(self.on_event_activated)
        content_layout.addWidget(self.events_list, 1)

        layout.addLayout(content_layout)

        # Status
        self.status_label = QLabel("No events loaded.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

        # Initial state
        self.api_key = os.getenv("FINNHUB_API_KEY", "").strip()
        self.default_symbols = self._gather_symbols()
        if not self.api_key:
            self.status_label.setText("FINNHUB_API_KEY is not set. Add it to .env to enable calendar events.")
        else:
            self.refresh_events()

    def on_date_changed(self):
        # Fetch events for selected date
        self.refresh_events()

    def _selected_date_range(self) -> tuple[str, str]:
        qd: QDate = self.calendar.selectedDate()
        end_date = qd.toString("yyyy-MM-dd")
        # Start range fixed per request
        start_date = "2024-01-01"
        return start_date, end_date

    def refresh_events(self):
        if not self.api_key:
            return
        start_date, end_date = self._selected_date_range()
        # Refresh symbols (watchlist/portfolio may change over time)
        self.default_symbols = self._gather_symbols()
        self.status_label.setText(f"Loading events for {start_date}…")
        self.events_list.clear()

        # Start worker thread
        self._worker = _CalendarFetchThread(
            self.api_key,
            start_date,
            end_date,
            self.default_symbols,
            include_earnings=self.chk_earnings.isChecked(),
            include_dividends=self.chk_dividends.isChecked(),
            include_economic=self.chk_economic.isChecked(),
            include_ipo=self.chk_ipo.isChecked(),
        )
        self._worker.data_ready.connect(self.on_events_ready)
        self._worker.error.connect(self.on_events_error)
        self._worker.start()

    def on_events_ready(self, events: list):
        if not events:
            self.events_list.addItem("No events found for selected date.")
            self.status_label.setText("Done.")
            return

        # Build a list with items and store event payload
        BIG_MOVERS = {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK.B","BRK.A","JPM","JNJ"}
        for ev in events:
            src = ev.get("_source", "")
            if src == "earnings":
                symbol = ev.get("symbol") or ev.get("ticker") or "?"
                date = ev.get("date") or ev.get("time") or ""
                est = ev.get("epsEstimate") or ev.get("epsEstimateCurrentYear") or ev.get("epsEst", "")
                text = f"🧾 Earnings • {symbol} • {date} • EPS est: {est}"
                if symbol in BIG_MOVERS:
                    text += " • High impact"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, ev)
                self.events_list.addItem(item)
            elif src == "dividend":
                symbol = ev.get("symbol", "?")
                date = ev.get("payDate") or ev.get("recordDate") or ev.get("exDate") or ""
                amt = ev.get("amount") or ev.get("cashAmount") or ev.get("adjDividend") or ""
                item = QListWidgetItem(f"💰 Dividend • {symbol} • {date} • {amt}")
                item.setData(Qt.ItemDataRole.UserRole, ev)
                self.events_list.addItem(item)
            elif src == "economic":
                date = ev.get("time") or ev.get("date") or ""
                country = ev.get("country", "")
                event = ev.get("event", ev.get("actual", "Economic event"))
                item = QListWidgetItem(f"🌐 Economic • {country} • {date} • {event}")
                item.setData(Qt.ItemDataRole.UserRole, ev)
                self.events_list.addItem(item)
            elif src == "ipo":
                symbol = ev.get("symbol") or ev.get("ipo") or ev.get("name") or "?"
                date = ev.get("date") or ev.get("ipoDate") or ev.get("time") or ""
                exchange = ev.get("exchange") or ev.get("market") or ""
                price = ev.get("price") or ev.get("priceLow") or ev.get("minPrice") or ""
                item = QListWidgetItem(f"🚀 IPO • {symbol} • {date} • {exchange} • {price}")
                item.setData(Qt.ItemDataRole.UserRole, ev)
                self.events_list.addItem(item)
            else:
                # Fallback stringify
                item = QListWidgetItem(str(ev))
                item.setData(Qt.ItemDataRole.UserRole, ev)
                self.events_list.addItem(item)
        self.status_label.setText(f"Loaded {len(events)} events.")

    def on_events_error(self, msg: str):
        self.events_list.clear()
        self.status_label.setText(f"Error loading events: {msg}")

    def _gather_symbols(self) -> list[str]:
        """Combine watchlist env, portfolio csv positions, and big movers."""
        env_syms = os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL").split(",")
        env_syms = [s.strip().upper() for s in env_syms if s.strip()]
        # portfolio from local CSV helper
        try:
            positions = get_portfolio_positions() or {}
            portfolio_syms = list(positions.keys())
        except Exception:
            portfolio_syms = []
        big_movers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"]
        # Preserve order preference: portfolio, env watchlist, then big movers
        out = []
        seen = set()
        for s in portfolio_syms + env_syms + big_movers:
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def on_event_activated(self, item: QListWidgetItem):
        ev = item.data(Qt.ItemDataRole.UserRole) or {}
        if ev.get("_source") == "earnings":
            self.open_earnings_forecast(ev)

    def open_earnings_forecast(self, ev: dict):
        """Open a dialog with a simple earnings preview/impact heuristic."""
        symbol = ev.get("symbol") or ev.get("ticker") or "?"
        date = ev.get("date") or ev.get("time") or ""
        eps_est = ev.get("epsEstimate") or ev.get("epsEstimateCurrentYear") or ev.get("epsEst", "?")
        rev_est = ev.get("revenueEstimate") or ev.get("revenueAvg") or ev.get("revenueEst", "?")
        big_movers = {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"}
        impact = "High" if symbol in big_movers else "Medium"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Earnings Preview • {symbol}")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Symbol: {symbol}"))
        v.addWidget(QLabel(f"Date/Time: {date}"))
        v.addWidget(QLabel(f"EPS Estimate: {eps_est}"))
        v.addWidget(QLabel(f"Revenue Estimate: {rev_est}"))
        v.addWidget(QLabel(f"Market Impact (heuristic): {impact}"))
        # Guidance text (simple heuristic narrative)
        note = QLabel(
            "If EPS/revenue beat consensus materially (≈>5%), short-term reaction is often positive; misses can weigh on the index if mega-caps are involved."
        )
        note.setWordWrap(True)
        v.addWidget(note)

        # Optional: compute recent-news sentiment and quick ML probability
        try:
            import datetime as dt
            # Event day window: previous N days to event date (configurable)
            event_date = dt.datetime.fromisoformat(str(date)) if date else dt.datetime.utcnow()
            window_days = int(self.sentiment_days_spin.value())
            sent = aggregate_symbol_sentiment(symbol, event_date - dt.timedelta(days=window_days), event_date)
            prob = predict_half_day_direction(
                symbol,
                event_date,
                sentiment_score=sent.get("avg_compound"),
                blend_weight=float(self.blend_spin.value()),
            )
            v.addWidget(QLabel(f"News sentiment (avg compound): {sent.get('avg_compound', 0.0):+.3f} from {sent.get('count',0)} articles"))
            v.addWidget(QLabel(f"ML short-term P(up next half-day): {prob.get('prob_up',0.5):.2%}"))
        except Exception as mlex:
            v.addWidget(QLabel(f"Forecast unavailable: {str(mlex)[:60]}…"))

        # Actions row: Train/Load (optional) + Close
        btn_row = QHBoxLayout()
        def do_train():
            try:
                import datetime as dt
                res = train_symbol_model(symbol, dt.datetime.utcnow(), lookback_days=240)
                QMessageBox.information(self, "Train Model", f"Result: {res}")
            except Exception as e:
                QMessageBox.critical(self, "Train Model", str(e))
        def do_load():
            try:
                mdl = load_saved_model(symbol)
                msg = "Loaded" if mdl is not None else "No saved model"
                QMessageBox.information(self, "Load Model", msg)
            except Exception as e:
                QMessageBox.critical(self, "Load Model", str(e))
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(do_train)
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(do_load)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(train_btn)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)
        dlg.setMinimumWidth(420)
        dlg.exec()
