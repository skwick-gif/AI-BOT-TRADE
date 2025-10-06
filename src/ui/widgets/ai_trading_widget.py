"""
AI Trading Widget (UI Skeleton)
Visual skeleton for automated AI-driven trading control panel.
No backend logic or dummy data—just structure and hooks.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QPlainTextEdit, QDialog,
    QTabWidget, QFormLayout, QListWidget, QListWidgetItem, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtGui import QFont
import random
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from core.ai_trading_config import AiTradingConfigManager, Strategy, AssetEntry
from services.ai_service import AIService


class AutomationSettingsDialog(QDialog):
    """Simple settings dialog with tabs (General / Profiles / Safety)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automation Settings")
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        # General tab (global defaults skeleton)
        general = QWidget()
        gform = QFormLayout(general)
        self.buy_threshold = QDoubleSpinBox()
        self.buy_threshold.setRange(0, 10)
        self.buy_threshold.setSingleStep(0.1)
        self.buy_threshold.setValue(8.0)
        self.sell_threshold = QDoubleSpinBox()
        self.sell_threshold.setRange(0, 10)
        self.sell_threshold.setSingleStep(0.1)
        self.sell_threshold.setValue(4.0)
        self.hysteresis = QSpinBox()
        self.hysteresis.setRange(1, 10)
        self.hysteresis.setValue(2)
        self.cooldown_min = QSpinBox()
        self.cooldown_min.setRange(1, 240)
        self.cooldown_min.setValue(3)
        self.sl_pct = QDoubleSpinBox()
        self.sl_pct.setSuffix(" %")
        self.sl_pct.setRange(0.1, 50.0)
        self.sl_pct.setSingleStep(0.1)
        self.sl_pct.setValue(3.0)
        self.tp_pct = QDoubleSpinBox()
        self.tp_pct.setSuffix(" %")
        self.tp_pct.setRange(0.1, 200.0)
        self.tp_pct.setSingleStep(0.1)
        self.tp_pct.setValue(6.0)
        self.global_interval = QComboBox()
        self.global_interval.addItems(["1m", "2m", "5m", "15m", "60m"]) 
        self.trading_hours_only = QCheckBox("Trading hours only")
        self.trading_hours_only.setChecked(True)
        gform.addRow("Buy ≥", self.buy_threshold)
        gform.addRow("Sell ≤", self.sell_threshold)
        gform.addRow("Hysteresis (cycles)", self.hysteresis)
        gform.addRow("Cooldown (min)", self.cooldown_min)
        gform.addRow("Stop-Loss %", self.sl_pct)
        gform.addRow("Take-Profit %", self.tp_pct)
        gform.addRow("Global Interval", self.global_interval)
        gform.addRow(self.trading_hours_only)
        tabs.addTab(general, "General")

        # Profiles tab (skeleton only)
        profiles = QWidget()
        pform = QFormLayout(profiles)
        pform.addRow(QLabel("Define/edit presets (Intraday, Swing, Long-term)."))
        tabs.addTab(profiles, "Profiles")

        # Safety tab (skeleton only)
        safety = QWidget()
        sform = QFormLayout(safety)
        self.daily_loss_limit = QDoubleSpinBox()
        self.daily_loss_limit.setPrefix("$")
        self.daily_loss_limit.setRange(0, 1_000_000)
        self.daily_loss_limit.setValue(0)
        self.max_trades_day = QSpinBox()
        self.max_trades_day.setRange(0, 1000)
        self.max_trades_day.setValue(0)
        sform.addRow("Daily loss limit", self.daily_loss_limit)
        sform.addRow("Max trades / day", self.max_trades_day)
        tabs.addTab(safety, "Safety")

        # Buttons row
        btns = QHBoxLayout()
        btns.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)


class AiTradingWidget(QWidget):
    """AI Trading control panel skeleton (no data wiring)."""

    # Signals (hooks only)
    connect_ibkr_requested = pyqtSignal()
    disconnect_ibkr_requested = pyqtSignal()
    test_api_requested = pyqtSignal()
    start_all_requested = pyqtSignal()
    stop_all_requested = pyqtSignal()
    # Internal: emitted when a background score is ready (symbol, score, error)
    score_ready = pyqtSignal(str, float, object)

    def __init__(self):
        super().__init__()
        self._ibkr_connected = False
        self._api_ok = False
        self._cfg = AiTradingConfigManager()  # persistent config
        self._global_interval = self._cfg.config.globals.interval
        self._paper_mode = (self._cfg.config.globals.mode == "Paper")
        # runtime state
        self._timers = {}  # symbol -> QTimer
        self._asset_rows = {}  # symbol -> row index
        self._rt_state = {}  # symbol -> { 'last_signal': str, 'h_count': int, 'cooldown_until': datetime|None }
        self._inflight = set()  # symbols currently being scored
        self._executor = ThreadPoolExecutor(max_workers=3)
        # short-lived cache for scores to reduce repeated calls
        self._score_cache = {}  # type: ignore[var-annotated]
        self._score_ttl_seconds = 10  # reuse score for a short window to reduce bursts
        # daily guardrails state (skeleton)
        self._daily_date = date.today()
        self._daily_trade_count = 0
        self._daily_pnl = 0.0  # placeholder until IBKR/exec wiring

        self._build_ui()
        self._load_from_config()
        # wire internal signal for background scoring results
        self.score_ready.connect(self._on_score_ready)

    # ---- Public hooks for future wiring ----
    def set_ibkr_service(self, service):
        """Receive IBKR service instance for live execution."""
        self._ibkr_service = service

    def set_ai_service(self, service: AIService):
        self._ai_service = service

    def set_api_status(self, ok: bool):
        self._api_ok = ok
        self._api_label.setText("Perplexity: OK" if ok else "Perplexity: Not set")

    def set_ibkr_status(self, connected: bool):
        self._ibkr_connected = connected
        self._ibkr_label.setText("IBKR: Connected" if connected else "IBKR: Disconnected")
        self._connect_btn.setText("Disconnect" if connected else "Connect")

    # ---- UI construction ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Top command bar
        top = QHBoxLayout()
        title = QLabel("AI Trading")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        top.addWidget(title)
        top.addStretch()

        self._ibkr_label = QLabel("IBKR: Disconnected")
        top.addWidget(self._ibkr_label)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self.connect_ibkr_requested.emit)
        top.addWidget(self._connect_btn)

        self._api_label = QLabel("Perplexity: Not set")
        top.addWidget(self._api_label)
        self._api_test_btn = QPushButton("Test")
        self._api_test_btn.clicked.connect(self.test_api_requested.emit)
        top.addWidget(self._api_test_btn)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Paper", "Live"])
        top.addWidget(self._mode_combo)

        self._global_interval_combo = QComboBox()
        self._global_interval_combo.addItems(["1m", "2m", "5m", "15m", "60m"])
        self._global_interval_combo.setCurrentText(self._global_interval)
        top.addWidget(QLabel("Interval"))
        top.addWidget(self._global_interval_combo)

        self._hours_only = QCheckBox("Trading hours only")
        self._hours_only.setChecked(self._cfg.config.globals.trading_hours_only)
        top.addWidget(self._hours_only)

        self._settings_btn = QPushButton("Settings ⚙")
        self._settings_btn.clicked.connect(self._open_settings)
        top.addWidget(self._settings_btn)

        self._start_all_btn = QPushButton("Start All")
        # Wire local start; can also emit outward if needed
        self._start_all_btn.clicked.connect(self._start_all)
        top.addWidget(self._start_all_btn)
        self._stop_all_btn = QPushButton("Stop All")
        self._stop_all_btn.clicked.connect(self._stop_all)
        top.addWidget(self._stop_all_btn)

        root.addLayout(top)

        # Tabbed area: Assets / Strategies / Settings / Logs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

    # --- Assets tab ---
        assets_tab = QWidget(); assets_layout = QVBoxLayout(assets_tab)
        add_group = QGroupBox("Add Asset")
        add_layout = QVBoxLayout(add_group)

        row1 = QHBoxLayout()
        self._symbol_edit = QLineEdit()
        self._symbol_edit.setPlaceholderText("Symbol (e.g., TNA)")
        row1.addWidget(QLabel("Symbol"))
        row1.addWidget(self._symbol_edit)

        self._qty_spin = QSpinBox()
        self._qty_spin.setRange(1, 1_000_000)
        self._qty_spin.setValue(100)
        row1.addWidget(QLabel("Quantity"))
        row1.addWidget(self._qty_spin)

        self._profile_combo = QComboBox()
        self._profile_combo.addItems(["Intraday", "Swing", "Long-term"])
        row1.addWidget(QLabel("Profile"))
        row1.addWidget(self._profile_combo)
        add_layout.addLayout(row1)

        # Advanced overrides (collapsible feel)
        self._advanced_btn = QPushButton("Advanced overrides ▸")
        self._advanced_btn.setCheckable(True)
        self._advanced_btn.toggled.connect(self._toggle_advanced)
        add_layout.addWidget(self._advanced_btn)

        self._advanced_frame = QFrame()
        self._advanced_frame.setVisible(False)
        adv = QVBoxLayout(self._advanced_frame)

        # Row: thresholds & gating
        row2 = QHBoxLayout()
        self._buy_thr = QDoubleSpinBox(); self._buy_thr.setRange(0, 10); self._buy_thr.setSingleStep(0.1); self._buy_thr.setValue(8.0)
        self._sell_thr = QDoubleSpinBox(); self._sell_thr.setRange(0, 10); self._sell_thr.setSingleStep(0.1); self._sell_thr.setValue(4.0)
        self._hys = QSpinBox(); self._hys.setRange(1, 10); self._hys.setValue(2)
        self._cool = QSpinBox(); self._cool.setRange(1, 240); self._cool.setValue(3)
        row2.addWidget(QLabel("Buy ≥")); row2.addWidget(self._buy_thr)
        row2.addWidget(QLabel("Sell ≤")); row2.addWidget(self._sell_thr)
        row2.addWidget(QLabel("Hysteresis")); row2.addWidget(self._hys)
        row2.addWidget(QLabel("Cooldown (min)")); row2.addWidget(self._cool)
        adv.addLayout(row2)

        # Row: risk
        row3 = QHBoxLayout()
        self._sl = QDoubleSpinBox(); self._sl.setRange(0.1, 50.0); self._sl.setSingleStep(0.1); self._sl.setValue(3.0); self._sl.setSuffix(" %")
        self._tp = QDoubleSpinBox(); self._tp.setRange(0.1, 200.0); self._tp.setSingleStep(0.1); self._tp.setValue(6.0); self._tp.setSuffix(" %")
        self._bracket = QCheckBox("Bracket")
        self._bracket.setChecked(True)
        row3.addWidget(QLabel("SL %")); row3.addWidget(self._sl)
        row3.addWidget(QLabel("TP %")); row3.addWidget(self._tp)
        row3.addWidget(self._bracket)
        adv.addLayout(row3)

        # Row: interval & model
        row4 = QHBoxLayout()
        self._use_global_interval = QCheckBox("Use global interval")
        self._use_global_interval.setChecked(True)
        self._custom_interval = QComboBox(); self._custom_interval.addItems(["1m", "2m", "5m", "15m", "60m"])
        self._custom_interval.setEnabled(False)
        self._use_global_interval.toggled.connect(lambda v: self._custom_interval.setEnabled(not v))
        row4.addWidget(self._use_global_interval)
        row4.addWidget(QLabel("Custom interval")); row4.addWidget(self._custom_interval)
        row4.addStretch()
        adv.addLayout(row4)

        add_layout.addWidget(self._advanced_frame)

        # Add button
        add_btn_row = QHBoxLayout()
        add_btn_row.addStretch()
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add_asset)
        add_btn_row.addWidget(self._add_btn)
        add_layout.addLayout(add_btn_row)
        assets_layout.addWidget(add_group)

        # Splitter: table (left) and details (right)
        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: assets table
        table_frame = QFrame(); table_layout = QVBoxLayout(table_frame); table_layout.setContentsMargins(0, 0, 0, 0)
        table_title = QLabel("Automation Assets")
        table_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        table_layout.addWidget(table_title)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "On", "Symbol", "Profile", "Qty", "Interval", "AI Score", "Status", "Actions"
        ])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(False)
        table_layout.addWidget(self._table)

        split.addWidget(table_frame)

        # Right: selected asset panel (skeleton)
        right = QFrame(); right_layout = QVBoxLayout(right)
        right_title = QLabel("Selected Asset")
        right_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(right_title)

        self._summary_label = QLabel("Score: —    Action: —    Confidence: —")
        right_layout.addWidget(self._summary_label)
        right_layout.addWidget(QLabel("Reason:"))
        self._reason = QPlainTextEdit(); self._reason.setReadOnly(True); self._reason.setPlaceholderText("Decision rationale will appear here…")
        right_layout.addWidget(self._reason)

        right_layout.addWidget(QLabel("Proposed SL/TP vs Current:"))
        sltp_row = QHBoxLayout()
        self._apply_sl_btn = QPushButton("Apply SL")
        self._apply_tp_btn = QPushButton("Apply TP")
        sltp_row.addWidget(self._apply_sl_btn)
        sltp_row.addWidget(self._apply_tp_btn)
        sltp_row.addStretch()
        right_layout.addLayout(sltp_row)

        right_layout.addWidget(QLabel("Next eval: —    Last latency: —"))

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        assets_layout.addWidget(split)

        self._tabs.addTab(assets_tab, "Assets")
        # selection updates right panel
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

    # --- Strategies tab ---
        strat_tab = QWidget(); strat_layout = QHBoxLayout(strat_tab)
        self._strat_list = QListWidget(); self._strat_list.itemClicked.connect(self._on_strat_selected)
        strat_layout.addWidget(self._strat_list, 1)
        # editor form
        strat_form = QFormLayout()
        self._s_name = QLineEdit(); self._s_buy = QDoubleSpinBox(); self._s_buy.setRange(0,10); self._s_buy.setSingleStep(0.1)
        self._s_sell = QDoubleSpinBox(); self._s_sell.setRange(0,10); self._s_sell.setSingleStep(0.1)
        self._s_hys = QSpinBox(); self._s_hys.setRange(1,10)
        self._s_cool = QSpinBox(); self._s_cool.setRange(1,240)
        self._s_sl = QDoubleSpinBox(); self._s_sl.setRange(0.1,50.0); self._s_sl.setSingleStep(0.1); self._s_sl.setSuffix(" %")
        self._s_tp = QDoubleSpinBox(); self._s_tp.setRange(0.1,200.0); self._s_tp.setSingleStep(0.1); self._s_tp.setSuffix(" %")
        self._s_int = QComboBox(); self._s_int.addItems(["1m","2m","5m","15m","60m"])
        strat_form.addRow("Name", self._s_name)
        strat_form.addRow("Buy ≥", self._s_buy)
        strat_form.addRow("Sell ≤", self._s_sell)
        strat_form.addRow("Hysteresis", self._s_hys)
        strat_form.addRow("Cooldown (min)", self._s_cool)
        strat_form.addRow("Stop-Loss %", self._s_sl)
        strat_form.addRow("Take-Profit %", self._s_tp)
        strat_form.addRow("Interval", self._s_int)
        # actions
        strat_btns = QHBoxLayout()
        self._s_save = QPushButton("Save/Update"); self._s_save.clicked.connect(self._save_strategy)
        self._s_delete = QPushButton("Delete"); self._s_delete.clicked.connect(self._delete_strategy)
        strat_btns.addWidget(self._s_save); strat_btns.addWidget(self._s_delete)
        form_wrap = QVBoxLayout(); form_right = QFrame(); form_right.setLayout(QVBoxLayout()); form_right.layout().addLayout(strat_form); form_right.layout().addLayout(strat_btns);
        strat_layout.addWidget(form_right, 2)
        self._tabs.addTab(strat_tab, "Strategies")

        # --- Settings tab ---
        settings_tab = QWidget(); st_layout = QFormLayout()
        self._set_mode = QComboBox(); self._set_mode.addItems(["Paper","Live"]) 
        self._set_mode.setCurrentText("Paper" if self._paper_mode else "Live")
        self._set_interval = QComboBox(); self._set_interval.addItems(["1m","2m","5m","15m","60m"]); self._set_interval.setCurrentText(self._global_interval)
        self._set_hours = QCheckBox("Trading hours only"); self._set_hours.setChecked(self._cfg.config.globals.trading_hours_only)
        # Safety
        self._set_daily_loss = QDoubleSpinBox(); self._set_daily_loss.setPrefix("$"); self._set_daily_loss.setRange(0, 1_000_000); self._set_daily_loss.setValue(self._cfg.config.globals.daily_loss_limit)
        self._set_max_trades = QSpinBox(); self._set_max_trades.setRange(0, 1000); self._set_max_trades.setValue(self._cfg.config.globals.max_trades_day)
        st_layout.addRow("Mode", self._set_mode)
        st_layout.addRow("Global Interval", self._set_interval)
        st_layout.addRow(self._set_hours)
        st_layout.addRow("Daily loss limit", self._set_daily_loss)
        st_layout.addRow("Max trades / day", self._set_max_trades)
        # Perplexity custom prompt
        self._use_custom_prompt = QCheckBox("Use custom Perplexity prompt")
        self._use_custom_prompt.setChecked(self._cfg.config.globals.use_custom_prompt)
        st_layout.addRow(self._use_custom_prompt)
        self._custom_prompt_edit = QTextEdit(); self._custom_prompt_edit.setPlaceholderText("Enter the exact prompt to send every interval, e.g. 'Score TNA on a 0-10 scale for intraday momentum. Output only a number.'")
        self._custom_prompt_edit.setPlainText(self._cfg.config.globals.custom_prompt or "")
        st_layout.addRow("Custom prompt", self._custom_prompt_edit)
        self._save_settings_btn = QPushButton("Save Settings"); self._save_settings_btn.clicked.connect(self._save_settings)
        self._test_prompt_btn = QPushButton("Test Prompt Now"); self._test_prompt_btn.clicked.connect(self._test_custom_prompt)
        self._insert_template_btn = QPushButton("Insert template"); self._insert_template_btn.clicked.connect(self._insert_recommended_prompt)
        st_btn_row = QHBoxLayout(); st_btn_row.addStretch(); st_btn_row.addWidget(self._save_settings_btn); st_btn_row.addWidget(self._test_prompt_btn); st_btn_row.addWidget(self._insert_template_btn)
        settings_v = QVBoxLayout(settings_tab)
        settings_v.addLayout(st_layout)
        settings_v.addLayout(st_btn_row)
        self._tabs.addTab(settings_tab, "Settings")

    # --- Logs tab ---
        logs_tab = QWidget(); lyt = QVBoxLayout(logs_tab); 
        lyt.addWidget(QLabel("Activity Log")); 
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setPlaceholderText("Execution and decision events will appear here…");
        lyt.addWidget(self._log)
        self._tabs.addTab(logs_tab, "Logs")

    # moved to Logs tab

    # ---- Helpers ----
    def _open_settings(self):
        dlg = AutomationSettingsDialog(self)
        dlg.exec()

    def _toggle_advanced(self, checked: bool):
        self._advanced_frame.setVisible(checked)
        self._advanced_btn.setText("Advanced overrides ▾" if checked else "Advanced overrides ▸")

    def _on_add_asset(self):
        symbol = self._symbol_edit.text().strip().upper()
        if not symbol:
            return
        qty = self._qty_spin.value()
        profile = self._profile_combo.currentText()
        interval = "Global" if self._use_global_interval.isChecked() else self._custom_interval.currentText()
        # persist
        self._cfg.add_asset(AssetEntry(
            symbol=symbol,
            quantity=qty,
            strategy=profile,
            use_global_interval=self._use_global_interval.isChecked(),
            custom_interval=self._custom_interval.currentText(),
            enabled=False,
        ))
        # update table UI
        row = self._table.rowCount()
        self._table.insertRow(row)
        # On/Off checkbox
        on_chk = QCheckBox()
        on_chk.setChecked(False)
        on_chk.toggled.connect(lambda checked, sym=symbol: self._on_asset_enabled_changed(sym, checked))
        self._table.setCellWidget(row, 0, on_chk)
        self._table.setItem(row, 1, QTableWidgetItem(symbol))
        self._table.setItem(row, 2, QTableWidgetItem(profile))
        self._table.setItem(row, 3, QTableWidgetItem(str(qty)))
        self._table.setItem(row, 4, QTableWidgetItem(interval))
        self._table.setItem(row, 5, QTableWidgetItem("—"))
        self._table.setItem(row, 6, QTableWidgetItem("Idle"))
        # Actions column
        action_frame = QFrame()
        h = QHBoxLayout(action_frame)
        h.setContentsMargins(0, 0, 0, 0)
        pause = QPushButton("Pause")
        close = QPushButton("Close")
        h.addWidget(pause); h.addWidget(close)
        self._table.setCellWidget(row, 7, action_frame)

        # Clear inputs for next add
        self._symbol_edit.clear()
        # track row mapping for new asset
        self._asset_rows[symbol.upper()] = row

    # ---- Config wiring helpers ----
    def _load_from_config(self):
        # top bar
        self._mode_combo.setCurrentText(self._cfg.config.globals.mode)
        self._global_interval_combo.setCurrentText(self._cfg.config.globals.interval)
        self._hours_only.setChecked(self._cfg.config.globals.trading_hours_only)
        # assets table
        self._table.setRowCount(0)
        for a in self._cfg.config.assets:
            row = self._table.rowCount(); self._table.insertRow(row)
            # On/Off checkbox
            on_chk = QCheckBox(); on_chk.setChecked(a.enabled)
            on_chk.toggled.connect(lambda checked, sym=a.symbol: self._on_asset_enabled_changed(sym, checked))
            self._table.setCellWidget(row, 0, on_chk)
            self._table.setItem(row, 1, QTableWidgetItem(a.symbol))
            self._table.setItem(row, 2, QTableWidgetItem(a.strategy))
            self._table.setItem(row, 3, QTableWidgetItem(str(a.quantity)))
            interval = "Global" if a.use_global_interval else a.custom_interval
            self._table.setItem(row, 4, QTableWidgetItem(interval))
            self._table.setItem(row, 5, QTableWidgetItem("—"))
            self._table.setItem(row, 6, QTableWidgetItem("Idle"))
            action_frame = QFrame(); h = QHBoxLayout(action_frame); h.setContentsMargins(0,0,0,0)
            pause = QPushButton("Pause"); close = QPushButton("Close")
            h.addWidget(pause); h.addWidget(close)
            self._table.setCellWidget(row, 7, action_frame)
            # track row mapping
            self._asset_rows[a.symbol.upper()] = row
            # start timers for enabled assets
            if a.enabled:
                self._start_timer_for_asset(a)
        # strategies list
        self._refresh_strategy_list()

        # bind top controls to persist
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._global_interval_combo.currentTextChanged.connect(self._on_interval_changed)
        self._hours_only.toggled.connect(self._on_hours_toggled)

    def _refresh_strategy_list(self):
        self._strat_list.clear()
        for name in sorted(self._cfg.list_strategies()):
            QListWidgetItem(name, self._strat_list)

    # ---- Runtime helpers ----
    def _interval_to_ms(self, interval: str) -> int:
        try:
            val = int(interval.replace("m", ""))
            return max(1, val) * 60_000
        except Exception:
            return 60_000

    def _find_asset(self, symbol: str) -> AssetEntry | None:
        for a in self._cfg.config.assets:
            if a.symbol.upper() == symbol.upper():
                return a
        return None

    def _get_strategy_for_asset(self, asset: AssetEntry) -> Strategy:
        s = self._cfg.get_strategy(asset.strategy)
        if s:
            return s
        return Strategy(
            name=asset.strategy or "Intraday",
            buy_threshold=8.0,
            sell_threshold=4.0,
            hysteresis=2,
            cooldown_min=3,
            sl_pct=3.0,
            tp_pct=6.0,
            interval="1m",
        )

    def _ensure_state(self, symbol: str) -> dict:
        sym = symbol.upper()
        st = self._rt_state.get(sym)
        if not st:
            st = {"last_signal": "HOLD", "h_count": 0, "cooldown_until": None}
            self._rt_state[sym] = st
        return st

    def _row_for_symbol(self, symbol: str) -> int:
        return self._asset_rows.get(symbol.upper(), -1)

    def _set_status(self, symbol: str, text: str):
        row = self._row_for_symbol(symbol)
        if row >= 0:
            self._table.setItem(row, 6, QTableWidgetItem(text))

    def _on_asset_enabled_changed(self, symbol: str, enabled: bool):
        a = self._find_asset(symbol)
        if not a:
            return
        a.enabled = enabled
        self._cfg.save()
        if enabled:
            self._start_timer_for_asset(a)
            self._set_status(symbol, "Scheduled")
            # run an immediate evaluation so the user sees feedback right away
            self._evaluate_asset(symbol)
        else:
            self._stop_timer_for_asset(symbol)
            self._set_status(symbol, "Stopped")

    def _start_timer_for_asset(self, asset: AssetEntry, *, start_jitter_ms: int | None = None):
        sym = asset.symbol.upper()
        # stop existing first
        self._stop_timer_for_asset(sym)
        # choose interval
        interval = self._cfg.config.globals.interval if asset.use_global_interval else asset.custom_interval
        ms = self._interval_to_ms(interval)
        t = QTimer(self)
        t.setInterval(ms)
        t.timeout.connect(lambda s=sym: self._evaluate_asset(s))
        if start_jitter_ms and start_jitter_ms > 0:
            # start the repeating timer after a jitter to stagger load
            QTimer.singleShot(start_jitter_ms, t.start)
        else:
            t.start()
        self._timers[sym] = t

    def _stop_timer_for_asset(self, symbol: str):
        sym = symbol.upper()
        t = self._timers.pop(sym, None)
        if t:
            t.stop()
            t.deleteLater()

    def _evaluate_asset(self, symbol: str):
        # trading hours gating
        if self._cfg.config.globals.trading_hours_only and not self._is_trading_hours():
            self._set_status(symbol, "Off-hours")
            return
        self._reset_daily_counters_if_new_day()
        asset = self._find_asset(symbol)
        if not asset:
            return
        # short-lived cache: if recent score exists, reuse it instead of calling AI
        try:
            last = self._score_cache.get(symbol.upper())
            if last:
                ts, val = last
                if (datetime.now() - ts).total_seconds() < self._score_ttl_seconds:
                    # process cached value as if it just arrived
                    self._on_score_ready(symbol, val, None)
                    return
        except Exception:
            pass
        # get AI score asynchronously to avoid blocking UI
        if not hasattr(self, "_ai_service") or not self._ai_service:
            self._append_log(f"[{symbol}] AI service not configured; skipping evaluation")
            return
        sym = asset.symbol.upper()
        if sym in self._inflight:
            # don't start another request if one is already running
            self._append_log(f"[{symbol}] scoring already in-flight; skipping")
            return
        self._inflight.add(sym)
        self._set_status(sym, "Scoring…")
        self._executor.submit(self._score_worker, sym)

    def _score_worker(self, symbol: str):
        try:
            # pass profile to prompt builder for better alignment
            asset = self._find_asset(symbol)
            prof = asset.strategy if asset else None
            # include thresholds from the asset's strategy to guide scoring
            thresholds = None
            if asset:
                strat = self._get_strategy_for_asset(asset)
                thresholds = {
                    "buy_threshold": float(strat.buy_threshold),
                    "sell_threshold": float(strat.sell_threshold),
                }
            if self._cfg.config.globals.use_custom_prompt and (self._cfg.config.globals.custom_prompt or "").strip():
                cp = (self._cfg.config.globals.custom_prompt or "")
                # Fill placeholders
                cp = cp.replace("{symbol}", symbol)
                prof_txt = (prof or "Intraday")
                cp = cp.replace("{profile}", str(prof_txt))
                if asset:
                    try:
                        bt = float(strat.buy_threshold)
                        st = float(strat.sell_threshold)
                        cp = cp.replace("{buy}", f"{bt:.1f}")
                        cp = cp.replace("{sell}", f"{st:.1f}")
                    except Exception:
                        pass
                score = float(self._ai_service.score_with_custom_prompt_numeric_sync(cp))
            else:
                score = float(self._ai_service.score_symbol_numeric_sync(symbol, profile=prof, thresholds=thresholds))
            self.score_ready.emit(symbol, score, None)
        except Exception as e:
            self.score_ready.emit(symbol, 0.0, e)

    def _on_score_ready(self, symbol: str, score: float, error: object):
        # called on the main thread via Qt signal
        self._inflight.discard(symbol.upper())
        if error is not None:
            self._append_log(f"[{symbol}] AI error: {error}")
            self._set_status(symbol, "AI error")
            return
        # update cache
        try:
            self._score_cache[symbol.upper()] = (datetime.now(), float(score))
        except Exception:
            pass
        asset = self._find_asset(symbol)
        if not asset:
            # asset might have been removed while scoring
            return
        strat = self._get_strategy_for_asset(asset)
        signal = "BUY" if score >= float(strat.buy_threshold) else ("SELL" if score <= float(strat.sell_threshold) else "HOLD")
        st = self._ensure_state(symbol)
        now = datetime.now()
        if st.get("cooldown_until") and now < st["cooldown_until"]:
            self._set_status(symbol, "Cooldown")
            self._update_score_and_panel(symbol, score, "COOLDOWN")
            self._append_log(f"[{symbol}] cooldown active, signal={signal}, score={score}")
            return
        # hysteresis cycles
        if signal == st.get("last_signal") and signal != "HOLD":
            st["h_count"] = st.get("h_count", 0) + 1
        else:
            st["h_count"] = 1 if signal != "HOLD" else 0
            st["last_signal"] = signal
        needed = max(1, int(strat.hysteresis))
        if signal == "HOLD":
            self._set_status(symbol, "HOLD")
            self._update_score_and_panel(symbol, score, "HOLD")
            # reduce log spam: only log meaningful transitions
            return
        if st["h_count"] >= needed:
            decision = signal
            # guardrails: max trades/day
            max_trades = int(self._cfg.config.globals.max_trades_day or 0)
            if max_trades > 0 and self._daily_trade_count >= max_trades:
                self._set_status(symbol, "Max trades/day reached")
                self._update_score_and_panel(symbol, score, "GUARDRAIL: MAX-TRADES")
                self._append_log(f"[{symbol}] blocked by guardrail: max trades/day ({max_trades})")
                return
            # guardrails: daily loss limit (placeholder until real PnL)
            loss_limit = float(self._cfg.config.globals.daily_loss_limit or 0.0)
            if loss_limit > 0 and (-self._daily_pnl) >= loss_limit:
                self._set_status(symbol, "Daily loss limit reached")
                self._update_score_and_panel(symbol, score, "GUARDRAIL: LOSS-LIMIT")
                self._append_log(f"[{symbol}] blocked by guardrail: daily loss limit (${loss_limit:.2f})")
                return
            st["h_count"] = 0
            st["last_signal"] = signal
            st["cooldown_until"] = now + timedelta(minutes=max(0, int(strat.cooldown_min)))
            # increment daily counter (placeholder execution)
            self._daily_trade_count += 1
            self._set_status(symbol, decision)
            self._update_score_and_panel(symbol, score, decision)
            self._append_log(f"[{symbol}] decision={decision} score={score} (cooldown {strat.cooldown_min}m)")
            # Execute action according to mode (Paper/Live)
            try:
                self._execute_decision(symbol, decision, int(asset.quantity), strat)
            except Exception as ex:
                self._append_log(f"[{symbol}] execution error: {ex}")
            # Execute trade/paper-trade according to mode
            try:
                self._execute_decision(symbol, decision, int(asset.quantity), strat)
            except Exception as ex:
                self._append_log(f"[{symbol}] execution error: {ex}")
        else:
            phase = f"{signal} ({st['h_count']}/{needed})"
            self._set_status(symbol, phase)
            self._update_score_and_panel(symbol, score, phase)

    def _reset_daily_counters_if_new_day(self):
        today = date.today()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_trade_count = 0
            self._daily_pnl = 0.0
            self._append_log("Daily counters reset")

    def _update_score_and_panel(self, symbol: str, score: float, status_text: str, reason: str = ""):
        row = self._row_for_symbol(symbol)
        if row >= 0:
            self._table.setItem(row, 5, QTableWidgetItem(str(score)))
            self._table.setItem(row, 6, QTableWidgetItem(status_text))
        sel = self._table.currentRow()
        if sel == row and hasattr(self, "_summary_label"):
            self._summary_label.setText(f"Score: {score}    Action: {status_text}    Confidence: —")
            if hasattr(self, "_reason"):
                a = self._find_asset(symbol)
                sname = a.strategy if a else "?"
                self._reason.setPlainText(f"Strategy: {sname}\nScore={score}\nState={status_text}\n{reason}")

    def _on_selection_changed(self):
        row = self._table.currentRow()
        if row < 0:
            return
        symbol_item = self._table.item(row, 1)
        score_item = self._table.item(row, 5)
        status_item = self._table.item(row, 6)
        sym = symbol_item.text() if symbol_item else "—"
        score = score_item.text() if score_item else "—"
        status = status_item.text() if status_item else "—"
        if hasattr(self, "_summary_label"):
            self._summary_label.setText(f"Score: {score}    Action: {status}    Confidence: —")
        if hasattr(self, "_reason"):
            self._reason.setPlainText(f"Selected {sym}. Latest status: {status}, score: {score}.")

    def _append_log(self, text: str):
        try:
            self._log.appendPlainText(text)
        except Exception:
            pass

    def _start_all(self):
        # If a symbol is typed but not added, add it automatically for convenience
        pending_sym = self._symbol_edit.text().strip().upper()
        if pending_sym and not self._find_asset(pending_sym):
            self._append_log(f"Adding pending symbol '{pending_sym}' before start")
            self._on_add_asset()

        # Enable and start timers for all assets, then evaluate once immediately
        for idx, a in enumerate(self._cfg.config.assets):
            sym = a.symbol
            # ensure UI checkbox is checked
            row = self._row_for_symbol(sym)
            if row >= 0:
                w = self._table.cellWidget(row, 0)
                if isinstance(w, QCheckBox) and not w.isChecked():
                    w.setChecked(True)
            # set enabled in config if needed
            if not a.enabled:
                a.enabled = True
            # start timer with jitter to spread load and schedule initial eval with jitter
            jitter = 300 + (idx % 10) * 120  # deterministic light staggering
            self._start_timer_for_asset(a, start_jitter_ms=jitter)
            self._set_status(sym, "Scheduled")
            self._schedule_initial_eval(sym, jitter_ms=jitter)
        self._cfg.save()
        if not self._cfg.config.assets:
            self._append_log("No assets to start. Add symbols via the Add button.")
        else:
            self._append_log("Started all assets")

    def _schedule_initial_eval(self, symbol: str, *, jitter_ms: int = 0):
        # schedule a one-shot evaluation after a short jitter to avoid bursts
        delay = max(0, int(jitter_ms))
        QTimer.singleShot(delay, lambda s=symbol: self._evaluate_asset(s))

    def _stop_all(self):
        for sym in list(self._timers.keys()):
            self._stop_timer_for_asset(sym)
            self._set_status(sym, "Stopped")
        self._append_log("Stopped all assets")

    def _is_trading_hours(self) -> bool:
        try:
            now = datetime.now(ZoneInfo("US/Eastern"))
            # Mon-Fri 9:30-16:00 ET
            if now.weekday() >= 5:
                return False
            start = now.replace(hour=9, minute=30, second=0, microsecond=0)
            end = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return start <= now <= end
        except Exception:
            # fallback: always true if timezone not available
            return True

    def _pause_asset(self, symbol: str):
        a = self._find_asset(symbol)
        if not a:
            return
        # uncheck checkbox and disable
        row = self._row_for_symbol(symbol)
        if row >= 0:
            w = self._table.cellWidget(row, 0)
            if isinstance(w, QCheckBox):
                w.setChecked(False)
        self._on_asset_enabled_changed(symbol, False)
        self._append_log(f"Paused {symbol}")

    def _close_asset(self, symbol: str):
        # stop timer and remove from config and table
        self._stop_timer_for_asset(symbol)
        self._cfg.remove_asset(symbol)
        row = self._row_for_symbol(symbol)
        if row >= 0:
            self._table.removeRow(row)
            self._rebuild_row_mapping()
        self._append_log(f"Removed {symbol}")

    def _rebuild_row_mapping(self):
        self._asset_rows.clear()
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 1)
            if item:
                self._asset_rows[item.text().upper()] = r

    def closeEvent(self, event):
        # ensure timers are stopped on close
        for sym in list(self._timers.keys()):
            self._stop_timer_for_asset(sym)
        # shut down background executor
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().closeEvent(event)

    def _on_strat_selected(self, item: QListWidgetItem):
        name = item.text()
        s = self._cfg.get_strategy(name)
        if not s:
            return
        self._s_name.setText(s.name)
        self._s_buy.setValue(s.buy_threshold)
        self._s_sell.setValue(s.sell_threshold)
        self._s_hys.setValue(s.hysteresis)
        self._s_cool.setValue(s.cooldown_min)
        self._s_sl.setValue(s.sl_pct)
        self._s_tp.setValue(s.tp_pct)
        self._s_int.setCurrentText(s.interval)

    def _save_strategy(self):
        name = self._s_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Strategy", "Name is required")
            return
        s = Strategy(
            name=name,
            buy_threshold=self._s_buy.value(),
            sell_threshold=self._s_sell.value(),
            hysteresis=self._s_hys.value(),
            cooldown_min=self._s_cool.value(),
            sl_pct=self._s_sl.value(),
            tp_pct=self._s_tp.value(),
            interval=self._s_int.currentText(),
        )
        self._cfg.upsert_strategy(s)
        self._refresh_strategy_list()

    def _delete_strategy(self):
        name = self._s_name.text().strip()
        if not name:
            return
        self._cfg.delete_strategy(name)
        self._s_name.clear()
        self._refresh_strategy_list()

    def _save_settings(self):
        self._cfg.update_globals(
            mode=self._set_mode.currentText(),
            interval=self._set_interval.currentText(),
            trading_hours_only=self._set_hours.isChecked(),
            daily_loss_limit=self._set_daily_loss.value(),
            max_trades_day=self._set_max_trades.value(),
        )
        # Save custom prompt flags
        self._cfg.config.globals.use_custom_prompt = self._use_custom_prompt.isChecked()
        self._cfg.config.globals.custom_prompt = self._custom_prompt_edit.toPlainText().strip()
        self._cfg.save()
        # reflect in top bar too
        self._mode_combo.setCurrentText(self._cfg.config.globals.mode)
        self._global_interval_combo.setCurrentText(self._cfg.config.globals.interval)
        self._hours_only.setChecked(self._cfg.config.globals.trading_hours_only)

    def _test_custom_prompt(self):
        if not hasattr(self, "_ai_service") or not self._ai_service:
            QMessageBox.warning(self, "AI Service", "AI service not configured.")
            return
        use_cp = self._use_custom_prompt.isChecked()
        prompt = self._custom_prompt_edit.toPlainText().strip()
        if use_cp and not prompt:
            QMessageBox.information(self, "Custom Prompt", "Please enter a custom prompt or uncheck 'Use custom prompt'.")
            return
        # Run once against the first enabled asset or the typed symbol
        symbol = None
        for a in self._cfg.config.assets:
            if a.enabled:
                symbol = a.symbol; break
        if not symbol:
            symbol = self._symbol_edit.text().strip().upper() or "AAPL"
        try:
            if use_cp:
                # Minimal numeric result expected; reuse numeric endpoint but override prompt
                val = self._ai_service.score_with_custom_prompt_numeric_sync(prompt)
                self._append_log(f"[TEST] Custom prompt → {val}")
                QMessageBox.information(self, "Prompt Test", f"Numeric response: {val}")
            else:
                # Default numeric scoring for the symbol
                val = self._ai_service.score_symbol_numeric_sync(symbol)
                self._append_log(f"[TEST] Default scoring for {symbol} → {val}")
                QMessageBox.information(self, "Prompt Test", f"{symbol}: {val}")
        except Exception as e:
            QMessageBox.critical(self, "Prompt Test", str(e))

    # top bar change handlers
    def _on_mode_changed(self, text: str):
        self._cfg.update_globals(mode=text)

    def _on_interval_changed(self, text: str):
        self._cfg.update_globals(interval=text)

    def _on_hours_toggled(self, checked: bool):
        self._cfg.update_globals(trading_hours_only=checked)

    def _insert_recommended_prompt(self):
        """Insert a recommended numeric-only prompt template with placeholders."""
        template = (
            "Score the opportunity for '{symbol}' on a 0-10 scale for {profile} trading horizon. "
            "Focus on price action, momentum, and near-term catalysts. "
            "Calibrate internally with thresholds BUY≥{buy} and SELL≤{sell}, "
            "but output ONLY the number (0-10). No text, no code, no units."
        )
        self._custom_prompt_edit.setPlainText(template)

    # ---- Execution helpers ----
    def _execute_decision(self, symbol: str, decision: str, qty: int, strat: Strategy):
        if decision not in ("BUY", "SELL"):
            return
        mode = (self._cfg.config.globals.mode or "Paper").strip()
        if mode.lower() == "paper":
            # Simulated trade using current price via yfinance
            try:
                from utils.trading_helpers import get_current_price, update_portfolio
                price = get_current_price(symbol)
                if price is None:
                    self._append_log(f"[{symbol}] Paper trade skipped: no price")
                    return
                update_portfolio(symbol, decision, qty, float(price))
                self._append_log(f"[{symbol}] Paper trade: {decision} {qty} @ {price:.2f}")
            except Exception as e:
                self._append_log(f"[{symbol}] Paper trade error: {e}")
            return
        # Live: require IBKR service
        if not hasattr(self, "_ibkr_service") or not getattr(self, "_ibkr_service", None):
            self._append_log(f"[{symbol}] Live trade skipped: IBKR not connected")
            return
        try:
            from utils.ibkr_trading_helpers import booktrade_ibkr
            res = booktrade_ibkr(symbol, qty, decision, order_type="MKT", ibkr_service=self._ibkr_service)
            self._append_log(f"[{symbol}] Live trade result: {res.success} {res.message}")
        except Exception as e:
            self._append_log(f"[{symbol}] Live trade error: {e}")

    # ---- Execution helpers ----
    def _execute_decision(self, symbol: str, decision: str, qty: int, strat: Strategy):
        if decision not in ("BUY", "SELL"):
            return
        mode = (self._cfg.config.globals.mode or "Paper").strip()
        if mode.lower() == "paper":
            # Simulated trade using current price via yfinance
            try:
                from utils.trading_helpers import get_current_price, update_portfolio
                price = get_current_price(symbol)
                if price is None:
                    self._append_log(f"[{symbol}] Paper trade skipped: no price")
                    return
                update_portfolio(symbol, decision, qty, float(price))
                self._append_log(f"[{symbol}] Paper trade: {decision} {qty} @ {price:.2f}")
            except Exception as e:
                self._append_log(f"[{symbol}] Paper trade error: {e}")
            return
        # Live: require IBKR service
        if not hasattr(self, "_ibkr_service") or not self._ibkr_service:
            self._append_log(f"[{symbol}] Live trade skipped: IBKR not connected")
            return
        try:
            from utils.ibkr_trading_helpers import booktrade_ibkr
            res = booktrade_ibkr(symbol, qty, decision, order_type="MKT", ibkr_service=self._ibkr_service)
            self._append_log(f"[{symbol}] Live trade result: {res.success} {res.message}")
        except Exception as e:
            self._append_log(f"[{symbol}] Live trade error: {e}")
