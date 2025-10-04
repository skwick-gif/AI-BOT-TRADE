"""
AI Trading Widget (UI Skeleton)
Visual skeleton for automated AI-driven trading control panel.
No backend logic or dummy data—just structure and hooks.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QPlainTextEdit, QDialog,
    QTabWidget, QFormLayout, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from core.ai_trading_config import AiTradingConfigManager, Strategy, AssetEntry


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

    def __init__(self):
        super().__init__()
        self._ibkr_connected = False
        self._api_ok = False
        self._cfg = AiTradingConfigManager()  # persistent config
        self._global_interval = self._cfg.config.globals.interval
        self._paper_mode = (self._cfg.config.globals.mode == "Paper")

        self._build_ui()
        self._load_from_config()

    # ---- Public hooks for future wiring ----
    def set_ibkr_service(self, service):
        """Placeholder to receive IBKR service later."""
        # No-op in skeleton
        pass

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
        self._start_all_btn.clicked.connect(self.start_all_requested.emit)
        top.addWidget(self._start_all_btn)
        self._stop_all_btn = QPushButton("Stop All")
        self._stop_all_btn.clicked.connect(self.stop_all_requested.emit)
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
        table_layout.addWidget(self._table)

        split.addWidget(table_frame)

        # Right: selected asset panel (skeleton)
        right = QFrame(); right_layout = QVBoxLayout(right)
        right_title = QLabel("Selected Asset")
        right_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(right_title)

        right_layout.addWidget(QLabel("Score: —    Action: —    Confidence: —"))
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
        st_layout.addRow("Mode", self._set_mode)
        st_layout.addRow("Global Interval", self._set_interval)
        st_layout.addRow(self._set_hours)
        self._save_settings_btn = QPushButton("Save Settings"); self._save_settings_btn.clicked.connect(self._save_settings)
        st_btn_row = QHBoxLayout(); st_btn_row.addStretch(); st_btn_row.addWidget(self._save_settings_btn)
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
        # On (placeholder text “Off”)
        self._table.setItem(row, 0, QTableWidgetItem("Off"))
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
            self._table.setItem(row, 0, QTableWidgetItem("On" if a.enabled else "Off"))
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
        )
        # reflect in top bar too
        self._mode_combo.setCurrentText(self._cfg.config.globals.mode)
        self._global_interval_combo.setCurrentText(self._cfg.config.globals.interval)
        self._hours_only.setChecked(self._cfg.config.globals.trading_hours_only)

    # top bar change handlers
    def _on_mode_changed(self, text: str):
        self._cfg.update_globals(mode=text)

    def _on_interval_changed(self, text: str):
        self._cfg.update_globals(interval=text)

    def _on_hours_toggled(self, checked: bool):
        self._cfg.update_globals(trading_hours_only=checked)
