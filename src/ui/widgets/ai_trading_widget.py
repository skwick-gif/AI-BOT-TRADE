"""
AI Trading Widget (UI Skeleton)
Visual skeleton for automated AI-driven trading control panel.
No backend logic or dummy data—just structure and hooks.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QPlainTextEdit, QDialog,
    QTabWidget, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


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
        self._global_interval = "1m"
        self._paper_mode = True

        self._build_ui()

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
        self._hours_only.setChecked(True)
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

        # Add Asset group
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

        root.addWidget(add_group)

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

        root.addWidget(split)

        # Bottom log
        log_title = QLabel("Activity Log")
        log_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        root.addWidget(log_title)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setPlaceholderText("Execution and decision events will appear here…")
        root.addWidget(self._log)

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
