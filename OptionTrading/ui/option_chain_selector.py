from __future__ import annotations
from typing import List, Optional, Tuple
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QLineEdit
from PyQt6.QtCore import Qt

class OptionChainSelectorDialog(QDialog):
    """Allows choosing expiry and strikes with help from IBKR secdef params.
    - Shows expirations (YYYYMMDD)
    - Shows available strikes (nearest N around spot)
    - Suggests strikes via % OTM inputs (for strategies like verticals / covered call)
    Returns tuple: (expiry, chosen_strikes[List[float]])
    """
    def __init__(self, spot: float, expirations: List[str], strikes: List[float], mode: str = "vertical", need_two_expiries: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Expiry & Strikes")
        self.resize(760, 520)
        self.spot = float(spot or 0.0)
        self.expirations = sorted(expirations)
        self.strikes = sorted(strikes)
        self.mode = mode
        self.need_two = bool(need_two_expiries)
        self.result_expiry: Optional[str] = None
        self.result_strikes: List[float] = []

        root = QVBoxLayout(self)

        # Header with spot
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"Spot: ${self.spot:.2f}"))
        hdr.addStretch(1)
        root.addLayout(hdr)

        # Expiry + OTM selectors
        row = QHBoxLayout()
        row.addWidget(QLabel("Expiry"))
        self.expiry = QComboBox(); self.expiry.addItems(self.expirations)
        row.addWidget(self.expiry)

        row.addSpacing(20)
        row.addWidget(QLabel("% OTM (lower/upper)"))
        self.lower_pct = QSpinBox(); self.lower_pct.setRange(-50, 200); self.lower_pct.setValue(1)
        self.upper_pct = QSpinBox(); self.upper_pct.setRange(-50, 200); self.upper_pct.setValue(5)
        row.addWidget(self.lower_pct); row.addWidget(self.upper_pct)

        row.addSpacing(20)
        self.btn_suggest = QPushButton("Suggest")
        self.btn_suggest.clicked.connect(self._suggest)
        row.addWidget(self.btn_suggest)

        root.addLayout(row)

        # Strikes table
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Strike","Select"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.tbl, 1)
        self._populate_strikes()

        # Output field
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Chosen strikes (comma-separated):"))
        self.ed_out = QLineEdit(); self.ed_out.setPlaceholderText("e.g., 180,190  (order matters per strategy)")
        out_row.addWidget(self.ed_out, 1)
        root.addLayout(out_row)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK"); cancel = QPushButton("Cancel")
        ok.clicked.connect(self._accept); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        root.addLayout(btns)

    def _populate_strikes(self, window: int = 20):
        # show nearest strikes around spot
        if not self.strikes:
            return
        # find nearest index
        closest = min(range(len(self.strikes)), key=lambda i: abs(self.strikes[i]-self.spot))
        lo = max(0, closest - window//2); hi = min(len(self.strikes), lo + window)
        view = self.strikes[lo:hi]
        self.tbl.setRowCount(0)
        for k in view:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            self.tbl.setItem(r,0,QTableWidgetItem(f"{k}"))
            btn = QPushButton("Add")
            btn.clicked.connect(lambda _, val=k: self._add_strike(val))
            self.tbl.setCellWidget(r,1,btn)

    def _add_strike(self, val: float):
        arr = [x.strip() for x in self.ed_out.text().split(",") if x.strip()]
        arr.append(str(val))
        self.ed_out.setText(", ".join(arr))

    def _suggest(self):
        lower = self.spot * (1 + self.lower_pct.value()/100.0)
        upper = self.spot * (1 + self.upper_pct.value()/100.0)
        # snap to nearest listed strikes
        def snap(x): return min(self.strikes, key=lambda k: abs(k-x)) if self.strikes else round(x,2)
        k1 = snap(lower); k2 = snap(upper)
        if self.mode == "covered_call":
            # for CC: only upper (call) strike
            self.ed_out.setText(str(k2))
        else:
            # vertical default
            self.ed_out.setText(f"{k1}, {k2}")

    def _accept(self):
        txt = self.ed_out.text().strip()
        if not txt:
            self.result_expiry = self.expiry.currentText()
            if self.need_two:
                self.result_expiry_far = getattr(self, 'expiry_far').currentText()
            self.result_strikes = []
            self.accept(); return
        arr = [float(x.strip()) for x in txt.split(",") if x.strip()]
        self.result_expiry = self.expiry.currentText()
        if self.need_two:
            self.result_expiry_far = getattr(self, 'expiry_far').currentText()
        self.result_strikes = arr
        self.accept()
