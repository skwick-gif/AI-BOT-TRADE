from __future__ import annotations
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QComboBox, QPushButton, QDialogButtonBox


class OrderTicketDialog(QDialog):
    def __init__(self, *, symbol: str, strategy_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Order Ticket — {strategy_name}")

        layout = QVBoxLayout(self)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"Symbol\n{symbol}"))
        hdr.addWidget(QLabel("Mode\nPAPER"))
        layout.addLayout(hdr)

        form = QFormLayout()
        self.qty = QLineEdit(self)
        self.qty.setText("1")
        self.pricing = QComboBox(self)
        self.pricing.addItems(["MID", "MARKET", "LIMIT"])
        self.limit = QLineEdit(self)
        self.limit.setText("3.85")
        self.tif = QComboBox(self)
        self.tif.addItems(["DAY", "GTC"]) 
        self.slippage = QLineEdit(self)
        self.slippage.setText("25")

        form.addRow("Quantity", self.qty)
        form.addRow("Pricing", self.pricing)
        form.addRow("Limit", self.limit)
        form.addRow("Time in Force", self.tif)
        form.addRow("Slippage (bps)", self.slippage)
        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview", self)
        self.send_btn = QPushButton("Send", self)
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.send_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self._result: Optional[Dict[str, Any]] = None
        self.send_btn.clicked.connect(self._on_send)

    def _on_send(self):
        try:
            self._result = {
                "quantity": int(self.qty.text() or "1"),
                "pricing": self.pricing.currentText(),
                "limit": float(self.limit.text() or 0),
                "tif": self.tif.currentText(),
                "slippage_bps": float(self.slippage.text() or 0),
            }
        except Exception:
            self._result = None
        self.accept()

    def result_payload(self) -> Optional[Dict[str, Any]]:
        return self._result
