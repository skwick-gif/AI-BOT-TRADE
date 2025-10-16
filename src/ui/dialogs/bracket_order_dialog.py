from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt

from services.ibkr_adapter_service import IBKRAdapterService
from typing import Optional


class BracketOrderDialog(QDialog):
    def __init__(self, parent=None, service: Optional[IBKRAdapterService] = None):
        super().__init__(parent)
        self.setWindowTitle("Place Bracket Order")
        self.resize(420, 300)
        self.service = service

        layout = QVBoxLayout(self)
        grid = QGridLayout()

        # Symbol
        grid.addWidget(QLabel("Symbol"), 0, 0)
        self.symbol_edit = QLineEdit("AAPL")
        grid.addWidget(self.symbol_edit, 0, 1)

        # Quantity
        grid.addWidget(QLabel("Quantity"), 1, 0)
        self.qty_edit = QLineEdit("1")
        grid.addWidget(self.qty_edit, 1, 1)

        # Action
        grid.addWidget(QLabel("Action"), 2, 0)
        self.action_combo = QComboBox()
        self.action_combo.addItems(["BUY", "SELL"])
        grid.addWidget(self.action_combo, 2, 1)

        # Entry type
        grid.addWidget(QLabel("Entry Type"), 3, 0)
        self.entry_combo = QComboBox()
        self.entry_combo.addItems(["MKT", "LMT"])
        grid.addWidget(self.entry_combo, 3, 1)

        # Entry price (for LMT)
        grid.addWidget(QLabel("Entry Price"), 4, 0)
        self.entry_price_edit = QLineEdit("")
        self.entry_price_edit.setPlaceholderText("Required if Entry=LMT")
        grid.addWidget(self.entry_price_edit, 4, 1)

        # Take profit
        grid.addWidget(QLabel("Take Profit"), 5, 0)
        self.tp_edit = QLineEdit("")
        self.tp_edit.setPlaceholderText("Optional")
        grid.addWidget(self.tp_edit, 5, 1)

        # Stop loss
        grid.addWidget(QLabel("Stop Loss"), 6, 0)
        self.sl_edit = QLineEdit("")
        self.sl_edit.setPlaceholderText("Optional")
        grid.addWidget(self.sl_edit, 6, 1)

        # Outside RTH
        grid.addWidget(QLabel("Outside RTH"), 7, 0)
        self.outside_chk = QCheckBox()
        grid.addWidget(self.outside_chk, 7, 1)

        layout.addLayout(grid)

        # Buttons
        self.place_btn = QPushButton("Place Bracket Order")
        self.place_btn.clicked.connect(self._place)
        layout.addWidget(self.place_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

    def _place(self):
        if not self.service or not self.service.is_connected():
            QMessageBox.warning(self, "Not Connected", "Connect to IBKR first.")
            return
        try:
            symbol = self.symbol_edit.text().strip().upper()
            qty = int(self.qty_edit.text().strip())
            action = self.action_combo.currentText()
            entry_type = self.entry_combo.currentText()
            entry_price = float(self.entry_price_edit.text().strip()) if self.entry_price_edit.text().strip() else None
            tp = float(self.tp_edit.text().strip()) if self.tp_edit.text().strip() else None
            sl = float(self.sl_edit.text().strip()) if self.sl_edit.text().strip() else None
            outside = self.outside_chk.isChecked()

            if entry_type == "LMT" and entry_price is None:
                QMessageBox.warning(self, "Missing Price", "Entry price is required for LMT.")
                return

            result = self.service.place_bracket_order(
                symbol=symbol,
                quantity=qty,
                action=action,
                entry_type=entry_type,
                entry_price=entry_price,
                take_profit=tp,
                stop_loss=sl,
                outside_rth=outside,
            )
            if result.get("success") or ("parentOrderId" in result):
                QMessageBox.information(self, "Order Placed", f"Bracket order placed.\n{result}")
                self.accept()
            else:
                QMessageBox.warning(self, "Order Failed", str(result))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
