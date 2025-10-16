from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton, QMessageBox, QCheckBox
)

from services.ibkr_adapter_service import IBKRAdapterService
from typing import Optional


class OcoOrderDialog(QDialog):
    def __init__(self, parent=None, service: Optional[IBKRAdapterService] = None):
        super().__init__(parent)
        self.setWindowTitle("Place OCO Orders")
        self.resize(420, 260)
        self.service = service

        layout = QVBoxLayout(self)
        grid = QGridLayout()

        grid.addWidget(QLabel("Symbol"), 0, 0)
        self.symbol_edit = QLineEdit("AAPL")
        grid.addWidget(self.symbol_edit, 0, 1)

        grid.addWidget(QLabel("Quantity"), 1, 0)
        self.qty_edit = QLineEdit("1")
        grid.addWidget(self.qty_edit, 1, 1)

        grid.addWidget(QLabel("Side"), 2, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItems(["SELL", "BUY"])  # usually SELL for a long position
        grid.addWidget(self.side_combo, 2, 1)

        grid.addWidget(QLabel("Limit Price (TP)"), 3, 0)
        self.lmt_edit = QLineEdit("")
        self.lmt_edit.setPlaceholderText("e.g. 200.50")
        grid.addWidget(self.lmt_edit, 3, 1)

        grid.addWidget(QLabel("Stop Price (SL)"), 4, 0)
        self.stp_edit = QLineEdit("")
        self.stp_edit.setPlaceholderText("e.g. 175.00")
        grid.addWidget(self.stp_edit, 4, 1)

        # Outside RTH
        self.outside_rth_chk = QCheckBox("Allow Outside RTH")
        grid.addWidget(self.outside_rth_chk, 5, 0, 1, 2)

        layout.addLayout(grid)

        self.place_btn = QPushButton("Place OCO")
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
            side = self.side_combo.currentText()
            lmt = float(self.lmt_edit.text().strip()) if self.lmt_edit.text().strip() else None
            stp = float(self.stp_edit.text().strip()) if self.stp_edit.text().strip() else None

            orders = []
            if lmt is not None:
                orders.append({
                    "action": side,
                    "orderType": "LMT",
                    "quantity": qty,
                    "price": lmt,
                })
            if stp is not None:
                orders.append({
                    "action": side,
                    "orderType": "STP",
                    "quantity": qty,
                    "stopPrice": stp,
                })
            if len(orders) < 2:
                QMessageBox.warning(self, "Missing Legs", "Provide at least a limit and a stop for OCO.")
                return

            result = self.service.place_oco_orders(symbol=symbol, orders=orders, outside_rth=self.outside_rth_chk.isChecked())
            if result.get("success") or result.get("orderIds"):
                QMessageBox.information(self, "OCO Placed", f"OCO orders placed.\n{result}")
                self.accept()
            else:
                QMessageBox.warning(self, "Order Failed", str(result))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
