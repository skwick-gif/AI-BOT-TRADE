from __future__ import annotations
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QCheckBox
from PyQt6.QtCore import Qt


class PreExecuteConfirmDialog(QDialog):
    """Show Net Debit, Greeks and leg detail and ask user to confirm before sending a real order."""

    def __init__(self, kpis: Optional[Dict[str, Any]] = None, legs: Optional[List[Dict[str, Any]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Order")
        self.resize(640, 360)
        self._confirmed = False
        self._dry_run = True

        layout = QVBoxLayout(self)

        if kpis:
            totals = kpis.get('totals', {})
            debit = totals.get('debit')
            layout.addWidget(QLabel(f"Net Debit: ${debit:,.2f}" if debit is not None else "Net Debit: -"))
            layout.addWidget(QLabel(f"Δ: {totals.get('delta', 0):.3f}   Γ: {totals.get('gamma', 0):.3f}   Θ: {totals.get('theta', 0):.3f}   V: {totals.get('vega', 0):.3f}"))
        else:
            layout.addWidget(QLabel("Pricing KPIs: -"))

        layout.addWidget(QLabel("Legs:"))
        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(["Position", "Strike", "Qty", "Market Prem", "Price"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        if legs:
            for leg in legs:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(leg.get('pos', ''))))
                self.table.setItem(r, 1, QTableWidgetItem(str(leg.get('strike', ''))))
                self.table.setItem(r, 2, QTableWidgetItem(str(leg.get('qty', ''))))
                mp = leg.get('premium')
                self.table.setItem(r, 3, QTableWidgetItem((f"${float(mp):.2f}") if mp is not None else "-"))
                cp = leg.get('price')
                self.table.setItem(r, 4, QTableWidgetItem((f"${float(cp):.2f}") if cp is not None else "-"))

        opts = QHBoxLayout()
        self.dry_chk = QCheckBox("Dry run / simulate (do not send to broker)", self)
        self.dry_chk.setChecked(True)
        opts.addWidget(self.dry_chk)
        opts.addStretch(1)
        self.ok_btn = QPushButton("Confirm & Continue", self)
        self.cancel_btn = QPushButton("Cancel", self)
        opts.addWidget(self.cancel_btn)
        opts.addWidget(self.ok_btn)
        layout.addLayout(opts)

        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_ok(self):
        self._confirmed = True
        self._dry_run = bool(self.dry_chk.isChecked())
        self.accept()

    def result(self) -> Dict[str, Any]:
        return {"confirmed": self._confirmed, "dry_run": self._dry_run}
