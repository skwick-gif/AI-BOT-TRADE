from __future__ import annotations
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal


class StrategyDetailsView(QWidget):
    executeRequested = pyqtSignal(dict)
    selectChainRequested = pyqtSignal()
    backRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StrategyDetailsView")
        self._strategy: Dict[str, Any] = {}
        self._current_price: Optional[float] = None

        root = QVBoxLayout(self)

        self.back_btn = QPushButton("← Back to recommendations", self)
        self.back_btn.clicked.connect(self.backRequested.emit)
        root.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.title = QLabel("Strategy")
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(self.title)

        self.price_lbl = QLabel("Current Price: -")
        root.addWidget(self.price_lbl)

        # KPI row
        kpi = QHBoxLayout()
        self.exp_profit = QLabel("Expected Profit: -")
        self.max_loss = QLabel("Max Loss: -")
        self.success_prob = QLabel("Success Probability: -")
        self.risk_reward = QLabel("Risk/Reward: -")
        kpi.addWidget(self.exp_profit)
        kpi.addWidget(self.max_loss)
        kpi.addWidget(self.success_prob)
        kpi.addWidget(self.risk_reward)
        root.addLayout(kpi)

        # Pricing / Greeks summary
        greeks_row = QHBoxLayout()
        self.net_debit_lbl = QLabel("Net Debit: -")
        self.delta_lbl = QLabel("Δ: -")
        self.gamma_lbl = QLabel("Γ: -")
        self.theta_lbl = QLabel("Θ: -")
        self.vega_lbl = QLabel("V: -")
        greeks_row.addWidget(self.net_debit_lbl)
        greeks_row.addWidget(self.delta_lbl)
        greeks_row.addWidget(self.gamma_lbl)
        greeks_row.addWidget(self.theta_lbl)
        greeks_row.addWidget(self.vega_lbl)
        root.addLayout(greeks_row)

        # Legs table
    # Columns: Position, Strike, Quantity, Market Premium, IV, Computed Price, Δ, Γ, Θ, V, Expiry
    self.table = QTableWidget(0, 11, self)
    self.table.setHorizontalHeaderLabels(["Position", "Strike", "Quantity", "Market Premium", "IV", "Computed Price", "Δ", "Γ", "Θ", "V", "Expiry"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(QLabel("Position Details"))
        root.addWidget(self.table)

        # Actions row: Select Chain + Execute
        actions = QHBoxLayout()
        self.chain_btn = QPushButton("Select Chain", self)
        actions.addWidget(self.chain_btn)
        actions.addStretch(1)
        self.exec_btn = QPushButton("⚡ Execute Trade & Send to Broker", self)
        actions.addWidget(self.exec_btn)
        self.exec_btn.clicked.connect(self._on_execute)
        self.chain_btn.clicked.connect(self.selectChainRequested.emit)
        root.addLayout(actions)

    def set_context(self, *, current_price: Optional[float]):
        self._current_price = current_price
        if current_price is not None:
            self.price_lbl.setText(f"Current Price: ${current_price:,.2f}")
        else:
            self.price_lbl.setText("Current Price: -")

    def set_strategy(self, strategy: Dict[str, Any], legs: List[Dict[str, Any]]):
        self._strategy = dict(strategy)
        name = strategy.get("name") or strategy.get("key") or "Strategy"
        self.title.setText(name)

        # KPIs
        exp = float(strategy.get("expected_profit", 0) or 0)
        mxl = float(strategy.get("max_loss", 0) or 0)
        sp = float(strategy.get("success_prob", 0.0) or 0.0)
        rr = (exp / mxl) if mxl > 0 else 0.0
        self.exp_profit.setText(f"Expected Profit: ${exp:,.0f}")
        self.max_loss.setText(f"Max Loss: ${mxl:,.0f}")
        self.success_prob.setText(f"Success Probability: {sp*100:.0f}%")
        self.risk_reward.setText(f"Risk/Reward: {rr:0.2f}")

        # Legs table
        self.table.setRowCount(0)
        for leg in legs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(leg.get("pos", ""))))
            self.table.setItem(row, 1, QTableWidgetItem(str(leg.get("strike", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(leg.get("qty", ""))))
            # Market Premium
            mp = leg.get('premium') if leg.get('premium') is not None else None
            self.table.setItem(row, 3, QTableWidgetItem((f"${float(mp):.2f}") if mp is not None else "-"))
            # IV
            iv = leg.get('iv') if leg.get('iv') is not None else None
            self.table.setItem(row, 4, QTableWidgetItem((f"{float(iv):.3f}") if iv is not None else "-"))
            # Computed price from pricing model
            cp = leg.get('price') if leg.get('price') is not None else None
            self.table.setItem(row, 5, QTableWidgetItem((f"${float(cp):.2f}") if cp is not None else "-"))
            # Greeks per leg
            d = leg.get('delta')
            g = leg.get('gamma')
            t = leg.get('theta')
            v = leg.get('vega')
            self.table.setItem(row, 6, QTableWidgetItem((f"{float(d):.4f}") if d is not None else "-"))
            self.table.setItem(row, 7, QTableWidgetItem((f"{float(g):.6f}") if g is not None else "-"))
            self.table.setItem(row, 8, QTableWidgetItem((f"{float(t):.4f}") if t is not None else "-"))
            self.table.setItem(row, 9, QTableWidgetItem((f"{float(v):.4f}") if v is not None else "-"))
            self.table.setItem(row, 10, QTableWidgetItem(str(leg.get("expiry", ""))))

    def set_pricing_kpis(self, kpis: Dict[str, Any]):
        """Receive computed pricing and Greeks and display them."""
        try:
            totals = kpis.get('totals', {})
            debit = totals.get('debit', None)
            if debit is None:
                self.net_debit_lbl.setText("Net Debit: -")
            else:
                self.net_debit_lbl.setText(f"Net Debit: ${debit:,.2f}")
            self.delta_lbl.setText(f"Δ: {totals.get('delta', 0):.3f}")
            self.gamma_lbl.setText(f"Γ: {totals.get('gamma', 0):.3f}")
            self.theta_lbl.setText(f"Θ: {totals.get('theta', 0):.3f}")
            self.vega_lbl.setText(f"V: {totals.get('vega', 0):.3f}")
        except Exception:
            pass

    def _on_execute(self):
        payload = {"strategy": self._strategy}
        self.executeRequested.emit(payload)
