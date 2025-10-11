from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout, QTableWidget, QHeaderView, QTableWidgetItem, QPushButton, QMessageBox
from services.pricing import price_greeks_pyvollib
from models.strategy import StrategyDetails

class StrategyDetailsPanel(QWidget):
    requestOrderTicket = pyqtSignal(StrategyDetails)
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent); self.details: Optional[StrategyDetails] = None
        root = QVBoxLayout(self)
        self.title = QLabel(""); self.title.setStyleSheet("font-size:16px; font-weight:600"); root.addWidget(self.title)
        kpi_box = QGroupBox("Metrics")
        grid = QGridLayout(kpi_box); self.kpi_profit = QLabel(""); self.kpi_loss = QLabel(""); self.kpi_prob = QLabel(""); self.kpi_rr = QLabel("")
        grid.addWidget(QLabel("Expected Profit"),0,0); grid.addWidget(self.kpi_profit,0,1); grid.addWidget(QLabel("Max Loss"),0,2); grid.addWidget(self.kpi_loss,0,3)
        grid.addWidget(QLabel("Success Probability"),1,0); grid.addWidget(self.kpi_prob,1,1); grid.addWidget(QLabel("Risk/Reward"),1,2); grid.addWidget(self.kpi_rr,1,3)
        root.addWidget(kpi_box)
        legs_box = QGroupBox("Position Details"); from PyQt6.QtWidgets import QVBoxLayout as _QVL; legs_layout=_QVL(legs_box)
        self.legs_table = QTableWidget(0,5); self.legs_table.setHorizontalHeaderLabels(["Position","Strike","Quantity","Premium","Expiry"]); self.legs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); legs_layout.addWidget(self.legs_table); root.addWidget(legs_box)
        greeks_box = QGroupBox("Greeks"); ggrid = QGridLayout(greeks_box); self.delta=QLabel(""); self.gamma=QLabel(""); self.theta=QLabel(""); self.vega=QLabel("")
        ggrid.addWidget(QLabel("Delta"),0,0); ggrid.addWidget(self.delta,0,1); ggrid.addWidget(QLabel("Gamma"),0,2); ggrid.addWidget(self.gamma,0,3)
        ggrid.addWidget(QLabel("Theta"),1,0); ggrid.addWidget(self.theta,1,1); ggrid.addWidget(QLabel("Vega"),1,2); ggrid.addWidget(self.vega,1,3); root.addWidget(greeks_box)
        self.btn_execute = QPushButton("⚡ Execute Trade & Send to Broker"); self.btn_execute.clicked.connect(self._emit_ticket); root.addWidget(self.btn_execute)
    def set_details(self, d: StrategyDetails):
        self.details = d; self.title.setText(f"{d.name} • Current Price: ${d.current_price:.2f} • {int(d.confidence*100)}% confidence")
        self.kpi_profit.setText(f"+${d.expected_profit:,.0f}"); self.kpi_loss.setText(f"${d.max_loss:,.0f}"); self.kpi_prob.setText(f"{int(d.success_prob*100)}%"); self.kpi_rr.setText(f"1:{d.risk_reward:.2f}")
        self.legs_table.setRowCount(0)
        # Compute fair value & Greeks via pricing service
        try:
                legs_payload = [{"pos": l.pos, "strike": float(l.strike), "qty": int(l.qty), "premium": (None if l.premium is None else float(l.premium)), "expiry": l.expiry} for l in d.legs]
                priced = price_greeks_pyvollib(legs_payload, spot=float(d.current_price))
                totals = priced.get("totals", {})
                # override panel greeks with computed totals
                self.delta.setText(f"{totals.get('delta',0):.3f}")
                self.gamma.setText(f"{totals.get('gamma',0):.4f}")
                self.theta.setText(f"{totals.get('theta',0):.2f}")
                self.vega.setText(f"{totals.get('vega',0):.2f}")
                computed_legs = priced.get("legs", [])
        except Exception:
            computed_legs = []

        for leg in d.legs: r=self.legs_table.rowCount(); self.legs_table.insertRow(r); self.legs_table.setItem(r,0,QTableWidgetItem(leg.pos)); self.legs_table.setItem(r,1,QTableWidgetItem(f"${leg.strike}")); self.legs_table.setItem(r,2,QTableWidgetItem(str(leg.qty))); self.legs_table.setItem(r,3,QTableWidgetItem(str(leg.premium if leg.premium is not None else '-'))); self.legs_table.setItem(r,4,QTableWidgetItem(leg.expiry))
        self.delta.setText(str(d.delta)); self.gamma.setText(str(d.gamma)); self.theta.setText(str(d.theta)); self.vega.setText(str(d.vega))
    def _emit_ticket(self):
        if not self.details: QMessageBox.warning(self,"Order","אין פרטי אסטרטגיה"); return
        self.requestOrderTicket.emit(self.details)
