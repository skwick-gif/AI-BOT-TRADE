from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Callable
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QWidget, QFrame, QMessageBox

@dataclass
class StrategyLeg:
    pos: Literal["Long Call", "Short Call", "Long Put", "Short Put"]
    strike: float
    qty: int
    premium: float | None
    expiry: str

@dataclass
class StrategyDetails:
    name: str
    current_price: float
    confidence: float
    expected_profit: float
    max_loss: float
    success_prob: float
    risk_reward: float
    legs: List[StrategyLeg]
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class OrderTicket:
    symbol: str
    quantity: int
    pricing_mode: Literal["MID", "LAST", "MARK", "LIMIT"]
    limit_price: Optional[float]
    tif: Literal["DAY", "GTC"]
    slippage_bps: int
    legs: List[StrategyLeg]
    mode: Literal["PAPER", "LIVE"]

class OrderTicketDialog(QDialog):
    orderSubmitted = pyqtSignal(dict)

    def __init__(self, symbol: str, strategy: StrategyDetails, account_margin_available: float, mode: Literal["PAPER","LIVE"]="PAPER", parent: QWidget | None=None, on_send: Callable[[OrderTicket], None] | None=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Order Ticket — {strategy.name}")
        self.setModal(True)
        self.setMinimumWidth(680)
        self.on_send = on_send
        self.symbol = symbol; self.strategy = strategy; self.account_margin_available = account_margin_available; self.mode = mode

        self.symbol_lbl = QLabel(f"{self.symbol}")
        self.mode_lbl = QLabel(self.mode); self.mode_lbl.setToolTip("מצב עבודה: PAPER = הדמיה, LIVE = מסחר אמיתי")

        self.qty = QSpinBox(); self.qty.setRange(1, 10_000); self.qty.setValue(1); self.qty.setToolTip("כמות יחידות אסטרטגיה. הכמות מוכפלת לכל ה-legs.")
        self.pricing = QComboBox(); self.pricing.addItems(["MID","LAST","MARK","LIMIT"]); self.pricing.setToolTip("שיטת תמחור: MID/LAST/MARK או LIMIT.")
        self.limit_price = QDoubleSpinBox(); self.limit_price.setRange(0.0, 1_000_000.0); self.limit_price.setDecimals(2); self.limit_price.setValue(self._suggest_limit()); self.limit_price.setToolTip("מחיר LIMIT כולל (נטו פרמיה). פעיל במצב LIMIT.")
        self.tif = QComboBox(); self.tif.addItems(["DAY","GTC"]); self.tif.setToolTip("תוקף: DAY או GTC")
        self.slippage = QSpinBox(); self.slippage.setRange(0,10_000); self.slippage.setValue(25); self.slippage.setToolTip("סליפאג' מותר (bps)")

        self.cost_lbl = QLabel(); self.margin_after_lbl = QLabel()

        # Layout
        root = QVBoxLayout(self)
        header = QHBoxLayout(); header.addWidget(QLabel("Symbol:")); header.addWidget(self.symbol_lbl); header.addSpacing(20); header.addWidget(QLabel("Mode:")); header.addWidget(self.mode_lbl); header.addStretch(1); root.addLayout(header)
        form = QFormLayout(); form.addRow("Quantity", self.qty); form.addRow("Pricing", self.pricing); form.addRow("Limit", self.limit_price); form.addRow("Time in Force", self.tif); form.addRow("Slippage (bps)", self.slippage); root.addLayout(form)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); root.addWidget(sep)
        metrics = QHBoxLayout(); metrics.addWidget(QLabel("Est. Cost/Max Loss:")); metrics.addWidget(self.cost_lbl); metrics.addSpacing(30); metrics.addWidget(QLabel("Margin After:")); metrics.addWidget(self.margin_after_lbl); metrics.addStretch(1); root.addLayout(metrics)
        btns = QHBoxLayout(); btns.addStretch(1); self.preview_btn = QPushButton("Preview"); self.send_btn = QPushButton("Send"); self.cancel_btn = QPushButton("Cancel"); btns.addWidget(self.preview_btn); btns.addWidget(self.send_btn); btns.addWidget(self.cancel_btn); root.addLayout(btns)

        # Logic
        self._refresh_metrics(); self._update_limit_enabled()
        self.qty.valueChanged.connect(self._refresh_metrics); self.pricing.currentTextChanged.connect(self._update_limit_enabled); self.limit_price.valueChanged.connect(self._refresh_metrics)
        self.preview_btn.clicked.connect(self._on_preview); self.send_btn.clicked.connect(self._on_send); self.cancel_btn.clicked.connect(self.reject)

    def _suggest_limit(self) -> float:
        if not self.strategy.legs: return 0.0
        premiums = [abs(l.premium or 0) for l in self.strategy.legs]
        return round(sum(premiums)/max(1,len(premiums)), 2)

    def _est_cost(self) -> float:
        return max(0.0, float(self.strategy.max_loss)) * max(1, self.qty.value())

    def _refresh_metrics(self) -> None:
        est = self._est_cost(); self.cost_lbl.setText(f"${est:,.2f}")
        self.margin_after_lbl.setText(f"${0.0:,.2f}")  # יוחלף בחישוב מרווח אמיתי אם יש לך נתוני חשבון

    def _update_limit_enabled(self) -> None:
        self.limit_price.setEnabled(self.pricing.currentText() == "LIMIT")

    def _validate(self) -> str | None:
        if self.pricing.currentText() == "LIMIT" and self.limit_price.value() <= 0: return "הזן מחיר LIMIT תקין (> 0)."
        return None

    def _on_preview(self) -> None:
        err = self._validate()
        if err: from PyQt6.QtWidgets import QMessageBox; QMessageBox.critical(self, "Preview", err); return
        from PyQt6.QtWidgets import QMessageBox; QMessageBox.information(self, "Preview", "ההזמנה נראית תקינה. ניתן לשלוח.")

    def _on_send(self) -> None:
        err = self._validate()
        if err: from PyQt6.QtWidgets import QMessageBox; QMessageBox.critical(self, "Order", err); return
        ticket = OrderTicket(symbol=self.symbol, quantity=int(self.qty.value()), pricing_mode=self.pricing.currentText(), limit_price=float(self.limit_price.value()) if self.pricing.currentText()=="LIMIT" else None, tif=self.tif.currentText(), slippage_bps=int(self.slippage.value()), legs=self.strategy.legs, mode=self.mode)
        if self.on_send: self.on_send(ticket)
        self.orderSubmitted.emit(asdict(ticket))
        from PyQt6.QtWidgets import QMessageBox; QMessageBox.information(self, "Order", "ההזמנה נשלחה.")
        self.accept()
