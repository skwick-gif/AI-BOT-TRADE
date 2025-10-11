from __future__ import annotations
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QPushButton, QHBoxLayout, QFrame, QGridLayout
from PyQt6.QtCore import Qt, pyqtSignal


class RecommendationsView(QWidget):
    strategySelected = pyqtSignal(dict)
    backRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecommendationsView")

        self._current_price = None

        root = QVBoxLayout(self)
        self.back_btn = QPushButton("← Back to recommendations", self)
        self.back_btn.setVisible(False)  # visible only in details view elsewhere
        root.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Analysis summary header
        self.header_price = QLabel("Current Price: -")
        self.header_price.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self.header_price)

        # Scrollable area for strategy cards (grid, 2 per row)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.list_container = QWidget()
        self.grid_layout = QGridLayout(self.list_container)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(12)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(self.list_container)
        root.addWidget(scroll, 1)

        # wiring
        self.back_btn.clicked.connect(self.backRequested.emit)

    def set_analysis_context(self, *, current_price: Optional[float] = None):
        self._current_price = current_price
        if current_price is not None:
            self.header_price.setText(f"Current Price: ${current_price:,.2f}")
        else:
            self.header_price.setText("Current Price: -")

    def set_recommendations(self, strategies: List[Dict[str, Any]]):
        # clear grid
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        # empty state
        if not strategies:
            msg = QLabel("No strategies found. Check API key or try another symbol.")
            msg.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.grid_layout.addWidget(msg, 0, 0)
            return
        # add cards in 2 columns
        cols = 2
        for idx, st in enumerate(strategies):
            r = idx // cols
            c = idx % cols
            card = self._create_card(st)
            self.grid_layout.addWidget(card, r, c)

    def _create_card(self, st: Dict[str, Any]) -> QWidget:
        w = QFrame(self)
        w.setFrameShape(QFrame.Shape.StyledPanel)
        w.setStyleSheet(
            "QFrame { border: 1px solid #d9d9d9; border-radius: 6px; }"
        )
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        # Header: Title + Confidence badge
        hdr = QHBoxLayout()
        title = QLabel(st.get("name") or st.get("key") or "Strategy")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        hdr.addWidget(title, 1)
        conf = float(st.get("confidence", 0.0) or 0.0) * 100
        badge = QLabel(f"{conf:.0f}% confidence")
        badge.setStyleSheet("padding: 2px 6px; border-radius: 10px; background:#e8f8ee; color:#1a7f37; font-size: 11px;")
        hdr.addWidget(badge, 0, alignment=Qt.AlignmentFlag.AlignRight)
        v.addLayout(hdr)

        # Blurb
        blurb = st.get("blurb") or ""
        if blurb:
            desc = QLabel(blurb)
            desc.setWordWrap(True)
            v.addWidget(desc)

        # Timeframe & Success Probability lines
        tf = st.get("timeframe") or ""
        if tf:
            v.addWidget(QLabel(f"Timeframe: {tf}"))
        sp = float(st.get("success_prob", 0.0) or 0.0) * 100
        v.addWidget(QLabel(f"Success Probability: {sp:.0f}%"))

        # KPI row: Expected Profit (green) and Max Loss (red)
        kpi = QHBoxLayout()
        ep = float(st.get('expected_profit', 0) or 0)
        ep_lbl = QLabel(f"Expected Profit  "+ (f"+${ep:,.0f}" if ep >= 0 else f"-${abs(ep):,.0f}"))
        ep_lbl.setStyleSheet("color: #0a8a0a; font-weight: 600;")
        kpi.addWidget(ep_lbl)
        kpi.addStretch(1)
        ml = float(st.get('max_loss', 0) or 0)
        ml_lbl = QLabel(f"Max Loss: ${ml:,.0f}")
        ml_lbl.setStyleSheet("color: #cc1b1b; font-weight: 600;")
        kpi.addWidget(ml_lbl)
        v.addLayout(kpi)

        # Action row
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("Select Strategy →", self)
        btn.clicked.connect(lambda: self.strategySelected.emit(st))
        row.addWidget(btn)
        v.addLayout(row)

        return w
