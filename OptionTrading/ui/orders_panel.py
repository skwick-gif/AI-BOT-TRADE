from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QHeaderView, QTableWidgetItem, QPushButton, QComboBox

@dataclass
class OrderRow:
    order_id: str
    symbol: str
    strategy_name: str
    qty: int
    pricing: str
    tif: str
    status: str = "PreSubmitted"
    filled: int = 0
    remaining: int = 0
    avg_price: float = 0.0

class OrdersPanel(QWidget):
    cancelRequested = pyqtSignal(str)     # order_id
    replaceRequested = pyqtSignal(str)    # order_id
    detailsRequested = pyqtSignal(str)    # order_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: Dict[str, OrderRow] = {}

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Orders"); title.setStyleSheet("font-weight:600")
        header.addWidget(title); header.addStretch(1)

        self.filter = QComboBox()
        self.filter.addItems(["All", "Working", "Filled", "Cancelled", "Rejected"])
        self.filter.currentIndexChanged.connect(self._apply_filter)
        header.addWidget(QLabel("Filter:"))
        header.addWidget(self.filter)
        root.addLayout(header)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Order ID","Symbol","Strategy","Qty","Pricing/TIF","Status","Filled","Avg Price","Actions"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table)

    def add_order(self, row: OrderRow):
        self.rows[row.order_id] = row
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(row.order_id))
        self.table.setItem(r, 1, QTableWidgetItem(row.symbol))
        self.table.setItem(r, 2, QTableWidgetItem(row.strategy_name))
        self.table.setItem(r, 3, QTableWidgetItem(str(row.qty)))
        self.table.setItem(r, 4, QTableWidgetItem(f"{row.pricing}/{row.tif}"))
        self.table.setItem(r, 5, QTableWidgetItem(row.status))
        self.table.setItem(r, 6, QTableWidgetItem(f"{row.filled}/{row.remaining}"))
        self.table.setItem(r, 7, QTableWidgetItem(f"{row.avg_price:.2f}"))

        actions = QWidget()
        h = QHBoxLayout(actions); h.setContentsMargins(0,0,0,0)
        btn_details = QPushButton("Details"); btn_details.clicked.connect(lambda _, oid=row.order_id: self.detailsRequested.emit(oid))
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(lambda _, oid=row.order_id: self.cancelRequested.emit(oid))
        btn_replace = QPushButton("Replace"); btn_replace.clicked.connect(lambda _, oid=row.order_id: self.replaceRequested.emit(oid))
        h.addWidget(btn_details); h.addWidget(btn_cancel); h.addWidget(btn_replace)
        self.table.setCellWidget(r, 8, actions)

    def update_status(self, order_id: str, status: str, filled: Optional[int]=None, remaining: Optional[int]=None, avg_price: Optional[float]=None):
        # find row
        for r in range(self.table.rowCount()):
            if self.table.item(r,0).text() == order_id:
                self.table.setItem(r, 5, QTableWidgetItem(status))
                if filled is not None or remaining is not None:
                    f = filled if filled is not None else self.rows[order_id].filled
                    rem = remaining if remaining is not None else self.rows[order_id].remaining
                    self.table.setItem(r, 6, QTableWidgetItem(f"{f}/{rem}"))
                    self.rows[order_id].filled = f; self.rows[order_id].remaining = rem
                if avg_price is not None:
                    self.table.setItem(r, 7, QTableWidgetItem(f"{avg_price:.2f}"))
                    self.rows[order_id].avg_price = avg_price
                self.rows[order_id].status = status
                break

    def _apply_filter(self):
        mode = self.filter.currentText()
        for r in range(self.table.rowCount()):
            st = self.table.item(r,5).text()
            show = True
            if mode == "Working": show = st in ("PreSubmitted","Submitted","PartiallyFilled")
            elif mode == "Filled": show = st == "Filled"
            elif mode == "Cancelled": show = st == "Cancelled"
            elif mode == "Rejected": show = st == "Rejected"
            self.table.setRowHidden(r, not show)
