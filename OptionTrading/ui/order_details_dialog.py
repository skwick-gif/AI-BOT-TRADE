from typing import List, Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QPushButton, QTextEdit

class OrderDetailsDialog(QDialog):
    replaceRequested = pyqtSignal(str)  # order_id
    cancelRequested = pyqtSignal(str)   # order_id

    def __init__(self, order_id: str, legs: List[Dict[str, Any]], history: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Order Details — {order_id}")
        self.resize(640, 480)
        self.order_id = order_id

        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"Order ID: {order_id}"))

        root.addWidget(QLabel("Legs"))
        legs_tbl = QTableWidget(0, 5); legs_tbl.setHorizontalHeaderLabels(["Position","Strike","Qty","Premium","Expiry"])
        legs_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for leg in legs:
            r = legs_tbl.rowCount(); legs_tbl.insertRow(r)
            legs_tbl.setItem(r,0,QTableWidgetItem(leg.get("pos",""))); legs_tbl.setItem(r,1,QTableWidgetItem(str(leg.get("strike",""))))
            legs_tbl.setItem(r,2,QTableWidgetItem(str(leg.get("qty","")))); legs_tbl.setItem(r,3,QTableWidgetItem(str(leg.get("premium",""))))
            legs_tbl.setItem(r,4,QTableWidgetItem(str(leg.get("expiry",""))))
        root.addWidget(legs_tbl)

        root.addWidget(QLabel("Status History"))
        hist = QTextEdit(); hist.setReadOnly(True)
        for ev in history:
            hist.append(f"[{ev.get('ts','')}] {ev.get('status','')}  {ev.get('info','')}")
        root.addWidget(hist)

        btns = QHBoxLayout()
        btn_replace = QPushButton("Replace"); btn_replace.clicked.connect(lambda: self.replaceRequested.emit(order_id))
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(lambda: self.cancelRequested.emit(order_id))
        btns.addStretch(1); btns.addWidget(btn_replace); btns.addWidget(btn_cancel)
        root.addLayout(btns)
