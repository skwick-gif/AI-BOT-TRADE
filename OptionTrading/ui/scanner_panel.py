from typing import Optional, List
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QComboBox, QSpinBox

class ScannerPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Stock Scanner"); title.setStyleSheet("font-weight:600"); header.addWidget(title)
        header.addSpacing(12); header.addWidget(QLabel("Source:"))
        self.source = QComboBox(); self.source.addItems(["Hybrid","IBKR-only","Local-only"]); header.addWidget(self.source)
        header.addSpacing(12); header.addWidget(QLabel("Max:"))
        self.max_items = QSpinBox(); self.max_items.setRange(1,50); self.max_items.setValue(50); header.addWidget(self.max_items)
        self.btn_scan = QPushButton("Scan"); header.addWidget(self.btn_scan)
        header.addStretch(1)
        layout.addLayout(header)

        self.table = QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["Symbol","Sector","Signal","Potential","Action"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); layout.addWidget(self.table)
        self.btn_scan.clicked.connect(self._emit_scan)
    def load_rows(self, rows: List[dict]):
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount(); self.table.insertRow(i)
            self.table.setItem(i,0,QTableWidgetItem(r.get("symbol",""))); self.table.setItem(i,1,QTableWidgetItem(r.get("sector","-"))); self.table.setItem(i,2,QTableWidgetItem(r.get("signal","-"))); self.table.setItem(i,3,QTableWidgetItem("High" if r.get("potential") else "Low"))
            btn = QPushButton("Select"); btn.clicked.connect(lambda _, s=r.get("symbol",""): self._bubble_symbol(s)); self.table.setCellWidget(i,4,btn)
    def _bubble_symbol(self, symbol: str):
        p=self.parent()
        while p is not None and not hasattr(p, "set_symbol_from_scanner"): p=p.parent()
        if p is not None: p.set_symbol_from_scanner(symbol)

    def _emit_scan(self):
        p=self.parent()
        while p is not None and not hasattr(p, "on_scan_bank_request"):
            p=p.parent()
        if p is not None:
            p.on_scan_bank_request(self.source.currentText(), int(self.max_items.value()))
