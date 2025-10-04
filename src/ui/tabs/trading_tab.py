"""
Trading tab for executing trades
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class TradingTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("🔧 Trading Tab - Under Construction"))
    
    def refresh_data(self):
        pass