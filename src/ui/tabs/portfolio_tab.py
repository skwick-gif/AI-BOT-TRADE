"""
Portfolio tab for viewing portfolio performance
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class PortfolioTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("💼 Portfolio Tab - Under Construction"))
    
    def refresh_data(self):
        pass