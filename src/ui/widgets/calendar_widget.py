"""
Calendar Widget
Simple calendar display with a placeholder for upcoming events.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QCalendarWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from utils.logger import get_logger


class CalendarWidget(QFrame):
    """Calendar section for the dashboard"""

    def __init__(self):
        super().__init__()
        self.logger = get_logger("Calendar")
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)

        # Title
        title = QLabel("Calendar")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Content
        content_layout = QHBoxLayout()

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        content_layout.addWidget(self.calendar, 2)

        # Placeholder for events list
        self.events_label = QLabel("Upcoming events\n\n• Earnings, dividends, and macro events will appear here.")
        self.events_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.events_label.setStyleSheet("color: #666;")
        content_layout.addWidget(self.events_label, 1)

        layout.addLayout(content_layout)
