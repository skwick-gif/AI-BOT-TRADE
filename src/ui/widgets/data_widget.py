from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget


class DataWidget(QWidget):
    """Dedicated widget for the top-level DATA tab.

    Contains two empty sub-tabs for now:
    - Daily Update
    - Report Viewer
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Empty placeholders for the two sub-tabs
        self.daily_update_tab = QWidget()
        self.report_viewer_tab = QWidget()

        self.tabs.addTab(self.daily_update_tab, "Daily Update")
        self.tabs.addTab(self.report_viewer_tab, "Report Viewer")
