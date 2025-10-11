from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt


class OptionsWidget(QWidget):
    """Top-level OPTIONS widget containing two sub-tabs: MAIN and BANK.

    This is a scaffold; you can extend each sub-tab with real controls.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Sub-tabs container
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # MAIN sub-tab
        try:
            from ui.widgets.options.main_panel import OptionsMainPanel
            strategies_path = "config/strategies.json"
            main_panel = OptionsMainPanel(strategies_path, self)
            self.tabs.addTab(main_panel, "MAIN")
        except Exception as e:
            fallback = QWidget()
            fl = QVBoxLayout(fallback)
            lbl = QLabel(f"MAIN failed to load: {e}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            fl.addWidget(lbl)
            self.tabs.addTab(fallback, "MAIN")

        # BANK sub-tab
        try:
            from ui.widgets.options.bank_panel import OptionsBankPanel
            bank_panel = OptionsBankPanel(self)
            self.tabs.addTab(bank_panel, "BANK")
        except Exception as e:
            fallback2 = QWidget()
            fl2 = QVBoxLayout(fallback2)
            lbl2 = QLabel(f"BANK failed to load: {e}")
            lbl2.setAlignment(Qt.AlignmentFlag.AlignLeft)
            fl2.addWidget(lbl2)
            self.tabs.addTab(fallback2, "BANK")
