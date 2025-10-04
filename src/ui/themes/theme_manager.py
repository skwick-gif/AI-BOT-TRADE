"""
Theme Manager for PyQt6 Application
Based on the provided CSS styling
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject
from pathlib import Path


class ThemeManager(QObject):
    """Manages application themes and styling"""
    
    def __init__(self):
        super().__init__()
        self.themes_dir = Path(__file__).parent
        self.current_theme = "dark"
    
    def apply_theme(self, app_or_widget, theme_name: str = "dark"):
        """Apply theme to application or widget"""
        self.current_theme = theme_name
        
        if theme_name == "dark":
            stylesheet = self.get_dark_theme()
        else:
            stylesheet = self.get_light_theme()
        
        # Apply to application or specific widget
        if hasattr(app_or_widget, 'setStyleSheet'):
            app_or_widget.setStyleSheet(stylesheet)
        elif isinstance(app_or_widget, QApplication):
            app_or_widget.setStyleSheet(stylesheet)
    
    def get_dark_theme(self) -> str:
        """Get dark theme stylesheet"""
        return """
/* Dark Theme Stylesheet */

/* Main Application */
QMainWindow {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #2b2b2b, stop: 1 #1e1e1e);
    color: #ffffff;
    border: none;
}

/* Central Widget */
QWidget {
    background-color: transparent;
    color: #ffffff;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9pt;
}

/* Tab Widget */
QTabWidget::pane {
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #2b2b2b, stop: 1 #1e1e1e);
}

QTabWidget::tab-bar {
    alignment: left;
}

QTabBar::tab {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3d3d3d, stop: 1 #2b2b2b);
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    min-width: 120px;
    padding: 8px 16px;
    margin: 2px;
    color: #ffffff;
    font-weight: bold;
}

QTabBar::tab:selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4CAF50, stop: 1 #45a049);
    border-color: #4CAF50;
}

QTabBar::tab:hover:!selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4d4d4d, stop: 1 #3d3d3d);
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4CAF50, stop: 1 #45a049);
    border: 2px solid #4CAF50;
    border-radius: 8px;
    color: white;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 80px;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #5CBF60, stop: 1 #4CAF50);
    border-color: #5CBF60;
}

QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #45a049, stop: 1 #3d8b40);
}

QPushButton:disabled {
    background: #3d3d3d;
    border-color: #2d2d2d;
    color: #888888;
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px;
    color: #ffffff;
    selection-background-color: #4CAF50;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #4CAF50;
}

/* ComboBox */
QComboBox {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px;
    color: #ffffff;
    min-width: 100px;
}

QComboBox:hover {
    border-color: #4CAF50;
}

QComboBox::drop-down {
    border: none;
    border-left: 2px solid #3d3d3d;
    border-radius: 0px 6px 6px 0px;
    background-color: #3d3d3d;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-style: solid;
    border-width: 4px 4px 0px 4px;
    border-color: transparent transparent #ffffff transparent;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    selection-background-color: #4CAF50;
    color: #ffffff;
}

/* Tables */
QTableWidget, QTreeWidget {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    color: #ffffff;
    gridline-color: #3d3d3d;
    alternate-background-color: #2a2a2a;
}

QTableWidget::item, QTreeWidget::item {
    padding: 8px;
    border: none;
}

QTableWidget::item:selected, QTreeWidget::item:selected {
    background-color: #4CAF50;
    color: white;
}

QHeaderView::section {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4d4d4d, stop: 1 #3d3d3d);
    border: 1px solid #3d3d3d;
    padding: 8px;
    color: #ffffff;
    font-weight: bold;
}

/* Scrollbars */
QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5d5d5d;
}

QScrollBar:horizontal {
    background-color: #2d2d2d;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #5d5d5d;
}

/* Menu Bar */
QMenuBar {
    background-color: #2b2b2b;
    color: #ffffff;
    border-bottom: 2px solid #3d3d3d;
}

QMenuBar::item {
    background-color: transparent;
    padding: 8px 16px;
}

QMenuBar::item:selected {
    background-color: #4CAF50;
    border-radius: 4px;
}

QMenu {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    color: #ffffff;
}

QMenu::item {
    padding: 8px 20px;
}

QMenu::item:selected {
    background-color: #4CAF50;
}

/* Status Bar */
QStatusBar {
    background-color: #2b2b2b;
    border-top: 2px solid #3d3d3d;
    color: #ffffff;
}

/* Tool Bar */
QToolBar {
    background-color: #2b2b2b;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    spacing: 4px;
    padding: 4px;
}

/* Progress Bar */
QProgressBar {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4CAF50, stop: 1 #45a049);
    border-radius: 4px;
}

/* Splitter */
QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

/* Group Box */
QGroupBox {
    font-weight: bold;
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0px 5px 0px 5px;
    color: #ffffff;
}

/* Labels */
QLabel {
    color: #ffffff;
    background-color: transparent;
}

/* Spin Box */
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    padding: 4px;
    color: #ffffff;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #4CAF50;
}
"""
    
    def get_light_theme(self) -> str:
        """Get light theme stylesheet"""
        return """
/* Light Theme Stylesheet */

/* Main Application */
QMainWindow {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #f5f5f5, stop: 1 #e0e0e0);
    color: #333333;
    border: none;
}

/* Central Widget */
QWidget {
    background-color: transparent;
    color: #333333;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9pt;
}

/* Tab Widget */
QTabWidget::pane {
    border: 2px solid #cccccc;
    border-radius: 8px;
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #ffffff, stop: 1 #f0f0f0);
}

QTabBar::tab {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #e0e0e0, stop: 1 #cccccc);
    border: 2px solid #cccccc;
    border-radius: 8px;
    min-width: 120px;
    padding: 8px 16px;
    margin: 2px;
    color: #333333;
    font-weight: bold;
}

QTabBar::tab:selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4CAF50, stop: 1 #45a049);
    border-color: #4CAF50;
    color: white;
}

QTabBar::tab:hover:!selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #f0f0f0, stop: 1 #e0e0e0);
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4CAF50, stop: 1 #45a049);
    border: 2px solid #4CAF50;
    border-radius: 8px;
    color: white;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 80px;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #5CBF60, stop: 1 #4CAF50);
    border-color: #5CBF60;
}

QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #45a049, stop: 1 #3d8b40);
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    border: 2px solid #cccccc;
    border-radius: 6px;
    padding: 8px;
    color: #333333;
    selection-background-color: #4CAF50;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #4CAF50;
}

/* Tables */
QTableWidget, QTreeWidget {
    background-color: #ffffff;
    border: 2px solid #cccccc;
    border-radius: 8px;
    color: #333333;
    gridline-color: #e0e0e0;
    alternate-background-color: #f5f5f5;
}

QTableWidget::item:selected, QTreeWidget::item:selected {
    background-color: #4CAF50;
    color: white;
}

QHeaderView::section {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #f0f0f0, stop: 1 #e0e0e0);
    border: 1px solid #cccccc;
    padding: 8px;
    color: #333333;
    font-weight: bold;
}

/* Menu Bar */
QMenuBar {
    background-color: #f0f0f0;
    color: #333333;
    border-bottom: 2px solid #cccccc;
}

QMenuBar::item:selected {
    background-color: #4CAF50;
    color: white;
    border-radius: 4px;
}

/* Status Bar */
QStatusBar {
    background-color: #f0f0f0;
    border-top: 2px solid #cccccc;
    color: #333333;
}
"""