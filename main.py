#!/usr/bin/env python3
"""
PyQt6 Trading Bot Application
Entry point for the trading application with AI agent integration
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QFont

from ui.windows.main_window import MainWindow
from core.config_manager import ConfigManager
from utils.logger import setup_logger
try:
    # Integrate ib_insync with Qt event loop when available
    from ib_insync import util as ib_util  # type: ignore
    ib_util.useQt()
except Exception:
    pass


def setup_application():
    """Setup application configuration and styling"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Trading Bot")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("TradingBot")
    app.setOrganizationDomain("tradingbot.local")
    
    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    return app


def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting AI Trading Bot PyQt6 Application")
    
    try:
        # Initialize configuration
        config = ConfigManager()
        logger.info("Config loaded")
        
        # Create and setup application
        app = setup_application()
        logger.info("QApplication created")

        # Create main window
        main_window = MainWindow()
        logger.info("Main window created")
        
        # Start in fullscreen-like mode (maximized for better Windows UX)
        main_window.showMaximized()
        logger.info("Main window shown")

        # Auto-connect disabled by user request
        # QTimer.singleShot(800, main_window.auto_connect_ibkr)

        logger.info("Application started successfully - entering event loop")

        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
