"""
InterReact Bridge Adapter
Adapter for connecting Python UI to InterReactBridge C# server
"""

import requests
import logging
from typing import Dict, Any, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread

logger = logging.getLogger(__name__)


class DataFetchThread(QThread):
    """Background thread for fetching data without blocking UI"""
    portfolio_fetched = pyqtSignal(list)
    account_fetched = pyqtSignal(dict)
    
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.running = True
    
    def run(self):
        """Fetch data in background"""
        try:
            # Fetch portfolio
            response = requests.get(f"{self.base_url}/portfolio", timeout=5)
            if response.status_code == 200:
                portfolio = response.json()
                self.portfolio_fetched.emit(portfolio)
                self.logger.debug(f"Background fetched {len(portfolio)} positions")
        except Exception as e:
            self.logger.debug(f"Background portfolio fetch failed: {e}")
        
        try:
            # Fetch account
            response = requests.get(f"{self.base_url}/account", timeout=5)
            if response.status_code == 200:
                account = response.json()
                self.account_fetched.emit(account)
                self.logger.debug(f"Background fetched account: {len(account)} fields")
        except Exception as e:
            self.logger.debug(f"Background account fetch failed: {e}")
    
    def stop(self):
        """Stop the thread"""
        self.running = False


class InterReactBridgeAdapter(QObject):
    """
    Adapter that makes InterReactBridge compatible with existing UI expectations.
    This adapter wraps the InterReactBridge REST API and provides the same interface
    as IBKRService, allowing seamless integration with the Dashboard.
    """
    
    # Signals for connection status
    connection_status_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    portfolio_updated = pyqtSignal(list)  # Emits portfolio data when fetched
    account_updated = pyqtSignal(dict)  # Emits account data when fetched
    
    def __init__(self, host: str = "localhost", port: int = 5080, auto_start_timer: bool = False):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self._connected = False
        self._tws_connected = False  # Cache TWS connection status
        self.logger = logger
        
        # Cache for data
        self._portfolio_cache = []
        self._account_cache = {}
        
        # Setup connection check timer (but don't start automatically)
        self.connection_timer = QTimer(self)
        self.connection_timer.timeout.connect(self._check_connection)  # Use sync version with reduced timeout
        
        # Setup data fetch timer
        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self._start_data_fetch)
        
        # Data fetch thread
        self._fetch_thread = None
        
        # Only start timer if explicitly requested (for compatibility with QApplication)
        if auto_start_timer:
            self.connection_timer.start(5000)  # Check every 5 seconds
            self.data_timer.start(10000)  # Fetch data every 10 seconds
        
        # Perform initial connection check (non-blocking, just sets flag)
        self._check_connection_sync()
        
        self.logger.info(f"InterReactBridgeAdapter initialized: {self.base_url}")
    
    def start_monitoring(self, interval_ms: int = 5000):
        """Start the connection monitoring timer"""
        if not self.connection_timer.isActive():
            # Start timer immediately - checks will happen async with reduced timeout
            # Don't block startup by checking immediately
            self.connection_timer.start(interval_ms)
            self.logger.info("Started connection monitoring")
        
        # Also start data fetching
        if not self.data_timer.isActive():
            # Fetch data after a short delay (don't block startup)
            QTimer.singleShot(2000, self._start_data_fetch)
            self.data_timer.start(10000)  # Then every 10 seconds
            self.logger.info("Started data fetching")
    
    def stop_monitoring(self):
        """Stop the connection monitoring timer"""
        if self.connection_timer.isActive():
            self.connection_timer.stop()
            self.logger.info("Stopped connection monitoring")
    
    def _check_connection_sync(self):
        """Synchronous connection check for initialization"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=1)
            self._connected = (response.status_code == 200)
        except Exception:
            self._connected = False
    
    def _check_connection(self):
        """
        Check if the bridge server is running and TWS connection status.
        WARNING: This method blocks! Use _check_connection_async() for non-blocking checks.
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=0.5)  # Reduced from 2s to 0.5s
            was_connected = self._connected
            self._connected = (response.status_code == 200)
            
            # Also check TWS connection status if bridge is connected
            if self._connected:
                try:
                    status_response = requests.get(f"{self.base_url}/connection-status", timeout=0.5)  # Reduced timeout
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        self._tws_connected = status_data.get('isConnected', False)
                except Exception:
                    self._tws_connected = False
            else:
                self._tws_connected = False
            
            if self._connected != was_connected:
                self.connection_status_changed.emit(self._connected)
                if self._connected:
                    self.logger.info(f"Connected to InterReactBridge (TWS: {'connected' if self._tws_connected else 'disconnected'})")
                else:
                    self.logger.warning("Disconnected from InterReactBridge")
        except Exception as e:
            if self._connected:
                self._connected = False
                self._tws_connected = False
                self.connection_status_changed.emit(False)
                self.logger.warning(f"Lost connection to InterReactBridge: {e}")
    

    
    def is_connected(self) -> bool:
        """Check if connected to the bridge (returns cached state)"""
        # Return cached state to avoid blocking on every call
        # State is updated by the background timer
        return self._connected
    
    def is_tws_connected(self) -> bool:
        """Check if TWS is connected (returns cached state)"""
        return self._tws_connected
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary from InterReactBridge.
        Returns data in format expected by UI:
        {
            "NetLiquidation": {"value": "123.45", "currency": "USD", "account": "U123"},
            "BuyingPower": {"value": "456.78", "currency": "USD", "account": "U123"},
            ...
        }
        """
        if not self.is_connected():
            self.logger.warning("Not connected to InterReactBridge")
            return {}
        
        try:
            response = requests.get(f"{self.base_url}/account", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self.logger.debug(f"Received account data: {len(data)} items")
            return data
            
        except requests.exceptions.Timeout:
            self.logger.error("Timeout getting account summary")
            self.error_occurred.emit("Timeout getting account data")
            return {}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting account summary: {e}")
            self.error_occurred.emit(f"Error getting account: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error getting account summary: {e}")
            self.error_occurred.emit(f"Unexpected error: {str(e)}")
            return {}
    
    def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        Get portfolio positions from InterReactBridge.
        Returns data in format expected by UI:
        [
            {
                "symbol": "AAPL",
                "position": 100,
                "average_cost": 150.0,
                "market_price": 155.0,
                "market_value": 15500.0,
                "unrealized_pnl": 500.0,
                "account": "U123"
            },
            ...
        ]
        """
        if not self.is_connected():
            self.logger.warning("Not connected to InterReactBridge")
            return []
        
        try:
            response = requests.get(f"{self.base_url}/portfolio", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Data already comes in correct format from C# endpoint
            self.logger.debug(f"Received portfolio data: {len(data)} positions")
            return data
            
        except requests.exceptions.Timeout:
            self.logger.error("Timeout getting portfolio")
            self.error_occurred.emit("Timeout getting portfolio data")
            return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting portfolio: {e}")
            self.error_occurred.emit(f"Error getting portfolio: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting portfolio: {e}")
            self.error_occurred.emit(f"Unexpected error: {str(e)}")
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Alias for get_portfolio() to match IBKRService interface"""
        return self.get_portfolio()
    
    def _start_data_fetch(self):
        """Start background data fetch thread"""
        if not self.is_connected() or not self.is_tws_connected():
            self.logger.debug("Skipping data fetch - not connected")
            return
        
        # Don't start new thread if one is already running
        if self._fetch_thread and self._fetch_thread.isRunning():
            self.logger.debug("Data fetch thread already running, skipping")
            return
        
        # Create and start thread
        self._fetch_thread = DataFetchThread(self.base_url)
        self._fetch_thread.portfolio_fetched.connect(self._on_portfolio_fetched)
        self._fetch_thread.account_fetched.connect(self._on_account_fetched)
        self._fetch_thread.finished.connect(self._on_fetch_finished)
        self._fetch_thread.start()
        self.logger.debug("Started background data fetch thread")
    
    def _on_portfolio_fetched(self, portfolio: list):
        """Handle portfolio data from background thread"""
        self._portfolio_cache = portfolio
        self.portfolio_updated.emit(portfolio)
        self.logger.debug(f"Portfolio updated: {len(portfolio)} positions")
    
    def _on_account_fetched(self, account: dict):
        """Handle account data from background thread"""
        self._account_cache = account
        self.account_updated.emit(account)
        self.logger.debug(f"Account updated: {len(account)} fields")
    
    def _on_fetch_finished(self):
        """Cleanup when fetch thread finishes"""
        self.logger.debug("Background fetch thread finished")
    

    
    def get_cached_portfolio(self) -> List[Dict[str, Any]]:
        """Get cached portfolio data (non-blocking)"""
        return self._portfolio_cache.copy()
    
    def get_cached_account(self) -> Dict[str, Any]:
        """Get cached account data (non-blocking)"""
        return self._account_cache.copy()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status from bridge"""
        if not self.is_connected():
            return {
                "isConnected": False,
                "message": "Bridge server not available"
            }
        
        try:
            response = requests.get(f"{self.base_url}/connection-status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "isConnected": False,
                    "message": f"Bridge returned status {response.status_code}"
                }
        except Exception as e:
            self.logger.error(f"Error getting connection status: {e}")
            return {
                "isConnected": False,
                "message": str(e)
            }
    
    def connect_to_ibkr(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 101) -> bool:
        """
        Request the bridge to connect to IBKR TWS/Gateway
        Note: TwsConnectionService connects automatically on startup,
        but this can be used to reconnect if needed.
        """
        if not self.is_connected():
            self.logger.error("Cannot connect to IBKR: Bridge server not available")
            return False
        
        try:
            params = {
                'host': host,
                'port': port,
                'clientId': client_id
            }
            response = requests.post(f"{self.base_url}/connect", 
                                   params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('connected'):
                    self.logger.info("Successfully connected to IBKR")
                    return True
                else:
                    self.logger.warning("Connection attempt returned false")
                    return False
            else:
                self.logger.error(f"Failed to connect to IBKR: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to IBKR: {e}")
            self.error_occurred.emit(f"Error connecting to IBKR: {str(e)}")
            return False
    
    def disconnect(self):
        """Stop the adapter"""
        self.connection_timer.stop()
        self._connected = False
        self.logger.info("InterReactBridgeAdapter stopped")
