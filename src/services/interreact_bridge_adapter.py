"""
InterReact Bridge Adapter
Adapter for connecting Python UI to InterReactBridge C# server
"""

import requests
import logging
from typing import Dict, Any, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


class InterReactBridgeAdapter(QObject):
    """
    Adapter that makes InterReactBridge compatible with existing UI expectations.
    This adapter wraps the InterReactBridge REST API and provides the same interface
    as IBKRService, allowing seamless integration with the Dashboard.
    """
    
    # Signals for connection status
    connection_status_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, host: str = "localhost", port: int = 5080):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self._connected = False
        self.logger = logger
        
        # Setup connection check timer
        self.connection_timer = QTimer()
        self.connection_timer.timeout.connect(self._check_connection)
        self.connection_timer.start(5000)  # Check every 5 seconds
        
        self.logger.info(f"InterReactBridgeAdapter initialized: {self.base_url}")
    
    def _check_connection(self):
        """Check if the bridge server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            was_connected = self._connected
            self._connected = (response.status_code == 200)
            
            if self._connected != was_connected:
                self.connection_status_changed.emit(self._connected)
                if self._connected:
                    self.logger.info("Connected to InterReactBridge")
                else:
                    self.logger.warning("Disconnected from InterReactBridge")
        except Exception as e:
            if self._connected:
                self._connected = False
                self.connection_status_changed.emit(False)
                self.logger.warning(f"Lost connection to InterReactBridge: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to the bridge"""
        return self._connected
    
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
