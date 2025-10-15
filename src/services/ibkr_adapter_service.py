"""
IBKR Adapter Service
Uses REST API bridge instead of gRPC
"""

import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import sys
import os

from core.config_manager import IBKRConfig
from utils.logger import get_logger


@dataclass
class OrderInfo:
    """Order information structure"""
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MKT, LMT, STP, etc.
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    parent_id: Optional[int] = None


class IBKRAdapterService:
    """IBKR service using REST API bridge"""

    def __init__(self, config: IBKRConfig):
        self.config = config
        self.logger = get_logger("IBKRAdapterService")
        self.base_url = "http://localhost:8080"  # REST bridge port
        self._connected = False
        self._connecting = False
        self.last_error = None

    def connect(self) -> bool:
        """Connect to IBKR via REST bridge"""
        if self._connecting:
            self.logger.info("Connect already in progress")
            return False

        try:
            self._connecting = True
            self.logger.info("Connecting to IBKR via REST bridge...")

            # Connect to IBKR through the REST bridge
            params = {
                'host': self.config.host,
                'port': self.config.port,
                'clientId': self.config.client_id
            }

            response = requests.post(f"{self.base_url}/connect", params=params, timeout=15)
            response.raise_for_status()

            result = response.json()
            if result.get('connected'):
                self._connected = True
                self.logger.info("Connected to IBKR successfully")
                return True
            else:
                self.logger.error("Failed to connect to IBKR")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to REST bridge: {e}")
            self.last_error = str(e)
            return False
        finally:
            self._connecting = False

    def disconnect(self):
        """Disconnect"""
        self._connected = False
        self.logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info"""
        if not self.is_connected():
            return {}

        try:
            response = requests.get(f"{self.base_url}/account", timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        if not self.is_connected():
            return []

        try:
            response = requests.get(f"{self.base_url}/portfolio", timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []

    def place_order(self, order_info: OrderInfo) -> bool:
        """Place an order"""
        if not self.is_connected():
            return False

        try:
            params = {
                'symbol': order_info.symbol,
                'secType': 'STK',  # Default to stock
                'exchange': 'SMART',
                'action': order_info.action.upper(),
                'quantity': order_info.quantity,
                'orderType': order_info.order_type.upper()
            }

            if order_info.limit_price:
                params['price'] = order_info.limit_price

            response = requests.post(f"{self.base_url}/order", params=params, timeout=15)
            response.raise_for_status()

            result = response.json()
            return result.get('success', False)

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return False

    def get_market_data(self, symbol: str, sec_type: str = 'STK', exchange: str = 'SMART', duration_seconds: int = 5) -> List[Dict[str, Any]]:
        """Get market data snapshot"""
        if not self.is_connected():
            return []

        try:
            params = {
                'symbol': symbol,
                'secType': sec_type,
                'exchange': exchange,
                'durationSeconds': duration_seconds
            }

            response = requests.get(f"{self.base_url}/marketdata", params=params, timeout=15)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return []

    def get_options_chain(self, underlying: str, exchange: str = 'SMART') -> List[Dict[str, Any]]:
        """Get options chain"""
        if not self.is_connected():
            return []

        try:
            params = {
                'underlying': underlying,
                'exchange': exchange
            }

            response = requests.get(f"{self.base_url}/optionschain", params=params, timeout=20)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to get options chain: {e}")
            return []

    def scan_market(self, scan_type: str = 'TOP_PERC_GAIN', number_of_rows: int = 10) -> List[Dict[str, Any]]:
        """Run market scanner"""
        if not self.is_connected():
            return []

        try:
            params = {
                'scanType': scan_type,
                'numberOfRows': number_of_rows
            }

            response = requests.get(f"{self.base_url}/scan", params=params, timeout=20)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to scan market: {e}")
            return []