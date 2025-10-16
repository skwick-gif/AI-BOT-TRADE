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
        # Allow overriding the REST bridge base URL via environment; default to 5080 (our launcher)
        self.base_url = os.getenv("IBKR_BRIDGE_URL", "http://localhost:5080").rstrip("/")
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

            # Try configured port first, then known alternatives
            ports_to_try = []
            try:
                initial_port = int(self.config.port)
            except Exception:
                initial_port = 4002
            # Common ports: IB Gateway Paper=4002, Live=4001; TWS Paper=7497, Live=7496
            candidate_ports = [initial_port, 4002, 4001, 7497, 7496]
            # Deduplicate while preserving order
            seen = set()
            for p in candidate_ports:
                if p not in seen and p > 0:
                    seen.add(p)
                    ports_to_try.append(p)

            # Store ports tried for UI diagnostics
            self.last_ports_tried = ports_to_try

            import socket
            def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
                try:
                    with socket.create_connection((host, port), timeout=timeout):
                        return True
                except Exception:
                    return False

            # Try a few clientIds in case the default is in use
            client_ids = []
            try:
                base_id = int(self.config.client_id)
            except Exception:
                base_id = 1
            for delta in range(0, 4):
                cid = base_id + delta
                if cid not in client_ids:
                    client_ids.append(cid)

            for port in ports_to_try:
                try:
                    # Quick local port check to avoid slow timeouts
                    if not _is_port_open(self.config.host, port):
                        self.logger.debug(f"Host {self.config.host}:{port} not accepting connections; skipping")
                        continue
                    for cid in client_ids:
                        params = {
                            'host': self.config.host,
                            'port': port,
                            'clientId': cid
                        }
                        self.logger.info(f"Attempting connect: {params}")
                        try:
                            response = requests.post(f"{self.base_url}/connect", params=params, timeout=12)
                            response.raise_for_status()
                            result = response.json()
                            if result.get('connected'):
                                self._connected = True
                                # Update our config with working values in-memory
                                try:
                                    self.config.port = port  # type: ignore[attr-defined]
                                    self.config.client_id = cid  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                self.logger.info(f"Connected to IBKR successfully on port {port} with clientId {cid}")
                                return True
                        except Exception as inner_e:
                            self.logger.debug(f"Connect attempt on port {port} clientId {cid} failed: {inner_e}")
                            continue
                except Exception as inner_e:
                    self.logger.debug(f"Connect attempt on port {port} failed: {inner_e}")

            # After attempts, query status and add helpful hint
            try:
                status = self.get_status()
            except Exception:
                status = {}
            if status:
                code = status.get('errorCode') or status.get('error_code')
                msg = status.get('errorMessage') or status.get('lastError') or status.get('error_message')
                hint = status.get('hint')
                status_host = status.get('host')
                status_port = status.get('port')
                parts = []
                if code:
                    parts.append(f"code={code}")
                if msg:
                    parts.append(f"message={msg}")
                if status_host and status_port:
                    parts.append(f"endpoint={status_host}:{status_port}")
                if hint:
                    parts.append(f"hint={hint}")
                extra = "; ".join(parts)
                if extra:
                    self.logger.error(f"All connect attempts failed: {extra}")
                    self.last_error = extra
                else:
                    self.logger.error("All connect attempts failed")
                    self.last_error = "All connect attempts failed"
            else:
                self.logger.error("All connect attempts failed")
                self.last_error = "All connect attempts failed"
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

    def get_status(self) -> Dict[str, Any]:
        """Return bridge connection status (/connect/status)."""
        try:
            response = requests.get(f"{self.base_url}/connect/status", timeout=5)
            response.raise_for_status()
            data = response.json()
            # Normalize keys
            if isinstance(data, dict):
                return data
            return {}
        except Exception as e:
            self.logger.debug(f"Failed to get status: {e}")
            return {}

    def probe(self, host: Optional[str] = None, port: Optional[int] = None, timeout_ms: int = 800) -> Dict[str, Any]:
        """Probe TCP reachability via bridge (/probe)."""
        h = host or getattr(self.config, 'host', '127.0.0.1')
        p = int(port or getattr(self.config, 'port', 4002) or 4002)
        try:
            params = { 'host': h, 'port': p, 'timeoutMs': timeout_ms }
            r = requests.get(f"{self.base_url}/probe", params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.logger.debug(f"Probe failed for {h}:{p}: {e}")
            return { 'reachable': False, 'error': str(e) }

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info"""
        if not self.is_connected():
            return {}

        try:
            response = requests.get(f"{self.base_url}/account", timeout=20)
            response.raise_for_status()
            raw = response.json()
            # Normalize into { TagName: { value, currency, account } }
            normalized: Dict[str, Any] = {}
            if isinstance(raw, list):
                for item in raw:
                    tag = item.get('Tag') or item.get('tag')
                    if not tag:
                        continue
                    normalized[tag] = {
                        'value': item.get('Value') if 'Value' in item else item.get('value'),
                        'currency': item.get('Currency') if 'Currency' in item else item.get('currency', 'USD'),
                        'account': item.get('Account') if 'Account' in item else item.get('account')
                    }
            elif isinstance(raw, dict):
                # Already in expected format
                normalized = raw
            return normalized
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}

    def get_account_summary(self) -> Dict[str, Any]:
        """Alias for get_account_info to match legacy UI API."""
        return self.get_account_info()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        if not self.is_connected():
            return []

        try:
            response = requests.get(f"{self.base_url}/portfolio", timeout=20)
            response.raise_for_status()
            raw = response.json()
            # Normalize keys to match UI expectations
            normalized: List[Dict[str, Any]] = []
            if isinstance(raw, list):
                for p in raw:
                    norm = {
                        'symbol': p.get('Symbol') or p.get('symbol'),
                        'position': p.get('Position') if 'Position' in p else p.get('position', 0),
                        'market_price': p.get('MarketPrice') if 'MarketPrice' in p else p.get('market_price', 0.0),
                        'market_value': p.get('MarketValue') if 'MarketValue' in p else p.get('market_value', 0.0),
                        'average_cost': p.get('AverageCost') if 'AverageCost' in p else p.get('average_cost', 0.0),
                        'unrealized_pnl': p.get('UnrealizedPnl') if 'UnrealizedPnl' in p else p.get('unrealized_pnl', 0.0),
                        'realized_pnl': p.get('RealizedPnl') if 'RealizedPnl' in p else p.get('realized_pnl', 0.0),
                    }
                    # Fallback compute market_value if missing
                    if not norm['market_value'] and norm['position'] and norm['average_cost']:
                        try:
                            norm['market_value'] = float(norm['position']) * float(norm['average_cost'])
                        except Exception:
                            pass
                    normalized.append(norm)
            elif isinstance(raw, dict):
                normalized.append(raw)
            return normalized
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """Alias for get_positions to match UI usage."""
        return self.get_positions()

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

    def health(self) -> bool:
        """Check REST bridge health endpoint"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            return bool(data)
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False

    def stream_live_data(self, symbol: str, sec_type: str = 'STK', exchange: str = 'SMART', on_tick=None, stop_flag: Optional[List[bool]] = None):
        """Stream live data via Server-Sent Events. Call on_tick(dict) for each tick.

        Args:
            symbol: symbol to stream
            sec_type: security type
            exchange: exchange
            on_tick: callback receiving a dict with price/tick data
            stop_flag: optional single-item list [False]; set to True to stop
        """
        if not self.is_connected():
            self.logger.warning("Not connected; stream_live_data aborted")
            return
        try:
            params = {
                'symbol': symbol,
                'secType': sec_type,
                'exchange': exchange
            }
            with requests.get(f"{self.base_url}/livedata", params=params, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if stop_flag and stop_flag[0]:
                        break
                    if not line:
                        continue
                    # SSE lines typically like: "data: {json}"
                    if isinstance(line, bytes):
                        try:
                            line = line.decode('utf-8', errors='ignore')
                        except Exception:
                            continue
                    if line.startswith('data:'):
                        payload = line[5:].strip()
                        try:
                            import json
                            data = json.loads(payload)
                            if on_tick:
                                on_tick(data)
                        except Exception:
                            # ignore malformed lines
                            pass
        except Exception as e:
            self.logger.error(f"Live data stream error for {symbol}: {e}")

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