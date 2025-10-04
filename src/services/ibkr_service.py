"""
Interactive Brokers Service
Handles connection and trading operations with IBKR
"""

import asyncio
import socket
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os
from datetime import datetime, timedelta
import re

from ib_insync import IB, Stock, Contract, Order, Trade, PortfolioItem
from ib_insync.objects import Position, AccountValue

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


class IBKRService:
    """Interactive Brokers service for trading operations"""
    
    def __init__(self, config: IBKRConfig):
        self.config = config
        self.logger = get_logger("IBKRService")
        self.ib = IB()
        self._connected = False
        self._account_updates_subscribed = False
        self._portfolio_fetch_in_progress = False
        self._connecting = False
        self.last_error = None
        
        # Setup event handlers
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        
        self.logger.info("IBKR Service initialized")
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers
        
        Returns:
            bool: True if connection successful
        """
        # Prevent concurrent connect attempts
        if self._connecting:
            self.logger.info("Connect already in progress; skipping duplicate request")
            return False
        self._connecting = True
        try:
            # We'll try a simple, robust order that worked previously.
            tried_ports = []
            self.last_error = None
            # Prefer an explicit env port if provided; otherwise try paper (7497) then live (7496)
            env_port = None
            try:
                env_port = int(os.getenv("IBKR_PORT", "")) if os.getenv("IBKR_PORT") else None
            except Exception:
                env_port = None
            if env_port:
                ports_to_try = [env_port, (7496 if env_port != 7496 else 7497)]
            else:
                ports_to_try = [7497, 7496]

            for port in ports_to_try:
                if port in tried_ports:
                    continue
                tried_ports.append(port)
                try:
                    self.logger.info(f"Connecting to IBKR at {self.config.host}:{port} (clientId={self.config.client_id})")
                    # Connect to IB Gateway or TWS
                    self.ib.connect(
                        host=self.config.host,
                        port=port,
                        clientId=self.config.client_id,
                        timeout=max(3, int(self.config.connect_timeout)),
                        readonly=False
                    )
                    self._connected = True
                    # Remember working port for this session
                    self.config.port = port
                    self.logger.info(f"Connected to IBKR successfully on port {port}")
                    # Discover accounts and subscribe to updates
                    try:
                        accounts = list(self.ib.managedAccounts() or [])
                        if accounts:
                            self.logger.info(f"Managed accounts available: {accounts}")
                            if self.config.account_code not in accounts:
                                self.config.account_code = accounts[0]
                        else:
                            self.logger.warning("No managed accounts returned by IBKR")
                        self._subscribe_account_updates()
                    except Exception as se:
                        self.logger.warning(f"Unable to enumerate accounts / subscribe to updates: {se}")
                    return True
                except Exception as e:
                    msg = str(e).strip() if str(e) else ""
                    if not msg:
                        msg = f"{type(e).__name__}"
                    self.last_error = msg
                    self.logger.warning(f"Connect attempt failed on port {port}: {msg}")
                    self._connected = False
                    continue
            self.logger.error(f"Failed to connect to IBKR. Last error: {self.last_error}")
            return False
        finally:
            self._connecting = False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        try:
            if self._connected:
                self.ib.disconnect()
                self._connected = False
                self.logger.info("Disconnected from IBKR")
        except Exception as e:
            self.logger.error(f"Error disconnecting from IBKR: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self._connected and self.ib.isConnected()
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary information
        
        Returns:
            dict: Account summary data
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            # Request account summary
            account_values = self.ib.accountSummary()
            
            # Convert to dictionary
            summary = {}
            for av in account_values:
                summary[av.tag] = {
                    'value': av.value,
                    'currency': av.currency,
                    'account': av.account
                }
            
            self.logger.debug("Retrieved account summary")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions (basic position data without market prices)
        
        Returns:
            list: List of position dictionaries
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            positions = self.ib.positions()
            
            position_list = []
            for pos in positions:
                position_data = {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'market_price': 0.0,  # Position object doesn't have market price
                    'market_value': pos.position * pos.avgCost,  # Calculate approximate value
                    'average_cost': pos.avgCost,
                    'unrealized_pnl': 0.0,  # Position object doesn't have PnL
                    'realized_pnl': 0.0,  # Position object doesn't have PnL
                    'contract': pos.contract
                }
                position_list.append(position_data)
            
            self.logger.debug(f"Retrieved {len(position_list)} positions")
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        Get portfolio items
        
        Returns:
            list: List of portfolio item dictionaries
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            if self._portfolio_fetch_in_progress:
                # Avoid overlapping calls that could block UI; return last known basic positions
                self.logger.debug("Portfolio fetch already in progress; returning positions as fallback")
                return self.get_positions()
            self._portfolio_fetch_in_progress = True
            
            portfolio = self.ib.portfolio()
            
            portfolio_list = []
            for item in portfolio:
                portfolio_data = {
                    'symbol': item.contract.symbol,
                    'position': item.position,
                    'market_price': item.marketPrice,
                    'market_value': item.marketValue,
                    'average_cost': item.averageCost,
                    'unrealized_pnl': item.unrealizedPNL,
                    'realized_pnl': item.realizedPNL,
                    'contract': item.contract
                }
                portfolio_list.append(portfolio_data)
            
            self.logger.info(f"Retrieved portfolio with {len(portfolio_list)} items")
            # Fallback: if still empty, return basic positions so UI shows something
            if not portfolio_list:
                positions = self.get_positions()
                if positions:
                    self.logger.info(f"Portfolio empty; returning {len(positions)} positions as fallback")
                    return positions
            return portfolio_list
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio: {e}")
            return []
        finally:
            self._portfolio_fetch_in_progress = False

    def _is_port_open(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Quickly check if a TCP port is open to avoid blocking connect attempts."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False
    
    def place_order(self, order_info: OrderInfo) -> Optional[Trade]:
        """
        Place a trading order
        
        Args:
            order_info: Order information
            
        Returns:
            Trade object if successful, None otherwise
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            # Create contract
            contract = Stock(order_info.symbol, 'SMART', 'USD')
            
            # Create order
            order = Order()
            order.action = order_info.action
            order.totalQuantity = order_info.quantity
            order.orderType = order_info.order_type
            
            if order_info.limit_price:
                order.lmtPrice = order_info.limit_price
            
            if order_info.stop_price:
                order.auxPrice = order_info.stop_price
            
            if order_info.parent_id:
                order.parentId = order_info.parent_id
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            self.logger.info(f"Order placed: {order_info.action} {order_info.quantity} {order_info.symbol}")
            return trade
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def place_bracket_order(
        self, 
        symbol: str, 
        action: str, 
        quantity: int, 
        limit_price: float,
        take_profit: float,
        stop_loss: float
    ) -> List[Trade]:
        """
        Place a bracket order (parent + profit taker + stop loss)
        
        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Number of shares
            limit_price: Limit price for parent order
            take_profit: Take profit price
            stop_loss: Stop loss price
            
        Returns:
            List of Trade objects
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Parent order
            parent = Order()
            parent.action = action
            parent.orderType = 'LMT'
            parent.totalQuantity = quantity
            parent.lmtPrice = limit_price
            parent.transmit = False  # Don't transmit until all orders are ready
            
            # Take profit order
            take_profit_action = 'SELL' if action == 'BUY' else 'BUY'
            profit_order = Order()
            profit_order.action = take_profit_action
            profit_order.orderType = 'LMT'
            profit_order.totalQuantity = quantity
            profit_order.lmtPrice = take_profit
            profit_order.parentId = parent.orderId
            profit_order.transmit = False
            
            # Stop loss order
            stop_order = Order()
            stop_order.action = take_profit_action
            stop_order.orderType = 'STP'
            stop_order.totalQuantity = quantity
            stop_order.auxPrice = stop_loss
            stop_order.parentId = parent.orderId
            stop_order.transmit = True  # Transmit all orders
            
            # Place bracket order
            parent_trade = self.ib.placeOrder(contract, parent)
            profit_trade = self.ib.placeOrder(contract, profit_order)
            stop_trade = self.ib.placeOrder(contract, stop_order)
            
            self.logger.info(f"Bracket order placed for {symbol}: Parent={limit_price}, TP={take_profit}, SL={stop_loss}")
            return [parent_trade, profit_trade, stop_trade]
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return []
    
    def cancel_order(self, order) -> bool:
        """
        Cancel an order
        
        Args:
            order: Order object to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            self.ib.cancelOrder(order)
            self.logger.info(f"Order cancelled: {order.orderId}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time market data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: Market data or None
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for data
            self.ib.sleep(1)
            
            market_data = {
                'symbol': symbol,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'close': ticker.close,
                'volume': ticker.volume,
                'high': ticker.high,
                'low': ticker.low,
                'timestamp': datetime.now()
            }
            
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        duration: str = "1 Y", 
        bar_size: str = "1 day"
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            duration: Duration string (e.g., "1 Y", "6 M", "1 M")
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
            
        Returns:
            list: Historical data bars
        """
        try:
            if not self.is_connected():
                raise Exception("Not connected to IBKR")
            
            contract = Stock(symbol, 'SMART', 'USD')
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            historical_data = []
            for bar in bars:
                bar_data = {
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                historical_data.append(bar_data)
            
            self.logger.debug(f"Retrieved {len(historical_data)} historical bars for {symbol}")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    # Event handlers
    def _on_connected(self):
        """Handle connection event"""
        self.logger.info("IBKR connection established")
    
    def _on_disconnected(self):
        """Handle disconnection event"""
        self._connected = False
        self.logger.warning("IBKR connection lost")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle error event"""
        self.last_error = f"{errorCode}: {errorString}"
        self.logger.error(f"IBKR Error {errorCode}: {errorString} (reqId: {reqId})")
        # If we attempted to subscribe with an invalid account code, clear it and allow retry
        if errorCode == 321 and "Invalid account code" in (errorString or ""):
            try:
                bad_acct = getattr(self.config, 'account_code', '')
                if bad_acct:
                    self.logger.warning(f"Clearing invalid IBKR account code '{bad_acct}' and will retry subscription later")
                    self.config.account_code = ""
                # Let system retry later
                self._account_updates_subscribed = False
            except Exception:
                pass

    def _subscribe_account_updates(self, account_code: Optional[str] = None):
        """Subscribe to account updates using the ib_insync-compatible call signature.

        Some ib_insync versions accept only a single positional parameter (subscribe). To remain
        compatible, we do not pass the account code here. The service still records discovered
        account codes in self.config.account_code for other calls that may use it.
        """
        if self._account_updates_subscribed:
            return
        try:
            # Use default signature to avoid version-specific arg mismatch (no account argument)
            self.ib.reqAccountUpdates(True)
            self.logger.info("Subscribed to account updates")
            self._account_updates_subscribed = True
        except Exception as e:
            # Avoid spamming; log and continue so the app remains usable.
            self.logger.warning(f"Account updates subscription encountered an error: {e}")
            # Do not mark as subscribed so future attempts can retry once accounts become available.
            self._account_updates_subscribed = False

    def ensure_account_updates(self):
        """Public helper to ensure account updates are subscribed.

        Safe to call repeatedly; will no-op if already subscribed. If not, it will try to
        subscribe using the configured or discovered account code without blocking the UI.
        """
        try:
            if not self._account_updates_subscribed and self.is_connected():
                self._subscribe_account_updates(None)
        except Exception as e:
            self.logger.debug(f"ensure_account_updates skipped due to: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if self.is_connected():
                self.disconnect()
        except:
            pass