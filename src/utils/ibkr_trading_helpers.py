"""
Enhanced IBKR trading functions ported from Streamlit version
Includes real trading execution, order management, and portfolio synchronization

Auto-connect policy:
- By default, these helpers will NOT auto-connect to IBKR. They require an
    already-connected IBKRService instance to be provided by the caller.
- To temporarily allow auto-connect from helpers, set environment variable
    IBKR_AUTO_CONNECT=true (not recommended for the PyQt app flow).
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from utils.logger import get_logger
from utils.trading_helpers import update_portfolio, PORTFOLIO_CSV

logger = get_logger("IBKRTradingHelpers")

# Auto-connect toggle (default: False)
AUTO_CONNECT = str(os.getenv("IBKR_AUTO_CONNECT", "false")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TradeResult:
    """Structure for trade execution results"""
    success: bool
    message: str
    order_id: Optional[int] = None
    ticker: str = ""
    action: str = ""
    shares: int = 0
    price: Optional[float] = None


def booktrade_ibkr(ticker: str, shares: int, action: str, order_type: str = "MKT", 
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None, 
                   limit_price: Optional[float] = None, ibkr_service=None) -> TradeResult:
    """
    Execute a trade through IBKR and update portfolio CSV
    
    Args:
        ticker: Stock symbol
        shares: Number of shares
        action: "BUY" or "SELL"
        order_type: Order type ("MKT", "LMT", "STP", "STP_LMT")
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        limit_price: Limit price for limit orders (optional)
        ibkr_service: IBKR service instance (optional)
        
    Returns:
        TradeResult with execution details
    """
    try:
        logger.info(f"Attempting to {action} {shares} shares of {ticker} via IBKR")
        
        # Import here to avoid circular imports
        from services.ibkr_service import IBKRService
        from core.config_manager import ConfigManager

        # Ensure we have a service instance only if auto-connect is allowed
        if not ibkr_service:
            if AUTO_CONNECT:
                config = ConfigManager()
                ibkr_service = IBKRService(config.ibkr)
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )

        # Require an active connection; optionally attempt auto-connect
        if not ibkr_service.is_connected():
            if AUTO_CONNECT:
                success = ibkr_service.connect()
                if not success:
                    return TradeResult(
                        success=False,
                        message="Failed to connect to IBKR",
                        ticker=ticker,
                        action=action,
                        shares=shares
                    )
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )
        
        # Create stock contract
        from ib_insync import Stock
        contract = Stock(ticker, 'SMART', 'USD')
        
        # Qualify the contract
        qualified_contracts = ibkr_service.ib.qualifyContracts(contract)
        if not qualified_contracts:
            return TradeResult(
                success=False,
                message=f"Could not qualify contract for {ticker}",
                ticker=ticker,
                action=action,
                shares=shares
            )
        
        contract = qualified_contracts[0]
        
        # Get current market price for reference
        ticker_data = ibkr_service.ib.reqMktData(contract, '', False, False)
        ibkr_service.ib.sleep(2)  # Wait for market data
        
        current_price = None
        if hasattr(ticker_data, 'last') and ticker_data.last:
            current_price = float(ticker_data.last)
        elif hasattr(ticker_data, 'close') and ticker_data.close:
            current_price = float(ticker_data.close)
        
        # Create order based on type
        from ib_insync import Order
        order = Order()
        order.action = action
        order.totalQuantity = shares
        order.orderType = order_type
        
        # Set price based on order type
        execution_price = current_price
        
        if order_type == "LMT":
            if limit_price:
                order.lmtPrice = limit_price
                execution_price = limit_price
            else:
                # Default to current price for limit orders
                order.lmtPrice = current_price
                execution_price = current_price
        
        elif order_type == "STP":
            if stop_loss:
                order.auxPrice = stop_loss
            else:
                return TradeResult(
                    success=False,
                    message="Stop price required for stop orders",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )
        
        elif order_type == "STP_LMT":
            if stop_loss and limit_price:
                order.auxPrice = stop_loss
                order.lmtPrice = limit_price
                execution_price = limit_price
            else:
                return TradeResult(
                    success=False,
                    message="Both stop price and limit price required for stop-limit orders",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )
        
        # Submit order
        trade = ibkr_service.ib.placeOrder(contract, order)
        
        # Wait for order to be processed
        ibkr_service.ib.sleep(3)
        
        # Check order status
        if trade.orderStatus.status in ['Filled', 'PartiallyFilled']:
            # Order executed successfully
            filled_price = trade.orderStatus.avgFillPrice or execution_price or current_price
            filled_shares = trade.orderStatus.filled
            
            # Update portfolio CSV
            update_portfolio(ticker, action, filled_shares, filled_price)
            
            logger.info(f"Trade executed: {action} {filled_shares} shares of {ticker} at ${filled_price:.2f}")
            
            return TradeResult(
                success=True,
                message=f"Order executed: {action} {filled_shares} shares of {ticker} at ${filled_price:.2f}",
                order_id=trade.order.orderId,
                ticker=ticker,
                action=action,
                shares=filled_shares,
                price=filled_price
            )
        
        elif trade.orderStatus.status in ['Submitted', 'PreSubmitted']:
            # Order submitted but not yet filled
            logger.info(f"Order submitted: {action} {shares} shares of {ticker}")
            
            return TradeResult(
                success=True,
                message=f"Order submitted: {action} {shares} shares of {ticker} (Status: {trade.orderStatus.status})",
                order_id=trade.order.orderId,
                ticker=ticker,
                action=action,
                shares=shares,
                price=execution_price
            )
        
        else:
            # Order failed or cancelled
            error_msg = f"Order failed: {trade.orderStatus.status}"
            logger.error(error_msg)
            
            return TradeResult(
                success=False,
                message=error_msg,
                ticker=ticker,
                action=action,
                shares=shares
            )
    
    except Exception as e:
        error_msg = f"Error executing trade: {str(e)}"
        logger.error(error_msg)
        
        return TradeResult(
            success=False,
            message=error_msg,
            ticker=ticker,
            action=action,
            shares=shares
        )


def create_bracket_order(ticker: str, shares: int, action: str, entry_price: float,
                        stop_loss: float, take_profit: float, ibkr_service=None) -> TradeResult:
    """
    Create a bracket order (parent + stop loss + take profit)
    
    Args:
        ticker: Stock symbol
        shares: Number of shares
        action: "BUY" or "SELL"
        entry_price: Entry price for the main order
        stop_loss: Stop loss price
        take_profit: Take profit price
        ibkr_service: IBKR service instance (optional)
        
    Returns:
        TradeResult with execution details
    """
    try:
        logger.info(f"Creating bracket order: {action} {shares} shares of {ticker}")
        
        # Import here to avoid circular imports
        from services.ibkr_service import IBKRService
        from core.config_manager import ConfigManager
        from ib_insync import Stock, Order

        if not ibkr_service:
            if AUTO_CONNECT:
                config = ConfigManager()
                ibkr_service = IBKRService(config.ibkr)
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )

        if not ibkr_service.is_connected():
            if AUTO_CONNECT:
                success = ibkr_service.connect()
                if not success:
                    return TradeResult(
                        success=False,
                        message="Failed to connect to IBKR",
                        ticker=ticker,
                        action=action,
                        shares=shares
                    )
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.",
                    ticker=ticker,
                    action=action,
                    shares=shares
                )
        
        # Create stock contract
        contract = Stock(ticker, 'SMART', 'USD')
        qualified_contracts = ibkr_service.ib.qualifyContracts(contract)
        if not qualified_contracts:
            return TradeResult(
                success=False,
                message=f"Could not qualify contract for {ticker}",
                ticker=ticker,
                action=action,
                shares=shares
            )
        
        contract = qualified_contracts[0]
        
        # Create parent order (entry order)
        parent = Order()
        parent.action = action
        parent.totalQuantity = shares
        parent.orderType = "LMT"
        parent.lmtPrice = entry_price
        parent.transmit = False  # Don't transmit yet
        
        # Create stop loss order
        stop_order = Order()
        stop_order.action = "SELL" if action == "BUY" else "BUY"  # Opposite of parent
        stop_order.totalQuantity = shares
        stop_order.orderType = "STP"
        stop_order.auxPrice = stop_loss
        stop_order.parentId = parent.orderId
        stop_order.transmit = False
        
        # Create take profit order
        profit_order = Order()
        profit_order.action = "SELL" if action == "BUY" else "BUY"  # Opposite of parent
        profit_order.totalQuantity = shares
        profit_order.orderType = "LMT"
        profit_order.lmtPrice = take_profit
        profit_order.parentId = parent.orderId
        profit_order.transmit = True  # Transmit all orders
        
        # Submit bracket order
        parent_trade = ibkr_service.ib.placeOrder(contract, parent)
        stop_trade = ibkr_service.ib.placeOrder(contract, stop_order)
        profit_trade = ibkr_service.ib.placeOrder(contract, profit_order)
        
        # Wait for orders to be processed
        ibkr_service.ib.sleep(3)
        
        logger.info(f"Bracket order created for {ticker}: Entry=${entry_price}, Stop=${stop_loss}, Target=${take_profit}")
        
        return TradeResult(
            success=True,
            message=f"Bracket order created: {action} {shares} shares of {ticker} at ${entry_price} (Stop: ${stop_loss}, Target: ${take_profit})",
            order_id=parent_trade.order.orderId,
            ticker=ticker,
            action=action,
            shares=shares,
            price=entry_price
        )
    
    except Exception as e:
        error_msg = f"Error creating bracket order: {str(e)}"
        logger.error(error_msg)
        
        return TradeResult(
            success=False,
            message=error_msg,
            ticker=ticker,
            action=action,
            shares=shares
        )


def cancel_order(order_id: int, ibkr_service=None) -> TradeResult:
    """
    Cancel an existing order
    
    Args:
        order_id: Order ID to cancel
        ibkr_service: IBKR service instance (optional)
        
    Returns:
        TradeResult with cancellation details
    """
    try:
        logger.info(f"Attempting to cancel order {order_id}")
        
        # Import here to avoid circular imports
        from services.ibkr_service import IBKRService
        from core.config_manager import ConfigManager

        if not ibkr_service:
            if AUTO_CONNECT:
                config = ConfigManager()
                ibkr_service = IBKRService(config.ibkr)
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again."
                )

        if not ibkr_service.is_connected():
            if AUTO_CONNECT:
                success = ibkr_service.connect()
                if not success:
                    return TradeResult(
                        success=False,
                        message="Failed to connect to IBKR"
                    )
            else:
                return TradeResult(
                    success=False,
                    message="IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again."
                )
        
        # Find the trade by order ID
        trades = ibkr_service.ib.trades()
        target_trade = None
        
        for trade in trades:
            if trade.order.orderId == order_id:
                target_trade = trade
                break
        
        if not target_trade:
            return TradeResult(
                success=False,
                message=f"Order {order_id} not found"
            )
        
        # Cancel the order
        ibkr_service.ib.cancelOrder(target_trade.order)
        ibkr_service.ib.sleep(2)
        
        logger.info(f"Order {order_id} cancellation requested")
        
        return TradeResult(
            success=True,
            message=f"Cancellation requested for order {order_id}",
            order_id=order_id
        )
    
    except Exception as e:
        error_msg = f"Error cancelling order {order_id}: {str(e)}"
        logger.error(error_msg)
        
        return TradeResult(
            success=False,
            message=error_msg
        )


def get_open_orders(ibkr_service=None) -> List[Dict[str, Any]]:
    """
    Get all open orders from IBKR
    
    Args:
        ibkr_service: IBKR service instance (optional)
        
    Returns:
        List of open orders
    """
    try:
        # Import here to avoid circular imports
        from services.ibkr_service import IBKRService
        from core.config_manager import ConfigManager

        if not ibkr_service:
            if AUTO_CONNECT:
                config = ConfigManager()
                ibkr_service = IBKRService(config.ibkr)
            else:
                logger.debug("IBKR not connected; get_open_orders returning empty list")
                return []

        if not ibkr_service.is_connected():
            if AUTO_CONNECT:
                success = ibkr_service.connect()
                if not success:
                    return []
            else:
                logger.debug("IBKR not connected; get_open_orders returning empty list")
                return []
        
        # Get all trades (which include orders)
        trades = ibkr_service.ib.trades()
        open_orders = []
        
        for trade in trades:
            # Only include open orders (not filled or cancelled)
            if trade.orderStatus.status in ['Submitted', 'PreSubmitted', 'PendingSubmit']:
                order_info = {
                    'order_id': trade.order.orderId,
                    'symbol': trade.contract.symbol,
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'order_type': trade.order.orderType,
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining,
                    'limit_price': getattr(trade.order, 'lmtPrice', None),
                    'stop_price': getattr(trade.order, 'auxPrice', None),
                    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Placeholder
                }
                open_orders.append(order_info)
        
        return open_orders
    
    except Exception as e:
        logger.error(f"Error getting open orders: {e}")
        return []


def sync_portfolio_with_ibkr(ibkr_service=None) -> Dict[str, Any]:
    """
    Synchronize local portfolio CSV with IBKR positions
    
    Args:
        ibkr_service: IBKR service instance (optional)
        
    Returns:
        Dictionary with sync results
    """
    try:
        logger.info("Synchronizing portfolio with IBKR positions")
        
        # Import here to avoid circular imports
        from services.ibkr_service import IBKRService
        from core.config_manager import ConfigManager

        if not ibkr_service:
            if AUTO_CONNECT:
                config = ConfigManager()
                ibkr_service = IBKRService(config.ibkr)
            else:
                return {
                    'success': False,
                    'message': 'IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.'
                }

        if not ibkr_service.is_connected():
            if AUTO_CONNECT:
                success = ibkr_service.connect()
                if not success:
                    return {
                        'success': False,
                        'message': 'Failed to connect to IBKR'
                    }
            else:
                return {
                    'success': False,
                    'message': 'IBKR not connected. Please connect from the app (Connection > Connect to IBKR) and try again.'
                }
        
        # Get IBKR positions
        ibkr_positions = {}
        positions = ibkr_service.ib.positions()
        
        for pos in positions:
            if pos.position != 0:  # Only non-zero positions
                symbol = pos.contract.symbol
                ibkr_positions[symbol] = {
                    'shares': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue,
                    'unrealized_pnl': pos.unrealizedPNL
                }
        
        # Get local portfolio positions
        from utils.trading_helpers import get_portfolio_positions
        local_positions = get_portfolio_positions()
        
        # Compare and report discrepancies
        discrepancies = []
        all_symbols = set(ibkr_positions.keys()) | set(local_positions.keys())
        
        for symbol in all_symbols:
            ibkr_shares = ibkr_positions.get(symbol, {}).get('shares', 0)
            local_shares = local_positions.get(symbol, 0)
            
            if abs(ibkr_shares - local_shares) > 0.001:  # Allow for small rounding differences
                discrepancies.append({
                    'symbol': symbol,
                    'ibkr_shares': ibkr_shares,
                    'local_shares': local_shares,
                    'difference': ibkr_shares - local_shares
                })
        
        sync_result = {
            'success': True,
            'ibkr_positions': len(ibkr_positions),
            'local_positions': len(local_positions),
            'discrepancies': discrepancies,
            'message': f'Sync complete. Found {len(discrepancies)} discrepancies.'
        }
        
        logger.info(f"Portfolio sync complete: {sync_result['message']}")
        return sync_result
    
    except Exception as e:
        error_msg = f"Error synchronizing portfolio: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'message': error_msg
        }