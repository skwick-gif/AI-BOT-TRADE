#!/usr/bin/env python3
import time
import sys
import os

# Add the tools directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ibkr_adapter_client import IBKRAdapterClient

def test_market_data_stream(client):
    print("Testing market data stream...")
    count = 0
    for data in client.stream_market_data('AAPL'):
        print(f"Received: {data.symbol} @ {data.price} vol: {data.volume}")
        count += 1
        if count >= 5:  # Stop after 5 messages
            break
    print("Market data stream test completed.")

def test_account_info(client):
    print("Testing account info...")
    account = client.get_account_info('12345')
    if account:
        print(f"Account: {account.account_id}, Balance: {account.balance}")
    else:
        print("Failed to get account info")

def test_place_order(client):
    print("Testing place order...")
    order = client.place_order('AAPL', 'BUY', 100, 150.0)
    if order:
        print(f"Order: {order.order_id}, Success: {order.success}, Message: {order.message}")
    else:
        print("Failed to place order")

def main():
    print("Starting integration test...")
    client = IBKRAdapterClient(port=7000)
    print("Client created")
    try:
        test_market_data_stream(client)
        test_account_info(client)
        test_place_order(client)
        print("DONE")
    finally:
        client.close()

if __name__ == "__main__":
    main()