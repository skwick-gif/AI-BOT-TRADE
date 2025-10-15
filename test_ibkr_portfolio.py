#!/usr/bin/env python3
"""
Test IBKR portfolio query
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ib_insync import IB
import time

def test_ibkr_portfolio():
    print("Testing IBKR portfolio query on port 7496...")

    ib = IB()

    try:
        # Connect to IBKR on port 7496
        ib.connect('127.0.0.1', 7496, clientId=1)

        # Wait for connection
        time.sleep(2)

        if ib.isConnected():
            print("✅ Connected to IBKR")

            # Request portfolio
            portfolio = ib.portfolio()

            print(f"Portfolio items: {len(portfolio)}")
            for item in portfolio:
                print(f"Symbol: {item.contract.symbol}, Position: {item.position}, Market Value: {item.marketValue}")

            # Get account values
            account_values = ib.accountValues()
            print(f"Account values: {len(account_values)}")
            for av in account_values[:5]:  # Show first 5
                print(f"{av.tag}: {av.value}")

            # Disconnect
            ib.disconnect()
            print("Disconnected.")
            return True
        else:
            print("❌ Failed to connect to IBKR")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_ibkr_portfolio()