#!/usr/bin/env python3
"""
Quick sanity check for IBKR connectivity using ib_insync (standalone, no Qt).
- Connects to TWS/Gateway using IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID (defaults 127.0.0.1/7497/1)
- Prints managed accounts and auto-selects a valid account code
- Subscribes to account updates for that account
- Prints count of account summary fields and positions

Run from repo root with your Python env activated.
"""
import os
import sys
from time import sleep

try:
    from ib_insync import IB
except Exception as e:
    print("ib_insync is not installed. Install with: pip install ib-insync")
    sys.exit(1)

HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
PORT = int(os.environ.get("IBKR_PORT", "7497"))
CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "1"))

print(f"[SANITY] Connecting to IBKR at {HOST}:{PORT} (clientId={CLIENT_ID})")
ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
print(f"[SANITY] Connected: {ib.isConnected()}")

accounts = list(ib.managedAccounts() or [])
print(f"[SANITY] Managed accounts: {accounts}")

acct = os.environ.get("IBKR_ACCOUNT_CODE")
if acct and acct not in accounts:
    print(f"[SANITY] Warning: IBKR_ACCOUNT_CODE='{acct}' not in managed accounts; ignoring.")
    acct = None
if not acct and accounts:
    acct = accounts[0]

if acct:
    ib.reqAccountUpdates(True, acct)
    print(f"[SANITY] Subscribed to account updates for '{acct}'")
else:
    ib.reqAccountUpdates(True)
    print("[SANITY] Subscribed to account updates (no account specified)")

# Give IB a moment to populate summary/portfolio internally
ib.sleep(1.0)

summary = ib.accountSummary()
print(f"[SANITY] Account summary fields: {len(summary)}")

portfolio = ib.portfolio()
print(f"[SANITY] Portfolio items: {len(portfolio)}")

positions = ib.positions()
print(f"[SANITY] Positions: {len(positions)}")

ib.disconnect()
print("[SANITY] Disconnected.")
