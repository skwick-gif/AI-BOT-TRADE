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
print("[SANITY] If TWS shows an 'Accept incoming connection?' dialog, please click 'Yes' within ~30s.")
print("[SANITY] Ensure Global Configuration > API > Settings: 'Enable ActiveX and Socket Clients' is checked, and the Socket Port matches.")
ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=30)
print(f"[SANITY] Connected: {ib.isConnected()}")

accounts = list(ib.managedAccounts() or [])
print(f"[SANITY] Managed accounts: {accounts}")

# Subscribe to account updates using an explicit valid account code when possible
try:
    acct = accounts[0] if accounts else ""
    try:
        ib.reqAccountUpdates(True, acct)
        print(f"[SANITY] Subscribed to account updates for account '{acct or 'ALL'}'")
    except TypeError as te:
        # Fallback for ib_insync builds that only accept (subscribe: bool)
        ib.reqAccountUpdates(True)
        print("[SANITY] Subscribed to account updates (no account arg)")
except Exception as e:
    print(f"[SANITY] Account updates subscription error: {e}")

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
