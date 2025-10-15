#!/usr/bin/env python3
"""
Test IBKR connection on port 7496
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ib_insync import IB
import time

def test_ibkr_connection():
    print("Testing IBKR connection...")

    ib = IB()

    # Try different ports
    ports = [7496, 4001, 7497]
    hosts = ['127.0.0.1', 'localhost']

    for host in hosts:
        for port in ports:
            print(f"Trying {host}:{port}...")
            try:
                ib.connect(host, port, clientId=1, timeout=10)  # Increase timeout

                # Wait a bit for connection
                time.sleep(3)

                if ib.isConnected():
                    print(f"✅ Successfully connected to IBKR on {host}:{port}")
                    try:
                        print(f"Server version: {ib.serverVersion()}")
                    except AttributeError:
                        print("Server version: N/A")
                    try:
                        print(f"Connection time: {ib.connectionTime()}")
                    except AttributeError:
                        print("Connection time: N/A")

                    # Disconnect
                    ib.disconnect()
                    print("Disconnected.")
                    return True
                else:
                    print(f"❌ Not connected on {host}:{port}")
                    ib.disconnect()

            except Exception as e:
                print(f"❌ Error connecting to {host}:{port}: {e}")
                try:
                    ib.disconnect()
                except:
                    pass

    print("❌ Failed to connect on all ports")
    return False

if __name__ == "__main__":
    test_ibkr_connection()