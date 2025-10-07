#!/usr/bin/env python3
"""
Quick test for IB Gateway connection on port 4001
"""
import sys
import subprocess
import traceback

# Ensure ib_insync is available
try:
    import ib_insync
except Exception:
    print('ib_insync not found, installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ib_insync'])
    try:
        import ib_insync
    except Exception:
        print('Failed to import ib_insync after installation')
        raise

from ib_insync import IB

ib = IB()
HOST = '127.0.0.1'
PORT = 4001  # IB Gateway default port
CLIENT_ID = 1

print(f'Testing IB Gateway connection to {HOST}:{PORT} clientId={CLIENT_ID}...')
try:
    connected = ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
    print('connect returned:', connected)
    print('ib.isConnected():', ib.isConnected())
    if ib.isConnected():
        try:
            print('Requesting current time...')
            print('Current time:', ib.reqCurrentTime())
        except Exception as e:
            print('Error requesting current time:', repr(e))
            traceback.print_exc()
        try:
            print('Requesting positions...')
            positions = ib.positions()
            print(f'Positions returned: {len(positions)}')
            for p in positions[:5]:  # Show first 5 positions
                print(f'  {p.contract.symbol}: {p.position} shares')
        except Exception as e:
            print('Error requesting positions:', repr(e))
            traceback.print_exc()
    else:
        print('Not connected after connect call')
        print('Possible issues:')
        print('- IB Gateway is not running')
        print('- API is not enabled in IB Gateway settings')
        print('- Port 4001 is blocked by firewall')
        print('- Need to accept connection dialog in IB Gateway')
except Exception as e:
    print('Exception during connect:', repr(e))
    print('Troubleshooting:')
    print('1. Make sure IB Gateway is running')
    print('2. Check that API connections are enabled')
    print('3. Verify port 4001 is configured in IB Gateway')
    print('4. Try accepting any connection dialogs that appear')
    traceback.print_exc()
finally:
    try:
        if ib.isConnected():
            ib.disconnect()
            print('Disconnected successfully.')
    except Exception:
        pass