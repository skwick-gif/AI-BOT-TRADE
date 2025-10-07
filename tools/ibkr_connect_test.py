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
PORT = 7496
CLIENT_ID = 1

print(f'Attempting IB connect to {HOST}:{PORT} clientId={CLIENT_ID}...')
try:
    connected = ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=5)
    print('connect returned:', connected)
    print('ib.isConnected():', ib.isConnected())
    if ib.isConnected():
        try:
            positions = ib.positions()
            print(f'Positions returned: {len(positions)}')
            for p in positions[:10]:
                print(p)
        except Exception as e:
            print('Error requesting positions:', repr(e))
            traceback.print_exc()
        try:
            accounts = ib.managedAccounts()
            print('Managed accounts:', accounts)
        except Exception as e:
            print('Error requesting managedAccounts:', repr(e))
            traceback.print_exc()
    else:
        print('Not connected after connect call; likely API handshake blocked or refused by TWS settings.')
except Exception as e:
    print('Exception during connect:', repr(e))
    traceback.print_exc()
finally:
    try:
        if ib.isConnected():
            ib.disconnect()
            print('Disconnected.')
    except Exception:
        pass

print('IB connect test finished.')
