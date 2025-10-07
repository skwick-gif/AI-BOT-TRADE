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
CLIENT_ID = 2

print(f'Attempting IB connect to {HOST}:{PORT} clientId={CLIENT_ID} with timeout=15...')
try:
    connected = ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
    print('connect returned:', connected)
    print('ib.isConnected():', ib.isConnected())
    if ib.isConnected():
        try:
            print('Requesting positions...')
            positions = ib.positions()
            print(f'Positions returned: {len(positions)}')
            for p in positions[:10]:
                print(p)
        except Exception as e:
            print('Error requesting positions:', repr(e))
            traceback.print_exc()
        try:
            print('Requesting managed accounts...')
            accounts = ib.managedAccounts()
            print('Managed accounts:', accounts)
        except Exception as e:
            print('Error requesting managedAccounts:', repr(e))
            traceback.print_exc()
        try:
            print('Requesting account summary (first 5)...')
            acct = accounts[0] if accounts else None
            if acct:
                summary = ib.accountSummary(acct)
                print('Account summary items:', len(summary))
                for s in summary[:5]:
                    print(s)
        except Exception as e:
            print('Error requesting account summary:', repr(e))
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

print('IB connect test 2 finished.')
