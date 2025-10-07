import sys
import subprocess
import logging
import traceback
import asyncio

# Ensure ib_insync is available
try:
    import ib_insync
except Exception:
    print('ib_insync not found, installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ib_insync'])
    import ib_insync

from ib_insync import IB

# Configure logging to show debug output from ib_insync and asyncio
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logging.getLogger('ib_insync').setLevel(logging.DEBUG)
logging.getLogger('asyncio').setLevel(logging.DEBUG)

# Also enable asyncio debug mode
try:
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
except Exception as e:
    print('Could not set asyncio loop debug:', e)

ib = IB()
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 3
TIMEOUT = 20

print(f'Attempting DEBUG IB connect to {HOST}:{PORT} clientId={CLIENT_ID} timeout={TIMEOUT}...')
try:
    connected = ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=TIMEOUT)
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
            positions = ib.positions()
            print(f'Positions: {len(positions)}')
        except Exception as e:
            print('Error requesting positions:', repr(e))
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

print('IB debug connect finished.')
