import sys
sys.path.insert(0, 'src')
from services.ibkr_adapter_service import IBKRAdapterService
from core.config_manager import ConfigManager

config = ConfigManager()
service = IBKRAdapterService(config.ibkr)

print("Attempting to connect to IBKR...")
connected = service.connect()
if connected:
    print("Connected successfully")
    print("Positions:", service.get_positions())
    print("Account Info:", service.get_account_info())
else:
    print("Failed to connect:", service.last_error)