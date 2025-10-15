#!/usr/bin/env python3
"""
Test IBKR connection using app's config
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.config_manager import ConfigManager
from services.ibkr_service import IBKRService

def test_app_ibkr_connection():
    print("Testing IBKR connection using app's configuration...")

    config_manager = ConfigManager()
    ibkr_config = config_manager.ibkr

    print(f"Config: host={ibkr_config.host}, port={ibkr_config.port}, client_id={ibkr_config.client_id}")

    service = IBKRService(ibkr_config)
    result = service.connect()

    print(f'Connection result: {result}')
    if result:
        print('✅ IBKR connected successfully in app!')
        service.ib.disconnect()
        return True
    else:
        print(f'❌ Connection failed: {service.last_error}')
        return False

if __name__ == "__main__":
    test_app_ibkr_connection()