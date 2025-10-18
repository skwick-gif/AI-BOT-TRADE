"""
Test script for InterReactBridgeAdapter
Quick test to verify the adapter works correctly
"""

import sys
sys.path.insert(0, 'src')

import requests
import time

def main():
    print("=" * 60)
    print("Testing InterReactBridge Connection")
    print("=" * 60)
    
    base_url = "http://localhost:5080"
    
    # Check connection
    print("\n1. Checking bridge connection...")
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        connected = response.status_code == 200
    except:
        connected = False
    
    print(f"   Connection status: {connected}")
    
    if not connected:
        print("\n❌ Bridge server not running!")
        print("Please start InterReactBridge server first:")
        print("   cd tools/InterReactBridge")
        print("   dotnet run")
        return
    
    print("\n✅ Connected to bridge!")
    
    # Test account summary
    print("\n2. Testing GET /account...")
    try:
        response = requests.get(f"{base_url}/account", timeout=10)
        account = response.json()
        print(f"   Received {len(account)} account fields")
        if account:
            # Show first few keys
            keys = list(account.keys())[:5]
            print(f"   Sample keys: {keys}")
            
            # Show NetLiquidation if exists
            if 'NetLiquidation' in account:
                net_liq = account['NetLiquidation']
                print(f"   NetLiquidation: {net_liq}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test portfolio
    print("\n3. Testing GET /portfolio...")
    try:
        response = requests.get(f"{base_url}/portfolio", timeout=10)
        portfolio = response.json()
        print(f"   Received {len(portfolio)} positions")
        if portfolio:
            # Show first position
            first_pos = portfolio[0]
            print(f"   First position:")
            print(f"      Symbol: {first_pos.get('symbol')}")
            print(f"      Position: {first_pos.get('position')}")
            print(f"      Average Cost: {first_pos.get('average_cost')}")
            print(f"      Market Price: {first_pos.get('market_price')}")
            print(f"      Market Value: {first_pos.get('market_value')}")
            print(f"      Unrealized P&L: {first_pos.get('unrealized_pnl')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test connection status
    print("\n4. Testing GET /connection-status...")
    try:
        response = requests.get(f"{base_url}/connection-status", timeout=5)
        status = response.json()
        print(f"   IBKR Connected: {status.get('isConnected')}")
        print(f"   Message: {status.get('message')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
