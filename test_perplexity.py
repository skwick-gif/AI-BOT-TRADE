#!/usr/bin/env python3
"""
Test script for Perplexity API connectivity
"""
import os
import asyncio
import sys
from pathlib import Path

# Add the project src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from core.config_manager import ConfigManager
    from services.ai_service import AIService
    from utils.logger import get_logger
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

async def test_perplexity_api():
    """Test Perplexity API connectivity"""
    print("🔌 Testing Perplexity API connectivity...")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if API key is configured
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ PERPLEXITY_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with your API key")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else api_key}")
    
    try:
        # Initialize config and service
        config = ConfigManager()
        print(f"✅ Config loaded - Model: {config.perplexity.model}")
        print(f"✅ Max tokens: {config.perplexity.max_tokens}")
        
        # Test API call
        async with AIService(config) as ai_service:
            print("\n🤖 Testing API call...")
            
            # Simple test message
            test_message = "Hello, can you confirm you're working? Please respond with just 'API test successful' if you receive this."
            
            response = await ai_service.get_ai_response(test_message)
            
            print(f"\n📝 Response received:")
            print(f"Length: {len(response)} characters")
            print(f"Content: {response[:200]}{'...' if len(response) > 200 else ''}")
            
            # Check if it's an error message
            if "error" in response.lower() or "not configured" in response.lower():
                print(f"\n❌ API Error: {response}")
                return False
            else:
                print(f"\n✅ API test successful!")
                return True
                
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    print("=" * 50)
    
    try:
        config = ConfigManager()
        print(f"✅ Perplexity API Key: {'*' * (len(config.perplexity.api_key) - 5) + config.perplexity.api_key[-5:] if config.perplexity.api_key else 'Not configured'}")
        print(f"✅ Model: {config.perplexity.model}")
        print(f"✅ Max tokens: {config.perplexity.max_tokens}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Perplexity API Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config_ok = test_config()
    
    if not config_ok:
        print("\n❌ Configuration test failed. Cannot proceed with API test.")
        return
    
    # Test 2: API connectivity
    api_ok = await test_perplexity_api()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"API Connectivity: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if config_ok and api_ok:
        print("\n🎉 All tests passed! Perplexity API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check your configuration.")

if __name__ == "__main__":
    asyncio.run(main())