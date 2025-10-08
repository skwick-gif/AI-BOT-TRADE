#!/usr/bin/env python3
"""
Demo script to test AI integration in Watchlist
"""
import sys
from pathlib import Path

# Add the project src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def demo_ai_integration():
    """Demo the AI integration functionality"""
    print("🤖 AI Integration Demo for Watchlist")
    print("=" * 50)
    
    # Test imports
    try:
        from services.ai_service import AIService
        from core.config_manager import ConfigManager
        print("✅ AI Service and Config Manager imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test config
    try:
        config = ConfigManager()
        if config.perplexity.api_key:
            print(f"✅ Perplexity API Key configured")
        else:
            print("❌ Perplexity API Key not configured")
            return
    except Exception as e:
        print(f"❌ Config error: {e}")
        return
    
    # Test AI service
    try:
        ai_service = AIService(config)
        print("✅ AI Service initialized")
        
        # Test numeric rating
        test_symbol = "AAPL"
        print(f"\n🧪 Testing numeric rating for {test_symbol}...")
        
        rating = ai_service.score_symbol_numeric_sync(
            test_symbol, 
            timeout=10.0,
            profile="swing"
        )
        print(f"✅ Rating: {rating}/10")
        
        # Test structured scoring (async)
        import asyncio
        
        async def test_structured():
            result = await ai_service.score_symbol(test_symbol, timeout=15.0)
            return result
        
        print(f"\n🧪 Testing structured scoring for {test_symbol}...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(test_structured())
            print(f"✅ Score: {result.get('score', 'N/A')}")
            print(f"✅ Action: {result.get('action', 'N/A')}")
            print(f"✅ Reason: {result.get('reason', 'N/A')}")
        finally:
            loop.close()
        
        print("\n🎉 All AI tests passed!")
        print("\n📝 Features available in Watchlist:")
        print("   • 🤖 AI button in each row for individual rating")
        print("   • 🤖 Rate All button for batch processing")
        print("   • AI Rating column shows 0-10 score")
        print("   • AI Prediction column shows BUY/SELL/HOLD + reason")
        print("   • Results are cached in symbol data")
        print("   • Batch processing includes delays to respect API limits")
        
    except Exception as e:
        print(f"❌ AI Service test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    demo_ai_integration()