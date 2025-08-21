#!/usr/bin/env python3
"""
Test script to verify Gemini API key setup is working correctly.
Run this from your backend directory to test the API key loading.
"""

import os
import sys
from dotenv import load_dotenv

def test_env_loading():
    """Test .env file loading from multiple locations"""
    print("🔍 Testing .env file loading...")
    
    # Try multiple locations for the .env file
    env_locations = [
        os.path.join(os.path.dirname(__file__), 'engine', '.env'),  # /backend/engine/.env
        os.path.join(os.path.dirname(__file__), '.env'),           # /backend/.env
        '.env'  # Current directory
    ]
    
    env_loaded = False
    for env_path in env_locations:
        if os.path.exists(env_path):
            print(f"✅ Found .env file at: {env_path}")
            load_dotenv(env_path)
            env_loaded = True
            break
        else:
            print(f"❌ Not found: {env_path}")
    
    if not env_loaded:
        print("⚠️ No .env file found!")
        return False
    
    # Test API key loading
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ GEMINI_API_KEY loaded successfully (length: {len(api_key)} chars)")
        if api_key.startswith("AIzaSy"):
            print("✅ API key format looks correct")
        else:
            print("⚠️ API key doesn't start with 'AIzaSy' - check if it's correct")
        return True
    else:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        return False

def test_gemini_client():
    """Test Gemini client initialization"""
    print("\n🤖 Testing Gemini client initialization...")
    
    try:
        from engine.gemini_client import initialize_client, get_usage_stats
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ No API key available for client test")
            return False
        
        # Initialize client
        client = initialize_client(api_key)
        if client:
            print("✅ Gemini client initialized successfully")
            
            # Test usage stats
            stats = get_usage_stats()
            print(f"✅ Usage stats: {stats}")
            return True
        else:
            print("❌ Failed to initialize Gemini client")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini client: {e}")
        return False

def test_preprocessor():
    """Test preprocessor with API key"""
    print("\n📊 Testing preprocessor with API key...")
    
    try:
        from engine.pre_processor import AdvancedBOQExtractor
        
        api_key = os.getenv("GEMINI_API_KEY")
        extractor = AdvancedBOQExtractor(gemini_api_key=api_key)
        
        if extractor.ai_enabled:
            print("✅ Preprocessor AI features enabled")
            return True
        else:
            print("❌ Preprocessor AI features disabled")
            return False
            
    except Exception as e:
        print(f"❌ Error testing preprocessor: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 BOQ API Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment Loading", test_env_loading),
        ("Gemini Client", test_gemini_client),
        ("Preprocessor", test_preprocessor)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your API setup is working correctly.")
    else:
        print("⚠️ Some tests failed. Check your .env file and API key.")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your .env file is in the /backend/engine/ folder")
        print("2. Check that GEMINI_API_KEY=your_actual_api_key (no quotes)")
        print("3. Verify your API key is valid at https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    main()