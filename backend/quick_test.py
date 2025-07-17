#!/usr/bin/env python3
"""
Quick test for Tensoe Backend
"""

import requests
import time

def test_server(port=8001):
    """Test server on specified port"""
    base_url = f"http://localhost:{port}"
    
    print(f"🔍 Testing server on port {port}...")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"📚 API Docs: {base_url}/docs")
            
            # Test ML endpoints
            try:
                ml_health = requests.get(f"{base_url}/ml/health", timeout=5)
                if ml_health.status_code == 200:
                    print("✅ ML endpoints available!")
                    print("🎉 System is ready for testing!")
                    
                    print(f"\n🚀 Ready to test ingestion!")
                    print(f"   Visit: {base_url}/docs")
                    print(f"   Try: POST {base_url}/ml/ingest/album")
                    return True
                else:
                    print("⚠️ ML endpoints not fully available yet")
            except:
                print("⚠️ ML endpoints not responding")
                
        else:
            print(f"❌ Server returned {response.status_code}")
            
    except requests.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("\n💡 To start server manually:")
        print(f"   uvicorn main:app --reload --port {port}")
        
    return False

if __name__ == "__main__":
    # Test both common ports
    for port in [8001, 8000]:
        if test_server(port):
            break
    else:
        print("\n❌ Server not found on common ports")
        print("💡 Start with: uvicorn main:app --reload") 