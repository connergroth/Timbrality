#!/usr/bin/env python3
"""
Simple ML System Test for Tensoe Backend

This script tests the ML integration without complex imports.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        return False

def test_ml_endpoints():
    """Test ML endpoints availability"""
    endpoints = [
        "/ml/health",
        "/ml/stats", 
        "/docs"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            results[endpoint] = response.status_code == 200
            if results[endpoint]:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"❌ {endpoint} - Failed ({response.status_code})")
        except Exception as e:
            results[endpoint] = False
            print(f"❌ {endpoint} - Error: {e}")
    
    return results

def test_ingestion(album_name="Kind of Blue", artist_name="Miles Davis"):
    """Test album ingestion"""
    print(f"\n🎵 Testing ingestion: {album_name} by {artist_name}")
    
    try:
        response = requests.post(f"{BASE_URL}/ml/ingest/album", 
                               json={
                                   "album_name": album_name,
                                   "artist_name": artist_name
                               },
                               timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ingestion successful: {result.get('message', 'No message')}")
            return True
        else:
            print(f"❌ Ingestion failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎵 Tensoe ML Integration Test")
    print("=" * 40)
    
    # Test 1: Server Health
    if not test_server_health():
        print("\n❌ Server is not running. Start it with:")
        print("   uvicorn main:app --reload")
        return False
    
    # Test 2: ML Endpoints
    print("\n🔍 Testing ML endpoints...")
    ml_results = test_ml_endpoints()
    
    # Test 3: Documentation
    print(f"\n📚 API Documentation: {BASE_URL}/docs")
    
    # Test 4: Optional ingestion test
    print("\n" + "=" * 40)
    test_ingestion_choice = input("🤔 Test album ingestion? (y/N): ").lower().strip()
    
    if test_ingestion_choice == 'y':
        album = input("📀 Album name (or press Enter for 'Kind of Blue'): ").strip() or "Kind of Blue"
        artist = input("🎤 Artist name (or press Enter for 'Miles Davis'): ").strip() or "Miles Davis"
        test_ingestion(album, artist)
    
    print("\n" + "=" * 40)
    print("🎉 Test complete!")
    print(f"📊 View stats: {BASE_URL}/ml/stats")
    print(f"📈 Analytics: {BASE_URL}/ml/analytics")
    print(f"📚 Full API docs: {BASE_URL}/docs")

if __name__ == "__main__":
    main() 