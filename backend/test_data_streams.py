#!/usr/bin/env python3
"""
Comprehensive Data Streams Test Suite
Tests all data sources: Spotify, Last.fm, AOTY, Database, ML Pipeline
"""
import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "details": []
}

def log_result(test_name: str, status: str, message: str, details: str = ""):
    """Log test result"""
    test_results["details"].append({
        "test": test_name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })
    
    if status == "PASS":
        test_results["passed"] += 1
        print(f"✅ {test_name}: {message}")
    elif status == "FAIL":
        test_results["failed"] += 1
        print(f"❌ {test_name}: {message}")
        if details:
            print(f"   Details: {details}")
    elif status == "WARN":
        test_results["warnings"] += 1
        print(f"⚠️  {test_name}: {message}")
        if details:
            print(f"   Details: {details}")


def test_spotify_connection():
    """Test Spotify API connectivity"""
    print("\n🎵 Testing Spotify API Connection...")
    try:
        from ingestion.spotify_fetcher import get_spotify_client, search_albums, get_album_tracks
        
        try:
            # Test Spotify client initialization
            client = get_spotify_client()
            log_result("Spotify Init", "PASS", "Spotify client initialized successfully")
            
            # Test search functionality
            results = search_albums("OK Computer Radiohead", limit=3)
            if results and len(results) > 0:
                album = results[0]
                log_result("Spotify Search", "PASS", f"Found album: {album.get('name', 'Unknown')} by {album.get('artist', 'Unknown')}")
                
                # Test getting album tracks
                tracks = get_album_tracks("OK Computer", "Radiohead")
                if tracks:
                    log_result("Spotify Tracks", "PASS", f"Retrieved {len(tracks)} tracks from album")
                else:
                    log_result("Spotify Tracks", "WARN", "No tracks returned for album")
            else:
                log_result("Spotify Search", "FAIL", "No search results returned")
                
        except Exception as e:
            log_result("Spotify API", "FAIL", "Spotify API connection failed", str(e))
            
    except ImportError as e:
        log_result("Spotify Import", "FAIL", "Could not import Spotify functions", str(e))
    except Exception as e:
        log_result("Spotify Test", "FAIL", "Spotify test failed", str(e))


def test_lastfm_connection():
    """Test Last.fm API connectivity"""
    print("\n🎶 Testing Last.fm API Connection...")
    try:
        from ingestion.lastfm_fetcher import get_lastfm_network, enrich_with_tags, get_artist_tags
        
        try:
            # Test Last.fm network initialization
            network = get_lastfm_network()
            log_result("Last.fm Init", "PASS", "Last.fm network initialized successfully")
            
            # Test track tag enrichment
            tags = enrich_with_tags("Paranoid Android", "Radiohead", max_tags=5)
            if tags:
                log_result("Last.fm Track Tags", "PASS", f"Found {len(tags)} tags: {', '.join(tags[:3])}")
            else:
                log_result("Last.fm Track Tags", "WARN", "No track tags returned")
                
            # Test artist tags
            artist_tags = get_artist_tags("Radiohead", max_tags=5)
            if artist_tags:
                log_result("Last.fm Artist Tags", "PASS", f"Found {len(artist_tags)} artist tags: {', '.join(artist_tags[:3])}")
            else:
                log_result("Last.fm Artist Tags", "WARN", "No artist tags returned")
                
        except Exception as e:
            log_result("Last.fm API", "FAIL", "Last.fm API connection failed", str(e))
            
    except ImportError as e:
        log_result("Last.fm Import", "FAIL", "Could not import Last.fm functions", str(e))
    except Exception as e:
        log_result("Last.fm Test", "FAIL", "Last.fm test failed", str(e))


async def test_aoty_scraping():
    """Test AOTY scraping functionality"""
    print("\n🏆 Testing AOTY Scraping...")
    try:
        from scraper.aoty_scraper import search_albums, get_album_url, close_browser
        
        # Test search functionality
        try:
            search_results = await search_albums("radiohead ok computer", limit=3)
            if search_results and len(search_results) > 0:
                log_result("AOTY Search", "PASS", f"Found {len(search_results)} search results")
                
                # Test specific album URL finding
                try:
                    url_result = await get_album_url("Radiohead", "OK Computer")
                    if url_result:
                        url, artist, title = url_result
                        log_result("AOTY Album URL", "PASS", f"Found: {artist} - {title}")
                    else:
                        log_result("AOTY Album URL", "WARN", "Could not find specific album URL")
                        
                except Exception as e:
                    log_result("AOTY Album URL", "FAIL", "Album URL search failed", str(e))
                    
            else:
                log_result("AOTY Search", "FAIL", "No search results returned")
                
        except Exception as e:
            log_result("AOTY Search", "FAIL", "AOTY search failed", str(e))
            
        # Cleanup
        try:
            await close_browser()
            log_result("AOTY Cleanup", "PASS", "Browser cleanup successful")
        except Exception as e:
            log_result("AOTY Cleanup", "WARN", "Browser cleanup warning", str(e))
            
    except ImportError as e:
        log_result("AOTY Import", "FAIL", "Could not import AOTY scraper", str(e))
    except Exception as e:
        log_result("AOTY Test", "FAIL", "AOTY test failed", str(e))





async def test_api_endpoints():
    """Test API endpoints are working"""
    print("\n🌐 Testing API Endpoints...")
    try:
        import httpx
        
        base_url = "http://localhost:8002"
        
        async with httpx.AsyncClient() as client:
            # Test root endpoint
            try:
                response = await client.get(f"{base_url}/")
                if response.status_code == 200:
                    log_result("API Root", "PASS", "Root endpoint accessible")
                else:
                    log_result("API Root", "FAIL", f"Root endpoint returned {response.status_code}")
            except Exception as e:
                log_result("API Root", "FAIL", "Could not reach root endpoint", str(e))
            
            # Test scraper status endpoint
            try:
                response = await client.get(f"{base_url}/scraper/status")
                if response.status_code == 200:
                    data = response.json()
                    log_result("API Scraper", "PASS", f"Scraper endpoint: {data.get('status', 'unknown')}")
                else:
                    log_result("API Scraper", "FAIL", f"Scraper endpoint returned {response.status_code}")
            except Exception as e:
                log_result("API Scraper", "FAIL", "Could not reach scraper endpoint", str(e))
            
            # Test ML endpoints
            try:
                response = await client.get(f"{base_url}/ml/status")
                if response.status_code == 200:
                    log_result("API ML", "PASS", "ML endpoints accessible")
                else:
                    log_result("API ML", "FAIL", f"ML endpoint returned {response.status_code}")
            except Exception as e:
                log_result("API ML", "FAIL", "Could not reach ML endpoint", str(e))
                
    except ImportError:
        log_result("API Test", "WARN", "httpx not available, skipping API endpoint tests")
    except Exception as e:
        log_result("API Test", "FAIL", "API endpoint test failed", str(e))


def print_summary():
    """Print test summary"""
    total_tests = test_results["passed"] + test_results["failed"] + test_results["warnings"]
    
    print("\n" + "="*60)
    print("🧪 DATA STREAMS TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"⚠️  Warnings: {test_results['warnings']}")
    
    if test_results["failed"] > 0:
        print(f"\n🚨 {test_results['failed']} tests failed - check configuration and API keys")
    elif test_results["warnings"] > 0:
        print(f"\n⚠️  {test_results['warnings']} warnings - some features may not work without proper API keys")
    else:
        print("\n🎉 All tests passed! Your data streams are ready to go!")
    
    # Show failed tests details
    if test_results["failed"] > 0:
        print("\n❌ FAILED TESTS:")
        for result in test_results["details"]:
            if result["status"] == "FAIL":
                print(f"   • {result['test']}: {result['message']}")
                if result["details"]:
                    print(f"     └─ {result['details']}")
    
    print(f"\n📊 Success Rate: {(test_results['passed'] / total_tests * 100):.1f}%")
    return test_results["failed"] == 0


async def main():
    """Run all data stream tests"""
    print("🚀 TENSOE DATA STREAMS TEST SUITE")
    print("Testing Spotify, Last.fm, and AOTY data sources...")
    print("="*60)
    
    # Run all tests
    test_spotify_connection()  # Synchronous
    test_lastfm_connection()   # Synchronous
    await test_aoty_scraping() # Async (uses Playwright)
    await test_api_endpoints() # Async (uses httpx)
    
    # Print summary and return exit code
    success = print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {str(e)}")
        sys.exit(1) 