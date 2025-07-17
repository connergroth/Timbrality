#!/usr/bin/env python3
"""
Test script for AOTY Scraper functionality
"""
import asyncio
import sys
import os

# Add the parent directory to the path so we can import from backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.aoty_scraper import (
    get_album_url,
    scrape_album,
    search_albums,
    close_browser
)


async def test_search():
    """Test album search functionality"""
    print("🔍 Testing album search...")
    try:
        results = await search_albums("radiohead ok computer", limit=3)
        print(f"✅ Found {len(results)} search results")
        for i, result in enumerate(results[:2]):
            print(f"   {i+1}. {result.artist} - {result.title} ({result.year}) Score: {result.score}")
        return True
    except Exception as e:
        print(f"❌ Search test failed: {str(e)}")
        return False


async def test_find_album():
    """Test finding specific album URL"""
    print("\n🎯 Testing album URL finding...")
    try:
        result = await get_album_url("Radiohead", "OK Computer")
        if result:
            url, artist, title = result
            print(f"✅ Found album: {artist} - {title}")
            print(f"   URL: {url}")
            return url
        else:
            print("❌ Album not found")
            return None
    except Exception as e:
        print(f"❌ Album URL test failed: {str(e)}")
        return None


async def test_scrape_album(url):
    """Test scraping full album data"""
    print("\n📖 Testing album scraping...")
    try:
        album = await scrape_album(url, "Radiohead", "OK Computer")
        print(f"✅ Scraped album: {album.artist} - {album.title}")
        print(f"   Cover: {album.cover_image[:50] + '...' if album.cover_image else 'None'}")
        print(f"   User Score: {album.user_score}")
        print(f"   Critic Score: {album.critic_score}")
        print(f"   Tracks: {len(album.tracks)}")
        print(f"   Critic Reviews: {len(album.critic_reviews)}")
        print(f"   User Reviews: {len(album.popular_reviews)}")
        print(f"   Genres: {', '.join(album.metadata.genres[:3])}")
        
        if album.tracks:
            print(f"   First track: {album.tracks[0].number}. {album.tracks[0].title} ({album.tracks[0].length})")
        
        return True
    except Exception as e:
        print(f"❌ Album scraping test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("🧪 AOTY Scraper Test Suite")
    print("=" * 40)
    
    passed = 0
    total = 3
    
    # Test 1: Search
    if await test_search():
        passed += 1
    
    # Test 2: Find album URL
    album_url = await test_find_album()
    if album_url:
        passed += 1
        
        # Test 3: Scrape album (only if we got a URL)
        if await test_scrape_album(album_url):
            passed += 1
    else:
        print("\n⚠️  Skipping album scraping test (no URL found)")
    
    # Cleanup
    print("\n🧹 Cleaning up browser instances...")
    try:
        await close_browser()
        print("✅ Browser cleanup successful")
    except Exception as e:
        print(f"⚠️  Browser cleanup warning: {str(e)}")
    
    # Results
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 