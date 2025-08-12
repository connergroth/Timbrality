#!/usr/bin/env python3
"""
Test script for the new CloudScraper-based AOTY scraper
"""

import sys
import os
import asyncio
import time
from typing import Optional, Tuple

# Add the backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
scraper_dir = os.path.join(backend_dir, 'scraper')
sys.path.insert(0, backend_dir)
sys.path.insert(0, scraper_dir)

try:
    from scraper.utils.scraper import get_album_url, scrape_album, get_user_profile, get_similar_albums
    from scraper.models import Album, UserProfile
except ImportError as e:
    print(f"[ERROR] Failed to import scraper modules: {e}")
    print("Make sure you're running from the backend directory")   
    sys.exit(1)


async def test_album_search():
    """Test album URL finding functionality"""
    print("\n" + "="*60)
    print("TESTING ALBUM SEARCH")
    print("="*60)
    
    test_cases = [
        ("Radiohead", "OK Computer"),
        ("Pink Floyd", "The Dark Side of the Moon"),
        ("The Beatles", "Abbey Road"),
        ("Nirvana", "Nevermind"),
        ("Invalid Artist", "Invalid Album")  # Should fail
    ]
    
    successful_tests = 0
    failed_tests = 0
    
    for artist, album in test_cases:
        print(f"\n[TEST] Searching for: {artist} - {album}")
        start_time = time.time()
        
        try:
            result = await get_album_url(artist, album)
            elapsed = time.time() - start_time
            
            if result:
                url, found_artist, found_title = result
                print(f"  [SUCCESS] Found: {found_artist} - {found_title}")
                print(f"  URL: {url}")
                print(f"  Time: {elapsed:.2f}s")
                successful_tests += 1
            else:
                print(f"  [NOT FOUND] No results for {artist} - {album}")
                print(f"  Time: {elapsed:.2f}s")
                if artist == "Invalid Artist":  # Expected to fail
                    successful_tests += 1
                else:
                    failed_tests += 1
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  [ERROR] {str(e)}")
            print(f"  Time: {elapsed:.2f}s")
            failed_tests += 1
        
        # Add small delay between requests
        await asyncio.sleep(1)
    
    print(f"\n[SUMMARY] Album Search: {successful_tests} passed, {failed_tests} failed")
    return successful_tests, failed_tests


async def test_album_scraping():
    """Test full album scraping functionality"""
    print("\n" + "="*60)
    print("TESTING ALBUM SCRAPING")
    print("="*60)
    
    # First get a valid album URL
    print(f"\n[TEST] Getting album URL for Radiohead - OK Computer")
    result = await get_album_url("Radiohead", "OK Computer")
    
    if not result:
        print("[ERROR] Cannot test scraping without valid album URL")
        return 0, 1
    
    url, artist, title = result
    print(f"  Found URL: {url}")
    
    # Now test scraping
    print(f"\n[TEST] Scraping album data...")
    start_time = time.time()
    
    try:
        album = await scrape_album(url, artist, title)
        elapsed = time.time() - start_time
        
        print(f"  [SUCCESS] Scraped album data")
        print(f"  Title: {album.title}")
        print(f"  Artist: {album.artist}")
        print(f"  User Score: {album.user_score}")
        print(f"  Ratings: {album.num_ratings}")
        print(f"  Tracks: {len(album.tracks)}")
        print(f"  Critic Reviews: {len(album.critic_reviews)}")
        print(f"  User Reviews: {len(album.popular_reviews)}")
        print(f"  Buy Links: {len(album.buy_links)}")
        print(f"  Must Hear: {album.is_must_hear}")
        print(f"  Time: {elapsed:.2f}s")
        
        # Show some track details
        if album.tracks:
            print(f"\n  Sample tracks:")
            for track in album.tracks[:3]:
                print(f"    {track.number}. {track.title} ({track.length})")
        
        return 1, 0
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERROR] Scraping failed: {str(e)}")
        print(f"  Time: {elapsed:.2f}s")
        return 0, 1


async def test_user_profile():
    """Test user profile scraping"""
    print("\n" + "="*60)
    print("TESTING USER PROFILE")
    print("="*60)
    
    test_users = [
        "evrynoiseatonce",  # Popular user
        "invalid_user_123"  # Should fail
    ]
    
    successful_tests = 0
    failed_tests = 0
    
    for username in test_users:
        print(f"\n[TEST] Getting profile for: {username}")
        start_time = time.time()
        
        try:
            profile = await get_user_profile(username)
            elapsed = time.time() - start_time
            
            if profile:
                print(f"  [SUCCESS] Found profile")
                print(f"  Username: {profile.username}")
                print(f"  Location: {profile.location}")
                print(f"  Member since: {profile.member_since}")
                print(f"  Stats: {profile.stats}")
                print(f"  Favorite albums: {len(profile.favorite_albums)}")
                print(f"  Recent reviews: {len(profile.recent_reviews)}")
                print(f"  Time: {elapsed:.2f}s")
                successful_tests += 1
            else:
                print(f"  [NOT FOUND] Profile not found")
                if username == "invalid_user_123":  # Expected to fail
                    successful_tests += 1
                else:
                    failed_tests += 1
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  [ERROR] {str(e)}")
            print(f"  Time: {elapsed:.2f}s")
            if username == "invalid_user_123" and "not found" in str(e).lower():
                successful_tests += 1  # Expected failure
            else:
                failed_tests += 1
        
        await asyncio.sleep(1)
    
    print(f"\n[SUMMARY] User Profile: {successful_tests} passed, {failed_tests} failed")
    return successful_tests, failed_tests


async def test_similar_albums():
    """Test similar albums functionality"""
    print("\n" + "="*60)
    print("TESTING SIMILAR ALBUMS")
    print("="*60)
    
    # First get a valid album URL
    print(f"\n[TEST] Getting similar albums for Radiohead - OK Computer")
    result = await get_album_url("Radiohead", "OK Computer")
    
    if not result:
        print("[ERROR] Cannot test similar albums without valid album URL")
        return 0, 1
    
    url, _, _ = result
    print(f"  Base URL: {url}")
    
    # Test similar albums
    print(f"\n[TEST] Getting similar albums...")
    start_time = time.time()
    
    try:
        similar_albums = await get_similar_albums(url)
        elapsed = time.time() - start_time
        
        print(f"  [SUCCESS] Found {len(similar_albums)} similar albums")
        print(f"  Time: {elapsed:.2f}s")
        
        # Show first few similar albums
        for i, album in enumerate(similar_albums[:3]):
            print(f"    {i+1}. {album.artist} - {album.title}")
            if album.user_score:
                print(f"       Score: {album.user_score}")
        
        return 1, 0
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERROR] Similar albums failed: {str(e)}")
        print(f"  Time: {elapsed:.2f}s")
        return 0, 1


async def test_anti_blocking_measures():
    """Test if scraper handles blocking gracefully"""
    print("\n" + "="*60)
    print("TESTING ANTI-BLOCKING MEASURES")
    print("="*60)
    
    print("\n[TEST] Making rapid consecutive requests...")
    
    rapid_test_cases = [
        ("Radiohead", "In Rainbows"),
        ("The Beatles", "Revolver"),
        ("Pink Floyd", "Wish You Were Here")
    ]
    
    successful_requests = 0
    blocked_requests = 0
    error_requests = 0
    
    for i, (artist, album) in enumerate(rapid_test_cases):
        print(f"\n  Request {i+1}: {artist} - {album}")
        start_time = time.time()
        
        try:
            result = await get_album_url(artist, album)
            elapsed = time.time() - start_time
            
            if result:
                _, found_artist, found_title = result
                print(f"    [SUCCESS] {found_artist} - {found_title} ({elapsed:.2f}s)")
                successful_requests += 1
            else:
                print(f"    [NOT FOUND] ({elapsed:.2f}s)")
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e).lower()
            
            if "403" in error_msg or "blocked" in error_msg or "rate limit" in error_msg:
                print(f"    [BLOCKED] Detected blocking: {str(e)[:100]}... ({elapsed:.2f}s)")
                blocked_requests += 1
            else:
                print(f"    [ERROR] Other error: {str(e)[:100]}... ({elapsed:.2f}s)")
                error_requests += 1
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    print(f"\n[SUMMARY] Anti-blocking test:")
    print(f"  Successful: {successful_requests}")
    print(f"  Blocked: {blocked_requests}")
    print(f"  Errors: {error_requests}")
    
    # Consider it a success if we get some successful requests
    # Blocking is expected with rapid requests
    if successful_requests > 0:
        return 1, 0
    else:
        return 0, 1


async def run_all_tests():
    """Run all test suites"""
    print("AOTY CloudScraper Integration Test Suite")
    print("="*70)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_passed = 0
    total_failed = 0
    
    # Run individual test suites
    test_suites = [
        ("Album Search", test_album_search),
        ("Album Scraping", test_album_scraping),
        ("User Profile", test_user_profile),
        ("Similar Albums", test_similar_albums),
        ("Anti-blocking", test_anti_blocking_measures),
    ]
    
    for suite_name, test_func in test_suites:
        try:
            passed, failed = await test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n[ERROR] Test suite '{suite_name}' crashed: {e}")
            total_failed += 1
        
        # Delay between test suites to be respectful
        await asyncio.sleep(2)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed)) * 100:.1f}%")
    print(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if total_failed == 0:
        print("\n[SUCCESS] All tests passed!")
        return True
    elif total_passed > total_failed:
        print(f"\n[PARTIAL SUCCESS] Most tests passed ({total_passed}/{total_passed + total_failed})")
        return True
    else:
        print(f"\n[FAILURE] Too many test failures ({total_failed}/{total_passed + total_failed})")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Tests cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] Test runner crashed: {e}")
        sys.exit(1)