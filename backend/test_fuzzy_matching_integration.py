#!/usr/bin/env python3
"""
Test script for fuzzy matching integration with AOTY scraper
This demonstrates how the fuzzy matcher handles various edge cases
that are common between Last.fm and AOTY data
"""

import sys
import os
import asyncio
from typing import List

# Add the backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

try:
    from services.music_matching_service import MusicMatchingService, LastfmTrack, MusicMatchResult
    from utils.fuzzy_matcher import MusicFuzzyMatcher
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    print("Make sure you're running from the backend directory")
    sys.exit(1)


async def test_fuzzy_matching_edge_cases():
    """Test fuzzy matching with various edge cases common between platforms"""
    print("Fuzzy Matching Edge Cases Test")
    print("=" * 70)
    
    matcher = MusicFuzzyMatcher()
    
    # Test cases covering common variations between Last.fm and AOTY
    test_cases = [
        # Artist name variations
        ("Artist Variations", [
            ("The Beatles", "Beatles"),
            ("The Strokes", "Strokes"),
            ("The National", "National"),
            ("Led Zeppelin", "Led Zeppelin"),  # No "The"
        ]),
        
        # Album title variations
        ("Album Title Variations", [
            ("OK Computer", "OK Computer"),  # Exact match
            ("Abbey Road (Remastered)", "Abbey Road"),  # Remaster suffix
            ("The Dark Side of the Moon", "Dark Side of the Moon"),  # "The" prefix
            ("In Rainbows - Deluxe Edition", "In Rainbows"),  # Deluxe edition
            ("Nevermind (2011 Remaster)", "Nevermind"),  # Year + remaster
            ("Sgt. Pepper's Lonely Hearts Club Band (2017 Stereo Mix)", "Sgt. Pepper's Lonely Hearts Club Band"),  # Complex suffix
        ]),
        
        # Track title variations (most complex)
        ("Track Title Variations", [
            ("Paranoid Android", "Paranoid Android"),  # Exact
            ("Karma Police (feat. Someone)", "Karma Police"),  # Feature credit
            ("Creep - 2009 Remaster", "Creep"),  # Remaster
            ("Everything In Its Right Place", "Everything in Its Right Place"),  # Capitalization
            ("No Surprises (Live)", "No Surprises"),  # Live version
            ("Let Down - Radio Edit", "Let Down"),  # Radio edit
            ("Exit Music (For a Film)", "Exit Music (For A Film)"),  # Article case
            ("Airbag (2017 - Remaster)", "Airbag"),  # Complex remaster
            ("Subterranean Homesick Alien [Remastered]", "Subterranean Homesick Alien"),  # Different brackets
        ]),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in test_cases:
        print(f"\n[{category.upper()}]")
        category_passed = 0
        
        for query, target in tests:
            total_tests += 1
            
            if "Artist" in category:
                result = matcher.match_artist(query, target)
            elif "Album" in category:
                result = matcher.match_album(query, target)
            else:  # Track
                result = matcher.match_track(query, target)
            
            if result and result.score >= 0.75:  # Consider 75%+ a pass
                passed_tests += 1
                category_passed += 1
                print(f"  [PASS] '{query}' -> '{target}': {result.score:.3f} ({result.confidence})")
            else:
                score_str = f"{result.score:.3f}" if result else "No match"
                print(f"  [FAIL] '{query}' -> '{target}': {score_str}")
        
        print(f"  Category result: {category_passed}/{len(tests)} passed")
    
    print(f"\n[SUMMARY] Fuzzy matching: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    return passed_tests, total_tests


async def test_real_world_lastfm_matching():
    """Test matching real-world Last.fm data with various inconsistencies"""
    print("\n" + "=" * 70)
    print("Real-World Last.fm Matching Test")
    print("=" * 70)
    
    service = MusicMatchingService()
    
    # Simulate real Last.fm data with common inconsistencies
    test_tracks = [
        # Perfect match case
        LastfmTrack("Paranoid Android", "Radiohead", "OK Computer", 100, True),
        
        # Feature credits (common in Last.fm)
        LastfmTrack("Karma Police (feat. Various Artists)", "Radiohead", "OK Computer", 85, False),
        
        # Remaster suffixes (common inconsistency)
        LastfmTrack("No Surprises - 2009 Remaster", "Radiohead", "OK Computer", 70, True),
        
        # Live versions
        LastfmTrack("Let Down (Live)", "Radiohead", "OK Computer", 45, False),
        
        # Capitalization differences
        LastfmTrack("everything in its right place", "radiohead", "kid a", 60, False),
        
        # Complex album titles with variations
        LastfmTrack("Airbag", "Radiohead", "OK Computer (Collector's Edition)", 55, False),
    ]
    
    print(f"\n[TEST] Processing {len(test_tracks)} real-world style tracks...")
    
    def detailed_progress(current, total, track_name, match_type):
        print(f"  [{current:2d}/{total}] '{track_name}' -> {match_type}")
    
    results = await service.batch_match_lastfm_tracks(
        test_tracks, 
        max_concurrent=2,  # Be gentle on AOTY
        progress_callback=detailed_progress
    )
    
    # Analyze results
    successful = 0
    album_only = 0
    failed = 0
    
    print(f"\n[DETAILED RESULTS]")
    for i, result in enumerate(results):
        track = test_tracks[i]
        print(f"\n{i+1}. {track.artist} - {track.name}")
        print(f"   Album: {track.album}")
        print(f"   Result: {result.match_type} (confidence: {result.confidence}, score: {result.score:.3f})")
        print(f"   Time: {result.processing_time:.2f}s")
        
        if result.match_type == "track_match":
            successful += 1
            if result.target_data and result.target_data.get("matched_track"):
                matched = result.target_data["matched_track"]
                print(f"   Matched: #{matched['number']} {matched['title']} ({matched['length']})")
                if result.target_data.get("match_details"):
                    details = result.target_data["match_details"]
                    print(f"   Normalized query: '{details['normalized_query']}'")
                    print(f"   Normalized target: '{details['normalized_target']}'")
        elif result.match_type == "album_only":
            album_only += 1
            print(f"   Album found but track not matched well enough")
        else:
            failed += 1
            print(f"   Failed: {result.target_data.get('error', 'Unknown error') if result.target_data else 'No data'}")
    
    # Statistics
    print(f"\n[RESULTS SUMMARY]")
    print(f"  Track matches: {successful}/{len(test_tracks)} ({(successful/len(test_tracks))*100:.1f}%)")
    print(f"  Album-only matches: {album_only}/{len(test_tracks)} ({(album_only/len(test_tracks))*100:.1f}%)")
    print(f"  Complete failures: {failed}/{len(test_tracks)} ({(failed/len(test_tracks))*100:.1f}%)")
    
    stats = service.get_match_statistics()
    print(f"\n[SERVICE STATISTICS]")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return successful, len(test_tracks)


async def test_batch_performance():
    """Test batch processing performance with larger dataset"""
    print("\n" + "=" * 70)
    print("Batch Performance Test")
    print("=" * 70)
    
    service = MusicMatchingService()
    
    # Create a larger batch of varied test data
    large_batch = []
    
    # Add multiple albums worth of tracks
    albums_to_test = [
        ("Radiohead", "OK Computer", ["Airbag", "Paranoid Android", "Subterranean Homesick Alien", "Exit Music (For a Film)"]),
        ("Pink Floyd", "The Dark Side of the Moon", ["Speak to Me", "Breathe", "On the Run", "Time"]),
        ("The Beatles", "Abbey Road", ["Come Together", "Something", "Maxwell's Silver Hammer", "Oh! Darling"]),
    ]
    
    for artist, album, tracks in albums_to_test:
        for track in tracks:
            # Add some variations to simulate real-world inconsistencies
            variations = [
                track,  # Original
                f"{track} - Remastered",  # Remaster
                f"{track} (2009 Remaster)",  # Year remaster
            ]
            
            for variant in variations:
                large_batch.append(LastfmTrack(variant, artist, album, 50, False))
    
    print(f"\n[BATCH TEST] Processing {len(large_batch)} tracks...")
    start_time = asyncio.get_event_loop().time()
    
    def batch_progress(current, total, track_name, match_type):
        if current % 5 == 0 or current == total:  # Only print every 5th for large batches
            print(f"  Progress: {current}/{total} ({(current/total)*100:.0f}%)")
    
    batch_results = await service.batch_match_lastfm_tracks(
        large_batch,
        max_concurrent=3,
        progress_callback=batch_progress
    )
    
    total_time = asyncio.get_event_loop().time() - start_time
    
    # Analyze batch performance
    successful = sum(1 for r in batch_results if r.match_type == "track_match")
    album_only = sum(1 for r in batch_results if r.match_type == "album_only")
    
    print(f"\n[BATCH RESULTS]")
    print(f"  Total processed: {len(batch_results)}")
    print(f"  Successful matches: {successful} ({(successful/len(batch_results))*100:.1f}%)")
    print(f"  Album-only matches: {album_only} ({(album_only/len(batch_results))*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per track: {total_time/len(batch_results):.3f}s")
    
    # Cache efficiency
    stats = service.get_match_statistics()
    print(f"  Cache hit rate: {stats['cache_hit_rate_percent']}%")
    print(f"  AOTY requests made: {stats['aoty_requests_made']}")
    
    return successful, len(batch_results)


async def run_all_integration_tests():
    """Run all integration tests"""
    print("AOTY Fuzzy Matching Integration Test Suite")
    print("=" * 80)
    
    total_passed = 0
    total_tests = 0
    
    # Run test suites
    test_suites = [
        ("Fuzzy Matching Edge Cases", test_fuzzy_matching_edge_cases),
        ("Real-World Last.fm Matching", test_real_world_lastfm_matching),
        ("Batch Performance", test_batch_performance),
    ]
    
    for suite_name, test_func in test_suites:
        try:
            passed, tests = await test_func()
            total_passed += passed
            total_tests += tests
            print(f"\n[{suite_name}] Result: {passed}/{tests} passed")
        except Exception as e:
            print(f"\n[ERROR] {suite_name} failed: {e}")
            total_tests += 1  # Count as one failed test
        
        # Small delay between test suites
        await asyncio.sleep(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Total Passed: {total_passed}")
    print(f"Total Tests: {total_tests}")
    if total_tests > 0:
        print(f"Success Rate: {(total_passed / total_tests) * 100:.1f}%")
    
    if total_passed >= total_tests * 0.8:  # 80% success rate
        print("\n[SUCCESS] Integration tests passed!")
        return True
    else:
        print("\n[FAILURE] Too many integration test failures")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Tests cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] Integration test runner crashed: {e}")
        sys.exit(1)