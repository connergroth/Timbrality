#!/usr/bin/env python3
"""
Test script for search query cleaning functionality
"""
import sys
import os

# Add the parent directory to the path so we can import from backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.search_cleaner import clean_search_query, extract_artist_and_album


def test_clean_search_query():
    """Test search query cleaning functionality"""
    print("🧪 Testing search query cleaning...")
    
    test_cases = [
        ("Denzel Curry Melt My Eyez See Your Future (Spotify)", "Denzel Curry Melt My Eyez See Your Future"),
        ("Radiohead OK Computer (Apple Music)", "Radiohead OK Computer"),
        ("Tyler The Creator Call Me If You Get Lost [Spotify]", "Tyler The Creator Call Me If You Get Lost"),
        ("Pink Floyd The Dark Side of the Moon (Remastered)", "Pink Floyd The Dark Side of the Moon"),
        ("Kendrick Lamar DAMN. (Deluxe Edition)", "Kendrick Lamar DAMN."),
        ("Mac Miller Swimming (Explicit)", "Mac Miller Swimming"),
        ("Clean query without suffixes", "Clean query without suffixes"),
        ("", ""),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for original, expected in test_cases:
        result = clean_search_query(original)
        if result == expected:
            print(f"✅ '{original}' -> '{result}'")
            passed += 1
        else:
            print(f"❌ '{original}' -> '{result}' (expected: '{expected}')")
    
    print(f"\n📊 Query cleaning results: {passed}/{total} tests passed")
    return passed == total


def test_extract_artist_and_album():
    """Test artist and album extraction functionality"""
    print("\n🧪 Testing artist and album extraction...")
    
    test_cases = [
        ("Denzel Curry Melt My Eyez See Your Future", ("Denzel Curry", "Melt My Eyez See Your Future")),
        ("Radiohead OK Computer", ("Radiohead", "OK Computer")),
        ("Tyler The Creator Call Me If You Get Lost", ("Tyler The Creator", "Call Me If You Get Lost")),
        ("Single", (None, "Single")),
        ("Artist - Album", ("Artist", "Album")),
        ("Album by Artist", ("Artist", "Album")),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for query, (expected_artist, expected_album) in test_cases:
        artist, album = extract_artist_and_album(query)
        if artist == expected_artist and album == expected_album:
            print(f"✅ '{query}' -> Artist: '{artist}', Album: '{album}'")
            passed += 1
        else:
            print(f"❌ '{query}' -> Artist: '{artist}', Album: '{album}' (expected: Artist: '{expected_artist}', Album: '{expected_album}')")
    
    print(f"\n📊 Extraction results: {passed}/{total} tests passed")
    return passed == total


def main():
    """Run all tests"""
    print("🧪 Search Query Cleaner Test Suite")
    print("=" * 45)
    
    all_passed = True
    
    # Test query cleaning
    if not test_clean_search_query():
        all_passed = False
    
    # Test artist/album extraction
    if not test_extract_artist_and_album():
        all_passed = False
    
    # Results
    print(f"\n📊 Overall Results: {'All tests passed!' if all_passed else 'Some tests failed'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 