#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_improved_aoty_scraper():
    """Test the improved AOTY scraper with better error handling"""
    
    print("Testing improved AOTY scraper...")
    print("=" * 50)
    
    try:
        from scraper.aoty_scraper import search_albums, get_album_url, close_browser
        
        # Test 1: Basic search
        print("\n1. Testing basic album search:")
        print("-" * 30)
        
        search_queries = [
            "OK Computer Radiohead",
            "The Dark Side of the Moon Pink Floyd", 
            "Thriller Michael Jackson"
        ]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            try:
                results = await search_albums(query, limit=3)
                if results:
                    print(f"✅ Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result.artist} - {result.title}")
                        if result.year:
                            print(f"     Year: {result.year}")
                        if result.score:
                            print(f"     Score: {result.score}")
                else:
                    print("❌ No results found")
            except Exception as e:
                print(f"❌ Search failed: {str(e)}")
        
        # Test 2: Specific album URL finding
        print("\n\n2. Testing specific album URL finding:")
        print("-" * 40)
        
        album_searches = [
            ("Radiohead", "OK Computer"),
            ("Pink Floyd", "The Dark Side of the Moon"),
            ("Nonexistent Artist", "Fake Album")
        ]
        
        for artist, album in album_searches:
            print(f"\nLooking for: {artist} - {album}")
            try:
                result = await get_album_url(artist, album)
                if result:
                    url, found_artist, found_album = result
                    print(f"✅ Found: {found_artist} - {found_album}")
                    print(f"   URL: {url}")
                else:
                    print("❌ Album not found")
            except Exception as e:
                print(f"❌ Search failed: {str(e)}")
        
        print("\n\n3. Testing timeout resilience:")
        print("-" * 35)
        
        # This should test our retry mechanism
        difficult_searches = [
            "very obscure album that probably doesn't exist",
            "test search with special chars !@#$%",
        ]
        
        for query in difficult_searches:
            print(f"\nTesting difficult search: '{query}'")
            try:
                results = await search_albums(query, limit=1)
                print(f"✅ Completed search (found {len(results)} results)")
            except Exception as e:
                print(f"⚠️  Search handled gracefully: {str(e)}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the backend directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        try:
            await close_browser()
            print("\n✅ Browser cleanup completed")
        except:
            pass
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_improved_aoty_scraper()) 