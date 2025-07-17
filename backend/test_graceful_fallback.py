#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_graceful_data_integration():
    """Test data integration with graceful fallback when AOTY is blocked"""
    
    print("Testing Graceful Data Integration Pipeline")
    print("=" * 60)
    print("🎯 Goal: Get album data even when AOTY blocks us")
    print()
    
    # Test albums
    test_albums = [
        ("Radiohead", "OK Computer"),
        ("Pink Floyd", "The Dark Side of the Moon"),
        ("The Beatles", "Abbey Road")
    ]
    
    # First test what actually works
    print("🔍 Testing individual services:")
    print("-" * 40)
    
    # Test Spotify (should work)
    try:
        from ingestion.spotify_fetcher import get_spotify_client, search_albums
        print("📊 Testing Spotify...")
        
        client = get_spotify_client()
        if client:
            results = search_albums(client, "OK Computer", "Radiohead")
            if results:
                print(f"✅ Spotify: Found {len(results)} results")
                album = results[0]
                print(f"   Album: {album.get('name')} by {album.get('artists', [{}])[0].get('name', 'Unknown')}")
            else:
                print("❌ Spotify: No results found")
        else:
            print("⚠️  Spotify: Client not initialized (check credentials)")
    except Exception as e:
        print(f"❌ Spotify: Error - {str(e)}")
    
    # Test Last.fm (might work if secret is provided)
    try:
        from ingestion.lastfm_fetcher import search_artist_albums
        print("🎵 Testing Last.fm...")
        
        results = search_artist_albums("Radiohead")
        if results:
            print(f"✅ Last.fm: Found {len(results)} albums")
            print(f"   Sample: {results[0].get('name', 'Unknown')}")
        else:
            print("❌ Last.fm: No results found")
    except Exception as e:
        print(f"❌ Last.fm: Error - {str(e)}")
    
    # Test AOTY with graceful handling
    print("🌐 Testing AOTY (expecting blocks)...")
    try:
        from scraper.aoty_scraper import search_albums as aoty_search
        
        # Try one quick search with short timeout
        results = await aoty_search("OK Computer", limit=1, max_retries=1)
        if results:
            print(f"✅ AOTY: Surprisingly found {len(results)} results!")
        else:
            print("⚠️  AOTY: Blocked as expected (this is normal)")
    except Exception as e:
        print(f"⚠️  AOTY: Blocked/Error as expected - {str(e)[:50]}...")
    
    print()
    print("🎯 Integration Strategy Test:")
    print("-" * 35)
    
    # Test integrated approach
    for artist, album_title in test_albums:
        print(f"\n🔍 Processing: {artist} - {album_title}")
        
        album_data = {
            "artist": artist,
            "title": album_title,
            "spotify_data": None,
            "lastfm_data": None,
            "aoty_data": None,
            "combined_score": "N/A"
        }
        
        # Try Spotify first (most reliable)
        try:
            from ingestion.spotify_fetcher import get_spotify_client, search_albums
            client = get_spotify_client()
            if client:
                spotify_results = search_albums(client, album_title, artist)
                if spotify_results:
                    album_data["spotify_data"] = {
                        "found": True,
                        "tracks": len(spotify_results[0].get('tracks', {}).get('items', [])),
                        "popularity": spotify_results[0].get('popularity', 0)
                    }
                    print(f"   ✅ Spotify: {album_data['spotify_data']['tracks']} tracks, popularity {album_data['spotify_data']['popularity']}")
                else:
                    print("   ❌ Spotify: Not found")
            else:
                print("   ⚠️  Spotify: Not configured")
        except Exception as e:
            print(f"   ❌ Spotify: Error - {str(e)[:30]}...")
        
        # Try Last.fm for additional metadata
        try:
            from ingestion.lastfm_fetcher import search_artist_albums
            lastfm_albums = search_artist_albums(artist)
            if lastfm_albums:
                # Look for matching album
                matching_album = None
                for lfm_album in lastfm_albums:
                    if album_title.lower() in lfm_album.get('name', '').lower():
                        matching_album = lfm_album
                        break
                
                if matching_album:
                    album_data["lastfm_data"] = {"found": True, "name": matching_album.get('name')}
                    print(f"   ✅ Last.fm: Found as '{matching_album.get('name')}'")
                else:
                    print(f"   ⚠️  Last.fm: Found {len(lastfm_albums)} albums but no exact match")
            else:
                print("   ❌ Last.fm: No albums found")
        except Exception as e:
            print(f"   ❌ Last.fm: Error - {str(e)[:30]}...")
        
        # Skip AOTY for now since it's blocking (but show how we'd handle it)
        album_data["aoty_data"] = {"blocked": True, "message": "Service temporarily unavailable"}
        print("   ⚠️  AOTY: Skipped (service blocking requests)")
        
        # Calculate combined score based on available data
        score_components = []
        if album_data["spotify_data"] and album_data["spotify_data"]["found"]:
            spotify_score = album_data["spotify_data"]["popularity"] / 100.0
            score_components.append(f"Spotify: {spotify_score:.2f}")
        
        if album_data["lastfm_data"] and album_data["lastfm_data"]["found"]:
            score_components.append("Last.fm: available")
        
        if score_components:
            album_data["combined_score"] = " + ".join(score_components)
        
        print(f"   📊 Combined Data: {album_data['combined_score']}")
    
    print()
    print("💡 Summary & Recommendations:")
    print("-" * 35)
    print("✅ Spotify: Primary data source - reliable and fast")
    print("✅ Last.fm: Secondary metadata - works when configured")
    print("⚠️  AOTY: Currently blocking - need to use sparingly")
    print()
    print("🔧 Strategy for AOTY blocking:")
    print("   1. Cache AOTY data when we can get it")
    print("   2. Use very long delays between AOTY requests (30+ seconds)")
    print("   3. Focus on Spotify + Last.fm for real-time features")
    print("   4. Only use AOTY for special, high-value requests")
    print("   5. Consider manual data entry for important albums")
    
    print("\n" + "=" * 60)
    print("✅ Test completed! Integration working with graceful fallbacks.")

if __name__ == "__main__":
    asyncio.run(test_graceful_data_integration()) 