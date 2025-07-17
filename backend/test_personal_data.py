#!/usr/bin/env python3
"""
Personal Data Integration Test
Fetches user's Last.fm tracks and cross-references with Spotify and AOTY

Updated to use CloudScraper for AOTY - much more reliable and faster!
"""
import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Results tracking
results = {
    "tracks_found": [],
    "tracks_failed": [],
    "spotify_matches": 0,
    "aoty_matches": 0,
    "total_processed": 0
}

def print_header():
    """Print a nice header"""
    print("🎵 PERSONAL MUSIC DATA INTEGRATION TEST")
    print("=" * 60)
    print("Fetching your personal tracks from Last.fm and cross-referencing")
    print("with Spotify and AOTY data sources...")
    print("=" * 60)


def get_lastfm_username():
    """Get Last.fm username from user input or config"""
    # You can hardcode your username here or make it configurable
    username = input("\n🎶 Enter your Last.fm username (or press Enter to use sample data): ").strip()
    
    if not username:
        print("📝 No username provided, using sample data...")
        return None
    
    return username


def get_sample_tracks() -> List[Dict]:
    """Fallback sample tracks if Last.fm fails"""
    return [
        {"name": "Paranoid Android", "artist": "Radiohead", "album": "OK Computer"},
        {"name": "Bohemian Rhapsody", "artist": "Queen", "album": "A Night at the Opera"},
        {"name": "Stairway to Heaven", "artist": "Led Zeppelin", "album": "Led Zeppelin IV"},
        {"name": "Hotel California", "artist": "Eagles", "album": "Hotel California"},
        {"name": "Sweet Child O' Mine", "artist": "Guns N' Roses", "album": "Appetite for Destruction"},
        {"name": "Imagine", "artist": "John Lennon", "album": "Imagine"},
        {"name": "Billie Jean", "artist": "Michael Jackson", "album": "Thriller"},
        {"name": "Smells Like Teen Spirit", "artist": "Nirvana", "album": "Nevermind"},
        {"name": "Purple Haze", "artist": "Jimi Hendrix", "album": "Are You Experienced"},
        {"name": "Like a Rolling Stone", "artist": "Bob Dylan", "album": "Highway 61 Revisited"}
    ]


def fetch_lastfm_tracks(username: str, limit: int = 10) -> List[Dict]:
    """Fetch user's top tracks from Last.fm"""
    try:
        from ingestion.lastfm_fetcher import get_lastfm_network
        
        print(f"🔍 Fetching top {limit} tracks for user: {username}")
        
        network = get_lastfm_network()
        user = network.get_user(username)
        
        # Get user's top tracks
        top_tracks = user.get_top_tracks(limit=limit)
        
        tracks = []
        for track_item in top_tracks:
            track = track_item.item
            try:
                # Get track details
                track_info = {
                    "name": track.get_name(),
                    "artist": track.get_artist().get_name(),
                    "playcount": track_item.weight,
                    "url": track.get_url()
                }
                
                # Try to get album info
                try:
                    album = track.get_album()
                    if album:
                        track_info["album"] = album.get_name()
                    else:
                        track_info["album"] = None
                except:
                    track_info["album"] = None
                
                tracks.append(track_info)
                
            except Exception as e:
                print(f"⚠️  Error processing track: {e}")
                continue
        
        print(f"✅ Successfully fetched {len(tracks)} tracks from Last.fm")
        return tracks
        
    except Exception as e:
        print(f"❌ Error fetching Last.fm data: {e}")
        print("📝 Using sample tracks instead...")
        return get_sample_tracks()[:limit]


def test_spotify_match(track_name: str, artist_name: str) -> Optional[Dict]:
    """Test if track exists on Spotify"""
    try:
        from ingestion.spotify_fetcher import get_spotify_client
        
        client = get_spotify_client()
        
        # Search for the track
        search_query = f"track:{track_name} artist:{artist_name}"
        results = client.search(q=search_query, type='track', limit=1)
        
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return {
                "found": True,
                "name": track['name'],
                "artist": ', '.join([artist['name'] for artist in track['artists']]),
                "album": track['album']['name'],
                "popularity": track['popularity'],
                "preview_url": track['preview_url'],
                "spotify_url": track['external_urls']['spotify']
            }
        else:
            return {"found": False}
            
    except Exception as e:
        print(f"⚠️  Spotify error for '{track_name}' by {artist_name}: {e}")
        return {"found": False, "error": str(e)}


async def test_aoty_match(track_name: str, artist_name: str, album_name: Optional[str] = None) -> Optional[Dict]:
    """Test if track's album exists on AOTY using CloudScraper"""
    try:
        from scraper.aoty_cloudscraper import search_albums, get_album_url
        
        # Search for the album if we have it, otherwise search for artist
        if album_name:
            search_query = f"{artist_name} {album_name}"
        else:
            search_query = f"{artist_name}"
        
        # Search for albums
        search_results = await search_albums(search_query, limit=3)
        
        if search_results:
            # Find best match
            best_match = None
            for result in search_results:
                if artist_name.lower() in result.artist.lower():
                    if album_name and album_name.lower() in result.title.lower():
                        best_match = result
                        break
                    elif not best_match:
                        best_match = result
            
            if best_match:
                return {
                    "found": True,
                    "album_title": best_match.title,
                    "artist": best_match.artist,
                    "year": best_match.year,
                    "score": best_match.score,
                    "url": best_match.url,
                    "cover_image": best_match.cover_image
                }
        
        return {"found": False}
        
    except Exception as e:
        print(f"⚠️  AOTY error for '{track_name}' by {artist_name}: {e}")
        return {"found": False, "error": str(e)}


async def process_track(track: Dict, index: int, total: int) -> Dict:
    """Process a single track across all platforms"""
    track_name = track['name']
    artist_name = track['artist']
    album_name = track.get('album')
    
    print(f"\n🎵 [{index}/{total}] Processing: {track_name} by {artist_name}")
    if album_name:
        print(f"   Album: {album_name}")
    
    result = {
        "track": track,
        "spotify": None,
        "aoty": None,
        "success": False
    }
    
    # Test Spotify
    print("   🔍 Searching Spotify...")
    spotify_result = test_spotify_match(track_name, artist_name)
    result["spotify"] = spotify_result
    
    if spotify_result and spotify_result.get("found"):
        print(f"   ✅ Found on Spotify: {spotify_result['name']} (popularity: {spotify_result['popularity']})")
        results["spotify_matches"] += 1
    else:
        print("   ❌ Not found on Spotify")
    
    # Test AOTY
    print("   🔍 Searching AOTY...")
    aoty_result = await test_aoty_match(track_name, artist_name, album_name)
    result["aoty"] = aoty_result
    
    if aoty_result and aoty_result.get("found"):
        print(f"   ✅ Found on AOTY: {aoty_result['album_title']} (score: {aoty_result.get('score', 'N/A')})")
        results["aoty_matches"] += 1
    else:
        print("   ❌ Not found on AOTY")
    
    # Determine success
    if (spotify_result and spotify_result.get("found")) or (aoty_result and aoty_result.get("found")):
        result["success"] = True
        results["tracks_found"].append(result)
    else:
        results["tracks_failed"].append(result)
    
    results["total_processed"] += 1
    
    return result


def print_summary(processed_tracks: List[Dict]):
    """Print comprehensive summary"""
    print("\n" + "=" * 60)
    print("📊 PERSONAL DATA INTEGRATION SUMMARY")
    print("=" * 60)
    
    total = results["total_processed"]
    found = len(results["tracks_found"])
    failed = len(results["tracks_failed"])
    
    print(f"Total Tracks Processed: {total}")
    print(f"✅ Successfully Matched: {found} ({found/total*100:.1f}%)")
    print(f"❌ Failed to Match: {failed} ({failed/total*100:.1f}%)")
    print(f"🎵 Spotify Matches: {results['spotify_matches']} ({results['spotify_matches']/total*100:.1f}%)")
    print(f"🏆 AOTY Matches: {results['aoty_matches']} ({results['aoty_matches']/total*100:.1f}%)")
    
    # Show best matches
    if results["tracks_found"]:
        print(f"\n🌟 BEST CROSS-PLATFORM MATCHES:")
        for i, track_result in enumerate(results["tracks_found"][:5], 1):
            track = track_result["track"]
            spotify = track_result["spotify"]
            aoty = track_result["aoty"]
            
            print(f"\n{i}. {track['name']} by {track['artist']}")
            
            if spotify and spotify.get("found"):
                print(f"   🎵 Spotify: ✅ (popularity: {spotify['popularity']})")
                if spotify.get('preview_url'):
                    print(f"      Preview: {spotify['preview_url']}")
            
            if aoty and aoty.get("found"):
                print(f"   🏆 AOTY: ✅ {aoty['album_title']} (score: {aoty.get('score', 'N/A')})")
    
    # Show failures
    if results["tracks_failed"]:
        print(f"\n❌ TRACKS NOT FOUND ON ANY PLATFORM:")
        for track_result in results["tracks_failed"][:3]:
            track = track_result["track"]
            print(f"   • {track['name']} by {track['artist']}")
    
    print(f"\n🎯 Data Integration Success Rate: {found/total*100:.1f}%")
    
    if found/total >= 0.8:
        print("🎉 Excellent! Your data integration is working great!")
    elif found/total >= 0.6:
        print("👍 Good data integration coverage!")
    else:
        print("⚠️  Some integration issues - check API configurations")


async def main():
    """Main execution function"""
    print_header()
    
    # Get username
    username = get_lastfm_username()
    
    # Fetch tracks
    if username:
        print(f"\n🎶 Fetching your personal data from Last.fm...")
        tracks = fetch_lastfm_tracks(username, limit=10)
    else:
        print(f"\n📝 Using sample popular tracks...")
        tracks = get_sample_tracks()[:10]
    
    if not tracks:
        print("❌ No tracks to process!")
        return 1
    
    print(f"\n🔄 Processing {len(tracks)} tracks across all platforms...")
    
    # Process each track
    processed_tracks = []
    for i, track in enumerate(tracks, 1):
        try:
            result = await process_track(track, i, len(tracks))
            processed_tracks.append(result)
            
            # Small delay to be nice to APIs
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"💥 Error processing track {i}: {e}")
            continue
    
    # Print summary
    print_summary(processed_tracks)
    
    # Cleanup (CloudScraper doesn't need browser cleanup, but keeping for compatibility)
    try:
        from scraper.aoty_cloudscraper import close_browser
        await close_browser()
    except:
        pass
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test crashed: {str(e)}")
        sys.exit(1) 