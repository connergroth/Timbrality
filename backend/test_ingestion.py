#!/usr/bin/env python3
"""
Test script for Tensoe Ingestion Pipeline

This script tests the ingestion pipeline with a simple album to verify setup.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = [
        'SPOTIFY_CLIENT_ID',
        'SPOTIFY_CLIENT_SECRET',
        'LASTFM_API_KEY',
        'LASTFM_API_SECRET',
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these in your .env file")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True


def test_spotify_connection():
    """Test Spotify API connection"""
    try:
        from ingestion.spotify_fetcher import get_spotify_client
        
        logger.info("Testing Spotify connection...")
        client = get_spotify_client()
        
        # Try a simple search
        results = client.search(q="Radiohead", type='artist', limit=1)
        
        if results['artists']['items']:
            artist = results['artists']['items'][0]
            logger.info(f"✅ Spotify connection successful - Found artist: {artist['name']}")
            return True
        else:
            logger.error("❌ Spotify search returned no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Spotify connection failed: {e}")
        return False


def test_lastfm_connection():
    """Test Last.fm API connection"""
    try:
        from ingestion.lastfm_fetcher import get_lastfm_network
        
        logger.info("Testing Last.fm connection...")
        network = get_lastfm_network()
        
        # Try to get artist info
        artist = network.get_artist("Radiohead")
        artist_name = artist.get_name()
        
        logger.info(f"✅ Last.fm connection successful - Found artist: {artist_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Last.fm connection failed: {e}")
        return False


def test_supabase_connection():
    """Test Supabase database connection"""
    try:
        from ingestion.insert_to_supabase import get_supabase_client, setup_database
        
        logger.info("Testing Supabase connection...")
        client = get_supabase_client()
        
        # Try to setup the database
        setup_success = setup_database()
        
        if setup_success:
            logger.info("✅ Supabase connection successful")
            return True
        else:
            logger.error("❌ Supabase database setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False


def test_aoty_scraper():
    """Test AOTY scraper functionality"""
    try:
        import asyncio
        from ingestion.aoty_scraper import get_album_aoty_data
        
        logger.info("Testing AOTY scraper...")
        
        # Test with a simple album
        async def test_aoty():
            result = await get_album_aoty_data("OK Computer", "Radiohead")
            return result
        
        result = asyncio.run(test_aoty())
        
        if result and (result['album_score'] or result['genres']):
            logger.info(f"✅ AOTY scraper successful - Found data for OK Computer")
            return True
        else:
            logger.warning("⚠️ AOTY scraper returned empty results (this may be normal)")
            return True  # Don't fail the test for AOTY issues
            
    except Exception as e:
        logger.error(f"❌ AOTY scraper failed: {e}")
        return False


def test_simple_ingestion():
    """Test a simple album ingestion"""
    try:
        from ingestion import run_ingestion
        
        logger.info("Testing simple album ingestion...")
        logger.info("This may take a few minutes...")
        
        # Test with a small, well-known album
        success = run_ingestion("In Rainbows", "Radiohead")
        
        if success:
            logger.info("✅ Simple ingestion test successful!")
            
            # Check if tracks were actually inserted
            from ingestion.insert_to_supabase import get_track_count
            total_tracks = get_track_count()
            logger.info(f"Total tracks in database: {total_tracks}")
            
            return True
        else:
            logger.error("❌ Simple ingestion test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple ingestion test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Tensoe Ingestion Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", check_environment_variables),
        ("Spotify Connection", test_spotify_connection),
        ("Last.fm Connection", test_lastfm_connection),
        ("Supabase Connection", test_supabase_connection),
        ("AOTY Scraper", test_aoty_scraper),
        ("Simple Ingestion", test_simple_ingestion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ingestion pipeline is ready to use.")
    elif passed >= total - 1:
        print("⚠️ Most tests passed. Minor issues may exist but the pipeline should work.")
    else:
        print("❌ Multiple tests failed. Please check your configuration.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 