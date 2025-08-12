#!/usr/bin/env python3
"""
Quick script to check sample track data from database
"""
import os
import sys
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from ingestion.insert_to_supabase import get_supabase_client

def check_sample_tracks():
    """Check a few sample tracks to see what data we have"""
    supabase = get_supabase_client()
    
    # Get Travis Scott tracks specifically to check for Spotify IDs
    result = supabase.table('tracks').select('*').ilike('artist', '%Travis Scott%').limit(3).execute()
    
    if result.data:
        print("Sample track data:")
        print("="*50)
        for i, track in enumerate(result.data, 1):
            print(f"\n{i}. {track.get('title', 'N/A')} by {track.get('artist', 'N/A')}")
            print(f"   Album: {track.get('album', 'N/A')}")
            print(f"   Track #: {track.get('track_number', 'N/A')}")
            print(f"   Cover URL: {'Yes' if track.get('cover_url') else 'No'}")
            print(f"   Spotify URL: {'Yes' if track.get('spotify_url') else 'No'}")
            print(f"   Spotify ID: {track.get('spotify_id', 'None')}")
            print(f"   Explicit: {track.get('explicit', 'None')}")
            print(f"   Release Date: {track.get('release_date', 'N/A')}")
            
            # Show all available fields
            print(f"   Available fields: {list(track.keys())}")

if __name__ == "__main__":
    check_sample_tracks()