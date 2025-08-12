#!/usr/bin/env python3
"""
Verify Database Data Quality Script

This script checks the quality and completeness of data inserted into the tracks table.
"""
import os
import sys
import json
from typing import Dict, List, Any

# Add the backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from ingestion.insert_to_supabase import (
    get_track_count, get_tracks_by_artist, get_training_dataset,
    get_supabase_client
)

def analyze_database_content():
    """Analyze the content and quality of database data"""
    
    print("DATABASE DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    # Get basic counts
    total_tracks = get_track_count()
    print(f"Total tracks in database: {total_tracks}")
    
    if total_tracks == 0:
        print("No tracks found in database!")
        return
    
    # Get sample data for analysis
    sample_tracks = get_training_dataset(limit=min(50, total_tracks))
    print(f"Analyzing sample of {len(sample_tracks)} tracks...")
    
    # Analyze data completeness
    print("\nDATA COMPLETENESS ANALYSIS:")
    print("-" * 30)
    
    fields_analysis = {
        'title': 0,
        'artist': 0,
        'album': 0,
        'genres': 0,
        'moods': 0,
        'aoty_score': 0,
        'duration_ms': 0,
        'popularity': 0,
        'track_number': 0,
        'release_date': 0,
        'cover_url': 0,
        'spotify_url': 0,
        'spotify_id': 0,
        'explicit': 0
    }
    
    for track in sample_tracks:
        for field in fields_analysis:
            if field in track and track[field]:
                if field in ['genres', 'moods']:
                    if isinstance(track[field], list) and len(track[field]) > 0:
                        fields_analysis[field] += 1
                else:
                    fields_analysis[field] += 1
    
    # Print completeness percentages
    for field, count in fields_analysis.items():
        percentage = (count / len(sample_tracks)) * 100
        print(f"{field:15}: {count:3}/{len(sample_tracks):3} ({percentage:5.1f}%)")
    
    # Analyze genre and mood diversity
    print("\nGENRE ANALYSIS:")
    print("-" * 15)
    all_genres = []
    for track in sample_tracks:
        if track.get('genres'):
            all_genres.extend(track['genres'])
    
    if all_genres:
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print(f"Total unique genres: {len(genre_counts)}")
        print("Top 10 genres:")
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (genre, count) in enumerate(sorted_genres[:10], 1):
            print(f"  {i:2}. {genre}: {count}")
    
    print("\nMOOD ANALYSIS:")
    print("-" * 14)
    all_moods = []
    for track in sample_tracks:
        if track.get('moods'):
            all_moods.extend(track['moods'])
    
    if all_moods:
        mood_counts = {}
        for mood in all_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        print(f"Total unique moods: {len(mood_counts)}")
        print("Top 10 moods:")
        sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (mood, count) in enumerate(sorted_moods[:10], 1):
            print(f"  {i:2}. {mood}: {count}")
    
    # Analyze AOTY scores
    print("\nAOTY SCORE ANALYSIS:")
    print("-" * 19)
    aoty_scores = [track['aoty_score'] for track in sample_tracks if track.get('aoty_score')]
    if aoty_scores:
        avg_score = sum(aoty_scores) / len(aoty_scores)
        min_score = min(aoty_scores)
        max_score = max(aoty_scores)
        print(f"Tracks with AOTY scores: {len(aoty_scores)}/{len(sample_tracks)}")
        print(f"Average AOTY score: {avg_score:.1f}")
        print(f"Score range: {min_score:.1f} - {max_score:.1f}")
    else:
        print("No AOTY scores found")
    
    # Show sample tracks
    print("\nSAMPLE TRACK DATA:")
    print("-" * 18)
    for i, track in enumerate(sample_tracks[:3], 1):
        print(f"\n{i}. {track['title']} by {track['artist']}")
        print(f"   Album: {track['album']}")
        print(f"   Genres: {track.get('genres', [])}")
        print(f"   Moods: {track.get('moods', [])}")
        print(f"   AOTY Score: {track.get('aoty_score', 'N/A')}")
        print(f"   Duration: {track.get('duration_ms', 'N/A')}ms")
        print(f"   Track #: {track.get('track_number', 'N/A')}")
        print(f"   Release Date: {track.get('release_date', 'N/A')}")
        print(f"   Cover URL: {'Yes' if track.get('cover_url') else 'No'}")
        print(f"   Spotify ID: {'Yes' if track.get('spotify_id') else 'No'}")
        print(f"   Explicit: {track.get('explicit', 'N/A')}")
    
    # Check for specific artists from our test
    print("\nTEST ARTIST VERIFICATION:")
    print("-" * 25)
    
    test_artists = ["Travis Scott", "Frank Ocean"]
    for artist in test_artists:
        artist_tracks = get_tracks_by_artist(artist, limit=5)
        print(f"\n{artist}: {len(artist_tracks)} tracks found")
        if artist_tracks:
            for track in artist_tracks[:3]:
                print(f"  - {track['title']} (Album: {track['album']})")

def check_data_for_ml():
    """Check if data is suitable for ML training"""
    print("\n" + "=" * 50)
    print("ML READINESS ASSESSMENT")
    print("=" * 50)
    
    # Get larger sample for ML assessment
    ml_sample = get_training_dataset(limit=100)
    
    if not ml_sample:
        print("No data available for ML training!")
        return
    
    print(f"Evaluating {len(ml_sample)} tracks for ML readiness...")
    
    # Check essential fields for ML
    essential_fields = ['genres', 'moods', 'aoty_score']
    field_coverage = {}
    
    for field in essential_fields:
        count = 0
        for track in ml_sample:
            if field in track and track[field]:
                if field in ['genres', 'moods']:
                    if isinstance(track[field], list) and len(track[field]) > 0:
                        count += 1
                else:
                    count += 1
        field_coverage[field] = (count / len(ml_sample)) * 100
    
    print("\nML FEATURE AVAILABILITY:")
    print("-" * 25)
    for field, coverage in field_coverage.items():
        status = "[GOOD]" if coverage >= 70 else "[OK]" if coverage >= 50 else "[POOR]"
        print(f"{status} {field:12}: {coverage:5.1f}% coverage")
    
    # Overall ML readiness
    avg_coverage = sum(field_coverage.values()) / len(field_coverage)
    print(f"\nOverall ML Readiness: {avg_coverage:.1f}%")
    
    if avg_coverage >= 70:
        print("[SUCCESS] Data is READY for ML training!")
    elif avg_coverage >= 50:
        print("[WARNING] Data has MODERATE quality for ML training")
    else:
        print("[FAIL] Data needs MORE enrichment before ML training")
    
    # Recommendations
    print("\nRECOMMENDations:")
    print("-" * 15)
    if field_coverage['genres'] < 70:
        print("- Increase genre coverage by adding more music sources")
    if field_coverage['moods'] < 70:
        print("- Enhance mood extraction from tags and descriptions")  
    if field_coverage['aoty_score'] < 70:
        print("- Improve AOTY matching to get more ratings")

if __name__ == "__main__":
    try:
        analyze_database_content()
        check_data_for_ml()
        
        print("\n" + "=" * 50)
        print("DATABASE VERIFICATION COMPLETE")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during database verification: {e}")
        sys.exit(1)