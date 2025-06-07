#!/usr/bin/env python3
"""
Example usage of the Tensoe Ingestion Pipeline

This script demonstrates various ways to use the ingestion pipeline
for collecting music data from Spotify, Last.fm, and AOTY.
"""

import asyncio
import logging
from pathlib import Path

# Import the ingestion functions
from ingestion import (
    run_ingestion,
    run_batch_ingestion,
    run_batch_from_csv,
    search_and_ingest
)
from ingestion.insert_to_supabase import (
    setup_database,
    get_track_count,
    get_tracks_by_artist,
    export_to_csv
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_single_album():
    """Example 1: Ingest a single album"""
    print("\n🎵 Example 1: Single Album Ingestion")
    print("=" * 50)
    
    # Ingest a classic album
    album_name = "Kid A"
    artist_name = "Radiohead"
    
    print(f"Ingesting '{album_name}' by {artist_name}...")
    success = run_ingestion(album_name, artist_name)
    
    if success:
        print("✅ Ingestion successful!")
    else:
        print("❌ Ingestion failed!")
    
    return success


def example_2_batch_ingestion():
    """Example 2: Ingest multiple albums"""
    print("\n🎵 Example 2: Batch Album Ingestion")
    print("=" * 50)
    
    # Define a list of albums to ingest
    album_list = [
        ("OK Computer", "Radiohead"),
        ("In Rainbows", "Radiohead"),
        ("Blonde", "Frank Ocean"),
        ("To Pimp a Butterfly", "Kendrick Lamar")
    ]
    
    print(f"Ingesting {len(album_list)} albums...")
    results = run_batch_ingestion(album_list, use_async=True)
    
    print(f"Results: {results['successful']} successful, {results['failed']} failed")
    
    if results['errors']:
        print("Errors encountered:")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    return results


def example_3_csv_ingestion():
    """Example 3: Ingest from CSV file"""
    print("\n🎵 Example 3: CSV Batch Ingestion")
    print("=" * 50)
    
    # Use the sample CSV file
    csv_file_path = Path(__file__).parent / "sample_albums.csv"
    
    if not csv_file_path.exists():
        print(f"❌ CSV file not found: {csv_file_path}")
        return None
    
    print(f"Ingesting albums from {csv_file_path}...")
    results = run_batch_from_csv(str(csv_file_path), use_async=True)
    
    print(f"Results: {results['successful']} successful, {results['failed']} failed")
    return results


def example_4_search_and_ingest():
    """Example 4: Search for albums and ingest"""
    print("\n🎵 Example 4: Search and Ingest")
    print("=" * 50)
    
    search_queries = [
        "indie rock 2023",
        "ambient electronic",
        "jazz fusion"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = search_and_ingest(query, limit=3)
        print(f"  Results: {results['successful']} successful, {results['failed']} failed")


def example_5_data_exploration():
    """Example 5: Explore ingested data"""
    print("\n🎵 Example 5: Data Exploration")
    print("=" * 50)
    
    # Get total track count
    total_tracks = get_track_count()
    print(f"Total tracks in database: {total_tracks}")
    
    if total_tracks > 0:
        # Get tracks by a specific artist
        radiohead_tracks = get_tracks_by_artist("Radiohead", limit=10)
        print(f"\nRadiohead tracks found: {len(radiohead_tracks)}")
        
        if radiohead_tracks:
            print("Sample tracks:")
            for track in radiohead_tracks[:3]:
                print(f"  - {track['title']} from {track['album']}")
                print(f"    Genres: {track['genres']}")
                print(f"    Moods: {track['moods']}")
                print(f"    AOTY Score: {track['aoty_score']}")
                print()


def example_6_data_export():
    """Example 6: Export data for ML training"""
    print("\n🎵 Example 6: Data Export")
    print("=" * 50)
    
    # Export data to CSV for ML training
    export_file = "training_data_export.csv"
    
    print(f"Exporting data to {export_file}...")
    success = export_to_csv(export_file, limit=1000)
    
    if success:
        print("✅ Export successful!")
        # Check if file exists and show size
        export_path = Path(export_file)
        if export_path.exists():
            size_mb = export_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
    else:
        print("❌ Export failed!")


async def example_7_async_processing():
    """Example 7: Advanced async processing"""
    print("\n🎵 Example 7: Advanced Async Processing")
    print("=" * 50)
    
    from ingestion.ingest_runner import run_ingestion_async
    
    # Process multiple albums concurrently
    albums = [
        ("The Dark Side of the Moon", "Pink Floyd"),
        ("Abbey Road", "The Beatles"),
        ("Nevermind", "Nirvana")
    ]
    
    print(f"Processing {len(albums)} albums concurrently...")
    
    # Create async tasks
    tasks = []
    for album_name, artist_name in albums:
        task = run_ingestion_async(album_name, artist_name)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = sum(1 for result in results if result is True)
    failed = len(results) - successful
    
    print(f"Async results: {successful} successful, {failed} failed")


def main():
    """Run all examples"""
    print("🎼 Tensoe Ingestion Pipeline Examples")
    print("=" * 60)
    
    # Setup database first
    print("Setting up database...")
    setup_database()
    
    # Run examples
    try:
        # Basic examples
        example_1_single_album()
        example_2_batch_ingestion()
        
        # Advanced examples (comment out if you don't have the CSV file)
        # example_3_csv_ingestion()
        
        # Search and data exploration
        example_4_search_and_ingest()
        example_5_data_exploration()
        example_6_data_export()
        
        # Async example
        print("\nRunning async example...")
        asyncio.run(example_7_async_processing())
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.exception("Error in main execution")


if __name__ == "__main__":
    main() 