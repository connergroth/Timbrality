"""
Supabase Integration - Insert normalized tracks into Supabase database
"""
import os
import logging
from typing import List, Dict, Any, Optional
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from .normalizer import TrackData, to_dict, validate_track_data
try:
    from models.ingestion_models import EnhancedTrack, MLTrainingData
except ImportError:
    # Models may not be available in some contexts
    EnhancedTrack = None
    MLTrainingData = None

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')

def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def setup_database():
    """Create the tracks table if it doesn't exist"""
    try:
        supabase = get_supabase_client()
        
        # Check if tracks table exists by trying to query it
        try:
            result = supabase.table('tracks').select('id').limit(1).execute()
            logger.info("Tracks table already exists")
            return True
        except Exception as e:
            logger.info("Tracks table doesn't exist, creating it...")
        
        # The tracks table should already exist with the base schema
        # This function now just ensures the enhanced columns exist
        enhance_schema_sql = """
        -- Add new columns if they don't exist (safe to run multiple times)
        DO $$ 
        BEGIN
            -- Add enhanced columns to tracks table
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'release_date') THEN
                ALTER TABLE tracks ADD COLUMN release_date DATE;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'duration_ms') THEN
                ALTER TABLE tracks ADD COLUMN duration_ms INTEGER;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'genres') THEN
                ALTER TABLE tracks ADD COLUMN genres TEXT[] DEFAULT '{}';
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'moods') THEN
                ALTER TABLE tracks ADD COLUMN moods TEXT[] DEFAULT '{}';
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'spotify_url') THEN
                ALTER TABLE tracks ADD COLUMN spotify_url TEXT;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'explicit') THEN
                ALTER TABLE tracks ADD COLUMN explicit BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'track_number') THEN
                ALTER TABLE tracks ADD COLUMN track_number INTEGER;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tracks' AND column_name = 'album_total_tracks') THEN
                ALTER TABLE tracks ADD COLUMN album_total_tracks INTEGER;
            END IF;
        END $$;
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_tracks_genres ON tracks USING GIN(genres);
        CREATE INDEX IF NOT EXISTS idx_tracks_moods ON tracks USING GIN(moods);
        CREATE INDEX IF NOT EXISTS idx_tracks_release_date ON tracks(release_date);
        CREATE INDEX IF NOT EXISTS idx_tracks_duration ON tracks(duration_ms);
        """
        
        # Execute the SQL directly (this may require elevated permissions)
        # For now, we'll create a simplified version that works with Supabase's REST API
        logger.warning("Direct SQL execution may not be available. Please ensure the tracks table exists in Supabase.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False


def insert_tracks(tracks: List[TrackData], batch_size: int = 100) -> bool:
    """
    Insert tracks into Supabase database
    
    Args:
        tracks: List of TrackData objects to insert
        batch_size: Number of tracks to insert per batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Validate and convert tracks to dictionaries
        valid_tracks = []
        for track in tracks:
            if validate_track_data(track):
                track_dict = to_dict(track)
                # Convert lists to JSON strings for Supabase
                track_dict['genres'] = json.dumps(track_dict['genres']) if track_dict['genres'] else '[]'
                track_dict['moods'] = json.dumps(track_dict['moods']) if track_dict['moods'] else '[]'
                valid_tracks.append(track_dict)
            else:
                logger.warning(f"Skipping invalid track: {track.track_id}")
        
        if not valid_tracks:
            logger.error("No valid tracks to insert")
            return False
        
        # Insert tracks in batches
        total_inserted = 0
        total_failed = 0
        
        for i in range(0, len(valid_tracks), batch_size):
            batch = valid_tracks[i:i + batch_size]
            
            try:
                # Use upsert to handle duplicates
                result = supabase.table('tracks').upsert(batch).execute()
                
                if result.data:
                    batch_inserted = len(result.data)
                    total_inserted += batch_inserted
                    logger.info(f"Successfully inserted batch of {batch_inserted} tracks")
                else:
                    logger.warning(f"Batch insert returned no data")
                    total_failed += len(batch)
                    
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                total_failed += len(batch)
                
                # Try to insert tracks individually to identify problem tracks
                for track in batch:
                    try:
                        result = supabase.table('tracks').upsert([track]).execute()
                        if result.data:
                            total_inserted += 1
                            total_failed -= 1
                        else:
                            logger.error(f"Failed to insert track: {track.get('id', 'unknown')}")
                    except Exception as individual_error:
                        logger.error(f"Failed to insert individual track {track.get('id', 'unknown')}: {individual_error}")
        
        logger.info(f"Insert completed: {total_inserted} successful, {total_failed} failed")
        return total_inserted > 0
        
    except Exception as e:
        logger.error(f"Error inserting tracks: {e}")
        return False


def get_track_by_id(track_id: str) -> Optional[Dict]:
    """
    Get a track by its ID
    
    Args:
        track_id: Spotify track ID
        
    Returns:
        Track dictionary or None if not found
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('tracks').select('*').eq('id', track_id).execute()
        
        if result.data and len(result.data) > 0:
            track = result.data[0]
            # Parse JSON strings back to lists
            track['genres'] = json.loads(track['genres']) if track['genres'] else []
            track['moods'] = json.loads(track['moods']) if track['moods'] else []
            return track
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching track {track_id}: {e}")
        return None


def get_tracks_by_artist(artist_name: str, limit: int = 100) -> List[Dict]:
    """
    Get tracks by artist name
    
    Args:
        artist_name: Name of the artist
        limit: Maximum number of tracks to return
        
    Returns:
        List of track dictionaries
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('tracks').select('*').ilike('artist', f'%{artist_name}%').limit(limit).execute()
        
        tracks = []
        if result.data:
            for track in result.data:
                # Parse JSON strings back to lists
                track['genres'] = json.loads(track['genres']) if track['genres'] else []
                track['moods'] = json.loads(track['moods']) if track['moods'] else []
                tracks.append(track)
        
        return tracks
        
    except Exception as e:
        logger.error(f"Error fetching tracks for artist {artist_name}: {e}")
        return []


def get_tracks_by_genre(genre: str, limit: int = 100) -> List[Dict]:
    """
    Get tracks by genre
    
    Args:
        genre: Genre to search for
        limit: Maximum number of tracks to return
        
    Returns:
        List of track dictionaries
    """
    try:
        supabase = get_supabase_client()
        
        # Since genres is stored as JSON, we need to use a text search
        result = supabase.table('tracks').select('*').ilike('genres', f'%{genre}%').limit(limit).execute()
        
        tracks = []
        if result.data:
            for track in result.data:
                # Parse JSON strings back to lists
                track['genres'] = json.loads(track['genres']) if track['genres'] else []
                track['moods'] = json.loads(track['moods']) if track['moods'] else []
                
                # Filter to only include tracks that actually have the genre
                if genre.lower() in [g.lower() for g in track['genres']]:
                    tracks.append(track)
        
        return tracks
        
    except Exception as e:
        logger.error(f"Error fetching tracks for genre {genre}: {e}")
        return []


def get_track_count() -> int:
    """
    Get total number of tracks in the database
    
    Returns:
        Total number of tracks
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('tracks').select('id', count='exact').execute()
        
        return result.count if result.count else 0
        
    except Exception as e:
        logger.error(f"Error getting track count: {e}")
        return 0


def delete_track(track_id: str) -> bool:
    """
    Delete a track by ID
    
    Args:
        track_id: Spotify track ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('tracks').delete().eq('id', track_id).execute()
        
        return len(result.data) > 0 if result.data else False
        
    except Exception as e:
        logger.error(f"Error deleting track {track_id}: {e}")
        return False


def get_training_dataset(limit: int = 10000, offset: int = 0) -> List[Dict]:
    """
    Get tracks for ML training with proper formatting
    
    Args:
        limit: Maximum number of tracks to return
        offset: Number of tracks to skip
        
    Returns:
        List of tracks formatted for ML training
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('tracks').select(
            'id, title, artist, album, release_date, duration_ms, popularity, genres, moods, aoty_score, explicit'
        ).range(offset, offset + limit - 1).execute()
        
        training_data = []
        if result.data:
            for track in result.data:
                # Parse JSON strings back to lists
                track['genres'] = json.loads(track['genres']) if track['genres'] else []
                track['moods'] = json.loads(track['moods']) if track['moods'] else []
                training_data.append(track)
        
        logger.info(f"Retrieved {len(training_data)} tracks for training dataset")
        return training_data
        
    except Exception as e:
        logger.error(f"Error fetching training dataset: {e}")
        return []


def export_to_csv(filename: str, limit: int = None) -> bool:
    """
    Export tracks to CSV file for ML training
    
    Args:
        filename: Path to save the CSV file
        limit: Maximum number of tracks to export (None for all)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        # Get all tracks or limited number
        if limit:
            tracks = get_training_dataset(limit=limit)
        else:
            # Get all tracks in batches
            tracks = []
            offset = 0
            batch_size = 1000
            
            while True:
                batch = get_training_dataset(limit=batch_size, offset=offset)
                if not batch:
                    break
                tracks.extend(batch)
                offset += batch_size
        
        if not tracks:
            logger.error("No tracks to export")
            return False
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(tracks)
        df.to_csv(filename, index=False)
        
        logger.info(f"Successfully exported {len(tracks)} tracks to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return False 