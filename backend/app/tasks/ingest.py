"""
Database Ingestion Module
Bulk upsert operations using Polars DataFrames and Supabase
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import polars as pl
from app.config import settings
from app.db import db_client
from app.models import (
    SongCore, SpotifyAttrs, AotyAttrs, LastfmStats,
    DBSong, DBSpotifyAttrs, DBAotyAttrs, DBLastfmStats
)

logger = logging.getLogger(__name__)


def validate_track_data(track_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean track data before database insertion
    
    Args:
        track_record: Raw track record dictionary
        
    Returns:
        Cleaned track record dictionary
    """
    cleaned_record = {}
    
    # Required fields
    if not track_record.get('title') or not track_record.get('artist'):
        raise ValueError(f"Missing required fields: title={track_record.get('title')}, artist={track_record.get('artist')}")
    
    # Clean and validate each field
    cleaned_record['title'] = str(track_record['title']).strip()[:255] if track_record.get('title') else None
    cleaned_record['artist'] = str(track_record['artist']).strip()[:255] if track_record.get('artist') else None
    cleaned_record['canonical_id'] = str(track_record['canonical_id']).strip() if track_record.get('canonical_id') else None
    
    # Optional string fields
    for field in ['album', 'isrc', 'spotify_id', 'spotify_url']:
        if track_record.get(field):
            cleaned_record[field] = str(track_record[field]).strip()[:255]
    
    # UUID fields (MusicBrainz)
    for field in ['mb_recording_id', 'mb_release_id']:
        if track_record.get(field):
            cleaned_record[field] = str(track_record[field]).strip()
    
    # Optional integer fields  
    for field in ['popularity', 'duration_ms', 'track_number']:
        if track_record.get(field) is not None:
            try:
                cleaned_record[field] = int(track_record[field])
            except (ValueError, TypeError):
                logger.warning(f"Invalid {field} value: {track_record.get(field)}")
    
    # Optional float fields
    for field in ['aoty_score', 'pred_energy', 'pred_valence', 'mood_confidence']:
        if track_record.get(field) is not None:
            try:
                cleaned_record[field] = float(track_record[field])
                # Validate range for prediction fields
                if field in ['pred_energy', 'pred_valence', 'mood_confidence']:
                    if not (0.0 <= cleaned_record[field] <= 1.0):
                        logger.warning(f"Invalid {field} value out of range [0,1]: {cleaned_record[field]}")
                        cleaned_record[field] = None
            except (ValueError, TypeError):
                logger.warning(f"Invalid {field} value: {track_record.get(field)}")
    
    # Boolean fields
    if track_record.get('explicit') is not None:
        cleaned_record['explicit'] = bool(track_record['explicit'])
    
    # Date fields
    if track_record.get('release_date'):
        cleaned_record['release_date'] = normalize_release_date(track_record['release_date'])
    
    # Array fields (ensure they're lists)
    for field in ['genres', 'moods']:
        if track_record.get(field):
            if isinstance(track_record[field], list):
                cleaned_record[field] = [str(item).strip() for item in track_record[field] if item]
            else:
                cleaned_record[field] = []
        else:
            cleaned_record[field] = []
    
    # Data source mask (smallint)
    if track_record.get('data_source_mask') is not None:
        try:
            cleaned_record['data_source_mask'] = int(track_record['data_source_mask'])
        except (ValueError, TypeError):
            logger.warning(f"Invalid data_source_mask value: {track_record.get('data_source_mask')}")
            cleaned_record['data_source_mask'] = 0
    
    # Vector field (will be None from ingestion, populated by ML service later)
    if track_record.get('track_vector') is not None:
        cleaned_record['track_vector'] = track_record['track_vector']
    
    return cleaned_record


def normalize_release_date(release_date: Optional[str]) -> Optional[str]:
    """
    Normalize Spotify release date to PostgreSQL date format.
    
    Spotify can return dates in formats:
    - "2017" (year only)
    - "2017-05" (year-month)  
    - "2017-05-12" (full date)
    
    PostgreSQL expects YYYY-MM-DD format.
    
    Args:
        release_date: Raw release date string from Spotify
        
    Returns:
        Normalized date string in YYYY-MM-DD format or None
    """
    if not release_date:
        return None
    
    release_date = release_date.strip()
    
    # Already in full format
    if len(release_date) == 10 and release_date.count('-') == 2:
        return release_date
    
    # Year-Month format: pad with -01 for day
    if len(release_date) == 7 and release_date.count('-') == 1:
        return f"{release_date}-01"
    
    # Year only: pad with -01-01
    if len(release_date) == 4 and release_date.isdigit():
        return f"{release_date}-01-01"
    
    # Invalid format
    logger.warning(f"Invalid release date format: {release_date}")
    return None


class IngestionProcessor:
    """Handles bulk ingestion of enriched song data into unified tracks table"""
    
    def __init__(self):
        self.chunk_size = settings.db_batch_size
        self.retry_attempts = settings.db_retry_attempts
        self.retry_delay = settings.db_retry_delay
    
    async def bulk_upsert_tracks(self, tracks_data: List[Dict[str, Any]]) -> int:
        """
        Bulk upsert tracks to unified tracks table
        
        Args:
            tracks_data: List of track dictionaries
            
        Returns:
            Number of successfully upserted tracks
        """
        logger.info(f"Bulk upserting {len(tracks_data)} tracks")
        
        df = pl.DataFrame(tracks_data)
        return await self._chunked_upsert(df, "tracks", db_client.upsert_tracks)
    
    
    async def _chunked_upsert(
        self, 
        df: pl.DataFrame, 
        table_name: str, 
        upsert_func
    ) -> int:
        """
        Perform chunked upsert with retry logic
        
        Args:
            df: Polars DataFrame to upsert
            table_name: Name of target table
            upsert_func: Function to perform upsert
            
        Returns:
            Number of successfully upserted records
        """
        total_records = df.height
        success_count = 0
        
        # Process in chunks
        for i in range(0, total_records, self.chunk_size):
            chunk_end = min(i + self.chunk_size, total_records)
            chunk_df = df.slice(i, chunk_end - i)
            
            # Convert to list of dicts for Supabase
            chunk_data = chunk_df.to_dicts()
            
            # Retry logic
            for attempt in range(self.retry_attempts):
                try:
                    result_count = await upsert_func(chunk_data, len(chunk_data))
                    success_count += result_count
                    logger.debug(f"Upserted chunk {i//self.chunk_size + 1} to {table_name}: {result_count} records")
                    break
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {table_name} chunk {i//self.chunk_size + 1}: {e}")
                    
                    if attempt < self.retry_attempts - 1:
                        # Exponential backoff
                        wait_time = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upsert chunk after {self.retry_attempts} attempts")
        
        logger.info(f"Successfully upserted {success_count}/{total_records} records to {table_name}")
        return success_count
    
    async def validate_counts(self) -> Dict[str, int]:
        """
        Validate record counts for tracks table
        
        Returns:
            Dictionary with table counts
        """
        logger.info("Validating record counts for tracks table")
        return await db_client.validate_counts()
    
    async def get_coverage_stats(self) -> Dict[str, Any]:
        """
        Get enrichment coverage statistics
        
        Returns:
            Dictionary with coverage percentages
        """
        return await db_client.get_coverage_stats()


async def prepare_song_ids_mapping(songs_with_canonical_ids: List[Tuple[SongCore, str]]) -> Dict[str, str]:
    """
    Create mapping from canonical_id to song_id after insertion
    
    Args:
        songs_with_canonical_ids: List of (SongCore, canonical_id) tuples
        
    Returns:
        Dictionary mapping canonical_id to song_id
    """
    mapping = {}
    
    # After songs are inserted, we need to fetch their IDs
    # This is a simplified version - in practice, you might want to batch this
    for song, canonical_id in songs_with_canonical_ids:
        try:
            db_song = await db_client.get_song_by_canonical_id(canonical_id)
            if db_song:
                mapping[canonical_id] = db_song["id"]
        except Exception as e:
            logger.error(f"Failed to get song ID for {canonical_id}: {e}")
    
    return mapping


async def process_enriched_pipeline_data(
    songs_with_ids: List[Tuple[SongCore, str]],  # (song, canonical_id)
    spotify_enrichments: List[Tuple[SongCore, Optional[SpotifyAttrs]]],
    aoty_enrichments: List[Tuple[SongCore, Optional[AotyAttrs]]]
) -> Dict[str, int]:
    """
    Process complete pipeline data and insert into unified tracks table
    
    Args:
        songs_with_ids: Songs with canonical IDs
        spotify_enrichments: Spotify enrichment results  
        aoty_enrichments: AOTY enrichment results
        
    Returns:
        Dictionary with insertion statistics
    """
    logger.info("Processing enriched pipeline data for unified tracks table...")
    
    # Create mappings for easy lookup
    spotify_map = {song.canonical_id if hasattr(song, 'canonical_id') else f"{song.artist}::{song.title}": attrs 
                   for song, attrs in spotify_enrichments if attrs is not None}
    aoty_map = {song.canonical_id if hasattr(song, 'canonical_id') else f"{song.artist}::{song.title}": attrs 
                for song, attrs in aoty_enrichments if attrs is not None}
    
    # Build unified track records
    tracks_to_insert = []
    for song, canonical_id in songs_with_ids:
        # Get enrichment data
        spotify_attrs = spotify_map.get(canonical_id)
        aoty_attrs = aoty_map.get(canonical_id)
        
        # Extract Spotify ID from enrichment results
        spotify_id = None
        if spotify_attrs and hasattr(spotify_attrs, 'track_id'):
            spotify_id = spotify_attrs.track_id
        elif spotify_attrs and hasattr(spotify_attrs, 'id'):
            spotify_id = spotify_attrs.id
        
        # Build unified track record matching tracks.sql schema
        track_record = {
            "title": song.title,
            "artist": song.artist,
            "canonical_id": canonical_id,
            "isrc": getattr(song, 'isrc', None),
            "spotify_id": spotify_id,
            "mb_recording_id": getattr(song, 'mb_recording_id', None),
            "mb_release_id": getattr(song, 'mb_release_id', None),
        }
        
        # Add Spotify data if available
        if spotify_attrs:
            track_record.update({
                "album": spotify_attrs.album_name,
                "popularity": spotify_attrs.popularity,
                "duration_ms": spotify_attrs.duration_ms,
                "release_date": normalize_release_date(spotify_attrs.release_date),
                "explicit": spotify_attrs.explicit,
                "track_number": spotify_attrs.track_number,
                "spotify_url": f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None,
            })
        
        # Add AOTY data if available
        if aoty_attrs:
            track_record.update({
                "aoty_score": aoty_attrs.user_score,
                "genres": aoty_attrs.genres or [],
                "moods": aoty_attrs.tags or [],
            })
        
        # Set data source mask based on available data
        # Bitmask: 1=LastFM, 2=Spotify, 4=AOTY, 8=MusicBrainz
        data_source_mask = 0
        data_source_mask |= 1  # Always have Last.fm as source (songs come from there)
        if spotify_attrs:
            data_source_mask |= 2
        if aoty_attrs:
            data_source_mask |= 4
        if track_record.get('mb_recording_id') or track_record.get('mb_release_id'):
            data_source_mask |= 8
        
        # Initialize ML prediction fields as None (to be populated by external ML service)
        track_record.update({
            "pred_energy": None,
            "pred_valence": None, 
            "mood_confidence": None,
            "track_vector": None,
            "data_source_mask": data_source_mask
        })
        
        # Add Last.fm data from song object
        if hasattr(song, 'playcount') and song.playcount and song.playcount > 0:
            # For now, we don't have a playcount field in tracks table
            # This could be added to the schema if needed
            pass
        
        # Validate and clean the track record
        try:
            validated_record = validate_track_data(track_record)
            tracks_to_insert.append(validated_record)
        except ValueError as e:
            logger.error(f"Invalid track data for {song.artist} - {song.title}: {e}")
            continue
    
    # Check for duplicate canonical IDs before insertion
    canonical_ids = [track.get('canonical_id') for track in tracks_to_insert if track.get('canonical_id')]
    unique_ids = set(canonical_ids)
    if len(canonical_ids) != len(unique_ids):
        logger.warning(f"Found {len(canonical_ids) - len(unique_ids)} duplicate canonical IDs")
        
        # Remove duplicates (keep first occurrence)
        seen_ids = set()
        deduplicated_tracks = []
        for track in tracks_to_insert:
            canonical_id = track.get('canonical_id')
            if canonical_id not in seen_ids:
                seen_ids.add(canonical_id)
                deduplicated_tracks.append(track)
            else:
                logger.debug(f"Removing duplicate track: {track.get('artist')} - {track.get('title')}")
        
        tracks_to_insert = deduplicated_tracks
        logger.info(f"Deduplicated to {len(tracks_to_insert)} unique tracks")
    
    # Insert tracks using database client
    logger.info(f"Inserting {len(tracks_to_insert)} tracks into unified tracks table...")
    tracks_inserted = await db_client.upsert_tracks(tracks_to_insert)
    
    # Validate final counts
    logger.info("Validating final counts...")
    final_counts = await db_client.validate_counts()
    
    stats = {
        "tracks_inserted": tracks_inserted,
        "final_counts": final_counts
    }
    
    logger.info(f"Ingestion complete: {stats}")
    return stats


# Convenience functions

async def bulk_upsert_tracks(tracks_data: List[Dict[str, Any]]) -> int:
    """Convenience function for bulk tracks upsert"""
    processor = IngestionProcessor()
    return await processor.bulk_upsert_tracks(tracks_data)


async def validate_counts() -> Dict[str, int]:
    """Convenience function for count validation"""
    processor = IngestionProcessor()
    return await processor.validate_counts()