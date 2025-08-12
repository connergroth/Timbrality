"""
Database client and helpers for Supabase PostgreSQL
"""
import logging
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Enhanced Supabase client with helper methods"""
    
    def __init__(self):
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get or create Supabase client"""
        if self._client is None:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key
            )
        return self._client
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            # Simple query to test connection
            result = self.client.from_("tracks").select("count").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_coverage_stats(self) -> Dict[str, Any]:
        """
        Get enrichment coverage statistics
        
        Aligns with architecture.md data source tracking
        """
        try:
            # Get counts by data source using data_source_mask
            result = self.client.from_("tracks").select("data_source_mask").execute()
            
            if not result.data:
                return {
                    "total_tracks": 0,
                    "lastfm_coverage": 0.0,
                    "spotify_coverage": 0.0,
                    "aoty_coverage": 0.0,
                    "musicbrainz_coverage": 0.0
                }
            
            total_tracks = len(result.data)
            lastfm_count = sum(1 for row in result.data if row["data_source_mask"] & 1)
            spotify_count = sum(1 for row in result.data if row["data_source_mask"] & 2)
            aoty_count = sum(1 for row in result.data if row["data_source_mask"] & 4)
            musicbrainz_count = sum(1 for row in result.data if row["data_source_mask"] & 8)
            
            return {
                "total_tracks": total_tracks,
                "lastfm_coverage": (lastfm_count / total_tracks * 100) if total_tracks > 0 else 0.0,
                "spotify_coverage": (spotify_count / total_tracks * 100) if total_tracks > 0 else 0.0,
                "aoty_coverage": (aoty_count / total_tracks * 100) if total_tracks > 0 else 0.0,
                "musicbrainz_coverage": (musicbrainz_count / total_tracks * 100) if total_tracks > 0 else 0.0,
                "source_counts": {
                    "lastfm": lastfm_count,
                    "spotify": spotify_count,
                    "aoty": aoty_count,
                    "musicbrainz": musicbrainz_count
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get coverage stats: {e}")
            return {
                "total_tracks": 0,
                "lastfm_coverage": 0.0,
                "spotify_coverage": 0.0,
                "aoty_coverage": 0.0,
                "musicbrainz_coverage": 0.0,
                "error": str(e)
            }
    
    async def get_total_songs(self) -> int:
        """Get total number of songs in database"""
        try:
            result = self.client.rpc("get_track_count").execute()
            return result.data if result.data else 0
        except Exception as e:
            logger.error(f"Failed to get total songs: {e}")
            return 0
    
    async def song_exists(self, canonical_id: str) -> bool:
        """Check if song exists by canonical ID"""
        try:
            result = self.client.from_("tracks").select("id").eq("canonical_id", canonical_id).limit(1).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to check song existence: {e}")
            return False
    
    async def get_song_by_canonical_id(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """Get song by canonical ID"""
        try:
            result = self.client.from_("tracks").select("*").eq("canonical_id", canonical_id).limit(1).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get song by canonical ID: {e}")
            return None
    
    async def upsert_tracks(self, tracks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Bulk upsert tracks with batching and detailed error handling"""
        success_count = 0
        
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i:i + batch_size]
            try:
                # Log a sample of what we're trying to insert for debugging
                if batch:
                    sample_track = batch[0]
                    logger.debug(f"Sample track data: {sample_track}")
                    
                    # Check for potential duplicate canonical_ids in this batch
                    canonical_ids = [track.get('canonical_id') for track in batch if track.get('canonical_id')]
                    unique_ids = set(canonical_ids)
                    if len(canonical_ids) != len(unique_ids):
                        logger.warning(f"Batch {i//batch_size + 1} contains duplicate canonical_ids: {len(canonical_ids)} vs {len(unique_ids)}")
                
                result = self.client.from_("tracks").upsert(
                    batch,
                    on_conflict="canonical_id",
                    ignore_duplicates=False
                ).execute()
                success_count += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} tracks")
                
            except Exception as e:
                logger.error(f"Failed to upsert tracks batch {i//batch_size + 1}: {e}")
                
                # Log detailed error information
                error_details = {
                    "batch_size": len(batch),
                    "batch_index": i//batch_size + 1,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                if hasattr(e, 'response'):
                    error_details["response_status"] = getattr(e.response, 'status_code', 'unknown')
                    error_details["response_text"] = getattr(e.response, 'text', 'unknown')
                
                logger.error(f"Detailed error info: {error_details}")
                
                # Log problematic track data (first few tracks in batch)
                if batch:
                    for j, track in enumerate(batch[:3]):  # Show first 3 tracks
                        logger.error(f"Track {j+1} in failed batch: {track}")
        
        return success_count
    
    async def get_track_count(self) -> int:
        """Get total number of tracks in database"""
        try:
            result = self.client.from_("tracks").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get track count: {e}")
            return 0
    
    async def validate_counts(self) -> Dict[str, int]:
        """Validate record counts for tracks table"""
        try:
            tracks_result = self.client.from_("tracks").select("id", count="exact").execute()
            
            return {
                "tracks": tracks_result.count or 0,
            }
        except Exception as e:
            logger.error(f"Failed to validate counts: {e}")
            return {"tracks": 0}


# Global database client instance
db_client = DatabaseClient()