"""
Main Pipeline Orchestrator
Coordinates the complete end-to-end ingestion process
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from app.config import settings
from app.models import PipelineStats, IngestionResult
from app.tasks.lastfm import pull_top_tracks, pull_recent_tracks
from app.tasks.spotify import enrich_tracks
from app.tasks.aoty import scrape_many
from app.tasks.unify import dedupe_songs, make_canonical_id_enhanced, fill_with_recent_tracks, validate_canonical_ids
from app.tasks.ingest import process_enriched_pipeline_data

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Main pipeline orchestrator for the complete ingestion process"""
    
    def __init__(self):
        self.stats = PipelineStats()
        self.start_time = 0
        self.errors = []
    
    async def run_complete_pipeline(
        self,
        max_songs: Optional[int] = None,
        skip_aoty: bool = False,
        dry_run: bool = False
    ) -> IngestionResult:
        """
        Run the complete ingestion pipeline
        
        Args:
            max_songs: Maximum number of songs to process (default from settings)
            skip_aoty: Skip AOTY scraping (faster testing)
            dry_run: Don't write to database, just process data
            
        Returns:
            IngestionResult with statistics and status
        """
        if max_songs is None:
            max_songs = settings.max_songs
        
        self.start_time = time.time()
        logger.info(f"Starting complete ingestion pipeline (max_songs={max_songs}, skip_aoty={skip_aoty}, dry_run={dry_run})")
        
        try:
            # Step 1: Pull top tracks from Last.fm
            logger.info("=== STEP 1: Pulling top tracks from Last.fm ===")
            raw_tracks = await self._pull_lastfm_tracks(max_songs * 3)  # Get 3x more for deduplication
            self.stats.lastfm_tracks_pulled = len(raw_tracks)
            
            # Step 2: Deduplicate to target count
            logger.info("=== STEP 2: Deduplicating tracks ===")
            deduped_tracks = await self._deduplicate_tracks(raw_tracks, max_songs)
            self.stats.songs_deduplicated = len(raw_tracks) - len(deduped_tracks)
            self.stats.final_songs_count = len(deduped_tracks)
            
            # Step 3: Enrich with Spotify data
            logger.info("=== STEP 3: Enriching with Spotify data ===")
            spotify_enrichments = await self._enrich_spotify(deduped_tracks)
            self.stats.spotify_enrichments = sum(1 for _, attrs in spotify_enrichments if attrs is not None)
            self.stats.spotify_coverage_pct = (self.stats.spotify_enrichments / len(deduped_tracks)) * 100
            
            # Step 4: Scrape AOTY data (optional)
            aoty_enrichments = []
            if not skip_aoty:
                logger.info("=== STEP 4: Scraping AOTY data ===")
                aoty_enrichments = await self._scrape_aoty(deduped_tracks)
                self.stats.aoty_enrichments = sum(1 for _, attrs in aoty_enrichments if attrs is not None)
                self.stats.aoty_coverage_pct = (self.stats.aoty_enrichments / len(deduped_tracks)) * 100
            else:
                logger.info("=== STEP 4: Skipping AOTY scraping ===")
                # Create empty AOTY results
                aoty_enrichments = [(song, None) for song in deduped_tracks]
            
            # Step 5: Prepare data for database insertion
            logger.info("=== STEP 5: Preparing canonical IDs ===")
            songs_with_canonical_ids = []
            spotify_lookup = {song.artist + "::" + song.title: attrs for song, attrs in spotify_enrichments}
            
            for song in deduped_tracks:
                lookup_key = song.artist + "::" + song.title
                spotify_attrs = spotify_lookup.get(lookup_key)
                canonical_id = make_canonical_id_enhanced(song, spotify_attrs)
                song.canonical_id = canonical_id  # Add to song object
                songs_with_canonical_ids.append((song, canonical_id))
            
            # Validate canonical IDs
            validation_stats = validate_canonical_ids(deduped_tracks)
            logger.info(f"Canonical ID validation: {validation_stats}")
            
            # Step 6: Database insertion (unless dry run)
            insertion_stats = {}
            if not dry_run:
                logger.info("=== STEP 6: Inserting into database ===")
                insertion_stats = await process_enriched_pipeline_data(
                    songs_with_canonical_ids,
                    spotify_enrichments,
                    aoty_enrichments
                )
            else:
                logger.info("=== STEP 6: Skipping database insertion (dry run) ===")
                insertion_stats = {
                    "songs_inserted": len(songs_with_canonical_ids),
                    "spotify_inserted": self.stats.spotify_enrichments,
                    "lastfm_inserted": len([s for s in deduped_tracks if s.playcount]),
                    "aoty_inserted": self.stats.aoty_enrichments,
                    "final_counts": {}
                }
            
            # Calculate final statistics
            self.stats.processing_time_seconds = time.time() - self.start_time
            
            # Create result
            result = IngestionResult(
                success=True,
                total_processed=len(deduped_tracks),
                successful_inserts=insertion_stats.get("songs_inserted", 0),
                failed_inserts=len(deduped_tracks) - insertion_stats.get("songs_inserted", 0),
                processing_time_seconds=self.stats.processing_time_seconds,
                errors=self.errors,
                coverage_stats={
                    "spotify_coverage_pct": self.stats.spotify_coverage_pct,
                    "aoty_coverage_pct": self.stats.aoty_coverage_pct,
                    "lastfm_tracks_pulled": self.stats.lastfm_tracks_pulled,
                    "final_songs_count": self.stats.final_songs_count,
                    "songs_deduplicated": self.stats.songs_deduplicated
                }
            )
            
            # Log summary
            await self._log_summary(result, insertion_stats, validation_stats)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.errors.append(str(e))
            
            return IngestionResult(
                success=False,
                total_processed=0,
                successful_inserts=0,
                failed_inserts=0,
                processing_time_seconds=time.time() - self.start_time,
                errors=self.errors
            )
    
    async def _pull_lastfm_tracks(self, max_tracks: int):
        """Pull tracks from Last.fm with error handling"""
        try:
            tracks = await pull_top_tracks(max_tracks)
            logger.info(f"Successfully pulled {len(tracks)} tracks from Last.fm")
            return tracks
        except Exception as e:
            logger.error(f"Failed to pull Last.fm tracks: {e}")
            self.errors.append(f"Last.fm error: {e}")
            raise
    
    async def _deduplicate_tracks(self, tracks, max_songs: int):
        """Deduplicate tracks and fill with recent if needed"""
        try:
            # Initial deduplication
            deduped = dedupe_songs(tracks, max_songs)
            
            # Fill with recent tracks if needed
            if len(deduped) < max_songs:
                canonical_ids = {getattr(song, 'canonical_id', '') for song in deduped}
                deduped = await fill_with_recent_tracks(deduped, max_songs, canonical_ids)
            
            logger.info(f"Deduplication complete: {len(deduped)} unique tracks")
            return deduped
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            self.errors.append(f"Deduplication error: {e}")
            return tracks[:max_songs]  # Fallback to simple truncation
    
    async def _enrich_spotify(self, tracks):
        """Enrich tracks with Spotify data"""
        try:
            enrichments = await enrich_tracks(tracks)
            successful = sum(1 for _, attrs in enrichments if attrs is not None)
            logger.info(f"Spotify enrichment: {successful}/{len(tracks)} tracks enhanced")
            return enrichments
        except Exception as e:
            logger.error(f"Spotify enrichment failed: {e}")
            self.errors.append(f"Spotify error: {e}")
            return [(track, None) for track in tracks]  # Return empty enrichments
    
    async def _scrape_aoty(self, tracks):
        """Scrape AOTY data for tracks"""
        try:
            enrichments = await scrape_many(tracks, settings.scrape_concurrency)
            successful = sum(1 for _, attrs in enrichments if attrs is not None)
            logger.info(f"AOTY scraping: {successful}/{len(tracks)} tracks enhanced")
            return enrichments
        except Exception as e:
            logger.error(f"AOTY scraping failed: {e}")
            self.errors.append(f"AOTY error: {e}")
            return [(track, None) for track in tracks]  # Return empty enrichments
    
    async def _log_summary(self, result: IngestionResult, insertion_stats: Dict, validation_stats: Dict):
        """Log comprehensive pipeline summary"""
        
        summary = f"""
        
=== PIPELINE COMPLETION SUMMARY ===
Status: {'SUCCESS' if result.success else 'FAILED'}
Processing Time: {result.processing_time_seconds:.2f} seconds

LAST.FM DATA:
- Tracks Pulled: {self.stats.lastfm_tracks_pulled}
- Tracks Deduplicated: {self.stats.songs_deduplicated}
- Final Track Count: {self.stats.final_songs_count}

ENRICHMENT COVERAGE:
- Spotify: {self.stats.spotify_enrichments}/{self.stats.final_songs_count} ({self.stats.spotify_coverage_pct:.1f}%)
- AOTY: {self.stats.aoty_enrichments}/{self.stats.final_songs_count} ({self.stats.aoty_coverage_pct:.1f}%)

CANONICAL IDS:
- ISRC: {validation_stats.get('isrc_count', 0)}
- MBID: {validation_stats.get('mbid_count', 0)}  
- Spotify: {validation_stats.get('spotify_count', 0)}
- Hash: {validation_stats.get('hash_count', 0)}

DATABASE INSERTION:
- Songs: {insertion_stats.get('songs_inserted', 0)}
- Spotify Attrs: {insertion_stats.get('spotify_inserted', 0)}
- Last.fm Stats: {insertion_stats.get('lastfm_inserted', 0)}
- AOTY Attrs: {insertion_stats.get('aoty_inserted', 0)}

ERRORS: {len(result.errors)}
{chr(10).join(f"- {error}" for error in result.errors) if result.errors else "- None"}

=== END SUMMARY ===
        """
        
        logger.info(summary)
        
        # Save summary to file
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"{settings.logs_dir}/pipeline_summary_{timestamp}.json"
            
            summary_data = {
                "timestamp": timestamp,
                "result": result.dict(),
                "stats": self.stats.dict(),
                "insertion_stats": insertion_stats,
                "validation_stats": validation_stats
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Pipeline summary saved to {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save summary file: {e}")


# Main entry point
async def main(
    max_songs: Optional[int] = None,
    skip_aoty: bool = False,
    dry_run: bool = False
) -> IngestionResult:
    """
    Main pipeline entry point
    
    Args:
        max_songs: Maximum number of songs to process
        skip_aoty: Skip AOTY scraping for faster execution
        dry_run: Process data but don't write to database
        
    Returns:
        IngestionResult with pipeline statistics
    """
    orchestrator = PipelineOrchestrator()
    return await orchestrator.run_complete_pipeline(max_songs, skip_aoty, dry_run)


# Synchronous wrapper for CLI
def run_pipeline_sync(
    max_songs: Optional[int] = None,
    skip_aoty: bool = False,
    dry_run: bool = False
) -> IngestionResult:
    """Synchronous wrapper for CLI usage"""
    return asyncio.run(main(max_songs, skip_aoty, dry_run))