"""
Ingestion and Data Pipeline API Routes
Aligns with architecture.md Section 6 API Surface
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.tasks.lastfm import pull_top_tracks, pull_recent_tracks
from app.tasks.spotify import enrich_tracks
from app.tasks.aoty import scrape_many
from app.tasks.unify import dedupe_songs
from app.tasks.ingest import process_enriched_pipeline_data
from app.db import db_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


class LastFmIngestRequest(BaseModel):
    """Request model for Last.fm data ingestion"""
    user_token: Optional[str] = None
    username: Optional[str] = None
    limit: int = 1000
    time_period: str = "overall"  # overall, 7day, 1month, 3month, 6month, 12month


class IngestJobStatus(BaseModel):
    """Response model for ingestion job status"""
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    message: str
    tracks_processed: int = 0
    tracks_inserted: int = 0


@router.post("/lastfm", response_model=IngestJobStatus)
async def ingest_lastfm(
    request: LastFmIngestRequest,
    background_tasks: BackgroundTasks
) -> IngestJobStatus:
    """
    Ingest Last.fm data for a user
    
    Architecture alignment: POST /ingest/lastfm
    User token → fetch & schedule scrobble import job
    """
    try:
        # Validate input
        if not request.user_token and not request.username:
            raise HTTPException(
                status_code=400,
                detail="Either user_token or username must be provided"
            )
        
        # For now, create a simple job ID
        job_id = f"lastfm_{request.username or 'token'}_{request.time_period}"
        
        # Queue background task for ingestion
        background_tasks.add_task(
            _run_lastfm_ingestion,
            job_id,
            request.username,
            request.limit,
            request.time_period
        )
        
        return IngestJobStatus(
            job_id=job_id,
            status="queued",
            message="Last.fm ingestion job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to queue Last.fm ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=IngestJobStatus)
async def get_ingestion_status(job_id: str) -> IngestJobStatus:
    """Get status of an ingestion job"""
    # TODO: Implement job status tracking (Redis/DB)
    # For now, return a placeholder
    return IngestJobStatus(
        job_id=job_id,
        status="completed",
        message="Job status tracking not yet implemented"
    )


@router.get("/stats")
async def get_ingestion_stats() -> Dict[str, Any]:
    """Get overall ingestion statistics"""
    try:
        track_count = await db_client.get_track_count()
        coverage_stats = await db_client.get_coverage_stats()
        
        return {
            "total_tracks": track_count,
            "coverage": coverage_stats,
            "last_updated": None  # TODO: Track last ingestion time
        }
        
    except Exception as e:
        logger.error(f"Failed to get ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_lastfm_ingestion(
    job_id: str,
    username: str,
    limit: int,
    time_period: str
):
    """
    Background task to run Last.fm ingestion
    
    Follows architecture.md Section 1 pipeline:
    1.1 Pull Last.fm data
    1.2 Pull Spotify metadata  
    1.3 Scrape AOTY
    1.4 Canonical track mapping
    1.5 Tag aggregation
    1.6 Feature snapshot
    """
    try:
        logger.info(f"Starting Last.fm ingestion job {job_id}")
        
        # Step 1: Pull Last.fm tracks
        logger.info("Step 1: Pulling Last.fm tracks...")
        lastfm_tracks = await pull_top_tracks(
            username=username,
            limit=limit,
            period=time_period
        )
        
        if not lastfm_tracks:
            logger.warning(f"No tracks found for user {username}")
            return
        
        # Step 2: Deduplicate and create canonical IDs
        logger.info("Step 2: Creating canonical track mapping...")
        songs_with_ids = await dedupe_songs(lastfm_tracks)
        
        # Step 3: Enrich with Spotify metadata
        logger.info("Step 3: Enriching with Spotify metadata...")
        spotify_enrichments = await enrich_tracks([song for song, _ in songs_with_ids])
        
        # Step 4: Scrape AOTY data (optional, can be skipped for performance)
        logger.info("Step 4: Scraping AOTY data...")
        aoty_enrichments = await scrape_many([song for song, _ in songs_with_ids])
        
        # Step 5: Process and insert into database
        logger.info("Step 5: Processing and inserting tracks...")
        result = await process_enriched_pipeline_data(
            songs_with_ids,
            spotify_enrichments,
            aoty_enrichments
        )
        
        logger.info(f"Completed Last.fm ingestion job {job_id}: {result}")
        
    except Exception as e:
        logger.error(f"Last.fm ingestion job {job_id} failed: {e}")
        # TODO: Update job status in database/Redis