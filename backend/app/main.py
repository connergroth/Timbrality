"""
FastAPI Application for Timbre Data Ingestion Service
Provides API endpoints for pipeline management and health monitoring
"""
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.config import settings
from app.db import db_client
from app.models import IngestionRequest, IngestionResult
from app.pipelines import PipelineOrchestrator

# Import new route modules
from routes.ingest_routes import router as ingest_router
from routes.recommendations_routes import router as recommendations_router
from routes.collaborative_filtering_routes import router as collaborative_filtering_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Timbre Music Recommendation API",
    description="ML-powered music recommendation engine with ingestion pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include route modules (aligns with architecture.md Section 6 API Surface)
app.include_router(ingest_router)
app.include_router(recommendations_router)
app.include_router(collaborative_filtering_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline state
current_pipeline: Optional[PipelineOrchestrator] = None
pipeline_status = {"running": False, "last_run": None, "last_result": None}


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    database_connected: bool
    pipeline_status: dict
    version: str = "1.0.0"


class PipelineStatusResponse(BaseModel):
    """Pipeline status response model"""
    running: bool
    last_run: Optional[datetime]
    last_result: Optional[IngestionResult]


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic service info"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        database_connected=await db_client.health_check(),
        pipeline_status=pipeline_status
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns:
        HealthResponse with service status and connectivity
    """
    db_connected = await db_client.health_check()
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.now(),
        database_connected=db_connected,
        pipeline_status=pipeline_status
    )


@app.get("/stats")
async def get_stats():
    """
    Get database and pipeline statistics
    
    Returns:
        Dictionary with various statistics
    """
    try:
        # Get database counts
        counts = await db_client.validate_counts()
        
        # Get coverage statistics
        coverage = await db_client.get_coverage_stats()
        
        return {
            "database_counts": counts,
            "coverage_stats": coverage,
            "pipeline_status": pipeline_status,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/start", response_model=dict)
async def start_pipeline(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start the ingestion pipeline in the background
    
    Args:
        request: Pipeline configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Dictionary with pipeline start confirmation
    """
    global pipeline_status
    
    if pipeline_status["running"]:
        raise HTTPException(
            status_code=409, 
            detail="Pipeline is already running"
        )
    
    # Start pipeline in background
    background_tasks.add_task(
        run_pipeline_background,
        request.max_tracks,
        request.skip_aoty,
        request.dry_run
    )
    
    pipeline_status["running"] = True
    pipeline_status["last_run"] = datetime.now()
    
    return {
        "message": "Pipeline started",
        "timestamp": datetime.now(),
        "config": request.dict()
    }


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """
    Get current pipeline status
    
    Returns:
        PipelineStatusResponse with current status
    """
    return PipelineStatusResponse(
        running=pipeline_status["running"],
        last_run=pipeline_status["last_run"],
        last_result=pipeline_status["last_result"]
    )


@app.post("/pipeline/stop")
async def stop_pipeline():
    """
    Stop the currently running pipeline (if possible)
    
    Returns:
        Dictionary with stop confirmation
    """
    global pipeline_status
    
    if not pipeline_status["running"]:
        raise HTTPException(
            status_code=400,
            detail="No pipeline is currently running"
        )
    
    # Note: Actual stopping would require more sophisticated state management
    # For now, just mark as not running
    pipeline_status["running"] = False
    
    return {
        "message": "Pipeline stop requested",
        "timestamp": datetime.now()
    }


async def run_pipeline_background(
    max_tracks: int,
    skip_aoty: bool,
    dry_run: bool
):
    """
    Background task to run the pipeline
    
    Args:
        max_tracks: Maximum tracks to process
        skip_aoty: Skip AOTY scraping
        dry_run: Don't write to database
    """
    global current_pipeline, pipeline_status
    
    try:
        logger.info(f"Starting background pipeline: max_tracks={max_tracks}, skip_aoty={skip_aoty}, dry_run={dry_run}")
        
        # Create and run pipeline
        current_pipeline = PipelineOrchestrator()
        result = await current_pipeline.run_complete_pipeline(
            max_tracks=max_tracks,
            skip_aoty=skip_aoty,
            dry_run=dry_run
        )
        
        # Update status
        pipeline_status["running"] = False
        pipeline_status["last_result"] = result
        
        logger.info(f"Background pipeline completed: success={result.success}")
        
    except Exception as e:
        logger.error(f"Background pipeline failed: {e}", exc_info=True)
        
        # Update status with error
        pipeline_status["running"] = False
        pipeline_status["last_result"] = IngestionResult(
            success=False,
            total_processed=0,
            successful_inserts=0,
            failed_inserts=0,
            processing_time_seconds=0,
            errors=[str(e)]
        )
    
    finally:
        current_pipeline = None


# Additional utility endpoints

@app.get("/database/counts")
async def get_database_counts():
    """Get current database record counts"""
    try:
        return await db_client.validate_counts()
    except Exception as e:
        logger.error(f"Failed to get database counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/database/coverage")
async def get_coverage_stats():
    """Get enrichment coverage statistics"""
    try:
        return await db_client.get_coverage_stats()
    except Exception as e:
        logger.error(f"Failed to get coverage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive data)"""
    return {
        "max_songs": settings.max_songs,
        "scrape_concurrency": settings.scrape_concurrency,
        "batch_size": settings.batch_size,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "spotify_rate_limit": settings.spotify_rate_limit,
        "lastfm_rate_limit": settings.lastfm_rate_limit,
        "aoty_rate_limit": settings.aoty_rate_limit,
    }


# Error handlers

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Timbre Data Ingestion Service starting up")
    
    # Test database connection
    db_connected = await db_client.health_check()
    if db_connected:
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Timbre Data Ingestion Service shutting down")
    
    # Clean up any running pipelines
    global pipeline_status
    if pipeline_status["running"]:
        logger.info("Stopping running pipeline due to shutdown")
        pipeline_status["running"] = False


if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )