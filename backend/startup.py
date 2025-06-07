"""
Tensoe Backend Startup Script

Initializes and connects all backend components:
- Database and models
- ML service and ingestion pipeline
- API server with all routes
- Configuration validation
"""

import logging
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config.settings import get_settings
from models.database import engine, Base
from models.ingestion_models import EnhancedTrack, EnhancedAlbum, EnhancedArtist
from services.ml_service import ml_service
from routes.ml_routes import ml_router
from routes.album_routes import router as album_router
from routes.user_routes import router as user_router  
from routes.metrics_routes import router as metrics_router

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    settings = get_settings()
    
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format=settings.logging.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.logging.log_file) if settings.logging.log_file else None
        ]
    )
    
    # Remove None handlers
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if h is not None]
    
    return logging.getLogger(__name__)


def initialize_database():
    """Initialize database tables"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
        
        # Log table information
        tables = Base.metadata.tables.keys()
        logger.info(f"Available tables: {', '.join(tables)}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def validate_system():
    """Validate system configuration and components"""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        # Validate settings
        settings.validate_required_settings()
        logger.info("✓ Configuration validation passed")
        
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✓ Database connection successful")
        
        # Test ML service
        stats = ml_service.get_ingestion_stats()
        logger.info(f"✓ ML service operational - {stats.total_tracks} tracks, {stats.total_albums} albums")
        
        # Log environment info
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.api.debug}")
        logger.info(f"Database: {settings.database.database_url.split('@')[-1] if '@' in settings.database.database_url else 'Local'}")
        
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info("🚀 Starting Tensoe Backend...")
    
    # Initialize components
    initialize_database()
    validate_system()
    
    logger.info("✅ Tensoe Backend startup complete!")
    logger.info("=" * 50)
    logger.info("Available endpoints:")
    logger.info("  📊 ML & Ingestion: /ml/*")
    logger.info("  🎵 Albums: /album/*") 
    logger.info("  👤 Users: /user/*")
    logger.info("  📈 Metrics: /metrics/*")
    logger.info("  📚 API Docs: /docs")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Tensoe Backend...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        **settings.get_api_info(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    from fastapi.middleware.cors import CORSMiddleware
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(ml_router, tags=["Machine Learning"])
    app.include_router(album_router, prefix="/album", tags=["Albums"])
    app.include_router(user_router, prefix="/user", tags=["Users"])
    app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
    
    # Root endpoint
    @app.get("/", summary="API Health Check", tags=["Health"])
    async def root():
        """Complete system health check"""
        try:
            # Get system stats
            ml_stats = ml_service.get_ingestion_stats()
            
            return {
                "message": "Tensoe Backend API - Complete ML Music Platform",
                "status": "healthy",
                "version": "1.0.0",
                "environment": settings.environment,
                "components": {
                    "database": "connected",
                    "ml_service": "operational",
                    "ingestion": "ready"
                },
                "data_summary": {
                    "total_tracks": ml_stats.total_tracks,
                    "total_albums": ml_stats.total_albums,
                    "total_artists": ml_stats.total_artists
                },
                "endpoints": {
                    "ml_ingestion": "/ml/ingest/album",
                    "training_data": "/ml/training-data",
                    "analytics": "/ml/analytics",
                    "documentation": "/docs"
                }
            }
        except Exception as e:
            return {
                "message": "Tensoe Backend API",
                "status": "degraded",
                "error": str(e),
                "version": "1.0.0"
            }
    
    # System info endpoint
    @app.get("/system", summary="System Information", tags=["Health"])
    async def system_info():
        """Detailed system information"""
        try:
            stats = ml_service.get_ingestion_stats()
            
            return {
                "system": "Tensoe Backend",
                "version": "1.0.0",
                "environment": settings.environment,
                "configuration": {
                    "database_pool_size": settings.database.pool_size,
                    "ingestion_batch_size": settings.ingestion.batch_size,
                    "ml_training_limit": settings.ml.training_data_limit,
                    "cache_ttl": settings.cache.default_ttl
                },
                "ingestion_stats": stats.dict(),
                "features": [
                    "Spotify API Integration",
                    "Last.fm Mood/Genre Enrichment", 
                    "AOTY Score Integration",
                    "ML Training Data Pipeline",
                    "Feature Matrix Generation",
                    "Batch Processing",
                    "Real-time Analytics"
                ]
            }
        except Exception as e:
            return {"error": f"System info unavailable: {e}"}
    
    return app


def run_server():
    """Run the server with optimal configuration"""
    logger = setup_logging()
    settings = get_settings()
    
    logger.info("🎵 Initializing Tensoe Backend...")
    
    # Create app
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.is_development,
        workers=1 if settings.is_development else 4
    )


if __name__ == "__main__":
    run_server() 