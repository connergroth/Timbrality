from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import asyncio

# Import config to ensure environment variables are loaded
import config

from routes.album_routes import router as album_router
from routes.user_routes import router as user_router
from routes.metrics_routes import router as metrics_router
from routes.ml_routes import ml_router
from routes.timbral_routes import timbral_router
from routes.scraper_routes import router as scraper_router
from routes.agent_routes import router as agent_router
from routes.playlist_routes import router as playlist_router
from routes.spotify_routes import router as spotify_router
from utils.metrics import metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize any resources on startup
    print("🚀 Starting Timbrality backend server...")
    yield
    # Clean up resources on shutdown
    print("🛑 Shutting down Timbrality backend server...")
    
    # Cancel any remaining tasks
    import asyncio
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    if tasks:
        print(f"Cancelling {len(tasks)} remaining tasks...")
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass  # Ignore cancellation exceptions
    
    print("✅ Server shutdown complete")


# Initialize the FastAPI app
app = FastAPI(
    title="Timbrality Backend API",
    description="""
    A comprehensive music discovery and recommendation platform backend.
    
    This API provides access to album information, user profiles, music recommendations,
    and more with advanced caching, rate limiting, and performance optimization.
    Powered by the Timbral ML engine for intelligent music analysis.
    """,
    version="1.0.0",
    contact={
        "name": "Conner Groth",
        "url": "https://github.com/connergroth",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add the SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limit error handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Handle asyncio cancellation gracefully - moved to route level since CancelledError 
# may not work properly as a global exception handler in all FastAPI versions

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    metrics.record_error()
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "path": request.url.path,
            "method": request.method,
        },
    )


# Include routers
app.include_router(album_router, prefix="/album", tags=["Albums"])
app.include_router(user_router, prefix="/user", tags=["Users"])
app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
app.include_router(ml_router, tags=["Machine Learning"])
app.include_router(timbral_router, tags=["Timbral ML Service"])
app.include_router(scraper_router, prefix="/scraper", tags=["AOTY Scraper"])
app.include_router(agent_router, prefix="/agent", tags=["AI Agent"])
app.include_router(playlist_router, prefix="/playlist", tags=["Playlists"])
app.include_router(spotify_router, tags=["Spotify"])


@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Timbrality Backend API",
        "status": "healthy",
        "version": "1.0.0",
        "ml_engine": "Timbral"
    }
