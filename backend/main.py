from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

from routes.album_routes import router as album_router
from routes.user_routes import router as user_router
from routes.metrics_routes import router as metrics_router
from routes.ml_routes import ml_router
from utils.metrics import metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize any resources on startup
    yield
    # Clean up resources on shutdown


# Initialize the FastAPI app
app = FastAPI(
    title="Tensoe Backend API",
    description="""
    A comprehensive music discovery and recommendation platform backend.
    
    This API provides access to album information, user profiles, music recommendations,
    and more with advanced caching, rate limiting, and performance optimization.
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


@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Tensoe Backend API",
        "status": "healthy",
        "version": "1.0.0"
    }
