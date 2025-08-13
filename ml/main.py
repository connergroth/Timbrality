"""
Main FastAPI application for Timbral music recommendation system.

This module serves as the entry point for the FastAPI application,
setting up the server, middleware, and routes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import logging

from timbral.api.routes import router
from timbral.config.settings import settings
from timbral import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Timbral Music Recommendation API",
    description="A hybrid NMF + BERT-based music recommendation system",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    """
    logger.info("Starting Timbral music recommendation API...")
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    """
    logger.info("Shutting down Timbral music recommendation API...")


@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        Basic API information
    """
    return {
        "name": "Timbral Music Recommendation API",
        "version": __version__,
        "description": "A hybrid NMF + BERT-based music recommendation system",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/ping")
async def ping():
    """
    Health check endpoint.
    
    Returns:
        Simple ping response
    """
    return {"message": "pong", "status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS if not settings.DEBUG else 1
    ) 