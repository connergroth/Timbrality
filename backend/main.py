from fastapi import FastAPI
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from app.routes.album_routes import router as album_router

# Initialize the FastAPI app
app = FastAPI()

# Create and attach the limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add the SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Include the album router
app.include_router(album_router, prefix="/album", tags=["Albums"])
