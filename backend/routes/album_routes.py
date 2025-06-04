from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.services.aoty_service import AOTYService
from app.middleware.aoty_middleware import enrich_album_with_aoty
from app.tasks.aoty_tasks import fetch_and_store_similar_albums

router = APIRouter()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/enrich/{album_id}")
async def enrich_album(album_id: str, db: Session = Depends(get_db)):
    """Enrich an album with AOTY metadata."""
    success = await enrich_album_with_aoty(db, album_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Album not found or enrichment failed")
    
    return {"message": "Album enriched successfully with AOTY data"}

@router.get("/similar/{album_id}")
async def get_similar_albums(
    album_id: str, 
    limit: int = Query(5, gt=0, le=10),
    db: Session = Depends(get_db)
):
    """Get similar albums for a given album."""
    success = await fetch_and_store_similar_albums(db, album_id, limit)
    
    if not success:
        raise HTTPException(status_code=404, detail="Album not found or similar albums fetch failed")
    
    # Get album to return updated data with similar albums
    db_album = db.query(Album).filter(Album.id == album_id).first()
    if not db_album or not db_album.tags or "similar_albums" not in db_album.tags:
        raise HTTPException(status_code=404, detail="Similar albums data not found")
    
    return {"similar_albums": db_album.tags["similar_albums"]}