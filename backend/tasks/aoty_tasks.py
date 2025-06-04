# backend/tasks/aoty_tasks.py
from sqlalchemy.orm import Session
from app.services.aoty_service import AOTYService
from app.models.album import Album
from app.models.artist import Artist

async def fetch_and_store_similar_albums(db: Session, album_id: str, limit: int = 5):
    """Fetches and stores similar albums for a given album ID."""
    # Get album from database
    db_album = db.query(Album).filter(Album.id == album_id).first()
    if not db_album:
        return False
    
    # Get artist from database
    db_artist = db.query(Artist).filter(Artist.id == db_album.artist_id).first()
    if not db_artist:
        return False
    
    # Initialize AOTY service
    aoty_service = AOTYService()
    
    try:
        # Fetch similar albums
        similar_albums = await aoty_service.get_similar_albums(
            db_artist.name, 
            db_album.title,
            limit
        )
        
        # Store similar albums as a relation or in album tags
        if similar_albums:
            # Option 1: Store in the album tags
            current_tags = db_album.tags or {}
            current_tags["similar_albums"] = [
                {
                    "title": album["title"],
                    "artist": album["artist"],
                    "score": album.get("user_score")
                }
                for album in similar_albums
            ]
            db_album.tags = current_tags
            db.commit()
            
            # Option 2: Store as actual database entries (more complex)
            # This would require creating records for the similar albums
            # and defining a relationship model between albums
            
        return True
    
    except Exception as e:
        print(f"Error fetching similar albums for {db_album.title}: {str(e)}")
        db.rollback()
        return False
    
    finally:
        await aoty_service.close()