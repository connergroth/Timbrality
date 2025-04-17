from sqlalchemy.orm import Session
from app.services.aoty_service import AOTYService
from app.models.album import Album as DBAlbum
from app.models.artist import Artist as DBArtist
from app.models.tracks.track import Track as DBTrack
from app.utils.matching import find_best_match

async def enrich_album_with_aoty(db: Session, album_id: str) -> bool:
    """
    Enriches an album in the database with AOTY metadata.
    Returns True if successful, False otherwise.
    """
    # Get album from database
    db_album = db.query(DBAlbum).filter(DBAlbum.id == album_id).first()
    if not db_album:
        return False
    
    # Get artist from database
    db_artist = db.query(DBArtist).filter(DBArtist.id == db_album.artist_id).first()
    if not db_artist:
        return False
    
    # Initialize AOTY service
    aoty_service = AOTYService()
    
    try:
        # Fetch album data from AOTY
        aoty_album = await aoty_service.get_album(db_artist.name, db_album.title)
        if not aoty_album:
            return False
        
        # Update album with AOTY metadata
        db_album.aoty_score = aoty_album.get("user_score")
        db_album.aoty_rating_count = aoty_album.get("num_ratings")
        
        # Extract and store tags if available
        genres = []
        is_must_hear = aoty_album.get("is_must_hear", False)
        
        # If metadata.genres exists (in new AOTY-API)
        if aoty_album.get("metadata") and aoty_album["metadata"].get("genres"):
            genres = aoty_album["metadata"]["genres"]
        
        # Update tags with genre info
        current_tags = db_album.tags or {}
        current_tags.update({
            "genres": genres,
            "is_must_hear": is_must_hear
        })
        db_album.tags = current_tags
        
        # Match album tracks with AOTY tracks if available
        if aoty_album.get("tracks"):
            db_tracks = db.query(DBTrack).filter(DBTrack.album_id == album_id).all()
            
            # Extract track titles for matching
            db_track_titles = [track.title for track in db_tracks]
            
            for aoty_track in aoty_album["tracks"]:
                # Find matching track in DB
                matched_title = find_best_match(
                    aoty_track["title"], 
                    db_track_titles,
                    score_cutoff=80
                )
                
                if matched_title:
                    matched_track = next(
                        (t for t in db_tracks if t.title == matched_title),
                        None
                    )
                    
                    if matched_track:
                        # Update track with AOTY data
                        matched_track.aoty_score = aoty_track.get("rating")
                        matched_track.tags = matched_track.tags or {}
                        matched_track.tags["aoty_length"] = aoty_track.get("length")
        
        # Commit changes to the database
        db.commit()
        return True
    
    except Exception as e:
        print(f"Error enriching album {db_album.title} with AOTY data: {str(e)}")
        db.rollback()
        return False
    
    finally:
        await aoty_service.close()