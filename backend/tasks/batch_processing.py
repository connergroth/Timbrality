# backend/tasks/batch_processing.py
import asyncio
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.models.album import Album
from app.middleware.aoty_middleware import enrich_album_with_aoty

async def batch_enrich_albums(limit: int = 100, skip: int = 0):
    """
    Batch process albums to enrich with AOTY data.
    Processes albums in chunks to avoid overloading the API.
    """
    db = SessionLocal()
    try:
        # Get total albums count
        total_albums = db.query(Album).count()
        
        processed = 0
        failed = 0
        
        # Process in chunks
        while skip < total_albums:
            # Get a batch of albums
            albums = db.query(Album).order_by(Album.id).offset(skip).limit(limit).all()
            
            # Process each album
            for album in albums:
                try:
                    success = await enrich_album_with_aoty(db, album.id)
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing album {album.id}: {str(e)}")
                    failed += 1
            
            # Update skip for next batch
            skip += limit
            
            print(f"Progress: {skip}/{total_albums} albums processed")
            
            # Optional: larger delay between batches
            await asyncio.sleep(5)
            
        return {
            "total": total_albums,
            "processed": processed,
            "failed": failed
        }
        
    finally:
        db.close()

# Function to run the task
def start_batch_processing():
    asyncio.run(batch_enrich_albums())