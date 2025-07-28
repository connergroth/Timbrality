# backend/tasks/batch_processing.py
import asyncio

try:
    from ingestion.insert_to_supabase import get_supabase_client, get_training_dataset
    from middleware.aoty_middleware import enrich_album_with_aoty
except ImportError:
    # Fallback functions if services not available
    def get_supabase_client():
        return None
    
    def get_training_dataset(*args, **kwargs):
        return []
    
    async def enrich_album_with_aoty(*args, **kwargs):
        return None

async def batch_enrich_albums(limit: int = 100, skip: int = 0):
    """
    Batch process albums to enrich with AOTY data using Supabase.
    Processes albums in chunks to avoid overloading the API.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            print("Supabase client not available")
            return {"total": 0, "processed": 0, "failed": 0}
        
        # Get albums from Supabase (tracks table contains album info)
        tracks_data = get_training_dataset(limit=1000)  # Get sample of tracks
        
        # Extract unique albums
        unique_albums = {}
        for track in tracks_data:
            album_key = f"{track.get('artist', '').lower()}-{track.get('album', '').lower()}"
            if album_key not in unique_albums and track.get('album') and track.get('artist'):
                unique_albums[album_key] = {
                    'artist': track.get('artist'),
                    'album': track.get('album')
                }
        
        total_albums = len(unique_albums)
        processed = 0
        failed = 0
        
        # Process each unique album
        for album_info in list(unique_albums.values())[skip:skip+limit]:
            try:
                result = await enrich_album_with_aoty(album_info['artist'], album_info['album'])
                if result:
                    processed += 1
                    print(f"Enriched: {album_info['artist']} - {album_info['album']}")
                else:
                    failed += 1
                    
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error processing album {album_info['album']} by {album_info['artist']}: {str(e)}")
                failed += 1
        
        print(f"Progress: {processed + failed}/{total_albums} albums processed")
        
        return {
            "total": total_albums,
            "processed": processed,
            "failed": failed
        }
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return {"total": 0, "processed": 0, "failed": 0}

# Function to run the task
def start_batch_processing():
    asyncio.run(batch_enrich_albums())