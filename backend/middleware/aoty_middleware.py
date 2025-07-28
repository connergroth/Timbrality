from sqlalchemy.orm import Session

try:
    from services.aoty_scraper_service import get_album_url, scrape_album
    from models.aoty_models import Album
    from utils.matching import find_best_match
except ImportError:
    # Fallback functions if services not available
    async def get_album_url(*args, **kwargs):
        return None
    
    async def scrape_album(*args, **kwargs):
        return None
    
    def find_best_match(*args, **kwargs):
        return None
    
    class Album:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

async def enrich_album_with_aoty(artist_name: str, album_title: str) -> dict:
    """
    Enriches an album with AOTY metadata using the scraper service.
    Returns album data if successful, None otherwise.
    """
    try:
        # Find the album URL using the scraper service
        album_url = await get_album_url(artist_name, album_title)
        if not album_url:
            return None
        
        # Scrape the album data
        album_data = await scrape_album(album_url)
        if not album_data:
            return None
        
        return album_data
    except Exception as e:
        print(f"Error enriching album {album_title} by {artist_name} with AOTY data: {str(e)}")
        return None