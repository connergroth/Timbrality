#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AOTY Scraper using cloudscraper - Much more effective at bypassing anti-bot measures.
This replaces the complex Playwright approach with a simpler, more reliable solution.
"""

import cloudscraper
import urllib.parse
import asyncio
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple

try:
    from ..models.aoty_models import SearchResult
    from ..config import BASE_URL
    from ..utils.search_cleaner import clean_search_query
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, backend_dir)
    
    from models.aoty_models import SearchResult
    from utils.search_cleaner import clean_search_query
    BASE_URL = "https://www.albumoftheyear.org"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

# Create cloudscraper instance
scraper = cloudscraper.create_scraper()


async def get_album_url(artist: str, album: str) -> Optional[Tuple[str, str, str]]:
    """
    Search for an album using cloudscraper.
    Much more reliable than Playwright for bypassing anti-bot measures.
    """
    search_query = urllib.parse.quote(f"{artist} {album}")
    url = f"{BASE_URL}/search/albums/?q={search_query}"

    try:
        print(f"🔍 Searching AOTY for: {artist} - {album}")
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")

        album_block = soup.select_one(".albumBlock")
        if album_block:
            album_link = album_block.select_one(".image a")
            if album_link:
                artist_elem = album_block.select_one(".artistTitle")
                title_elem = album_block.select_one(".albumTitle")
                
                if artist_elem and title_elem:
                    album_url = f"{BASE_URL}{album_link['href']}"
                    found_artist = artist_elem.text.strip()
                    found_title = title_elem.text.strip()
                    
                    print(f"✅ Found: {found_artist} - {found_title}")
                    return (album_url, found_artist, found_title)
                    
    except Exception as e:
        print(f"❌ Error searching for {artist} - {album}: {str(e)}")
        return None

    print(f"❌ No results found for: {artist} - {album}")
    return None


async def search_albums(query: str, limit: int = 10) -> List[SearchResult]:
    """
    Search for albums and return results using cloudscraper.
    """
    # Clean the search query to remove platform suffixes and other noise
    cleaned_query = clean_search_query(query)
    search_query = urllib.parse.quote(cleaned_query)
    url = f"{BASE_URL}/search/albums/?q={search_query}"

    try:
        print(f"🔍 Searching AOTY for: {query}")
        if cleaned_query != query:
            print(f"🧹 Cleaned query: '{query}' -> '{cleaned_query}'")
        
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")
        results = []

        album_blocks = soup.select(".albumBlock")[:limit]
        
        for block in album_blocks:
            try:
                album_link = block.select_one(".image a")
                if not album_link:
                    continue
                    
                artist_elem = block.select_one(".artistTitle")
                title_elem = block.select_one(".albumTitle")
                
                if not (artist_elem and title_elem):
                    continue
                
                album_url = f"{BASE_URL}{album_link['href']}"
                artist_name = artist_elem.text.strip()
                album_title = title_elem.text.strip()
                
                # Extract additional data
                cover_image = None
                cover_elem = block.select_one(".image img")
                if cover_elem:
                    cover_image = cover_elem.get("src")
                
                year = None
                year_elem = block.select_one(".albumYear")
                if year_elem and year_elem.text.strip().isdigit():
                    year = int(year_elem.text.strip())
                
                score = None
                score_elem = block.select_one(".albumScore")
                if score_elem and score_elem.text.strip() != "NR":
                    try:
                        score = float(score_elem.text.strip())
                    except ValueError:
                        pass
                
                results.append(SearchResult(
                    title=album_title,
                    artist=artist_name,
                    url=album_url,
                    cover_image=cover_image,
                    year=year,
                    score=score
                ))
                
            except Exception as e:
                print(f"Error parsing search result: {str(e)}")
                continue
        
        print(f"✅ Found {len(results)} results for '{cleaned_query}'")
        return results
        
    except Exception as e:
        print(f"❌ Search error for '{cleaned_query}': {str(e)}")
        return []


async def close_browser():
    """Cleanup function - not needed for cloudscraper but kept for compatibility"""
    pass


if __name__ == "__main__":
    async def test_cloudscraper():
        print("Testing CloudScraper AOTY Integration")
        print("=" * 45)
        
        # Test search
        print("\n🔍 Testing search:")
        results = await search_albums("OK Computer", limit=3)
        print(f"Search results: {len(results)}")
        
        # Test album URL finding
        print("\n🔍 Testing album URL finding:")
        result = await get_album_url("Radiohead", "OK Computer")
        if result:
            url, artist, title = result
            print(f"Found: {artist} - {title}")
            print(f"URL: {url}")
        
        print("\n✅ CloudScraper test completed!")
    
    asyncio.run(test_cloudscraper()) 