"""
HTTP-based fallback scraper for AOTY when Playwright times out.
This uses simple HTTP requests and BeautifulSoup parsing.
"""

import requests
import urllib.parse
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup
import time

try:
    from ..config import BASE_URL, HEADERS, SCRAPER_DELAY
except ImportError:
    # Fallback for direct execution
    BASE_URL = "https://www.albumoftheyear.org"
    SCRAPER_DELAY = 1.5
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    }

def http_search_albums(query: str, limit: int = 10, max_retries: int = 2) -> List[dict]:
    """
    Simple HTTP-based album search as fallback when Playwright fails.
    Returns basic album information.
    """
    search_query = urllib.parse.quote(query)
    url = f"{BASE_URL}/search/albums/?q={search_query}"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(SCRAPER_DELAY * attempt)
            
            print(f"HTTP fallback search attempt {attempt + 1} for: {query}")
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Look for album blocks with various possible class names
            album_blocks = soup.find_all(['div', 'article'], class_=lambda x: x and ('album' in x.lower() or 'block' in x.lower()))
            
            if not album_blocks:
                # Try simpler selectors
                album_blocks = soup.find_all('div', class_=['albumBlock', 'searchResult'])
            
            print(f"Found {len(album_blocks)} potential album blocks")
            
            for block in album_blocks[:limit]:
                try:
                    # Extract album link
                    album_link = block.find('a', href=lambda x: x and '/album/' in x)
                    if not album_link:
                        continue
                    
                    href = album_link.get('href')
                    album_url = f"{BASE_URL}{href}" if href.startswith('/') else href
                    
                    # Extract text content for artist and title
                    text_content = block.get_text(separator=' ', strip=True)
                    
                    # Try to find artist and title in various ways
                    artist_name = ""
                    album_title = ""
                    
                    # Look for specific elements
                    artist_elem = block.find(['span', 'div', 'a'], class_=lambda x: x and 'artist' in x.lower())
                    title_elem = block.find(['span', 'div', 'a'], class_=lambda x: x and 'title' in x.lower())
                    
                    if artist_elem:
                        artist_name = artist_elem.get_text(strip=True)
                    if title_elem:
                        album_title = title_elem.get_text(strip=True)
                    
                    # If we couldn't find specific elements, try parsing the link text
                    if not artist_name or not album_title:
                        link_text = album_link.get_text(strip=True)
                        if ' - ' in link_text:
                            parts = link_text.split(' - ', 1)
                            if len(parts) == 2:
                                artist_name = artist_name or parts[0]
                                album_title = album_title or parts[1]
                    
                    if artist_name and album_title:
                        results.append({
                            'artist': artist_name,
                            'title': album_title,
                            'url': album_url,
                            'source': 'http_fallback'
                        })
                    
                except Exception as e:
                    print(f"Error parsing album block: {e}")
                    continue
            
            return results
            
        except requests.Timeout:
            print(f"HTTP request timed out (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return []
        except requests.RequestException as e:
            print(f"HTTP request failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return []
        except Exception as e:
            print(f"HTTP fallback error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return []
    
    return []

def http_find_album_url(artist: str, album: str, max_retries: int = 2) -> Optional[Tuple[str, str, str]]:
    """
    HTTP-based album URL finder as fallback.
    """
    query = f"{artist} {album}"
    results = http_search_albums(query, limit=3, max_retries=max_retries)
    
    if not results:
        return None
    
    # Return the first result
    result = results[0]
    return (result['url'], result['artist'], result['title'])

if __name__ == "__main__":
    # Test the HTTP fallback
    print("Testing HTTP fallback scraper...")
    
    test_queries = [
        "OK Computer Radiohead",
        "Dark Side of the Moon Pink Floyd"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        results = http_search_albums(query, limit=2)
        if results:
            print(f"✅ Found {len(results)} results:")
            for result in results:
                print(f"  - {result['artist']} - {result['title']}")
        else:
            print("❌ No results found") 