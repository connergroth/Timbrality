"""
Enhanced Album of the Year (AOTY) Scraping Module
Uses cloudscraper for better anti-bot bypass with conservative rate limiting
"""
import asyncio
import csv
import json
import logging
import os
import random
import time
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, quote_plus
import urllib.parse
from app.config import settings
from app.models import SongCore, AotyAttrs

# Try to import cloudscraper (more effective than httpx for anti-bot)
try:
    import cloudscraper
    from bs4 import BeautifulSoup
    HAS_CLOUDSCRAPER = True
except ImportError:
    import httpx
    from selectolax.parser import HTMLParser
    HAS_CLOUDSCRAPER = False
    logging.warning("cloudscraper not available, falling back to httpx (less effective against anti-bot)")

logger = logging.getLogger(__name__)


class AOTYConservativeScraper:
    """Conservative AOTY scraper with caching and cloudscraper for anti-bot bypass"""
    
    def __init__(self):
        self.base_url = "https://www.albumoftheyear.org"
        self.search_url = f"{self.base_url}/search/albums/"
        
        # Cache for storing results to minimize requests
        self.cache_file = os.path.join(settings.data_dir, "aoty_cache.json")
        self.cache = {}
        self.cache_duration = 24 * 60 * 60  # 24 hours
        self.load_cache()
        
        # Rate limiting - very conservative
        self.last_request_time = 0
        self.minimum_delay = max(settings.scrape_delay_sec, 10.0)  # At least 10 seconds
        self.request_count = 0
        self.max_requests_per_session = 20  # Conservative limit
        
        # Circuit breaker for anti-bot detection
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.circuit_breaker_until = 0
        
        # Initialize cloudscraper or httpx
        if HAS_CLOUDSCRAPER:
            self.scraper = cloudscraper.create_scraper(
                browser='chrome',
                delay=random.uniform(2, 5)
            )
            logger.info("Using cloudscraper for AOTY (better anti-bot protection)")
        else:
            self.scraper = None
            logger.warning("Using httpx fallback for AOTY (limited anti-bot protection)")
        
        # Failed tracks for CSV export
        self.failed_tracks = []
    
    def load_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached AOTY entries")
        except Exception as e:
            logger.error(f"Failed to load AOTY cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save AOTY cache: {e}")
    
    def _get_cache_key(self, artist: str, title: str) -> str:
        """Generate cache key for artist/title"""
        return f"{artist.lower().strip()}::{title.lower().strip()}"
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = cache_entry['timestamp']
        current_time = time.time()
        return (current_time - cache_time) < self.cache_duration
    
    def _should_make_request(self) -> Tuple[bool, float]:
        """Check if we should make a request and how long to wait"""
        # Check circuit breaker
        now = time.time()
        if now < self.circuit_breaker_until:
            remaining = self.circuit_breaker_until - now
            logger.warning(f"AOTY circuit breaker active for {remaining:.1f}s more")
            return False, remaining
        
        # Check session request limit
        if self.request_count >= self.max_requests_per_session:
            logger.warning(f"Reached AOTY session limit ({self.max_requests_per_session} requests)")
            return False, 3600  # Wait 1 hour
        
        # Check rate limiting
        time_since_last = now - self.last_request_time
        if time_since_last < self.minimum_delay:
            wait_time = self.minimum_delay - time_since_last
            return False, wait_time
        
        return True, 0.0
    
    async def _make_request_cloudscraper(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Make request using cloudscraper (runs in thread pool)"""
        try:
            # Add random delay to appear more human
            await asyncio.sleep(random.uniform(2, 5))
            
            # Run in thread pool since cloudscraper is synchronous
            response_text = await asyncio.to_thread(
                lambda: self.scraper.get(url, params=params).text
            )
            
            self.last_request_time = time.time()
            self.request_count += 1
            self.consecutive_failures = 0  # Reset on success
            
            return response_text
            
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Cloudscraper request failed: {e}")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                # Activate circuit breaker for 30 minutes
                self.circuit_breaker_until = time.time() + 1800
                logger.error(f"AOTY circuit breaker activated after {self.consecutive_failures} failures")
            
            return None
    
    async def _make_request_httpx(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Enhanced fallback request using httpx with better anti-bot measures"""
        
        # Rotate user agents for better anti-bot protection
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        
        # Add referer for non-search requests
        if not url.endswith('/search/albums/'):
            headers["Referer"] = self.base_url
        
        try:
            timeout = httpx.Timeout(30.0, connect=10.0)
            
            # Use HTTP/2 and follow redirects
            async with httpx.AsyncClient(
                timeout=timeout, 
                follow_redirects=True,
                http2=True,
                headers=headers
            ) as client:
                
                # Add random delay before request
                await asyncio.sleep(random.uniform(3, 8))
                
                response = await client.get(url, params=params)
                
                # Handle specific status codes
                if response.status_code == 403:
                    logger.warning("httpx: 403 Forbidden - possible bot detection")
                    raise Exception("403 Forbidden")
                elif response.status_code == 429:
                    logger.warning("httpx: 429 Too Many Requests - rate limited")
                    raise Exception("429 Rate Limited")
                elif response.status_code == 503:
                    logger.warning("httpx: 503 Service Unavailable - possible anti-bot")
                    raise Exception("503 Service Unavailable")
                
                response.raise_for_status()
                
                self.last_request_time = time.time() 
                self.request_count += 1
                self.consecutive_failures = 0
                
                return response.text
                
        except httpx.HTTPStatusError as e:
            self.consecutive_failures += 1
            status_code = e.response.status_code
            logger.error(f"httpx HTTP error {status_code}: {e}")
            
            # Activate circuit breaker for certain error codes
            if status_code in [403, 503]:
                self.circuit_breaker_until = time.time() + 3600  # 1 hour for bot detection
                logger.error("AOTY circuit breaker activated due to anti-bot detection")
            elif self.consecutive_failures >= self.max_consecutive_failures:
                self.circuit_breaker_until = time.time() + 1800  # 30 minutes for other failures
                logger.error("AOTY circuit breaker activated due to consecutive failures")
            
            return None
            
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"httpx request failed: {e}")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.circuit_breaker_until = time.time() + 1800
                logger.error("AOTY circuit breaker activated")
            
            return None
    
    async def _make_request(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Make HTTP request with anti-bot measures"""
        can_request, wait_time = self._should_make_request()
        
        if not can_request:
            logger.debug(f"Skipping AOTY request, need to wait {wait_time:.1f}s")
            return None
        
        # Add conservative delay before request
        await asyncio.sleep(random.uniform(1, 3))
        
        if HAS_CLOUDSCRAPER:
            return await self._make_request_cloudscraper(url, params)
        else:
            return await self._make_request_httpx(url, params)
    
    async def scrape_aoty(self, song: SongCore) -> Optional[AotyAttrs]:
        """
        Scrape AOTY data for a single song with caching
        
        Args:
            song: SongCore object to search for
            
        Returns:
            AotyAttrs object or None if not found/failed
        """
        cache_key = self._get_cache_key(song.artist, song.title)
        
        # Check cache first
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug(f"Using cached AOTY data for {song.artist} - {song.title}")
                data = cache_entry.get('data')
                return AotyAttrs(**data) if data else None
        
        try:
            # Search for the album
            search_query = f"{song.artist} {song.title}"
            logger.debug(f"Searching AOTY for: {search_query}")
            
            search_params = {"q": urllib.parse.quote(search_query)}
            search_html = await self._make_request(self.search_url, search_params)
            
            if not search_html:
                self._add_failed_track(song, "Search request failed")
                self._cache_result(cache_key, None, "request_failed")
                return None
            
            # Parse search results
            attrs = await self._parse_search_results(search_html, song)
            
            # Cache the result
            if attrs:
                self._cache_result(cache_key, attrs.dict(), "found")
                logger.debug(f"Successfully scraped AOTY data for {song.artist} - {song.title}")
            else:
                self._cache_result(cache_key, None, "not_found")
                logger.debug(f"No AOTY data found for {song.artist} - {song.title}")
            
            return attrs
            
        except Exception as e:
            logger.error(f"Failed to scrape AOTY for {song.artist} - {song.title}: {e}")
            self._add_failed_track(song, str(e))
            self._cache_result(cache_key, None, f"error: {str(e)[:100]}")
            return None
    
    def _cache_result(self, cache_key: str, data: Optional[dict], status: str):
        """Cache a search result"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'status': status
        }
        self.save_cache()
    
    async def _parse_search_results(self, html: str, song: SongCore) -> Optional[AotyAttrs]:
        """Parse search results to find album data"""
        try:
            if HAS_CLOUDSCRAPER:
                return await self._parse_with_beautifulsoup(html, song)
            else:
                return await self._parse_with_htmlparser(html, song)
        except Exception as e:
            logger.error(f"Failed to parse search results: {e}")
            return None
    
    async def _parse_with_beautifulsoup(self, html: str, song: SongCore) -> Optional[AotyAttrs]:
        """Parse using BeautifulSoup (preferred)"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for album blocks
        album_blocks = soup.select('.albumBlock')[:3]  # Check first 3 results
        
        for block in album_blocks:
            try:
                # Get album link
                album_link = block.select_one('.image a')
                if not album_link:
                    continue
                
                # Get basic info
                artist_elem = block.select_one('.artistTitle')
                title_elem = block.select_one('.albumTitle')
                
                if not (artist_elem and title_elem):
                    continue
                
                album_url = urljoin(self.base_url, album_link.get('href', ''))
                found_artist = artist_elem.get_text(strip=True)
                found_title = title_elem.get_text(strip=True)
                
                # Extract score
                user_score = None
                score_elem = block.select_one('.albumScore')
                if score_elem and score_elem.get_text(strip=True) != "NR":
                    try:
                        user_score = float(score_elem.get_text(strip=True))
                    except (ValueError, TypeError):
                        pass
                
                # Extract year
                year = None
                year_elem = block.select_one('.albumYear')
                if year_elem and year_elem.get_text(strip=True).isdigit():
                    year = int(year_elem.get_text(strip=True))
                
                # For now, return basic data (could scrape individual album page for more)
                if user_score is not None and user_score > 0:
                    return AotyAttrs(
                        user_score=user_score,
                        rating_count=50,  # Placeholder - would need album page scrape
                        tags=None,
                        genres=None,
                        album_url=album_url,
                        album_title=found_title
                    )
                    
            except Exception as e:
                logger.debug(f"Error parsing album block: {e}")
                continue
        
        return None
    
    async def _parse_with_htmlparser(self, html: str, song: SongCore) -> Optional[AotyAttrs]:
        """Parse using HTMLParser (fallback)"""
        parser = HTMLParser(html)
        
        # Look for album links
        album_links = parser.css("a[href*='/album/']")[:3]
        
        for link in album_links:
            try:
                album_url = urljoin(self.base_url, link.attributes.get("href", ""))
                
                # Extract basic score if visible in search results
                parent = link.parent
                if parent:
                    score_text = parent.text(strip=True)
                    # Look for score patterns
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if score_match:
                        try:
                            user_score = float(score_match.group(1))
                            if 0 < user_score <= 100:  # Valid AOTY score range
                                return AotyAttrs(
                                    user_score=user_score,
                                    rating_count=50,  # Placeholder
                                    tags=None,
                                    genres=None,
                                    album_url=album_url,
                                    album_title=None
                                )
                        except (ValueError, TypeError):
                            pass
                            
            except Exception as e:
                logger.debug(f"Error parsing link: {e}")
                continue
        
        return None
    
    def _add_failed_track(self, song: SongCore, reason: str):
        """Add failed track to CSV export list"""
        self.failed_tracks.append({
            "artist": song.artist,
            "title": song.title,
            "reason": reason,
            "timestamp": time.time()
        })
    
    async def save_failed_tracks(self):
        """Export failed tracks to CSV for manual review"""
        if not self.failed_tracks:
            return
        
        try:
            filename = f"{settings.data_dir}/aoty_todo.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['artist', 'title', 'reason', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.failed_tracks)
            
            logger.info(f"Saved {len(self.failed_tracks)} failed tracks to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save failed tracks CSV: {e}")
    
    async def scrape_many(self, songs: List[SongCore], concurrency: Optional[int] = None) -> List[Tuple[SongCore, Optional[AotyAttrs]]]:
        """
        Scrape AOTY data for multiple songs with controlled concurrency and conservative limits
        
        Args:
            songs: List of SongCore objects to scrape
            concurrency: Number of concurrent requests (ignored for conservative approach)
            
        Returns:
            List of tuples (SongCore, AotyAttrs or None)
        """
        logger.info(f"Starting conservative AOTY scraping for {len(songs)} tracks")
        logger.info(f"Using {'cloudscraper' if HAS_CLOUDSCRAPER else 'httpx fallback'}")
        
        results = []
        successful = 0
        cached = 0
        
        # Process songs sequentially for maximum anti-bot protection
        for i, song in enumerate(songs):
            try:
                # Check if we're hitting session limits
                if self.request_count >= self.max_requests_per_session:
                    logger.warning(f"Reached AOTY session limit, remaining {len(songs) - i} tracks will use cache only")
                
                attrs = await self.scrape_aoty(song)
                
                if attrs:
                    successful += 1
                    logger.debug(f"✓ AOTY data found for {song.artist} - {song.title}")
                else:
                    # Check if it was from cache
                    cache_key = self._get_cache_key(song.artist, song.title)
                    if cache_key in self.cache:
                        cached += 1
                
                results.append((song, attrs))
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"AOTY progress: {i + 1}/{len(songs)} processed, {successful} found")
                
            except Exception as e:
                logger.error(f"Failed to scrape {song.artist} - {song.title}: {e}")
                self._add_failed_track(song, str(e))
                results.append((song, None))
        
        # Save failed tracks and cache
        await self.save_failed_tracks()
        self.save_cache()
        
        logger.info(f"AOTY scraping complete: {successful}/{len(songs)} found, {cached} from cache, {self.request_count} requests made")
        
        return results


# Global scraper instance for session persistence
_scraper_instance = None

def get_scraper_instance() -> AOTYConservativeScraper:
    """Get or create global scraper instance"""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = AOTYConservativeScraper()
    return _scraper_instance


async def scrape_aoty(song: SongCore) -> Optional[AotyAttrs]:
    """
    Scrape AOTY data for a single song using conservative scraper
    
    Args:
        song: SongCore object to search for
        
    Returns:
        AotyAttrs object or None if not found
    """
    scraper = get_scraper_instance()
    return await scraper.scrape_aoty(song)


async def scrape_many(songs: List[SongCore], concurrency: Optional[int] = None) -> List[Tuple[SongCore, Optional[AotyAttrs]]]:
    """
    Scrape AOTY data for multiple songs using conservative approach
    
    Args:
        songs: List of SongCore objects to scrape
        concurrency: Ignored - uses sequential processing for anti-bot protection
        
    Returns:
        List of tuples (SongCore, AotyAttrs or None)
    """
    scraper = get_scraper_instance()
    return await scraper.scrape_many(songs, concurrency)


def get_aoty_cache_stats() -> dict:
    """Get AOTY cache statistics"""
    scraper = get_scraper_instance()
    valid_entries = sum(1 for entry in scraper.cache.values() if scraper._is_cache_valid(entry))
    
    return {
        "total_entries": len(scraper.cache),
        "valid_entries": valid_entries,
        "expired_entries": len(scraper.cache) - valid_entries,
        "cache_file": scraper.cache_file,
        "session_requests": scraper.request_count,
        "max_session_requests": scraper.max_requests_per_session,
        "circuit_breaker_active": scraper.circuit_breaker_until > time.time(),
        "using_cloudscraper": HAS_CLOUDSCRAPER
    }


def clear_aoty_cache():
    """Clear AOTY cache (use with caution)"""
    scraper = get_scraper_instance()
    scraper.cache.clear()
    scraper.save_cache()
    logger.info("AOTY cache cleared")