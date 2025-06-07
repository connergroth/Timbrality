import asyncio
import urllib.parse
from typing import Optional, List, Dict, Tuple, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Page
from fastapi import HTTPException

from app.config import BASE_URL, PLAYWRIGHT_HEADLESS, PLAYWRIGHT_TIMEOUT
from app.models.aoty_models import (
    Album, 
    Track, 
    CriticReview, 
    AlbumUserReview, 
    ProfileUserReview,
    UserProfile, 
    BuyLink, 
    SearchResult,
    AlbumMetadata,
    UserStats
)

# Cache for the Playwright browser instance
_browser = None
_browser_context = None


async def get_browser():
    """Get or create a shared Playwright browser instance"""
    global _browser, _browser_context
    
    if _browser is None:
        playwright = await async_playwright().start()
        _browser = await playwright.chromium.launch(headless=PLAYWRIGHT_HEADLESS)
        
    if _browser_context is None:
        _browser_context = await _browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
    
    return _browser_context


async def new_page():
    """Create a new page from the browser context"""
    context = await get_browser()
    return await context.new_page()


def parse_number(text: str | int) -> int:
    """Convert string numbers with commas to integers"""
    if isinstance(text, int):
        return text
    
    # Remove commas and other non-numeric characters
    clean_text = ''.join(c for c in str(text) if c.isdigit())
    return int(clean_text) if clean_text else 0


async def get_album_url(artist: str, album: str) -> Optional[Tuple[str, str, str]]:
    """
    Search for an album and return its URL, artist name, and album title.
    """
    search_query = urllib.parse.quote(f"{artist} {album}")
    url = f"{BASE_URL}/search/albums/?q={search_query}"

    try:
        page = await new_page()
        try:
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="domcontentloaded")
            
            # Wait for search results to load
            await page.wait_for_selector(".albumBlock", timeout=PLAYWRIGHT_TIMEOUT)
            
            # Check if any album blocks exist
            album_blocks = await page.query_selector_all(".albumBlock")
            if not album_blocks or len(album_blocks) == 0:
                return None
                
            # Get the first album block
            album_block = album_blocks[0]
            
            # Extract album URL
            album_link = await album_block.query_selector(".image a")
            if not album_link:
                return None
                
            href = await album_link.get_attribute("href")
            album_url = f"{BASE_URL}{href}"
            
            # Extract artist name
            artist_elem = await album_block.query_selector(".artistTitle")
            artist_name = await artist_elem.text_content() if artist_elem else ""
            
            # Extract album title
            title_elem = await album_block.query_selector(".albumTitle")
            album_title = await title_elem.text_content() if title_elem else ""
            
            if artist_name and album_title:
                return (album_url, artist_name.strip(), album_title.strip())
                
            return None
            
        finally:
            await page.close()
    except PlaywrightTimeoutError:
        raise HTTPException(status_code=503, detail="Timeout searching for album")
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error searching for album: {str(e)}"
        )


async def scrape_album(url: str, artist: str, title: str) -> Album:
    """Scrape album information using Playwright"""
    try:
        page = await new_page()
        try:
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="domcontentloaded")
            
            # Wait for album page to load
            await page.wait_for_selector(".albumPage", timeout=PLAYWRIGHT_TIMEOUT)
            
            # Extract cover image URL
            cover_image = None
            img_elem = await page.query_selector(".albumImage img")
            if img_elem:
                cover_image = await img_elem.get_attribute("src")
            
            # Extract user score
            user_score = None
            score_elem = await page.query_selector(".albumUserScore a")
            if score_elem:
                score_title = await score_elem.get_attribute("title")
                if score_title:
                    try:
                        user_score = float(score_title)
                    except ValueError:
                        pass
            
            # Extract critic score
            critic_score = None
            critic_elem = await page.query_selector(".albumCriticScore")
            if critic_elem:
                critic_text = await critic_elem.text_content()
                try:
                    if critic_text and critic_text.strip() != "NR":
                        critic_score = float(critic_text.strip())
                except ValueError:
                    pass
            
            # Extract number of ratings
            num_ratings = 0
            ratings_elem = await page.query_selector(".albumUserScore .numRatings")
            if ratings_elem:
                ratings_text = await ratings_elem.text_content()
                num_ratings = parse_number(ratings_text)
            
            # Extract number of reviews
            num_reviews = 0
            reviews_elem = await page.query_selector(".albumReviewCount")
            if reviews_elem:
                reviews_text = await reviews_elem.text_content()
                num_reviews = parse_number(reviews_text)
            
            # Create basic album object
            album = Album(
                title=title,
                artist=artist,
                url=url,
                cover_image=cover_image,
                user_score=user_score,
                critic_score=critic_score,
                num_ratings=num_ratings,
                num_reviews=num_reviews,
                metadata=AlbumMetadata(),
                tracks=[],
                critic_reviews=[],
                popular_reviews=[],
                buy_links=[]
            )
            
            return album
            
        finally:
            await page.close()
    except PlaywrightTimeoutError:
        raise HTTPException(status_code=503, detail="Timeout accessing album page")
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error scraping album: {str(e)}"
        )


async def search_albums(query: str, limit: int = 10) -> List[SearchResult]:
    """Search for albums matching the query"""
    search_query = urllib.parse.quote(query)
    url = f"{BASE_URL}/search/albums/?q={search_query}"
    
    try:
        page = await new_page()
        try:
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="domcontentloaded")
            
            # Wait for search results to load
            await page.wait_for_selector(".albumBlock", timeout=PLAYWRIGHT_TIMEOUT)
            
            # Get album blocks
            album_blocks = await page.query_selector_all(".albumBlock")
            results = []
            
            for i, block in enumerate(album_blocks[:limit]):
                try:
                    # Extract album URL
                    album_link = await block.query_selector(".image a")
                    if not album_link:
                        continue
                        
                    href = await album_link.get_attribute("href")
                    album_url = f"{BASE_URL}{href}"
                    
                    # Extract artist name
                    artist_elem = await block.query_selector(".artistTitle")
                    artist = await artist_elem.text_content() if artist_elem else ""
                    
                    # Extract album title
                    title_elem = await block.query_selector(".albumTitle")
                    title = await title_elem.text_content() if title_elem else ""
                    
                    # Extract cover image
                    cover_image = None
                    img_elem = await block.query_selector(".image img")
                    if img_elem:
                        cover_image = await img_elem.get_attribute("src")
                    
                    # Extract year
                    year = None
                    year_elem = await block.query_selector(".albumYear")
                    if year_elem:
                        year_text = await year_elem.text_content()
                        try:
                            year = int(year_text.strip())
                        except ValueError:
                            pass
                    
                    # Extract score
                    score = None
                    score_elem = await block.query_selector(".albumScore")
                    if score_elem:
                        score_text = await score_elem.text_content()
                        try:
                            if score_text and score_text.strip() != "NR":
                                score = float(score_text.strip())
                        except ValueError:
                            pass
                    
                    if artist and title:
                        results.append(SearchResult(
                            title=title.strip(),
                            artist=artist.strip(),
                            url=album_url,
                            cover_image=cover_image,
                            year=year,
                            score=score
                        ))
                        
                except Exception as e:
                    print(f"Error parsing search result {i}: {str(e)}")
                    continue
            
            return results
            
        finally:
            await page.close()
    except PlaywrightTimeoutError:
        raise HTTPException(status_code=503, detail="Timeout searching for albums")
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error searching albums: {str(e)}"
        )


async def get_similar_albums(url: str, limit: int = 5) -> List[Album]:
    """Get similar albums for a given album URL"""
    try:
        page = await new_page()
        try:
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="domcontentloaded")
            
            # Look for similar albums section
            similar_albums = []
            similar_blocks = await page.query_selector_all(".similarAlbums .albumBlock")
            
            for i, block in enumerate(similar_blocks[:limit]):
                try:
                    # Extract album URL
                    album_link = await block.query_selector(".image a")
                    if not album_link:
                        continue
                        
                    href = await album_link.get_attribute("href")
                    similar_url = f"{BASE_URL}{href}"
                    
                    # Extract artist name
                    artist_elem = await block.query_selector(".artistTitle")
                    artist = await artist_elem.text_content() if artist_elem else ""
                    
                    # Extract album title
                    title_elem = await block.query_selector(".albumTitle")
                    title = await title_elem.text_content() if title_elem else ""
                    
                    # Extract cover image
                    cover_image = None
                    img_elem = await block.query_selector(".image img")
                    if img_elem:
                        cover_image = await img_elem.get_attribute("src")
                    
                    if artist and title:
                        similar_albums.append(Album(
                            title=title.strip(),
                            artist=artist.strip(),
                            url=similar_url,
                            cover_image=cover_image,
                            metadata=AlbumMetadata()
                        ))
                        
                except Exception as e:
                    print(f"Error parsing similar album {i}: {str(e)}")
                    continue
            
            return similar_albums
            
        finally:
            await page.close()
    except PlaywrightTimeoutError:
        raise HTTPException(status_code=503, detail="Timeout accessing similar albums")
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error getting similar albums: {str(e)}"
        )


async def get_user_profile(username: str) -> UserProfile:
    """Get user profile information from AOTY"""
    url = f"{BASE_URL}/user/{username}/"
    
    try:
        page = await new_page()
        try:
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="domcontentloaded")
            
            # Wait for user profile page to load
            await page.wait_for_selector(".userProfile", timeout=PLAYWRIGHT_TIMEOUT)
            
            # Extract basic profile info
            location = None
            about = None
            member_since = None
            
            # Extract location
            location_elem = await page.query_selector(".userLocation")
            if location_elem:
                location = await location_elem.text_content()
                location = location.strip() if location else None
            
            # Extract about text
            about_elem = await page.query_selector(".userAbout")
            if about_elem:
                about = await about_elem.text_content()
                about = about.strip() if about else None
            
            # Extract member since
            member_elem = await page.query_selector(".memberSince")
            if member_elem:
                member_since = await member_elem.text_content()
                member_since = member_since.strip() if member_since else None
            
            # Extract user stats
            stats = UserStats()
            
            # Extract ratings count
            ratings_elem = await page.query_selector(".userStats .ratings")
            if ratings_elem:
                ratings_text = await ratings_elem.text_content()
                stats.ratings = parse_number(ratings_text)
            
            # Extract reviews count
            reviews_elem = await page.query_selector(".userStats .reviews")
            if reviews_elem:
                reviews_text = await reviews_elem.text_content()
                stats.reviews = parse_number(reviews_text)
            
            # Extract lists count
            lists_elem = await page.query_selector(".userStats .lists")
            if lists_elem:
                lists_text = await lists_elem.text_content()
                stats.lists = parse_number(lists_text)
            
            # Extract followers count
            followers_elem = await page.query_selector(".userStats .followers")
            if followers_elem:
                followers_text = await followers_elem.text_content()
                stats.followers = parse_number(followers_text)
            
            # Create basic user profile
            user_profile = UserProfile(
                username=username,
                location=location,
                about=about,
                member_since=member_since,
                stats=stats,
                favorite_albums=[],
                recent_reviews=[],
                social_links={},
                rating_distribution={}
            )
            
            return user_profile
            
        finally:
            await page.close()
    except PlaywrightTimeoutError:
        raise HTTPException(status_code=503, detail="Timeout accessing user profile")
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error getting user profile: {str(e)}"
        ) 