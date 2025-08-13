import cloudscraper
import urllib.parse
import asyncio

from bs4 import BeautifulSoup, Tag
from typing import Optional, Final, Dict
from ..models import (
    Album,
    Track,
    CriticReview,
    AlbumUserReview,
    ProfileUserReview,
    UserProfile,
    BuyLink,
    Artist,
)
from fastapi import HTTPException

BASE_URL: Final = "https://www.albumoftheyear.org"
HEADERS: Final = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

scraper = cloudscraper.create_scraper()


def parse_number(text: str | int) -> int:
    """Convert string numbers with commas to integers"""
    if isinstance(text, int):
        return text
    return int(str(text).replace(",", ""))


async def get_album_url(artist: str, album: str) -> Optional[tuple[str, str, str]]:
    """
    Search for an album.
    """
    search_query = urllib.parse.quote(f"{artist} {album}")
    url = f"{BASE_URL}/search/albums/?q={search_query}"

    try:
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")

        if album_block := soup.select_one(".albumBlock"):
            if album_link := album_block.select_one(".image a"):
                return (
                    f"{BASE_URL}{album_link['href']}",
                    album_block.select_one(".artistTitle").text.strip(),
                    album_block.select_one(".albumTitle").text.strip(),
                )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error accessing album site: {str(e)}"
        )

    return None


def parse_tracks(soup: BeautifulSoup) -> list[Track]:
    """Extract track information from the album page."""
    tracks = []
    for row in soup.select(".trackListTable tr"):
        number = int(row.select_one(".trackNumber").text)
        title = row.select_one(".trackTitle a").text
        length = row.select_one(".length").text if row.select_one(".length") else ""
        rating = None
        num_ratings = 0

        if rating_elem := row.select_one(".trackRating span"):
            rating = int(rating_elem.text)
            
            # Extract number of ratings from title attribute
            # Format: title="3187 Ratings"
            title_attr = rating_elem.get("title", "")
            if title_attr and "Ratings" in title_attr:
                try:
                    # Extract number from "3187 Ratings" format
                    rating_text = title_attr.replace("Ratings", "").strip()
                    num_ratings = parse_number(rating_text)
                except (ValueError, AttributeError):
                    pass

        featured_artists = []
        if featured_elem := row.select_one(".featuredArtists"):
            featured_artists = [a.text for a in featured_elem.select("a")]

        tracks.append(
            Track(
                number=number,
                title=title,
                length=length,
                rating=rating,
                num_ratings=num_ratings,
                featured_artists=featured_artists,
            )
        )

    return tracks


def parse_critic_reviews(soup: BeautifulSoup) -> list[CriticReview]:
    """Extract critic reviews from the album page."""
    reviews = []
    for review in soup.select("#critics .albumReviewRow"):
        author = (
            review.select_one(".author a").text
            if review.select_one(".author a")
            else "Unknown"
        )
        publication = review.select_one(".publication a").text
        rating_text = review.select_one(".albumReviewRating").text
        rating = int(rating_text) if rating_text.isdigit() else 0
        text = review.select_one(".albumReviewText").text.strip()

        reviews.append(
            CriticReview(
                author=author,
                publication=publication,
                rating=rating,
                text=text,
            )
        )

    return reviews


async def parse_user_reviews_full(album_url: str) -> list[AlbumUserReview]:
    """Extract popular user reviews by fetching the full popular reviews page."""
    all_reviews = []
    
    try:
        # Construct the user reviews URL
        # AOTY URL pattern: /album/{id}-{artist}-{name}/user-reviews/
        if album_url.endswith('/'):
            album_url = album_url[:-1]  # Remove trailing slash
        popular_reviews_url = f"{album_url}/user-reviews/"
        
        # Fetch the popular reviews page
        response_text = await asyncio.to_thread(
            lambda: scraper.get(popular_reviews_url, headers=HEADERS).text
        )
        soup = BeautifulSoup(response_text, "html.parser")
        
        # Extract all reviews from the popular reviews page
        for review in soup.select(".albumReviewRow"):
            try:
                author = review.select_one(".userReviewName a").text
                rating = None

                if rating_elem := review.select_one(".rating"):
                    if rating_elem.text != "NR":
                        rating = int(rating_elem.text)

                text = review.select_one(".albumReviewText").text.strip()
                likes = 0

                if likes_elem := review.select_one(".review_likes"):
                    # Try to get from <a> element first (some pages), then direct text
                    if link_elem := likes_elem.select_one("a"):
                        likes = int(link_elem.text)
                    else:
                        likes = int(likes_elem.text.strip())

                # Skip very short reviews (likely spam/jokes)
                if len(text.split()) < 5:  # Less than 5 words
                    continue
                    
                all_reviews.append(
                    AlbumUserReview(
                        author=author,
                        rating=rating,
                        text=text[:2500],  # Truncate very long reviews at 2500 chars
                        likes=likes,
                    )
                )
            except (AttributeError, ValueError):
                continue

    except Exception as e:
        # Fallback to regular review parsing if popular reviews page fails
        return []

    # Apply mixed selection strategy with all 25 popular reviews
    selected_reviews = []
    seen_authors = set()
    
    # 1. Top 5 by likes (most popular)
    by_likes = sorted(all_reviews, key=lambda x: x.likes, reverse=True)
    for review in by_likes[:5]:
        if review.author not in seen_authors:
            selected_reviews.append(review)
            seen_authors.add(review.author)
    
    # 2. Top 5 by length (detailed reviews)  
    # Filter for substantial reviews (30+ words) with decent engagement
    detailed_reviews = [r for r in all_reviews 
                       if len(r.text.split()) >= 30 and r.likes >= 5]
    by_length = sorted(detailed_reviews, key=lambda x: len(x.text), reverse=True)
    
    for review in by_length[:5]:
        if review.author not in seen_authors and len(selected_reviews) < 10:
            selected_reviews.append(review)
            seen_authors.add(review.author)
    
    # 3. Fill remaining slots with highest quality remaining reviews
    remaining = [r for r in all_reviews if r.author not in seen_authors]
    # Sort by combined score: likes + length bonus
    remaining_scored = sorted(remaining, 
                            key=lambda x: x.likes + (len(x.text.split()) / 10), 
                            reverse=True)
    
    for review in remaining_scored:
        if len(selected_reviews) >= 10:
            break
        selected_reviews.append(review)
        seen_authors.add(review.author)

    return selected_reviews


def parse_user_reviews(soup: BeautifulSoup, section_id: str) -> list[AlbumUserReview]:
    """Fallback function for basic review parsing (kept for compatibility)."""
    reviews = []
    for review in soup.select(f"#{section_id} .albumReviewRow"):
        try:
            author = review.select_one(".userReviewName a").text
            rating = None

            if rating_elem := review.select_one(".rating"):
                if rating_elem.text != "NR":
                    rating = int(rating_elem.text)

            text = review.select_one(".albumReviewText").text.strip()
            likes = 0

            if likes_elem := review.select_one(".review_likes"):
                # Try to get from <a> element first (some pages), then direct text
                if link_elem := likes_elem.select_one("a"):
                    likes = int(link_elem.text)
                else:
                    likes = int(likes_elem.text.strip())

            if len(text.split()) < 5:  # Skip very short reviews
                continue
                
            reviews.append(
                AlbumUserReview(
                    author=author,
                    rating=rating,
                    text=text[:2500],
                    likes=likes,
                )
            )
        except (AttributeError, ValueError):
            continue

    return reviews[:10]  # Limit to 10 reviews


def parse_buy_links(soup: BeautifulSoup) -> list[BuyLink]:
    """Extract buy links from the album page."""
    buy_links = []
    if buy_buttons := soup.select_one(".buyButtons"):
        for link in buy_buttons.select("a"):
            platform = link.get("title", "").strip()
            url = link.get("href", "").strip()
            if platform and url:
                buy_links.append(BuyLink(platform=platform, url=url))
    return buy_links


async def scrape_album(url: str, artist: str, title: str) -> Album:
    """
    Scrape album information from albumoftheyear.org.
    """
    try:
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")

        user_score = None
        if score_elem := soup.select_one(".albumUserScore a"):
            try:
                user_score = float(score_elem.get("title", 0))
            except (ValueError, TypeError):
                pass

        num_ratings = 0
        # Look for user ratings link pattern: <a href="...user-reviews/?type=ratings"><strong>10,522</strong>&nbsp;ratings</a>
        if ratings_elem := soup.select_one('a[href*="user-reviews/?type=ratings"] strong'):
            try:
                num_ratings = parse_number(ratings_elem.text)
            except ValueError:
                pass

        tracks_task = asyncio.create_task(asyncio.to_thread(parse_tracks, soup))
        critic_reviews_task = asyncio.create_task(
            asyncio.to_thread(parse_critic_reviews, soup)
        )
        buy_links_task = asyncio.create_task(asyncio.to_thread(parse_buy_links, soup))
        
        # Try to get full popular reviews, fallback to basic parsing
        try:
            popular_reviews = await parse_user_reviews_full(url)
        except Exception:
            # Fallback to basic review parsing
            popular_reviews = await asyncio.to_thread(parse_user_reviews, soup, "users")

        # wait for all parsing tasks to complete
        tracks, critic_reviews, buy_links = await asyncio.gather(
            tracks_task,
            critic_reviews_task,
            buy_links_task,
        )

        return Album(
            title=title,
            artist=artist,
            user_score=user_score,
            num_ratings=num_ratings,
            tracks=tracks,
            critic_reviews=critic_reviews,
            popular_reviews=popular_reviews,
            is_must_hear=bool(soup.select_one(".mustHearButton")),
            buy_links=buy_links,
        )

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error accessing album site: {str(e)}"
        )


async def get_similar_albums(url: str) -> list[Album]:
    """
    Get similar albums for a given album URL.
    """
    try:
        if not url.endswith("/"):
            url += "/"
        similar_url = f"{url}similar/"
        print(f"\nFetching similar albums from: {similar_url}")

        response_text = await asyncio.to_thread(
            lambda: scraper.get(similar_url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")
        similar_albums = []

        for album_block in soup.select(".albumBlock"):
            try:
                if album_link := album_block.select_one(".image a"):
                    album_url = f"{BASE_URL}{album_link['href']}"
                    artist = album_block.select_one(".artistTitle").text.strip()
                    title = album_block.select_one(".albumTitle").text.strip()

                    album = await scrape_album(album_url, artist, title)
                    similar_albums.append(album)
            except Exception as e:
                print(f"Error processing album: {str(e)}")
                continue

        return similar_albums

    except Exception as e:
        print(f"Error in get_similar_albums: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Error accessing similar albums page: {str(e)}"
        )


def extract_basic_info(
    soup: BeautifulSoup,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract basic user profile information"""
    about = soup.select_one(".aboutUser")
    about_text = about.text.strip() if about else None

    location = soup.select_one(".profileLocation")
    location_text = location.text.strip() if location else None

    member_since = None
    for div in soup.select(".rightBox div"):
        if div.text.startswith("Member since"):
            member_since = div.text.replace("Member since ", "").strip()
            break

    return about_text, location_text, member_since


def extract_rating_distribution(soup: BeautifulSoup) -> Dict[str, int]:
    """Extract rating distribution data"""
    dist = {}
    for row in soup.select(".dist .distRow"):
        label = row.select_one(".distLabel")
        count = row.select_one(".distCount")
        if label and count:
            rating_range = label.text.strip()
            count_text = count.text.strip()
            dist[rating_range] = parse_number(count_text)
    return dist


def extract_review(review: Tag) -> Optional[ProfileUserReview]:
    """Extract a single review from review element"""
    try:
        album_title = review.select_one(".albumTitle")
        album_artist = review.select_one(".artistTitle")
        rating = review.select_one(".rating")
        review_text = review.select_one(".albumReviewText")
        likes = review.select_one(".review_likes")
        timestamp = review.select_one(".actionContainer[title]")

        if all([album_title, album_artist, rating, review_text]):
            return ProfileUserReview(
                album_title=album_title.text.strip(),
                album_artist=album_artist.text.strip(),
                rating=parse_number(rating.text.strip()),
                review_text=review_text.text.strip(),
                likes=parse_number(likes.text.strip()) if likes else 0,
                timestamp=timestamp.text.strip() if timestamp else "",
            )
    except (AttributeError, ValueError):
        return None
    return None


def extract_social_links(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract social media links"""
    socials = {}
    for link in soup.select(".profileLink"):
        try:
            icon = link.select_one(".logo i")
            url = link.select_one("a")
            if icon and url:
                platform = icon["class"][1].replace("fa-", "")
                socials[platform] = url["href"]
        except (IndexError, KeyError):
            continue
    return socials


def extract_stats(soup: BeautifulSoup) -> Dict[str, int]:
    """Extract user statistics"""
    stats = {}
    stat_containers = soup.select(".profileStatContainer")
    if len(stat_containers) >= 4:
        for idx, key in enumerate(["ratings", "reviews", "lists", "followers"]):
            stat = stat_containers[idx].select_one(".profileStat")
            stats[key] = parse_number(stat.text.strip()) if stat else 0
    return stats


async def get_user_profile(username: str) -> Optional[UserProfile]:
    """
    Scrape user profile information from albumoftheyear.org

    Args:
        username: The username to fetch profile for

    Returns:
        UserProfile object containing the user's information

    Raises:
        HTTPException: If user doesn't exist or there's an error accessing the profile
    """
    url = f"{BASE_URL}/user/{username}/"

    try:
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )
        soup = BeautifulSoup(response_text, "html.parser")

        # check if user exists by looking for profile elements
        if not soup.select_one(".profileHeadLeft"):
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")

        about_text, location_text, member_since = extract_basic_info(soup)

        rating_dist = extract_rating_distribution(soup)
        reviews = [
            r
            for r in (
                extract_review(review) for review in soup.select(".albumReviewRow")[:5]
            )
            if r
        ]
        socials = extract_social_links(soup)
        stats = extract_stats(soup)

        favorite_albums = [
            title.text.strip()
            for album in soup.select("#favAlbumsBlock .albumBlock")
            if (title := album.select_one(".albumTitle"))
        ]

        return UserProfile(
            username=username,
            location=location_text,
            about=about_text,
            member_since=member_since,
            stats=stats,
            rating_distribution=rating_dist,
            favorite_albums=favorite_albums,
            recent_reviews=reviews,
            social_links=socials,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error accessing user profile: {str(e)}"
        )


async def get_artist_url(artist_name: str) -> Optional[str]:
    """
    Search for an artist and return their AOTY URL.
    """
    search_query = urllib.parse.quote(artist_name)
    url = f"{BASE_URL}/search/artists/?q={search_query}"

    try:
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")

        if artist_block := soup.select_one(".artistBlock"):
            if artist_link := artist_block.select_one("a"):
                return f"{BASE_URL}{artist_link['href']}"
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error accessing artist search: {str(e)}"
        )

    return None


async def scrape_artist(url: str, name: str) -> Artist:
    """
    Scrape artist information from albumoftheyear.org.
    """
    try:
        response_text = await asyncio.to_thread(
            lambda: scraper.get(url, headers=HEADERS).text
        )

        soup = BeautifulSoup(response_text, "html.parser")

        user_score = None
        if score_elem := soup.select_one(".artistUserScore"):
            try:
                user_score = float(score_elem.text.strip())
            except (ValueError, TypeError):
                pass

        num_ratings = 0
        # Extract number of ratings using pattern: <div class="text">Based on <strong>177,497</strong>&nbsp;ratings</div>
        text_divs = soup.select(".text")
        for text_elem in text_divs:
            if "ratings" in text_elem.text and "Based on" in text_elem.text:
                try:
                    if strong_elem := text_elem.select_one("strong"):
                        num_ratings = parse_number(strong_elem.text.strip())
                        break
                except (ValueError, AttributeError):
                    continue

        # Extract genre
        genre = None
        if genre_elem := soup.select_one(".artistGenre"):
            genre = genre_elem.text.strip()

        # Extract formed date
        formed = None
        if formed_elem := soup.select_one(".artistFormed"):
            formed = formed_elem.text.strip()

        # Extract location
        location = None
        if location_elem := soup.select_one(".artistLocation"):
            location = location_elem.text.strip()

        # Extract album titles
        albums = []
        for album_elem in soup.select(".albumBlock .albumTitle"):
            albums.append(album_elem.text.strip())

        return Artist(
            name=name,
            user_score=user_score,
            num_ratings=num_ratings,
            genre=genre,
            formed=formed,
            location=location,
            albums=albums,
        )

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error accessing artist page: {str(e)}"
        )
