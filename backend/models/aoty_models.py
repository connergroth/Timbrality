from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class SearchResult(BaseModel):
    title: str
    artist: str
    url: str
    year: Optional[int] = None
    score: Optional[float] = None
    user_score: Optional[float] = None
    type: str = "album"  # "album", "artist", "song"

class Album(BaseModel):
    title: str
    artist: str
    url: str
    year: Optional[int] = None
    genres: Optional[List[str]] = []
    rating: Optional[float] = None
    user_rating: Optional[float] = None
    ratings_count: Optional[int] = None
    user_ratings_count: Optional[int] = None
    critics_score: Optional[float] = None
    user_score: Optional[float] = None
    cover_url: Optional[str] = None
    description: Optional[str] = None
    tracks: Optional[List['Track']] = []
    similar_albums: Optional[List[str]] = []
    buy_links: Optional[List['BuyLink']] = []

class Track(BaseModel):
    title: str
    position: Optional[int] = None
    duration: Optional[str] = None
    rating: Optional[float] = None
    url: Optional[str] = None

class CriticReview(BaseModel):
    critic_name: str
    publication: str
    score: Optional[float] = None
    review_text: Optional[str] = None
    url: Optional[str] = None
    date: Optional[datetime] = None

class AlbumUserReview(BaseModel):
    username: str
    rating: float
    review_text: Optional[str] = None
    date: Optional[datetime] = None
    helpful_count: Optional[int] = None

class ProfileUserReview(BaseModel):
    album_title: str
    artist: str
    rating: float
    review_text: Optional[str] = None
    date: Optional[datetime] = None
    album_url: Optional[str] = None

class UserProfile(BaseModel):
    username: str
    url: str
    profile_picture: Optional[str] = None
    location: Optional[str] = None
    member_since: Optional[datetime] = None
    reviews_count: Optional[int] = None
    ratings_count: Optional[int] = None
    lists_count: Optional[int] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    favorite_genres: Optional[List[str]] = []

class BuyLink(BaseModel):
    platform: str
    url: str
    price: Optional[str] = None

class AlbumMetadata(BaseModel):
    aoty_id: Optional[str] = None
    spotify_id: Optional[str] = None
    lastfm_url: Optional[str] = None
    musicbrainz_id: Optional[str] = None
    discogs_url: Optional[str] = None
    bandcamp_url: Optional[str] = None

class UserStats(BaseModel):
    ratings: int = 0
    reviews: int = 0
    lists: int = 0
    followers: int = 0

# Update forward references
Album.model_rebuild()