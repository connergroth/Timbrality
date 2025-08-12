"""
Unit tests for Spotify module with mocked HTTP requests
"""
import pytest
import respx
import httpx
import json
from unittest.mock import patch, AsyncMock
from app.tasks.spotify import SpotifyClient, search_track, enrich_tracks
from app.models import SongCore, SpotifyAttrs


@pytest.fixture
def spotify_client():
    """Create a SpotifyClient instance for testing"""
    with patch("app.tasks.spotify.settings") as mock_settings:
        mock_settings.spotify_client_id = "test_client_id"
        mock_settings.spotify_client_secret = "test_client_secret"
        mock_settings.spotify_rate_limit = 100
        mock_settings.request_timeout = 30
        mock_settings.redis_url = "redis://localhost:6379/0"
        yield SpotifyClient()


@pytest.fixture
def sample_token_response():
    """Sample Spotify OAuth token response"""
    return {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600
    }


@pytest.fixture
def sample_search_response():
    """Sample Spotify search API response"""
    return {
        "tracks": {
            "items": [
                {
                    "id": "4u7EnebtmKWzUH433cf5Qv",
                    "name": "Bohemian Rhapsody",
                    "artists": [{"id": "1dfeR4HaWDbWqFHLkxsg1d", "name": "Queen"}],
                    "album": {
                        "id": "6i6folBtxKV28WX3msQ4FE",
                        "name": "A Night at the Opera",
                        "release_date": "1975-11-21"
                    },
                    "duration_ms": 355000,
                    "popularity": 85,
                    "explicit": False
                }
            ]
        }
    }




@respx.mock
@pytest.mark.asyncio
async def test_spotify_get_access_token(spotify_client, sample_token_response):
    """Test OAuth token acquisition"""
    
    # Mock Redis to return None (no cached token)
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock token endpoint
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        token = await spotify_client._get_access_token()
        
        assert token == "test_access_token"
        assert spotify_client._access_token == "test_access_token"


@respx.mock
@pytest.mark.asyncio
async def test_spotify_search_track_success(
    spotify_client, 
    sample_token_response, 
    sample_search_response
):
    """Test successful track search"""
    
    # Mock Redis (no cache)
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock token endpoint
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        # Mock search endpoint
        respx.get("https://api.spotify.com/v1/search").mock(
            return_value=httpx.Response(200, json=sample_search_response)
        )
        
        attrs = await spotify_client.search_track("Queen", "Bohemian Rhapsody")
        
        assert attrs is not None
        assert isinstance(attrs, SpotifyAttrs)
        assert attrs.duration_ms == 355000
        assert attrs.popularity == 85
        assert attrs.album_id == "6i6folBtxKV28WX3msQ4FE"
        assert attrs.artist_id == "1dfeR4HaWDbWqFHLkxsg1d"
        assert attrs.album_name == "A Night at the Opera"
        assert attrs.release_date == "1975-11-21"
        assert attrs.explicit is False


@respx.mock
@pytest.mark.asyncio
async def test_spotify_search_track_not_found(spotify_client, sample_token_response):
    """Test track not found scenario"""
    
    empty_response = {"tracks": {"items": []}}
    
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        respx.get("https://api.spotify.com/v1/search").mock(
            return_value=httpx.Response(200, json=empty_response)
        )
        
        attrs = await spotify_client.search_track("Unknown Artist", "Unknown Track")
        
        assert attrs is None


@respx.mock
@pytest.mark.asyncio
async def test_spotify_cached_result(spotify_client):
    """Test retrieval of cached search results"""
    
    cached_attrs = SpotifyAttrs(
        duration_ms=300000,
        popularity=75,
        album_id="cached_album",
        artist_id="cached_artist"
    )
    
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = cached_attrs.json()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        attrs = await spotify_client.search_track("Cached Artist", "Cached Track")
        
        assert attrs is not None
        assert attrs.duration_ms == 300000
        assert attrs.popularity == 75
        assert attrs.album_id == "cached_album"


@respx.mock
@pytest.mark.asyncio
async def test_spotify_rate_limit_handling(spotify_client, sample_token_response):
    """Test 429 rate limit handling with exponential backoff"""
    
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        # Mock 429 response with Retry-After header
        respx.get("https://api.spotify.com/v1/search").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "1"}),
                httpx.Response(200, json={"tracks": {"items": []}})
            ]
        )
        
        with patch("asyncio.sleep") as mock_sleep:
            attrs = await spotify_client.search_track("Test", "Test")
            
            # Should have slept due to rate limit
            mock_sleep.assert_called()
            assert attrs is None  # Empty response


@respx.mock
@pytest.mark.asyncio
async def test_spotify_token_refresh_on_401(spotify_client, sample_token_response):
    """Test token refresh when receiving 401 unauthorized"""
    
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock token endpoint (called twice due to refresh)
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        # Mock 401 response followed by success
        respx.get("https://api.spotify.com/v1/search").mock(
            side_effect=[
                httpx.Response(401),
                httpx.Response(200, json={"tracks": {"items": []}})
            ]
        )
        
        attrs = await spotify_client.search_track("Test", "Test")
        
        # Should have attempted token refresh
        assert len(respx.calls) >= 3  # Token call + 401 + token call + retry


@pytest.mark.asyncio
async def test_spotify_enrich_tracks_batch():
    """Test batch enrichment of multiple tracks"""
    
    songs = [
        SongCore(artist="Queen", title="Bohemian Rhapsody", source="test"),
        SongCore(artist="Beatles", title="Hey Jude", source="test"),
        SongCore(artist="Led Zeppelin", title="Stairway to Heaven", source="test")
    ]
    
    mock_attrs = SpotifyAttrs(duration_ms=300000, popularity=80)
    
    with patch("app.tasks.spotify.SpotifyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.search_track.return_value = mock_attrs
        MockClient.return_value = mock_instance
        
        client = SpotifyClient()
        client.search_track = mock_instance.search_track
        
        results = await client.enrich_tracks_batch(songs, batch_size=2)
        
        assert len(results) == 3
        for song, attrs in results:
            assert isinstance(song, SongCore)
            assert attrs == mock_attrs


@pytest.mark.asyncio
async def test_spotify_enrich_tracks_with_exceptions():
    """Test batch enrichment handling exceptions"""
    
    songs = [
        SongCore(artist="Good Artist", title="Good Track", source="test"),
        SongCore(artist="Bad Artist", title="Bad Track", source="test")
    ]
    
    mock_attrs = SpotifyAttrs(duration_ms=300000, popularity=80)
    
    async def mock_search_track(artist, title):
        if artist == "Bad Artist":
            raise Exception("API Error")
        return mock_attrs
    
    with patch("app.tasks.spotify.SpotifyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.search_track.side_effect = mock_search_track
        MockClient.return_value = mock_instance
        
        client = SpotifyClient()
        client.search_track = mock_instance.search_track
        
        results = await client.enrich_tracks_batch(songs, batch_size=2)
        
        assert len(results) == 2
        assert results[0][1] == mock_attrs  # Successful
        assert results[1][1] is None  # Failed


@pytest.mark.asyncio
async def test_search_track_function():
    """Test the module-level search_track function"""
    
    mock_attrs = SpotifyAttrs(duration_ms=300000, popularity=80)
    
    with patch("app.tasks.spotify.SpotifyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.search_track.return_value = mock_attrs
        MockClient.return_value = mock_instance
        
        result = await search_track("Test Artist", "Test Track")
        
        assert result == mock_attrs
        mock_instance.search_track.assert_called_once_with("Test Artist", "Test Track")


@pytest.mark.asyncio
async def test_enrich_tracks_function():
    """Test the module-level enrich_tracks function"""
    
    songs = [SongCore(artist="test", title="test", source="test")]
    mock_results = [(songs[0], SpotifyAttrs(duration_ms=300000))]
    
    with patch("app.tasks.spotify.SpotifyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.enrich_tracks_batch.return_value = mock_results
        MockClient.return_value = mock_instance
        
        result = await enrich_tracks(songs)
        
        assert result == mock_results
        mock_instance.enrich_tracks_batch.assert_called_once()


@respx.mock
@pytest.mark.asyncio
async def test_spotify_redis_unavailable(spotify_client, sample_token_response, sample_search_response):
    """Test graceful handling when Redis is unavailable"""
    
    # Mock Redis to fail connection
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.side_effect = Exception("Redis connection failed")
        mock_redis.return_value = mock_redis_instance
        
        respx.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(200, json=sample_token_response)
        )
        
        respx.get("https://api.spotify.com/v1/search").mock(
            return_value=httpx.Response(200, json=sample_search_response)
        )
        
        # Should still work without Redis
        attrs = await spotify_client.search_track("Queen", "Bohemian Rhapsody")
        assert attrs is not None