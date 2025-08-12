"""
Unit tests for Last.fm module with mocked HTTP requests
"""
import pytest
import respx
import httpx
from unittest.mock import patch, AsyncMock
from app.tasks.lastfm import LastfmClient, pull_top_tracks
from app.models import SongCore


@pytest.fixture
def lastfm_client():
    """Create a LastfmClient instance for testing"""
    with patch("app.tasks.lastfm.settings") as mock_settings:
        mock_settings.lastfm_api_key = "test_api_key"
        mock_settings.lastfm_username = "test_user"
        mock_settings.lastfm_rate_limit = 200
        mock_settings.data_dir = "/tmp"
        yield LastfmClient()


@pytest.fixture
def sample_lastfm_response():
    """Sample Last.fm API response"""
    return {
        "toptracks": {
            "track": [
                {
                    "name": "Bohemian Rhapsody",
                    "artist": {"name": "Queen"},
                    "playcount": "150"
                },
                {
                    "name": "Stairway to Heaven",
                    "artist": {"name": "Led Zeppelin"},
                    "playcount": "125"
                },
                {
                    "name": "Hotel California",
                    "artist": {"name": "Eagles"},
                    "playcount": "100"
                }
            ],
            "@attr": {
                "page": "1",
                "perPage": "3",
                "totalPages": "1",
                "total": "3"
            }
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_lastfm_get_top_tracks_success(lastfm_client, sample_lastfm_response):
    """Test successful Last.fm top tracks fetch"""
    
    # Mock the Last.fm API response
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=httpx.Response(200, json=sample_lastfm_response)
    )
    
    # Mock file writing
    with patch("builtins.open"), patch("json.dump"):
        tracks = await lastfm_client.get_top_tracks(max_tracks=3)
    
    assert len(tracks) == 3
    
    # Check first track
    first_track = tracks[0]
    assert isinstance(first_track, SongCore)
    assert first_track.artist == "queen"  # Normalized
    assert first_track.title == "bohemian rhapsody"  # Normalized
    assert first_track.playcount == 150
    assert first_track.source == "lastfm"
    
    # Check second track
    second_track = tracks[1]
    assert second_track.artist == "led zeppelin"
    assert second_track.title == "stairway to heaven"
    assert second_track.playcount == 125
    
    # Check third track
    third_track = tracks[2]
    assert third_track.artist == "eagles"
    assert third_track.title == "hotel california"
    assert third_track.playcount == 100


@respx.mock
@pytest.mark.asyncio
async def test_lastfm_rate_limiting(lastfm_client):
    """Test that rate limiting is applied"""
    import time
    
    response_data = {
        "toptracks": {
            "track": [{"name": "Test", "artist": {"name": "Test"}, "playcount": "1"}]
        }
    }
    
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    start_time = time.time()
    
    with patch("builtins.open"), patch("json.dump"):
        await lastfm_client.get_top_tracks(max_tracks=1)
    
    # Should have applied rate limiting delay
    elapsed = time.time() - start_time
    assert elapsed >= 0.25  # Minimum jitter time


@respx.mock
@pytest.mark.asyncio
async def test_lastfm_429_rate_limit_handling(lastfm_client):
    """Test handling of 429 rate limit responses"""
    
    # Mock 429 response followed by success
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"toptracks": {"track": []}})
        ]
    )
    
    with patch("builtins.open"), patch("json.dump"), patch("asyncio.sleep") as mock_sleep:
        tracks = await lastfm_client.get_top_tracks(max_tracks=1)
    
    # Should have slept due to rate limit
    mock_sleep.assert_called()
    assert len(tracks) == 0


@respx.mock
@pytest.mark.asyncio
async def test_lastfm_empty_response(lastfm_client):
    """Test handling of empty Last.fm response"""
    
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=httpx.Response(200, json={"toptracks": {"track": []}})
    )
    
    with patch("builtins.open"), patch("json.dump"):
        tracks = await lastfm_client.get_top_tracks(max_tracks=10)
    
    assert len(tracks) == 0


@respx.mock
@pytest.mark.asyncio
async def test_lastfm_malformed_response(lastfm_client):
    """Test handling of malformed Last.fm response"""
    
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=httpx.Response(200, json={"error": "Invalid API key"})
    )
    
    with patch("builtins.open"), patch("json.dump"):
        tracks = await lastfm_client.get_top_tracks(max_tracks=10)
    
    assert len(tracks) == 0


@pytest.mark.asyncio
async def test_lastfm_track_parsing():
    """Test track data parsing logic"""
    client = LastfmClient()
    
    # Test with artist as object
    track_data = {
        "name": "Test Track",
        "artist": {"name": "Test Artist"},
        "playcount": "50"
    }
    
    track = client._parse_track(track_data)
    assert track is not None
    assert track.title == "test track"
    assert track.artist == "test artist"
    assert track.playcount == 50
    
    # Test with artist as string
    track_data_str = {
        "name": "Another Track",
        "artist": "Another Artist",
        "playcount": "75"
    }
    
    track = client._parse_track(track_data_str)
    assert track is not None
    assert track.title == "another track"
    assert track.artist == "another artist"
    assert track.playcount == 75
    
    # Test with missing data
    invalid_data = {"name": "", "artist": ""}
    track = client._parse_track(invalid_data)
    assert track is None


@pytest.mark.asyncio
async def test_lastfm_recent_tracks(lastfm_client):
    """Test recent tracks functionality"""
    
    recent_response = {
        "recenttracks": {
            "track": [
                {
                    "name": "Recent Track 1",
                    "artist": {"#text": "Recent Artist 1"},
                    "date": {"uts": "1234567890"}
                },
                {
                    "name": "Recent Track 2", 
                    "artist": {"#text": "Recent Artist 2"},
                    "@attr": {"nowplaying": "true"}  # Should be skipped
                },
                {
                    "name": "Recent Track 3",
                    "artist": {"#text": "Recent Artist 3"},
                    "date": {"uts": "1234567880"}
                }
            ]
        }
    }
    
    with respx.mock:
        respx.get("https://ws.audioscrobbler.com/2.0/").mock(
            return_value=httpx.Response(200, json=recent_response)
        )
        
        with patch("builtins.open"), patch("json.dump"):
            tracks = await lastfm_client.get_recent_tracks(max_tracks=10)
    
    # Should have 2 tracks (skipping the "now playing" one)
    assert len(tracks) == 2
    assert tracks[0].title == "recent track 1"
    assert tracks[1].title == "recent track 3"


@pytest.mark.asyncio
async def test_pull_top_tracks_function():
    """Test the module-level pull_top_tracks function"""
    
    with patch("app.tasks.lastfm.LastfmClient") as MockClient:
        mock_instance = AsyncMock()
        mock_tracks = [SongCore(artist="test", title="test", source="lastfm")]
        mock_instance.get_top_tracks.return_value = mock_tracks
        MockClient.return_value = mock_instance
        
        result = await pull_top_tracks(100)
        
        assert result == mock_tracks
        mock_instance.get_top_tracks.assert_called_once_with(100)


@pytest.mark.asyncio 
async def test_save_raw_data(lastfm_client):
    """Test raw data saving functionality"""
    
    tracks = [SongCore(artist="test", title="test", source="lastfm")]
    
    with patch("builtins.open", create=True) as mock_open, \\
         patch("json.dump") as mock_dump, \\
         patch("os.path.join", return_value="/tmp/test.json"):
        
        await lastfm_client._save_raw_data(tracks, "test")
        
        mock_open.assert_called_once()
        mock_dump.assert_called_once()
        
        # Check that the data structure is correct
        call_args = mock_dump.call_args[0]
        data = call_args[0]
        
        assert "metadata" in data
        assert "tracks" in data
        assert data["metadata"]["username"] == "test_user"
        assert data["metadata"]["track_count"] == 1
        assert len(data["tracks"]) == 1