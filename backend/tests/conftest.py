"""
Pytest configuration and shared fixtures
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import os
import tempfile


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch("app.config.settings") as mock:
        mock.lastfm_api_key = "test_lastfm_key"
        mock.lastfm_username = "test_user"
        mock.spotify_client_id = "test_spotify_id"
        mock.spotify_client_secret = "test_spotify_secret"
        mock.supabase_url = "https://test.supabase.co"
        mock.supabase_service_role_key = "test_supabase_key"
        mock.redis_url = "redis://localhost:6379/0"
        mock.max_songs = 10000
        mock.scrape_concurrency = 4
        mock.scrape_delay_sec = 2.0
        mock.batch_size = 50
        mock.db_batch_size = 2000
        mock.spotify_rate_limit = 100
        mock.lastfm_rate_limit = 200
        mock.aoty_rate_limit = 30
        mock.request_timeout = 30
        mock.max_concurrent_requests = 10
        mock.min_aoty_rating_count = 5
        mock.log_level = "INFO"
        mock.data_dir = "/tmp/test_data"
        mock.logs_dir = "/tmp/test_logs"
        mock.exports_dir = "/tmp/test_exports"
        yield mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing"""
    mock_client = MagicMock()
    
    # Mock table operations
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[])
    
    mock_client.from_.return_value = mock_table
    mock_client.table.return_value = mock_table
    mock_client.rpc.return_value = mock_table
    
    return mock_client


@pytest.fixture
def sample_song_data():
    """Sample song data for testing"""
    return {
        "artist": "Queen",
        "title": "Bohemian Rhapsody", 
        "mbid": "b1e26560-60e5-4236-bbdb-9aa5a8d5ee19",
        "playcount": 150,
        "source": "lastfm"
    }


@pytest.fixture
def sample_spotify_data():
    """Sample Spotify API response data"""
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


@pytest.fixture
def sample_audio_features():
    """Sample Spotify audio features data"""
    return {
        "danceability": 0.3,
        "energy": 0.619,
        "key": 7,
        "loudness": -10.559,
        "mode": 0,
        "speechiness": 0.0594,
        "acousticness": 0.145,
        "instrumentalness": 0.0000184,
        "liveness": 0.158,
        "valence": 0.279,
        "tempo": 72.038,
        "time_signature": 4
    }


@pytest.fixture
def sample_lastfm_data():
    """Sample Last.fm API response data"""
    return {
        "toptracks": {
            "track": [
                {
                    "name": "Bohemian Rhapsody",
                    "artist": {"name": "Queen"},
                    "playcount": "150",
                    "mbid": "b1e26560-60e5-4236-bbdb-9aa5a8d5ee19"
                },
                {
                    "name": "Stairway to Heaven",
                    "artist": {"name": "Led Zeppelin"},
                    "playcount": "125",
                    "mbid": ""
                }
            ]
        }
    }


@pytest.fixture
def mock_aoty_html():
    """Sample AOTY HTML for testing scraping"""
    return """
    <html>
        <body>
            <div class="albumUserScore">84</div>
            <div class="numReviews">1,234 ratings</div>
            <div class="genre">
                <a href="/genre/rock">Rock</a>
                <a href="/genre/progressive-rock">Progressive Rock</a>
            </div>
            <h1 class="albumTitle">A Night at the Opera</h1>
        </body>
    </html>
    """


@pytest.fixture(autouse=True)
def setup_test_environment(mock_settings, temp_dir):
    """Automatically set up test environment for all tests"""
    # Create test directories
    os.makedirs(f"{temp_dir}/data", exist_ok=True)
    os.makedirs(f"{temp_dir}/logs", exist_ok=True)
    os.makedirs(f"{temp_dir}/exports", exist_ok=True)
    
    # Update mock settings with temp dir
    mock_settings.data_dir = f"{temp_dir}/data"
    mock_settings.logs_dir = f"{temp_dir}/logs"
    mock_settings.exports_dir = f"{temp_dir}/exports"


@pytest.fixture
def mock_db_client():
    """Mock database client for testing"""
    with patch("app.db.db_client") as mock:
        mock.health_check.return_value = True
        mock.get_coverage_stats.return_value = {
            "total_songs": 1000,
            "spotify_coverage": 85.5,
            "lastfm_coverage": 100.0,
            "aoty_coverage": 45.2
        }
        mock.get_total_songs.return_value = 1000
        mock.song_exists.return_value = False
        mock.get_song_by_canonical_id.return_value = None
        mock.upsert_songs.return_value = 100
        mock.upsert_spotify_attrs.return_value = 85
        mock.upsert_lastfm_stats.return_value = 100
        mock.upsert_aoty_attrs.return_value = 45
        mock.validate_counts.return_value = {
            "songs": 1000,
            "spotify_attrs": 850,
            "lastfm_stats": 1000,
            "aoty_attrs": 450
        }
        yield mock


# Async test utilities

async def async_return(value):
    """Helper to return a value asynchronously"""
    return value


def async_mock(*args, **kwargs):
    """Create an async mock"""
    mock = AsyncMock(*args, **kwargs)
    return mock


# Test markers

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Skip markers for missing dependencies

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark all async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Mark tests based on file names
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)