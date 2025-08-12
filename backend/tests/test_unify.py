"""
Unit tests for unification and deduplication module
"""
import pytest
from app.tasks.unify import (
    make_canonical_id_enhanced,
    dedupe_songs,
    _normalize_artist_name,
    _normalize_track_title,
    _select_best_song,
    validate_canonical_ids,
    analyze_duplicates
)
from app.models import SongCore, SpotifyAttrs


@pytest.fixture
def sample_songs():
    """Create sample songs for testing"""
    return [
        SongCore(
            artist="Queen",
            title="Bohemian Rhapsody",
            playcount=150,
            source="lastfm"
        ),
        SongCore(
            artist="Led Zeppelin", 
            title="Stairway to Heaven",
            spotify_id="5CQ30WqJwcep0pYcV4AMNc",
            playcount=125,
            source="lastfm"
        ),
        SongCore(
            artist="Eagles",
            title="Hotel California",
            isrc="USEK10001234",
            playcount=100,
            source="lastfm"
        ),
        SongCore(
            artist="Pink Floyd",
            title="Comfortably Numb",
            playcount=90,
            source="lastfm"
        )
    ]


@pytest.fixture
def duplicate_songs():
    """Create songs with duplicates for testing"""
    return [
        SongCore(artist="Queen", title="Bohemian Rhapsody", playcount=150, isrc="test-isrc"),
        SongCore(artist="queen", title="bohemian rhapsody", playcount=100),  # Duplicate (case diff)
        SongCore(artist="The Beatles", title="Hey Jude", playcount=120),
        SongCore(artist="Beatles", title="Hey Jude", playcount=80),  # Duplicate (missing "The")
        SongCore(artist="Led Zeppelin", title="Stairway to Heaven (Remaster)", playcount=110),
        SongCore(artist="Led Zeppelin", title="Stairway to Heaven", playcount=95),  # Duplicate (remaster)
    ]


def test_make_canonical_id_with_isrc():
    """Test canonical ID generation with ISRC (highest priority)"""
    song = SongCore(
        artist="Test Artist",
        title="Test Title",
        isrc="USRC12345678",
        spotify_id="test-spotify"
    )
    
    canonical_id = make_canonical_id_enhanced(song)
    assert canonical_id == "isrc:USRC12345678"




def test_make_canonical_id_with_spotify_id():
    """Test canonical ID generation with Spotify ID (second priority)"""
    song = SongCore(
        artist="Test Artist",
        title="Test Title",
        spotify_id="5CQ30WqJwcep0pYcV4AMNc"
    )
    
    canonical_id = make_canonical_id_enhanced(song)
    assert canonical_id == "spotify:5CQ30WqJwcep0pYcV4AMNc"


def test_make_canonical_id_with_spotify_attrs():
    """Test canonical ID generation with Spotify attributes"""
    song = SongCore(artist="Test Artist", title="Test Title")
    spotify_attrs = SpotifyAttrs(artist_id="test-artist-id")
    
    canonical_id = make_canonical_id_enhanced(song, spotify_attrs)
    assert canonical_id == "spotify:test-artist-id"


def test_make_canonical_id_hash_fallback():
    """Test canonical ID generation falling back to hash"""
    song = SongCore(artist="Test Artist", title="Test Title")
    
    canonical_id = make_canonical_id_enhanced(song)
    assert canonical_id.startswith("hash:")
    assert len(canonical_id) == 21  # "hash:" + 16 character hex


def test_normalize_artist_name():
    """Test artist name normalization"""
    
    # Test "the" prefix removal
    assert _normalize_artist_name("The Beatles") == "beatles"
    assert _normalize_artist_name("the beatles") == "beatles"
    
    # Test featuring removal
    assert _normalize_artist_name("Artist feat. Other") == "artist"
    assert _normalize_artist_name("Artist ft. Other") == "artist"
    assert _normalize_artist_name("Artist featuring Other") == "artist"
    
    # Test whitespace normalization
    assert _normalize_artist_name("  Artist   Name  ") == "artist name"
    
    # Test basic case
    assert _normalize_artist_name("Queen") == "queen"


def test_normalize_track_title():
    """Test track title normalization"""
    
    # Test remaster/remix removal
    assert _normalize_track_title("Song (Remaster)") == "song"
    assert _normalize_track_title("Song (2010 Remaster)") == "song"
    assert _normalize_track_title("Song (Remix)") == "song"
    assert _normalize_track_title("Song (Live)") == "song"
    assert _normalize_track_title("Song (Acoustic)") == "song"
    
    # Test featuring removal
    assert _normalize_track_title("Song feat. Artist") == "song"
    assert _normalize_track_title("Song ft. Artist") == "song"
    
    # Test version removal
    assert _normalize_track_title("Song - Version 1") == "song"
    
    # Test basic case
    assert _normalize_track_title("Bohemian Rhapsody") == "bohemian rhapsody"


def test_select_best_song():
    """Test selection of best song from duplicates"""
    
    songs = [
        SongCore(artist="Queen", title="Test", playcount=50),
        SongCore(artist="Queen", title="Test", playcount=100, spotify_id="test-spotify"),
        SongCore(artist="Queen", title="Test", playcount=75, isrc="test-isrc")
    ]
    
    best = _select_best_song(songs)
    
    # Should select the one with ISRC (highest metadata score)
    assert best.isrc == "test-isrc"


def test_select_best_song_by_playcount():
    """Test selection prioritizing playcount when metadata is equal"""
    
    songs = [
        SongCore(artist="Queen", title="Test", playcount=50),
        SongCore(artist="Queen", title="Test", playcount=150),
        SongCore(artist="Queen", title="Test", playcount=100)
    ]
    
    best = _select_best_song(songs)
    assert best.playcount == 150


def test_dedupe_songs(sample_songs):
    """Test basic song deduplication"""
    
    # Add some duplicates
    duplicate = SongCore(
        artist="queen",  # Different case
        title="bohemian rhapsody",  # Different case
        playcount=75,
        source="test"
    )
    
    songs_with_dupe = sample_songs + [duplicate]
    
    deduped = dedupe_songs(songs_with_dupe, max_songs=10)
    
    # Should have removed the duplicate
    assert len(deduped) == len(sample_songs)
    
    # Should have canonical IDs assigned
    for song in deduped:
        assert hasattr(song, 'canonical_id')
        assert song.canonical_id is not None


def test_dedupe_songs_advanced_duplicates(duplicate_songs):
    """Test deduplication with complex duplicate scenarios"""
    
    deduped = dedupe_songs(duplicate_songs, max_songs=10)
    
    # Should identify and remove duplicates
    assert len(deduped) < len(duplicate_songs)
    
    # Check that we get unique canonical IDs
    canonical_ids = [song.canonical_id for song in deduped]
    assert len(canonical_ids) == len(set(canonical_ids))


def test_dedupe_songs_max_limit(sample_songs):
    """Test that deduplication respects max_songs limit"""
    
    deduped = dedupe_songs(sample_songs, max_songs=2)
    
    assert len(deduped) == 2


def test_validate_canonical_ids():
    """Test canonical ID validation"""
    
    songs = [
        SongCore(artist="Test1", title="Test1"),
        SongCore(artist="Test2", title="Test2"),
        SongCore(artist="Test3", title="Test3")
    ]
    
    # Add canonical IDs
    songs[0].canonical_id = "isrc:TEST123"
    songs[1].canonical_id = "spotify:test-spotify"
    songs[2].canonical_id = "hash:abcd1234"
    
    stats = validate_canonical_ids(songs)
    
    assert stats["total_songs"] == 3
    assert stats["isrc_count"] == 1
    assert stats["spotify_count"] == 1
    assert stats["hash_count"] == 1
    assert stats["unique_ids"] == 3
    assert stats["duplicates"] == 0


def test_validate_canonical_ids_with_duplicates():
    """Test canonical ID validation with duplicates"""
    
    songs = [
        SongCore(artist="Test1", title="Test1"),
        SongCore(artist="Test2", title="Test2")
    ]
    
    # Assign same canonical ID (duplicate)
    songs[0].canonical_id = "hash:same123"
    songs[1].canonical_id = "hash:same123"
    
    stats = validate_canonical_ids(songs)
    
    assert stats["total_songs"] == 2
    assert stats["unique_ids"] == 1
    assert stats["duplicates"] == 1


def test_analyze_duplicates():
    """Test duplicate analysis functionality"""
    
    songs = [
        SongCore(artist="Queen", title="Bohemian Rhapsody", playcount=150),
        SongCore(artist="queen", title="bohemian rhapsody", playcount=100),  # Duplicate
        SongCore(artist="Led Zeppelin", title="Stairway to Heaven", playcount=120),
        SongCore(artist="Beatles", title="Hey Jude", playcount=90)  # Unique
    ]
    
    duplicates = analyze_duplicates(songs)
    
    # Should find one duplicate group
    assert len(duplicates) == 1
    
    duplicate_group = duplicates[0]
    assert duplicate_group["count"] == 2
    assert len(duplicate_group["songs"]) == 2
    
    # Check that the songs in the group are the Queen tracks
    song_titles = [s["title"] for s in duplicate_group["songs"]]
    assert "Bohemian Rhapsody" in song_titles


def test_edge_case_empty_strings():
    """Test handling of empty strings in song data"""
    
    song = SongCore(
        artist="",
        title="Test Title",
        isrc="",
        spotify_id=""
    )
    
    canonical_id = make_canonical_id_enhanced(song)
    # Should fall back to hash since all IDs are empty
    assert canonical_id.startswith("hash:")


def test_edge_case_whitespace_only():
    """Test handling of whitespace-only strings"""
    
    song = SongCore(
        artist="  Test Artist  ",
        title="  Test Title  ",
        isrc="   "
    )
    
    canonical_id = make_canonical_id_enhanced(song)
    # Should fall back to hash since ISRC is whitespace-only
    assert canonical_id.startswith("hash:")


def test_normalization_special_characters():
    """Test normalization with special characters"""
    
    # Test with accented characters
    artist = _normalize_artist_name("Björk")
    title = _normalize_track_title("Café del Mar")
    
    # Should handle unicode normalization
    assert "bjork" in artist.lower()
    assert "cafe" in title.lower()


def test_dedupe_preserves_metadata():
    """Test that deduplication preserves important metadata"""
    
    songs = [
        SongCore(
            artist="Queen",
            title="Bohemian Rhapsody", 
            playcount=150,
            isrc="test-isrc",
            source="lastfm"
        )
    ]
    
    deduped = dedupe_songs(songs, max_songs=10)
    
    assert len(deduped) == 1
    song = deduped[0]
    
    # Should preserve original data
    assert song.artist == "Queen"  # Original case preserved 
    assert song.title == "Bohemian Rhapsody"
    assert song.playcount == 150
    assert song.isrc == "test-isrc"
    assert song.source == "lastfm"
    
    # Should have canonical ID
    assert hasattr(song, 'canonical_id')
    assert song.canonical_id.startswith("isrc:")