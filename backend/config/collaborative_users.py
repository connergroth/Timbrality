"""
Configuration file for collaborative filtering users.
Add Last.fm usernames here to fetch their data for collaborative filtering.
"""

# List of Last.fm usernames to fetch data from for collaborative filtering
# Add real Last.fm usernames here - these should be users with public profiles
# and interesting music taste that you want to analyze

COLLABORATIVE_USERS = [
    # Replace these with actual Last.fm usernames
    'connergroth',  # Your own Last.fm username
    
    # Example users (replace with real usernames):
    # 'Giannistaz',
    # 'indie_lover',
    # 'jazz_enthusiast',
    # 'rock_fanatic',
    # 'electronic_music_fan',
    
    # You can also add users from specific communities or with similar taste:
    # 'pitchfork_reader',
    # 'npr_music_fan',
    # 'kexp_listener',
]

# Configuration for data fetching limits
FETCH_LIMITS = {
    'tracks_per_user': 100,    # Number of top tracks to fetch per user
    'albums_per_user': 50,     # Number of top albums to fetch per user
    'artists_per_user': 30,    # Number of top artists to fetch per user
}

# Rate limiting settings (to avoid hitting Last.fm API limits)
RATE_LIMITING = {
    'delay_between_users': 1.0,  # Seconds to wait between processing users
    'max_requests_per_minute': 30,  # Last.fm allows 30 requests per minute
}

# Data quality filters
DATA_FILTERS = {
    'min_playcount': 1,        # Minimum playcount to consider a track/album/artist
    'min_tracks_per_user': 10, # Minimum tracks a user must have to be included
    'exclude_users_with_private_profiles': True,
}

# Collaborative filtering algorithm settings
ALGORITHM_SETTINGS = {
    'similarity_threshold': 0.1,  # Minimum similarity score to consider users similar
    'max_similar_users': 50,      # Maximum number of similar users to consider
    'min_shared_items': 5,        # Minimum shared tracks/albums to calculate similarity
}

def get_collaborative_users():
    """Get the list of collaborative users"""
    return COLLABORATIVE_USERS.copy()

def get_fetch_limits():
    """Get the fetch limits configuration"""
    return FETCH_LIMITS.copy()

def get_rate_limiting():
    """Get the rate limiting configuration"""
    return RATE_LIMITING.copy()

def get_data_filters():
    """Get the data quality filters"""
    return DATA_FILTERS.copy()

def get_algorithm_settings():
    """Get the algorithm settings"""
    return ALGORITHM_SETTINGS.copy()

def add_collaborative_user(username: str):
    """Add a new user to the collaborative users list"""
    if username not in COLLABORATIVE_USERS:
        COLLABORATIVE_USERS.append(username)
        return True
    return False

def remove_collaborative_user(username: str):
    """Remove a user from the collaborative users list"""
    if username in COLLABORATIVE_USERS:
        COLLABORATIVE_USERS.remove(username)
        return True
    return False

def validate_username(username: str) -> bool:
    """Basic validation for Last.fm usernames"""
    if not username or len(username) < 1:
        return False
    
    # Last.fm usernames are typically alphanumeric with underscores and hyphens
    import re
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, username))




