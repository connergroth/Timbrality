"""
Configuration for Tensoe Ingestion Pipeline

Now using centralized settings from config.settings
"""
import os
try:
    from config.settings import settings
except ImportError:
    # Fallback to environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    class FallbackSettings:
        def __init__(self):
            self.spotify = type('obj', (object,), {
                'client_id': os.getenv('SPOTIFY_CLIENT_ID'),
                'client_secret': os.getenv('SPOTIFY_CLIENT_SECRET'),
                'request_timeout': 30
            })()
            self.lastfm = type('obj', (object,), {
                'api_key': os.getenv('LASTFM_API_KEY'),
                'api_secret': os.getenv('LASTFM_API_SECRET')
            })()
            self.database = type('obj', (object,), {
                'supabase_url': os.getenv('SUPABASE_URL'),
                'supabase_anon_key': os.getenv('SUPABASE_ANON_KEY')
            })()
            self.ingestion = type('obj', (object,), {
                'batch_size': 50,
                'max_concurrent': 10,
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'max_genres_per_track': 10,
                'max_moods_per_track': 15
            })()
            self.logging = type('obj', (object,), {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            })()
        
        def get_redis_url(self):
            return os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    get_settings = lambda: FallbackSettings()

# Get centralized settings
settings = get_settings()

# API Configuration
SPOTIFY_CLIENT_ID = settings.spotify.client_id
SPOTIFY_CLIENT_SECRET = settings.spotify.client_secret

LASTFM_API_KEY = settings.lastfm.api_key
LASTFM_API_SECRET = settings.lastfm.api_secret

SUPABASE_URL = settings.database.supabase_url
SUPABASE_ANON_KEY = settings.database.supabase_anon_key

# Rate Limiting Configuration
SPOTIFY_RATE_LIMIT = 100  # requests per minute
LASTFM_RATE_LIMIT = 200   # requests per minute
AOTY_RATE_LIMIT = 30      # requests per minute

# Batch Processing Configuration  
DEFAULT_BATCH_SIZE = settings.ingestion.batch_size
MAX_PARALLEL_REQUESTS = settings.ingestion.max_concurrent
REQUEST_TIMEOUT = settings.spotify.request_timeout  # seconds

# Data Processing Configuration
MAX_GENRES_PER_TRACK = settings.ingestion.max_genres_per_track
MAX_MOODS_PER_TRACK = settings.ingestion.max_moods_per_track
MAX_TAGS_FROM_LASTFM = 10

# Database Configuration
DB_BATCH_SIZE = settings.ingestion.batch_size
DB_RETRY_ATTEMPTS = settings.ingestion.retry_attempts
DB_RETRY_DELAY = settings.ingestion.retry_delay  # seconds

# Logging Configuration
LOG_LEVEL = settings.logging.level
LOG_FORMAT = settings.logging.format

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), 'exports')

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, EXPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Validation Configuration
REQUIRED_TRACK_FIELDS = ['track_id', 'title', 'artist']
OPTIONAL_TRACK_FIELDS = [
    'album', 'release_date', 'duration_ms', 'popularity', 
    'genres', 'moods', 'aoty_score', 'explicit'
]

# Export Configuration
CSV_EXPORT_FIELDS = [
    'id', 'title', 'artist', 'album', 'release_date', 
    'duration_ms', 'popularity', 'genres', 'moods', 
    'aoty_score', 'explicit'
] 