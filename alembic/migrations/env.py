from logging.config import fileConfig
from sqlalchemy import create_engine
from sqlalchemy import pool
from alembic import context

# Import SQLAlchemy Base and models
from app.models.database import Base
from app.models.user import User  # Import all your models
from app.models.track import Track
from app.models.album import Album
from app.models.artist import Artist
from app.models.recommendation import Recommendation
from app.models.playlist import Playlist
from app.models.compatibility import Compatibility
from app.models.album_compatibility import AlbumCompatability
from app.models.artist_compatibility import ArtistCompatibility
from app.models.track_compatibility import TrackCompatibility
from app.models.listening_history import ListeningHistory

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the metadata to track models
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_engine('postgresql://postgres:postgronner34@localhost:5432/Sonance')

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()