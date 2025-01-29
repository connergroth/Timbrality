from app.models.database import SessionLocal  # For database session
from app.models.album import Album  # Model imports
from app.models.artist import Artist
from app.models.track import Track
from app.models.user import User
from app.models.track_listening_history import TrackListeningHistory
from app.models.recommendation import Recommendation
from app.models.album_compatibility import AlbumCompatability
from app.models.artist_compatibility import ArtistCompatibility
from app.models.track_compatibility import TrackCompatibility
from app.models.compatibility import Compatibility
from sqlalchemy.sql import func  # For SQLAlchemy functions like func.now()
import pandas as pd  # For DataFrame manipulation 

def insert_data_to_db(data, table_name):
    """
    Insert data into the PostgreSQL database based on the table name.

    :param data: A DataFrame containing the data to insert.
    :param table_name: The name of the table where data will be inserted.
    """
    session = SessionLocal()

    if table_name == 'albums':
        for _, row in data.iterrows():
            session.add(Album(
                title=row['title'],
                artist=row['artist'],
                release_date=row['release_date'],
                genre=row['genre'],
                aoty_score=row['aoty_score'],
                cover_url=row['cover_url']
            ))
    elif table_name == 'artists':
        for _, row in data.iterrows():
            session.add(Artist(
                name=row['name'],
                genre=row['genre'],
                popularity=row['popularity'],
                aoty_score=row['aoty_score']
            ))
    elif table_name == 'tracks':
        for _, row in data.iterrows():
            session.add(Track(
                title=row['title'],
                artist=row['artist'],
                album=row['album'],
                genre=row['genre'],
                popularity=row['popularity'],
                aoty_score=row['aoty_score'],
                audio_features=row['audio_features'],  # JSON field
                cover_url=row['cover_url']
            ))
    elif table_name == 'users':
        for _, row in data.iterrows():
            session.add(User(
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash']
            ))
    elif table_name == 'track_listening_histories':
        for _, row in data.iterrows():
            session.add(TrackListeningHistory(
                user_id=row['user_id'],
                track_id=row['track_id'],
                play_count=row['play_count'],
                timestamp=func.now()
            ))
    elif table_name == 'recommendations':
        for _, row in data.iterrows():
            session.add(Recommendation(
                user_id=row['user_id'],
                track_id=row['track_id'],
                album=row['album'],
                recommendation_score=row['recommendation_score']
            ))
    elif table_name == 'album_compatibilities':
        for _, row in data.iterrows():
            session.add(AlbumCompatability(
                user_id_1=row['user_id_1'],
                user_id_2=row['user_id_2'],
                album_id=row['album_id'],
                compatibility_score=row['compatibility_score']
            ))
    elif table_name == 'artist_compatibilities':
        for _, row in data.iterrows():
            session.add(ArtistCompatibility(
                user_id_1=row['user_id_1'],
                user_id_2=row['user_id_2'],
                artist_id=row['artist_id'],
                compatibility_score=row['compatibility_score']
            ))
    elif table_name == 'track_compatibilities':
        for _, row in data.iterrows():
            session.add(TrackCompatibility(
                user_id_1=row['user_id_1'],
                user_id_2=row['user_id_2'],
                track_id=row['track_id'],
                compatibility_score=row['compatibility_score']
            ))
    elif table_name == 'compatibilities':
        for _, row in data.iterrows():
            session.add(Compatibility(
                user_id_1=row['user_id_1'],
                user_id_2=row['user_id_2'],
                compatibility_score=row['compatibility_score']
            ))

    session.commit()
    session.close()
