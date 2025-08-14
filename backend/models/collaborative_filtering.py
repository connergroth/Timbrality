"""
SQLAlchemy models for collaborative filtering tables
"""
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Float, Text, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid


class LastfmUser(Base):
    """Last.fm user profiles for collaborative filtering"""
    __tablename__ = "lastfm_users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lastfm_username = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    real_name = Column(String)
    country = Column(String)
    age = Column(Integer)
    gender = Column(String)
    subscriber = Column(Boolean, default=False)
    playcount_total = Column(Integer, default=0)
    playlists_count = Column(Integer, default=0)
    registered_date = Column(DateTime)
    last_updated = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    data_fetch_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    track_interactions = relationship("UserTrackInteraction", back_populates="user")
    album_interactions = relationship("UserAlbumInteraction", back_populates="user")
    artist_interactions = relationship("UserArtistInteraction", back_populates="user")


class UserTrackInteraction(Base):
    """User-track interactions (plays, loves, skips, etc.)"""
    __tablename__ = "user_track_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lastfm_user_id = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False)
    interaction_type = Column(String, default="play")  # 'play', 'love', 'skip', 'repeat'
    play_count = Column(Integer, default=0)
    user_loved = Column(Boolean, default=False)
    last_played = Column(DateTime)
    tags = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("LastfmUser", back_populates="track_interactions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('lastfm_user_id', 'track_id', name='user_track_interactions_unique'),
    )


class UserAlbumInteraction(Base):
    """User-album interactions"""
    __tablename__ = "user_album_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lastfm_user_id = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    album_title = Column(String, nullable=False)
    album_artist = Column(String, nullable=False)
    play_count = Column(Integer, default=0)
    user_loved = Column(Boolean, default=False)
    last_played = Column(DateTime)
    tags = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("LastfmUser", back_populates="album_interactions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('lastfm_user_id', 'album_title', 'album_artist', name='user_album_interactions_unique'),
    )


class UserArtistInteraction(Base):
    """User-artist interactions"""
    __tablename__ = "user_artist_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lastfm_user_id = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    artist_name = Column(String, nullable=False)
    play_count = Column(Integer, default=0)
    user_loved = Column(Boolean, default=False)
    last_played = Column(DateTime)
    tags = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("LastfmUser", back_populates="artist_interactions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('lastfm_user_id', 'artist_name', name='user_artist_interactions_unique'),
    )


class UserSimilarity(Base):
    """Pre-calculated user similarity scores for collaborative filtering"""
    __tablename__ = "user_similarities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id_1 = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    user_id_2 = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    similarity_type = Column(String, default="cosine")  # 'cosine', 'pearson', 'jaccard'
    shared_tracks_count = Column(Integer, default=0)
    shared_albums_count = Column(Integer, default=0)
    shared_artists_count = Column(Integer, default=0)
    calculated_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id_1', 'user_id_2', name='user_similarities_unique'),
        CheckConstraint('similarity_score >= -1 AND similarity_score <= 1', name='user_similarities_score_check'),
    )


class CollaborativeRecommendation(Base):
    """Generated collaborative filtering recommendations"""
    __tablename__ = "collaborative_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_user_id = Column(UUID(as_uuid=True), ForeignKey("lastfm_users.id"), nullable=False)
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False)
    recommendation_score = Column(Float, nullable=False)
    algorithm_type = Column(String, default="user_based")  # 'user_based', 'item_based', 'hybrid'
    confidence_score = Column(Float, default=0.0)
    reason = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('target_user_id', 'track_id', name='collaborative_recommendations_unique'),
        CheckConstraint('recommendation_score >= 0 AND recommendation_score <= 1', name='collaborative_recommendations_score_check'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='collaborative_recommendations_confidence_check'),
    )


class DataFetchLog(Base):
    """Logs for monitoring data fetching progress"""
    __tablename__ = "data_fetch_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lastfm_username = Column(String, nullable=False)
    fetch_type = Column(String, nullable=False)  # 'tracks', 'albums', 'artists', 'profile'
    status = Column(String, default="pending")  # 'pending', 'success', 'failed'
    tracks_fetched = Column(Integer, default=0)
    albums_fetched = Column(Integer, default=0)
    artists_fetched = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)





