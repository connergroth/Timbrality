"""
Last.fm Session Processing and Feature Extraction.

This module processes Last.fm scrobble data to derive sessions, calculate
discovery rates, and extract user activity features for the ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a listening session."""
    user_id: str
    start_time: datetime
    end_time: datetime
    tracks: List[Dict[str, Any]]
    session_length_minutes: float
    unique_artists: int
    unique_albums: int
    unique_tracks: int
    repeat_rate: float
    skip_rate: float = 0.0  # Will be calculated if skip data available


@dataclass
class UserActivityFeatures:
    """User activity features derived from session analysis."""
    user_id: str
    
    # Session statistics
    total_sessions: int
    avg_session_length: float
    total_listening_time: float
    sessions_per_week: float
    
    # Discovery metrics
    discovery_rate: float  # New artists per week
    exploration_score: float  # Diversity of content
    novelty_seeking: float  # Preference for new releases
    
    # Listening patterns
    repeat_rate: float  # How often tracks are repeated
    skip_rate: float  # Rate of skipping tracks
    genre_diversity: float  # Number of unique genres
    temporal_consistency: float  # Consistency of listening times
    
    # Social/trend following
    mainstream_score: float  # How much user follows popular music
    early_adopter_score: float  # How quickly user discovers new music
    
    # Activity level
    total_plays: int
    plays_per_day: float
    active_days: int
    longest_streak: int


class SessionProcessor:
    """Processes Last.fm scrobble data to extract sessions and user features."""
    
    def __init__(
        self,
        session_gap_minutes: int = 30,
        min_session_tracks: int = 3,
        skip_threshold_seconds: int = 30
    ):
        """
        Initialize session processor.
        
        Args:
            session_gap_minutes: Minutes between tracks to consider new session
            min_session_tracks: Minimum tracks required for a valid session
            skip_threshold_seconds: Threshold to consider a track "skipped"
        """
        self.session_gap = timedelta(minutes=session_gap_minutes)
        self.min_session_tracks = min_session_tracks
        self.skip_threshold = skip_threshold_seconds
        
    def process_user_scrobbles(
        self, 
        scrobbles_df: pd.DataFrame,
        user_id: str
    ) -> Tuple[List[Session], UserActivityFeatures]:
        """
        Process scrobbles for a single user to extract sessions and features.
        
        Args:
            scrobbles_df: DataFrame with columns ['timestamp', 'artist', 'album', 'track', 'duration']
            user_id: User identifier
            
        Returns:
            Tuple of (sessions_list, user_activity_features)
        """
        # Sort by timestamp
        df = scrobbles_df.sort_values('timestamp').copy()
        
        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract sessions
        sessions = self._extract_sessions(df, user_id)
        
        # Calculate user activity features
        features = self._calculate_user_features(sessions, df, user_id)
        
        return sessions, features
    
    def _extract_sessions(self, df: pd.DataFrame, user_id: str) -> List[Session]:
        """Extract listening sessions from chronological scrobbles."""
        sessions = []
        current_session_tracks = []
        last_timestamp = None
        
        for _, row in df.iterrows():
            current_timestamp = row['timestamp']
            
            # Start new session if gap is too large or this is the first track
            if (last_timestamp is None or 
                current_timestamp - last_timestamp > self.session_gap):
                
                # Save previous session if it meets minimum requirements
                if len(current_session_tracks) >= self.min_session_tracks:
                    session = self._create_session(current_session_tracks, user_id)
                    sessions.append(session)
                
                # Start new session
                current_session_tracks = []
            
            # Add track to current session
            track_data = {
                'timestamp': current_timestamp,
                'artist': row.get('artist', ''),
                'album': row.get('album', ''),
                'track': row.get('track', ''),
                'duration': row.get('duration', 0),
                'artist_mbid': row.get('artist_mbid', ''),
                'album_mbid': row.get('album_mbid', ''),
                'track_mbid': row.get('track_mbid', '')
            }
            current_session_tracks.append(track_data)
            last_timestamp = current_timestamp
        
        # Don't forget the last session
        if len(current_session_tracks) >= self.min_session_tracks:
            session = self._create_session(current_session_tracks, user_id)
            sessions.append(session)
        
        logger.info(f"Extracted {len(sessions)} sessions for user {user_id}")
        return sessions
    
    def _create_session(self, tracks: List[Dict], user_id: str) -> Session:
        """Create a Session object from a list of tracks."""
        if not tracks:
            raise ValueError("Cannot create session from empty track list")
        
        start_time = tracks[0]['timestamp']
        end_time = tracks[-1]['timestamp']
        session_length = (end_time - start_time).total_seconds() / 60.0  # minutes
        
        # Count unique entities
        unique_artists = len(set(track['artist'] for track in tracks))
        unique_albums = len(set(track['album'] for track in tracks))
        unique_tracks = len(set(track['track'] for track in tracks))
        
        # Calculate repeat rate
        total_tracks = len(tracks)
        repeat_rate = 1.0 - (unique_tracks / total_tracks) if total_tracks > 0 else 0.0
        
        # Calculate skip rate if duration data available
        skip_rate = self._calculate_skip_rate(tracks)
        
        return Session(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            tracks=tracks,
            session_length_minutes=session_length,
            unique_artists=unique_artists,
            unique_albums=unique_albums,
            unique_tracks=unique_tracks,
            repeat_rate=repeat_rate,
            skip_rate=skip_rate
        )
    
    def _calculate_skip_rate(self, tracks: List[Dict]) -> float:
        """Calculate skip rate for a session if duration data is available."""
        if not tracks:
            return 0.0
        
        skipped_tracks = 0
        valid_tracks = 0
        
        for i, track in enumerate(tracks[:-1]):  # Exclude last track
            duration = track.get('duration', 0)
            if duration > 0:
                valid_tracks += 1
                
                # Calculate listening time based on next track timestamp
                next_track = tracks[i + 1]
                listening_time = (next_track['timestamp'] - track['timestamp']).total_seconds()
                
                # Consider skipped if listened to less than threshold or 50% of track
                skip_threshold = min(self.skip_threshold, duration * 0.5)
                if listening_time < skip_threshold:
                    skipped_tracks += 1
        
        return skipped_tracks / valid_tracks if valid_tracks > 0 else 0.0
    
    def _calculate_user_features(
        self, 
        sessions: List[Session], 
        df: pd.DataFrame,
        user_id: str
    ) -> UserActivityFeatures:
        """Calculate comprehensive user activity features from sessions."""
        
        if not sessions:
            # Return default features for users with no valid sessions
            return UserActivityFeatures(
                user_id=user_id,
                total_sessions=0,
                avg_session_length=0.0,
                total_listening_time=0.0,
                sessions_per_week=0.0,
                discovery_rate=0.0,
                exploration_score=0.0,
                novelty_seeking=0.0,
                repeat_rate=0.0,
                skip_rate=0.0,
                genre_diversity=0.0,
                temporal_consistency=0.0,
                mainstream_score=0.5,
                early_adopter_score=0.0,
                total_plays=len(df),
                plays_per_day=0.0,
                active_days=0,
                longest_streak=0
            )
        
        # Basic session statistics
        total_sessions = len(sessions)
        avg_session_length = np.mean([s.session_length_minutes for s in sessions])
        total_listening_time = sum(s.session_length_minutes for s in sessions)
        
        # Calculate time span and sessions per week
        first_session = min(sessions, key=lambda s: s.start_time)
        last_session = max(sessions, key=lambda s: s.end_time)
        time_span_days = (last_session.end_time - first_session.start_time).days
        sessions_per_week = (total_sessions / max(time_span_days, 1)) * 7
        
        # Discovery rate calculation
        discovery_rate = self._calculate_discovery_rate(df)
        
        # Exploration and diversity metrics
        exploration_score = self._calculate_exploration_score(sessions)
        genre_diversity = self._calculate_genre_diversity(df)
        
        # Novelty seeking (preference for new releases)
        novelty_seeking = self._calculate_novelty_seeking(df)
        
        # Listening patterns
        repeat_rate = np.mean([s.repeat_rate for s in sessions])
        skip_rate = np.mean([s.skip_rate for s in sessions])
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(sessions)
        
        # Social/trend metrics
        mainstream_score = self._calculate_mainstream_score(df)
        early_adopter_score = self._calculate_early_adopter_score(df)
        
        # Activity level
        total_plays = len(df)
        active_days = self._calculate_active_days(df)
        plays_per_day = total_plays / max(time_span_days, 1)
        longest_streak = self._calculate_longest_streak(df)
        
        return UserActivityFeatures(
            user_id=user_id,
            total_sessions=total_sessions,
            avg_session_length=avg_session_length,
            total_listening_time=total_listening_time,
            sessions_per_week=sessions_per_week,
            discovery_rate=discovery_rate,
            exploration_score=exploration_score,
            novelty_seeking=novelty_seeking,
            repeat_rate=repeat_rate,
            skip_rate=skip_rate,
            genre_diversity=genre_diversity,
            temporal_consistency=temporal_consistency,
            mainstream_score=mainstream_score,
            early_adopter_score=early_adopter_score,
            total_plays=total_plays,
            plays_per_day=plays_per_day,
            active_days=active_days,
            longest_streak=longest_streak
        )
    
    def _calculate_discovery_rate(self, df: pd.DataFrame) -> float:
        """Calculate rate of discovering new artists per week."""
        df_sorted = df.sort_values('timestamp')
        
        # Track first time each artist is encountered
        seen_artists = set()
        new_artists_by_week = defaultdict(int)
        
        for _, row in df_sorted.iterrows():
            artist = row['artist']
            week = row['timestamp'].isocalendar()[1]  # Week number
            year = row['timestamp'].year
            week_key = f"{year}-W{week:02d}"
            
            if artist not in seen_artists:
                seen_artists.add(artist)
                new_artists_by_week[week_key] += 1
        
        # Calculate average new artists per week
        if new_artists_by_week:
            return np.mean(list(new_artists_by_week.values()))
        return 0.0
    
    def _calculate_exploration_score(self, sessions: List[Session]) -> float:
        """Calculate how much user explores different content within sessions."""
        if not sessions:
            return 0.0
        
        diversity_scores = []
        for session in sessions:
            if len(session.tracks) > 1:
                # Measure diversity as ratio of unique artists to total tracks
                diversity = session.unique_artists / len(session.tracks)
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_genre_diversity(self, df: pd.DataFrame) -> float:
        """Calculate user's genre diversity (requires genre data)."""
        # This would need genre mapping from artist names
        # For now, estimate based on artist diversity
        unique_artists = df['artist'].nunique()
        total_plays = len(df)
        
        return min(unique_artists / max(total_plays, 1), 1.0)
    
    def _calculate_novelty_seeking(self, df: pd.DataFrame) -> float:
        """Calculate user's preference for new releases."""
        # Estimate based on how recent the music is (would need release dates)
        # For now, return a placeholder based on temporal patterns
        if len(df) == 0:
            return 0.0
        
        # Simple heuristic: users who listen to music in later periods might prefer newer content
        recent_threshold = df['timestamp'].max() - timedelta(days=180)
        recent_plays = df[df['timestamp'] > recent_threshold]
        
        return len(recent_plays) / len(df)
    
    def _calculate_temporal_consistency(self, sessions: List[Session]) -> float:
        """Calculate consistency of listening times (daily patterns)."""
        if not sessions:
            return 0.0
        
        # Extract hour of day for each session
        session_hours = [s.start_time.hour for s in sessions]
        
        # Calculate standard deviation (lower = more consistent)
        hour_std = np.std(session_hours)
        
        # Convert to consistency score (0-1, higher = more consistent)
        return max(0, 1 - (hour_std / 12))  # Normalize by half-day
    
    def _calculate_mainstream_score(self, df: pd.DataFrame) -> float:
        """Calculate how mainstream user's music taste is."""
        # This would ideally use popularity data
        # For now, estimate based on artist play counts
        artist_counts = df['artist'].value_counts()
        
        # Users with very concentrated listening might be less mainstream
        concentration = (artist_counts.iloc[0] / len(df)) if len(artist_counts) > 0 else 0
        
        # Return inverse of concentration (more diverse = more mainstream assumption)
        return max(0, 1 - concentration)
    
    def _calculate_early_adopter_score(self, df: pd.DataFrame) -> float:
        """Calculate how quickly user adopts new music."""
        # This would need release date data and popularity trends
        # For now, return a placeholder
        return 0.3  # Average early adopter score
    
    def _calculate_active_days(self, df: pd.DataFrame) -> int:
        """Calculate number of days user was active."""
        if len(df) == 0:
            return 0
        
        unique_dates = df['timestamp'].dt.date.nunique()
        return unique_dates
    
    def _calculate_longest_streak(self, df: pd.DataFrame) -> int:
        """Calculate longest consecutive day streak."""
        if len(df) == 0:
            return 0
        
        unique_dates = sorted(df['timestamp'].dt.date.unique())
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(unique_dates)):
            if (unique_dates[i] - unique_dates[i-1]).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak


class LastfmDataProcessor:
    """High-level processor for Last.fm data extraction and feature generation."""
    
    def __init__(self, session_processor: SessionProcessor = None):
        """Initialize with optional custom session processor."""
        self.session_processor = session_processor or SessionProcessor()
    
    def process_user_data(
        self, 
        user_scrobbles: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process scrobble data for multiple users.
        
        Args:
            user_scrobbles: Dict mapping user_id to scrobbles DataFrame
            
        Returns:
            Dict mapping user_id to processed data (sessions + features)
        """
        processed_data = {}
        
        for user_id, scrobbles_df in user_scrobbles.items():
            try:
                sessions, features = self.session_processor.process_user_scrobbles(
                    scrobbles_df, user_id
                )
                
                processed_data[user_id] = {
                    'sessions': sessions,
                    'features': features,
                    'raw_scrobbles_count': len(scrobbles_df)
                }
                
                logger.info(f"Processed user {user_id}: {len(sessions)} sessions, "
                           f"{features.total_plays} total plays")
                
            except Exception as e:
                logger.error(f"Failed to process user {user_id}: {e}")
                processed_data[user_id] = None
        
        return processed_data
    
    def export_features_for_ml(
        self, 
        processed_data: Dict[str, Dict[str, Any]],
        normalize_features: bool = True
    ) -> pd.DataFrame:
        """
        Export user features in format suitable for ML training.
        
        Args:
            processed_data: Output from process_user_data
            normalize_features: Whether to normalize features across activity levels
            
        Returns:
            DataFrame with user features for ML pipeline
        """
        features_list = []
        
        for user_id, data in processed_data.items():
            if data is None:
                continue
            
            features = data['features']
            
            # Convert to dictionary for DataFrame
            feature_dict = {
                'user_id': user_id,
                'total_sessions': features.total_sessions,
                'avg_session_length': features.avg_session_length,
                'total_listening_time': features.total_listening_time,
                'sessions_per_week': features.sessions_per_week,
                'discovery_rate': features.discovery_rate,
                'exploration_score': features.exploration_score,
                'novelty_seeking': features.novelty_seeking,
                'repeat_rate': features.repeat_rate,
                'skip_rate': features.skip_rate,
                'genre_diversity': features.genre_diversity,
                'temporal_consistency': features.temporal_consistency,
                'mainstream_score': features.mainstream_score,
                'early_adopter_score': features.early_adopter_score,
                'total_plays': features.total_plays,
                'plays_per_day': features.plays_per_day,
                'active_days': features.active_days,
                'longest_streak': features.longest_streak
            }
            
            features_list.append(feature_dict)
        
        df = pd.DataFrame(features_list)
        
        if normalize_features and len(df) > 1:
            df = self._normalize_session_features(df)
        
        return df
    
    def _normalize_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize session-derived features to handle different activity levels.
        
        Key normalizations:
        - Discovery rate → per active week (not total weeks)
        - Streak length → percentile rank
        - Session metrics → relative to user's own activity
        """
        df_norm = df.copy()
        
        # 1. Discovery rate per active week (not calendar weeks)
        if 'discovery_rate' in df_norm.columns and 'active_days' in df_norm.columns:
            active_weeks = df_norm['active_days'] / 7.0
            active_weeks = active_weeks.clip(lower=0.1)  # Avoid division by zero
            df_norm['discovery_rate_per_active_week'] = df_norm['discovery_rate'] / active_weeks
            
        # 2. Streak length as percentile (0-1 scale)
        if 'longest_streak' in df_norm.columns:
            df_norm['longest_streak_percentile'] = df_norm['longest_streak'].rank(pct=True)
            
        # 3. Sessions per week relative to user's active period
        if 'sessions_per_week' in df_norm.columns and 'active_days' in df_norm.columns:
            # Normalize by actual activity span vs calendar span
            activity_density = df_norm['active_days'] / (df_norm['total_sessions'] * 7 / df_norm['sessions_per_week']).clip(lower=1)
            df_norm['session_density'] = df_norm['sessions_per_week'] * activity_density
            
        # 4. Listening intensity (total time relative to active days)
        if 'total_listening_time' in df_norm.columns and 'active_days' in df_norm.columns:
            df_norm['listening_intensity'] = df_norm['total_listening_time'] / df_norm['active_days'].clip(lower=1)
            
        # 5. Z-score normalization for key continuous features
        continuous_features = [
            'avg_session_length', 'exploration_score', 'novelty_seeking',
            'repeat_rate', 'skip_rate', 'genre_diversity', 'temporal_consistency',
            'mainstream_score', 'early_adopter_score', 'plays_per_day'
        ]
        
        for feature in continuous_features:
            if feature in df_norm.columns:
                mean_val = df_norm[feature].mean()
                std_val = df_norm[feature].std()
                if std_val > 0:
                    df_norm[f'{feature}_normalized'] = (df_norm[feature] - mean_val) / std_val
                else:
                    df_norm[f'{feature}_normalized'] = 0.0
        
        # 6. Activity level binning for stratified analysis
        if 'total_plays' in df_norm.columns:
            df_norm['activity_level'] = pd.qcut(
                df_norm['total_plays'], 
                q=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
        # 7. User type classification based on behavior patterns
        df_norm['user_type'] = self._classify_user_types(df_norm)
        
        return df_norm
    
    def _classify_user_types(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify users into behavioral types for better generalization.
        
        Types:
        - explorer: High discovery rate, high genre diversity
        - loyalist: Low discovery rate, high repeat rate
        - casual: Low activity, irregular patterns
        - binger: High session length, low session frequency
        - social: High mainstream score, follows trends
        """
        user_types = []
        
        for _, row in df.iterrows():
            discovery = row.get('discovery_rate', 0)
            diversity = row.get('genre_diversity', 0)
            repeat = row.get('repeat_rate', 0)
            session_length = row.get('avg_session_length', 0)
            sessions_per_week = row.get('sessions_per_week', 0)
            mainstream = row.get('mainstream_score', 0.5)
            total_plays = row.get('total_plays', 0)
            
            # Define thresholds (could be data-driven percentiles)
            high_discovery = discovery > df['discovery_rate'].quantile(0.7)
            high_diversity = diversity > df['genre_diversity'].quantile(0.7)
            high_repeat = repeat > df['repeat_rate'].quantile(0.7)
            long_sessions = session_length > df['avg_session_length'].quantile(0.7)
            low_frequency = sessions_per_week < df['sessions_per_week'].quantile(0.3)
            high_mainstream = mainstream > df['mainstream_score'].quantile(0.7)
            low_activity = total_plays < df['total_plays'].quantile(0.3)
            
            # Classification logic
            if low_activity:
                user_type = 'casual'
            elif high_discovery and high_diversity:
                user_type = 'explorer'
            elif high_repeat and not high_discovery:
                user_type = 'loyalist'
            elif long_sessions and low_frequency:
                user_type = 'binger'
            elif high_mainstream:
                user_type = 'social'
            else:
                user_type = 'balanced'
                
            user_types.append(user_type)
        
        return pd.Series(user_types)
    
    def save_processed_data(
        self, 
        processed_data: Dict[str, Dict[str, Any]], 
        output_path: str
    ):
        """Save processed data to file."""
        # Convert sessions to serializable format
        serializable_data = {}
        
        for user_id, data in processed_data.items():
            if data is None:
                continue
            
            sessions_data = []
            for session in data['sessions']:
                session_dict = {
                    'user_id': session.user_id,
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat(),
                    'session_length_minutes': session.session_length_minutes,
                    'unique_artists': session.unique_artists,
                    'unique_albums': session.unique_albums,
                    'unique_tracks': session.unique_tracks,
                    'repeat_rate': session.repeat_rate,
                    'skip_rate': session.skip_rate,
                    'track_count': len(session.tracks)
                }
                sessions_data.append(session_dict)
            
            features = data['features']
            features_dict = {
                'user_id': features.user_id,
                'total_sessions': features.total_sessions,
                'avg_session_length': features.avg_session_length,
                'total_listening_time': features.total_listening_time,
                'sessions_per_week': features.sessions_per_week,
                'discovery_rate': features.discovery_rate,
                'exploration_score': features.exploration_score,
                'novelty_seeking': features.novelty_seeking,
                'repeat_rate': features.repeat_rate,
                'skip_rate': features.skip_rate,
                'genre_diversity': features.genre_diversity,
                'temporal_consistency': features.temporal_consistency,
                'mainstream_score': features.mainstream_score,
                'early_adopter_score': features.early_adopter_score,
                'total_plays': features.total_plays,
                'plays_per_day': features.plays_per_day,
                'active_days': features.active_days,
                'longest_streak': features.longest_streak
            }
            
            serializable_data[user_id] = {
                'sessions': sessions_data,
                'features': features_dict,
                'raw_scrobbles_count': data['raw_scrobbles_count']
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Processed data saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the session processor
    
    # Sample scrobbles data
    sample_scrobbles = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'artist': ['Artist A', 'Artist B', 'Artist C'] * 34,
        'album': ['Album 1', 'Album 2', 'Album 3'] * 34,
        'track': [f'Track {i}' for i in range(100)],
        'duration': [180, 240, 200] * 34  # seconds
    })
    
    # Process sessions
    processor = SessionProcessor()
    sessions, features = processor.process_user_scrobbles(sample_scrobbles, 'user_123')
    
    print(f"Found {len(sessions)} sessions")
    print(f"User features: discovery_rate={features.discovery_rate:.2f}, "
          f"avg_session_length={features.avg_session_length:.1f}min")
