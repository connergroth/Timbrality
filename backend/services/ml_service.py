"""
ML Service - Complete integration for machine learning pipeline

This service connects the ingestion pipeline with data preparation and model serving,
providing a unified interface for ML operations.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

try:
    from models.ingestion_models import (
        EnhancedTrack, 
        EnhancedAlbum, 
        EnhancedArtist,
        MLTrainingData,
        IngestionStats,
        get_ml_training_data
    )
    from ingestion.ingest_runner import run_ingestion, run_batch_ingestion
    from ingestion.insert_to_supabase import (
        get_supabase_client, 
        get_training_dataset, 
        export_to_csv,
        get_track_count,
        get_tracks_by_genre,
        get_tracks_by_artist
    )
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.ingestion_models import (
        EnhancedTrack, 
        EnhancedAlbum, 
        EnhancedArtist,
        MLTrainingData,
        IngestionStats,
        get_ml_training_data
    )
    try:
        from ingestion.ingest_runner import run_ingestion, run_batch_ingestion
        from ingestion.insert_to_supabase import (
            get_supabase_client, 
            get_training_dataset, 
            export_to_csv,
            get_track_count,
            get_tracks_by_genre,
            get_tracks_by_artist
        )
    except ImportError:
        # Fallback functions if ingestion modules fail
        def run_ingestion(*args, **kwargs):
            return False
        
        async def run_batch_ingestion(*args, **kwargs):
            return {'total': 0, 'successful': 0, 'failed': 0, 'errors': []}
        
        def get_training_dataset(*args, **kwargs):
            return []
        
        def export_to_csv(*args, **kwargs):
            return False
            
        def get_supabase_client():
            return None
            
        def get_track_count():
            return 0
            
        def get_tracks_by_genre(*args, **kwargs):
            return []
            
        def get_tracks_by_artist(*args, **kwargs):
            return []

logger = logging.getLogger(__name__)


class MLService:
    """Comprehensive ML service for Timbre backend"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
    # ========== Data Ingestion Integration ==========
    
    async def ingest_album_async(self, album_name: str, artist_name: str) -> bool:
        """Ingest a single album asynchronously"""
        try:
            success = await run_ingestion(album_name, artist_name)
            if success:
                logger.info(f"Successfully ingested album '{album_name}' by {artist_name}")
            return success
        except Exception as e:
            logger.error(f"Error ingesting album '{album_name}' by {artist_name}: {e}")
            return False
    
    def ingest_albums_batch(self, album_list: List[Tuple[str, str]]) -> Dict[str, int]:
        """Ingest multiple albums in batch"""
        try:
            results = run_batch_ingestion(album_list, use_async=True)
            logger.info(f"Batch ingestion completed: {results['successful']} successful, {results['failed']} failed")
            return results
        except Exception as e:
            logger.error(f"Error in batch ingestion: {e}")
            return {'total': len(album_list), 'successful': 0, 'failed': len(album_list), 'errors': [str(e)]}
    
    # ========== Data Retrieval & Analysis ==========
    
    def get_ingestion_stats(self) -> IngestionStats:
        """Get comprehensive ingestion statistics"""
        try:
            if not self.supabase:
                return IngestionStats()
                
            # Get total track count
            total_tracks = get_track_count()
            
            # Get tracks for analysis
            tracks_data = get_training_dataset(limit=10000)  # Sample for stats
            
            # Calculate statistics
            tracks_with_genres = sum(1 for track in tracks_data if track.get('genres') and len(track['genres']) > 0)
            tracks_with_moods = sum(1 for track in tracks_data if track.get('moods') and len(track['moods']) > 0)
            tracks_with_aoty_scores = sum(1 for track in tracks_data if track.get('aoty_score') is not None)
            tracks_with_audio_features = sum(1 for track in tracks_data if track.get('audio_features') and len(track.get('audio_features', {})) > 0)
            
            # Calculate averages
            genre_counts = [len(track.get('genres', [])) for track in tracks_data]
            mood_counts = [len(track.get('moods', [])) for track in tracks_data]
            avg_genres = np.mean(genre_counts) if genre_counts else 0.0
            avg_moods = np.mean(mood_counts) if mood_counts else 0.0
            
            # Get latest ingestion timestamp
            latest_ingestion = None
            if tracks_data:
                latest_track = max(tracks_data, key=lambda x: x.get('created_at', ''), default=None)
                if latest_track and latest_track.get('created_at'):
                    latest_ingestion = latest_track['created_at']
            
            return IngestionStats(
                total_tracks=total_tracks,
                total_albums=0,  # Albums not tracked separately in Supabase
                total_artists=0,  # Artists not tracked separately in Supabase
                successful_inserts=tracks_with_genres,  # Use as proxy
                failed_inserts=0,
                processing_time=0.0,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting ingestion stats: {e}")
            return IngestionStats()
    
    def get_tracks_for_training(self, limit: int = 10000, min_features: int = 3) -> List[Dict]:
        """Get tracks suitable for ML training with quality filters"""
        try:
            # Get tracks from Supabase
            all_tracks = get_training_dataset(limit=limit)
            
            # Filter tracks based on quality criteria
            filtered_tracks = []
            for track in all_tracks:
                if (track.get('popularity') is not None and
                    track.get('genres') and len(track['genres']) >= min_features and
                    track.get('duration_ms') is not None):
                    filtered_tracks.append(track)
            
            logger.info(f"Retrieved {len(filtered_tracks)} tracks for ML training")
            return filtered_tracks
            
        except Exception as e:
            logger.error(f"Error retrieving tracks for training: {e}")
            return []
    
    def get_ml_training_data(self, limit: int = 10000, include_audio_features: bool = True) -> List[Dict]:
        """Get properly formatted ML training data"""
        try:
            tracks = self.get_tracks_for_training(limit)
            
            if include_audio_features:
                # Filter tracks that have audio features
                tracks = [t for t in tracks if t.get('audio_features') and len(t.get('audio_features', {})) > 0]
            
            logger.info(f"Prepared {len(tracks)} tracks for ML training")
            return tracks
            
        except Exception as e:
            logger.error(f"Error preparing ML training data: {e}")
            return []
    
    # ========== Data Analysis & Insights ==========
    
    def get_genre_distribution(self, limit: int = 20) -> Dict[str, int]:
        """Get distribution of genres across tracks"""
        try:
            # Get tracks from Supabase
            tracks = get_training_dataset(limit=10000)  # Sample for genre analysis
            
            genre_counts = {}
            for track in tracks:
                for genre in track.get('genres', []):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Sort by count and limit
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_genres[:limit])
            
        except Exception as e:
            logger.error(f"Error getting genre distribution: {e}")
            return {}
    
    def get_mood_distribution(self, limit: int = 20) -> Dict[str, int]:
        """Get distribution of moods across tracks"""
        try:
            # Get tracks from Supabase
            tracks = get_training_dataset(limit=10000)  # Sample for mood analysis
            
            mood_counts = {}
            for track in tracks:
                for mood in track.get('moods', []):
                    mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            # Sort by count and limit
            sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_moods[:limit])
            
        except Exception as e:
            logger.error(f"Error getting mood distribution: {e}")
            return {}
    
    def get_popularity_insights(self) -> Dict[str, Any]:
        """Get insights about track popularity"""
        try:
            # Get tracks from Supabase
            tracks = get_training_dataset(limit=10000)  # Sample for popularity analysis
            
            popularities = [t.get('popularity') for t in tracks if t.get('popularity') is not None]
            
            if not popularities:
                return {}
            
            return {
                'total_tracks': len(popularities),
                'average_popularity': np.mean(popularities),
                'median_popularity': np.median(popularities),
                'std_popularity': np.std(popularities),
                'min_popularity': min(popularities),
                'max_popularity': max(popularities),
                'popularity_quartiles': {
                    'q1': np.percentile(popularities, 25),
                    'q2': np.percentile(popularities, 50),
                    'q3': np.percentile(popularities, 75)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting popularity insights: {e}")
            return {}
    
    # ========== Model Preparation ==========
    
    def prepare_feature_matrix(self, tracks: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector for ML training"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(tracks)
            
            # Extract numerical features
            numerical_features = [
                'duration_ms', 'popularity', 'track_number', 
                'album_total_tracks', 'release_year'
            ]
            
            # Handle audio features
            audio_feature_columns = []
            if df['audio_features'].notna().any():
                # Extract common audio features
                audio_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                                'instrumentalness', 'liveness', 'valence', 'tempo']
                
                for feature in audio_features:
                    df[f'audio_{feature}'] = df['audio_features'].apply(
                        lambda x: x.get(feature, 0) if isinstance(x, dict) else 0
                    )
                    audio_feature_columns.append(f'audio_{feature}')
            
            # One-hot encode genres (top genres only)
            top_genres = self.get_top_genres(limit=20)
            for genre in top_genres:
                df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if isinstance(x, list) and genre in x else 0)
            
            # One-hot encode moods (top moods only)
            top_moods = self.get_top_moods(limit=15)
            for mood in top_moods:
                df[f'mood_{mood}'] = df['moods'].apply(lambda x: 1 if isinstance(x, list) and mood in x else 0)
            
            # Binary features
            df['is_explicit'] = df['explicit'].astype(int)
            
            # Select features for training
            feature_columns = (
                numerical_features + 
                audio_feature_columns + 
                [f'genre_{g}' for g in top_genres] +
                [f'mood_{m}' for m in top_moods] +
                ['is_explicit']
            )
            
            # Remove columns with all NaN or missing values
            feature_columns = [col for col in feature_columns if col in df.columns and df[col].notna().any()]
            
            X = df[feature_columns].fillna(0)  # Fill missing values
            y = df['target_score'].fillna(df['aoty_score']).fillna(50)  # Use AOTY score as target
            
            logger.info(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing feature matrix: {e}")
            return pd.DataFrame(), pd.Series()
    
    def get_top_genres(self, limit: int = 20) -> List[str]:
        """Get most common genres"""
        genre_dist = self.get_genre_distribution(limit)
        return list(genre_dist.keys())
    
    def get_top_moods(self, limit: int = 15) -> List[str]:
        """Get most common moods"""
        mood_dist = self.get_mood_distribution(limit)
        return list(mood_dist.keys())
    
    # ========== Export Functions ==========
    
    def export_training_data(self, filename: str, limit: int = 10000) -> bool:
        """Export training data to CSV"""
        try:
            ml_data = self.get_ml_training_data(limit)
            
            if not ml_data:
                logger.error("No training data available for export")
                return False
            
            # Convert to DataFrame and save
            df = pd.DataFrame([track.dict() for track in ml_data])
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(ml_data)} training samples to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return False
    
    def export_feature_matrix(self, filename: str, limit: int = 10000) -> bool:
        """Export prepared feature matrix to CSV"""
        try:
            ml_data = self.get_ml_training_data(limit)
            X, y = self.prepare_feature_matrix(ml_data)
            
            if X.empty:
                logger.error("No feature matrix available for export")
                return False
            
            # Combine features and target
            export_df = X.copy()
            export_df['target'] = y
            export_df.to_csv(filename, index=False)
            
            logger.info(f"Exported feature matrix to {filename}: {X.shape[0]} samples, {X.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feature matrix: {e}")
            return False
    
    # ========== Recommendation Preparation ==========
    
    def get_track_embeddings_data(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Get track data formatted for embedding generation"""
        try:
            if not self.supabase:
                return []
                
            # Get tracks by IDs from Supabase
            result = self.supabase.table('tracks').select('*').in_('id', track_ids).execute()
            
            embeddings_data = []
            if result.data:
                for track in result.data:
                    # Parse JSON strings back to lists
                    genres = json.loads(track.get('genres', '[]')) if track.get('genres') else []
                    moods = json.loads(track.get('moods', '[]')) if track.get('moods') else []
                    
                    data = {
                        'track_id': track.get('id'),
                        'title': track.get('title'),
                        'artist': track.get('artist'),
                        'album': track.get('album'),
                        'genres': genres,
                        'moods': moods,
                        'audio_features': track.get('audio_features', {}),
                        'popularity': track.get('popularity'),
                        'duration_ms': track.get('duration_ms'),
                        'explicit': track.get('explicit'),
                        'aoty_score': track.get('aoty_score')
                    }
                    embeddings_data.append(data)
            
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Error getting track embeddings data: {e}")
            return []
    
    def find_similar_tracks_by_features(self, track_id: str, limit: int = 10) -> List[Dict]:
        """Find similar tracks based on features (simple implementation)"""
        try:
            if not self.supabase:
                return []
                
            # Get target track
            target_result = self.supabase.table('tracks').select('*').eq('id', track_id).execute()
            
            if not target_result.data:
                return []
                
            target_track = target_result.data[0]
            target_genres = set(json.loads(target_track.get('genres', '[]')))
            
            # Get all tracks for comparison
            all_tracks = get_training_dataset(limit=5000)  # Limit for performance
            
            # Filter by shared genres
            candidates = []
            
            for track in all_tracks:
                if track.get('id') == track_id:
                    continue
                    
                track_genres = set(track.get('genres', []))
                shared_genres = len(target_genres.intersection(track_genres))
                
                if shared_genres > 0:
                    # Simple similarity score
                    target_popularity = target_track.get('popularity', 50)
                    track_popularity = track.get('popularity', 50)
                    popularity_diff = abs(target_popularity - track_popularity)
                    similarity_score = shared_genres * 10 - popularity_diff * 0.1
                    candidates.append((track, similarity_score))
            
            # Sort by similarity and return top results
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [track for track, score in candidates[:limit]]
            
        except Exception as e:
            logger.error(f"Error finding similar tracks: {e}")
            return []


# Global ML service instance
ml_service = MLService()


# Helper functions for easy access
def get_training_data(limit: int = 10000) -> List[Dict]:
    """Get ML training data"""
    return ml_service.get_ml_training_data(limit)


def get_feature_matrix(limit: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
    """Get prepared feature matrix for ML training"""
    training_data = get_training_data(limit)
    return ml_service.prepare_feature_matrix(training_data)


def ingest_album(album_name: str, artist_name: str) -> bool:
    """Simple interface to ingest a single album"""
    return asyncio.run(ml_service.ingest_album_async(album_name, artist_name))


def get_stats() -> IngestionStats:
    """Get ingestion statistics"""
    return ml_service.get_ingestion_stats() 