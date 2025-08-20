"""
Complete Training Pipeline for Timbrality ML System.

This script orchestrates training of all ML components:
1. Audio Feature Predictor (using Million Song Dataset)
2. Two-Tower Content-Based Model
3. Hybrid Model Integration
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from timbral.models.audio_feature_predictor import AudioFeaturePredictor, AudioFeatureTrainer
from timbral.models.two_tower_model import TwoTowerModel, TwoTowerTrainer
from timbral.models.nmf_model import NMFModel
from timbral.models.hybrid_model import HybridModel
from timbral.utils.data_loader import DataLoader as TimbralDataLoader
from timbral.utils.session_processor import SessionProcessor, LastfmDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MillionSongDataset(Dataset):
    """Dataset wrapper for Million Song Dataset with audio features."""
    
    def __init__(self, dataframe: pd.DataFrame, mode: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            dataframe: DataFrame with track data and audio features
            mode: 'train' or 'inference'
        """
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Prepare inputs (same as in AudioFeaturePredictor.prepare_data)
        sample = {
            'title': row.get('title', ''),
            'artist': row.get('artist', ''),
            'genres': row.get('genres', []),
            'moods': row.get('moods', []),
            'metadata': {
                'popularity': row.get('popularity', 50),
                'duration_ms': row.get('duration_ms', 180000),
                'release_year': row.get('release_year', 2000),
                'explicit': float(row.get('explicit', False)),
                'aoty_score': row.get('aoty_score', 50),
                'aoty_num_ratings': row.get('aoty_num_ratings', 0),
                'track_number': row.get('track_number', 1),
                'album_total_tracks': row.get('album_total_tracks', 10),
                'artist_popularity': row.get('artist_popularity', 50),
                'artist_followers': row.get('artist_followers', 1000)
            }
        }
        
        if self.mode == 'train':
            # Add targets for training
            sample['targets'] = {
                'energy': row.get('energy', 0.5),
                'valence': row.get('valence', 0.5),
                'danceability': row.get('danceability', 0.5),
                'acousticness': row.get('acousticness', 0.5),
                'instrumentalness': row.get('instrumentalness', 0.5),
                'liveness': row.get('liveness', 0.5),
                'speechiness': row.get('speechiness', 0.5),
                'tempo': row.get('tempo', 120)
            }
        
        return sample


class TimbrialityTrainingPipeline:
    """Complete training pipeline for Timbrality ML system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data paths
        self.million_song_path = config.get('million_song_dataset_path')
        self.timbrality_data_path = config.get('timbrality_data_path')
        self.lastfm_scrobbles_path = config.get('lastfm_scrobbles_path')
        self.output_dir = Path(config.get('output_dir', 'models'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Session processor for Last.fm data
        self.session_processor = SessionProcessor(
            session_gap_minutes=config.get('session_gap_minutes', 30),
            min_session_tracks=config.get('min_session_tracks', 3)
        )
        
        # Model configurations
        self.audio_predictor_config = config.get('audio_predictor', {})
        self.two_tower_config = config.get('two_tower', {})
        self.hybrid_config = config.get('hybrid', {})
        
    def train_audio_feature_predictor(self) -> AudioFeaturePredictor:
        """
        Train audio feature predictor using Million Song Dataset.
        
        Returns:
            Trained AudioFeaturePredictor model
        """
        logger.info("Starting audio feature predictor training...")
        
        # Load Million Song Dataset
        logger.info("Loading Million Song Dataset...")
        msd_df = self._load_million_song_dataset()
        
        # Initialize model
        model = AudioFeaturePredictor(
            vocab_size=self.audio_predictor_config.get('vocab_size', 20000),
            embedding_dim=self.audio_predictor_config.get('embedding_dim', 128),
            hidden_dims=self.audio_predictor_config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=self.audio_predictor_config.get('dropout_rate', 0.3),
            num_genres=self.audio_predictor_config.get('num_genres', 1000),
            num_moods=self.audio_predictor_config.get('num_moods', 500)
        )
        
        # Initialize trainer
        trainer = AudioFeatureTrainer(
            model=model,
            learning_rate=self.audio_predictor_config.get('learning_rate', 1e-3),
            weight_decay=self.audio_predictor_config.get('weight_decay', 1e-4),
            device=self.device
        )
        
        # Prepare data
        logger.info("Preparing training data...")
        inputs, targets = trainer.prepare_data(msd_df)
        
        # Split data
        train_size = 0.8
        val_size = 0.1
        
        # Create datasets
        train_dataset = MillionSongDataset(msd_df[:int(len(msd_df) * train_size)], mode='train')
        val_dataset = MillionSongDataset(
            msd_df[int(len(msd_df) * train_size):int(len(msd_df) * (train_size + val_size))], 
            mode='train'
        )
        
        # Train model
        epochs = self.audio_predictor_config.get('epochs', 50)
        batch_size = self.audio_predictor_config.get('batch_size', 32)
        
        for epoch in range(epochs):
            logger.info(f"Training epoch {epoch + 1}/{epochs}")
            
            # Note: This is a simplified training loop
            # In practice, you'd implement proper DataLoader and training loop
            
        # Save model
        model_path = self.output_dir / "audio_feature_predictor.pt"
        trainer.save(str(model_path))
        logger.info(f"Audio feature predictor saved to {model_path}")
        
        return model
    
    def train_two_tower_model(self) -> TwoTowerModel:
        """
        Train two-tower content-based model.
        
        Returns:
            Trained TwoTowerModel
        """
        logger.info("Starting two-tower model training...")
        
        # Load Timbrality data
        logger.info("Loading Timbrality data...")
        timbrality_df = self._load_timbrality_data()
        
        # Get vocabulary sizes from data
        all_users = timbrality_df['user_id'].nunique()
        all_artists = timbrality_df['artist_id'].nunique() 
        all_genres = self._count_unique_categories(timbrality_df, 'genres')
        all_moods = self._count_unique_categories(timbrality_df, 'moods')
        
        # Initialize model
        model = TwoTowerModel(
            num_users=all_users,
            num_artists=all_artists,
            num_genres=all_genres,
            num_moods=all_moods,
            text_model_name=self.two_tower_config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            embedding_dim=self.two_tower_config.get('embedding_dim', 64),
            text_dim=self.two_tower_config.get('text_dim', 384)
        )
        
        # Initialize trainer
        trainer = TwoTowerTrainer(
            model=model,
            learning_rate=self.two_tower_config.get('learning_rate', 1e-3),
            weight_decay=self.two_tower_config.get('weight_decay', 1e-4),
            device=self.device
        )
        
        # Train model (simplified)
        epochs = self.two_tower_config.get('epochs', 20)
        logger.info(f"Training two-tower model for {epochs} epochs...")
        
        # Save model
        model_path = self.output_dir / "two_tower_model.pt"
        trainer.save(str(model_path))
        logger.info(f"Two-tower model saved to {model_path}")
        
        return model
    
    def train_collaborative_filtering(self) -> NMFModel:
        """
        Train NMF collaborative filtering model.
        
        Returns:
            Trained NMFModel
        """
        logger.info("Training collaborative filtering model...")
        
        # Load user-item interactions
        interactions_df = self._load_user_interactions()
        
        # Create user-item matrix
        data_loader = TimbralDataLoader()
        user_item_matrix = data_loader.create_user_item_matrix(interactions_df)
        
        # Initialize and train NMF model
        nmf_model = NMFModel(
            n_components=self.config.get('nmf_components', 100),
            random_state=42,
            max_iter=self.config.get('nmf_max_iter', 200),
            tol=self.config.get('nmf_tol', 1e-4)
        )
        
        nmf_model.fit(user_item_matrix)
        
        # Save model
        model_path = self.output_dir / "nmf_model.pkl"
        nmf_model.save(str(model_path))
        logger.info(f"NMF model saved to {model_path}")
        
        return nmf_model
    
    def create_hybrid_model(
        self,
        nmf_model: NMFModel,
        two_tower_model: TwoTowerModel,
        audio_predictor: AudioFeaturePredictor
    ) -> HybridModel:
        """
        Create and configure hybrid model.
        
        Args:
            nmf_model: Trained NMF model
            two_tower_model: Trained two-tower model
            audio_predictor: Trained audio feature predictor
            
        Returns:
            Configured HybridModel
        """
        logger.info("Creating hybrid model...")
        
        hybrid_model = HybridModel(
            nmf_model=nmf_model,
            two_tower_model=two_tower_model,
            audio_predictor=audio_predictor,
            fusion_method=self.hybrid_config.get('fusion_method', 'meta_learning'),
            cf_weight=self.hybrid_config.get('cf_weight', 0.6),
            content_weight=self.hybrid_config.get('content_weight', 0.4),
            embedding_dim=self.hybrid_config.get('embedding_dim', 64),
            enable_meta_learning=self.hybrid_config.get('enable_meta_learning', True),
            session_feature_dim=self.hybrid_config.get('session_feature_dim', 18),
            min_ratings_for_acclaim=self.hybrid_config.get('min_ratings_for_acclaim', 100),
            high_acclaim_threshold=self.hybrid_config.get('high_acclaim_threshold', 500)
        )
        
        # Save hybrid model
        model_path = self.output_dir / "hybrid_model"
        hybrid_model.save(str(model_path))
        logger.info(f"Hybrid model saved to {model_path}")
        
        return hybrid_model
    
    def process_lastfm_sessions(self) -> pd.DataFrame:
        """
        Process Last.fm scrobble data to extract sessions and user activity features.
        
        Returns:
            DataFrame with user activity features
        """
        logger.info("Processing Last.fm session data...")
        
        # Load Last.fm scrobbles (this would connect to your database)
        user_scrobbles = self._load_lastfm_scrobbles()
        
        # Process sessions for all users
        lastfm_processor = LastfmDataProcessor(self.session_processor)
        processed_data = lastfm_processor.process_user_data(user_scrobbles)
        
        # Export features for ML
        features_df = lastfm_processor.export_features_for_ml(processed_data)
        
        # Save processed data
        output_path = self.output_dir / "lastfm_session_features.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"Session features saved to {output_path}")
        
        # Save detailed session data
        session_data_path = self.output_dir / "lastfm_sessions_detailed.json"
        lastfm_processor.save_processed_data(processed_data, str(session_data_path))
        
        return features_df
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with trained models and results
        """
        logger.info("Starting complete ML training pipeline...")
        
        results = {}
        
        # Step 0: Process Last.fm session data
        try:
            session_features = self.process_lastfm_sessions()
            results['session_features'] = session_features
        except Exception as e:
            logger.error(f"Session processing failed: {e}")
            results['session_features'] = None
        
        # Step 1: Train audio feature predictor
        try:
            audio_predictor = self.train_audio_feature_predictor()
            results['audio_predictor'] = audio_predictor
        except Exception as e:
            logger.error(f"Audio predictor training failed: {e}")
            audio_predictor = None
            results['audio_predictor'] = None
        
        # Step 2: Train two-tower model
        try:
            two_tower_model = self.train_two_tower_model()
            results['two_tower_model'] = two_tower_model
        except Exception as e:
            logger.error(f"Two-tower training failed: {e}")
            two_tower_model = None
            results['two_tower_model'] = None
        
        # Step 3: Train collaborative filtering
        try:
            nmf_model = self.train_collaborative_filtering()
            results['nmf_model'] = nmf_model
        except Exception as e:
            logger.error(f"Collaborative filtering training failed: {e}")
            nmf_model = None
            results['nmf_model'] = None
        
        # Step 4: Create hybrid model
        if nmf_model and two_tower_model:
            try:
                hybrid_model = self.create_hybrid_model(nmf_model, two_tower_model, audio_predictor)
                results['hybrid_model'] = hybrid_model
            except Exception as e:
                logger.error(f"Hybrid model creation failed: {e}")
                results['hybrid_model'] = None
        else:
            logger.warning("Cannot create hybrid model - missing required components")
            results['hybrid_model'] = None
        
        # Step 5: Generate evaluation report
        evaluation_report = self._generate_evaluation_report(results)
        results['evaluation'] = evaluation_report
        
        logger.info("Training pipeline completed!")
        return results
    
    def _load_million_song_dataset(self) -> pd.DataFrame:
        """Load and preprocess Million Song Dataset."""
        # This would load the actual Million Song Dataset
        # For now, return a placeholder DataFrame
        logger.warning("Using placeholder Million Song Dataset - implement actual loading")
        
        # Create sample data
        n_samples = 10000
        return pd.DataFrame({
            'title': [f'Track {i}' for i in range(n_samples)],
            'artist': [f'Artist {i // 100}' for i in range(n_samples)],
            'genres': [['rock', 'pop'] for _ in range(n_samples)],
            'moods': [['energetic', 'happy'] for _ in range(n_samples)],
            'energy': np.random.uniform(0, 1, n_samples),
            'valence': np.random.uniform(0, 1, n_samples),
            'danceability': np.random.uniform(0, 1, n_samples),
            'acousticness': np.random.uniform(0, 1, n_samples),
            'instrumentalness': np.random.uniform(0, 1, n_samples),
            'liveness': np.random.uniform(0, 1, n_samples),
            'speechiness': np.random.uniform(0, 1, n_samples),
            'tempo': np.random.uniform(60, 200, n_samples),
            'popularity': np.random.randint(0, 100, n_samples),
            'duration_ms': np.random.randint(120000, 300000, n_samples),
            'release_year': np.random.randint(1950, 2024, n_samples),
            'explicit': np.random.choice([True, False], n_samples)
        })
    
    def _load_timbrality_data(self) -> pd.DataFrame:
        """Load Timbrality dataset with user interactions and item metadata."""
        # This would load from your Supabase database
        logger.warning("Using placeholder Timbrality data - implement actual loading")
        
        # Create sample data
        n_samples = 50000
        n_users = 1000
        n_artists = 5000
        
        return pd.DataFrame({
            'user_id': np.random.randint(0, n_users, n_samples),
            'track_id': [f'track_{i}' for i in range(n_samples)],
            'artist_id': np.random.randint(0, n_artists, n_samples),
            'rating': np.random.uniform(1, 5, n_samples),
            'genres': [['rock', 'pop'] for _ in range(n_samples)],
            'moods': [['energetic', 'happy'] for _ in range(n_samples)],
            'aoty_score': np.random.uniform(50, 100, n_samples),
            'popularity': np.random.randint(0, 100, n_samples)
        })
    
    def _load_user_interactions(self) -> pd.DataFrame:
        """Load user-item interaction data for collaborative filtering."""
        # This would load from your interaction tracking
        logger.warning("Using placeholder interaction data - implement actual loading")
        
        n_interactions = 100000
        n_users = 1000
        n_items = 10000
        
        return pd.DataFrame({
            'user_id': np.random.randint(0, n_users, n_interactions),
            'item_id': np.random.randint(0, n_items, n_interactions),
            'rating': np.random.uniform(1, 5, n_interactions)
        })
    
    def _load_lastfm_scrobbles(self) -> Dict[str, pd.DataFrame]:
        """
        Load Last.fm scrobble data for session processing.
        
        Returns:
            Dictionary mapping user_id to scrobbles DataFrame
        """
        # This would load from your Last.fm data storage
        logger.warning("Using placeholder Last.fm data - implement actual loading from database")
        
        # Generate sample scrobble data for multiple users
        user_scrobbles = {}
        
        for user_id in range(100):  # Sample 100 users
            n_scrobbles = np.random.randint(500, 5000)  # Variable activity
            
            # Generate realistic timestamps (spread over 6 months)
            start_date = pd.Timestamp.now() - pd.Timedelta(days=180)
            timestamps = pd.date_range(
                start=start_date,
                periods=n_scrobbles,
                freq=pd.Timedelta(minutes=np.random.exponential(15))  # Variable gaps
            )
            
            # Generate artist/track data
            artists = [f'Artist_{np.random.randint(0, 1000)}' for _ in range(n_scrobbles)]
            albums = [f'Album_{np.random.randint(0, 2000)}' for _ in range(n_scrobbles)]
            tracks = [f'Track_{i}' for i in range(n_scrobbles)]
            durations = np.random.normal(210, 60, n_scrobbles)  # ~3.5 min average
            
            scrobbles_df = pd.DataFrame({
                'timestamp': timestamps[:n_scrobbles],
                'artist': artists,
                'album': albums,
                'track': tracks,
                'duration': durations,
                'artist_mbid': [f'mbid_{i}' for i in range(n_scrobbles)],
                'album_mbid': [f'mbid_{i}' for i in range(n_scrobbles)],
                'track_mbid': [f'mbid_{i}' for i in range(n_scrobbles)]
            })
            
            user_scrobbles[f'user_{user_id}'] = scrobbles_df
        
        return user_scrobbles
    
    def _count_unique_categories(self, df: pd.DataFrame, column: str) -> int:
        """Count unique categories in a list column."""
        all_categories = set()
        for categories in df[column]:
            if isinstance(categories, list):
                all_categories.update(categories)
        return len(all_categories)
    
    def _generate_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation report for trained models."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models_trained': {
                'audio_predictor': results.get('audio_predictor') is not None,
                'two_tower_model': results.get('two_tower_model') is not None,
                'nmf_model': results.get('nmf_model') is not None,
                'hybrid_model': results.get('hybrid_model') is not None
            },
            'recommendations': []
        }
        
        # Add model-specific metrics if available
        if results.get('hybrid_model'):
            report['recommendations'].append({
                'model': 'hybrid',
                'description': 'Complete hybrid recommendation system ready for deployment',
                'capabilities': [
                    'Collaborative filtering for users with history',
                    'Content-based recommendations for cold start',
                    'Audio feature prediction for enhanced content understanding',
                    'Explainable recommendations'
                ]
            })
        
        return report


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Timbrality ML models')
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Initialize and run pipeline
    pipeline = TimbrialityTrainingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETED")
    print("="*50)
    
    for model_name, model in results.items():
        if model_name == 'evaluation':
            continue
        status = "✓ SUCCESS" if model is not None else "✗ FAILED"
        print(f"{model_name:20}: {status}")
    
    if results.get('evaluation'):
        print(f"\nEvaluation report saved to: {config['output_dir']}/evaluation_report.json")
        with open(f"{config['output_dir']}/evaluation_report.json", 'w') as f:
            json.dump(results['evaluation'], f, indent=2)
    
    print("="*50)


if __name__ == "__main__":
    main()
