"""
Audio Feature Predictor using Million Song Dataset.

This module implements a model to predict Spotify-like audio features
from text metadata and available track information.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)


class AudioFeaturePredictor(nn.Module):
    """
    Neural network to predict audio features from text and metadata.
    
    Predicts: energy, valence, danceability, acousticness, instrumentalness,
    liveness, speechiness, tempo (normalized 0-1)
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        num_genres: int = 1000,
        num_moods: int = 500
    ):
        """
        Initialize audio feature predictor.
        
        Args:
            vocab_size: Size of text vocabulary
            embedding_dim: Dimension of text embeddings
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            num_genres: Number of unique genres
            num_moods: Number of unique moods
        """
        super().__init__()
        
        # Text embeddings for title/artist
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Genre and mood embeddings
        self.genre_embedding = nn.Embedding(num_genres, 64)
        self.mood_embedding = nn.Embedding(num_moods, 32)
        
        # Metadata processing (NO popularity or user-specific data to avoid leakage)
        self.metadata_linear = nn.Linear(6, 32)  # duration, year, track_number, explicit, etc.
        
        # Calculate input dimension
        # Text features (title + artist averaged) + genre + mood + metadata
        input_dim = embedding_dim + 64 + 32 + 32
        
        # Hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layers for different audio features
        self.energy_head = nn.Linear(prev_dim, 1)
        self.valence_head = nn.Linear(prev_dim, 1)
        self.danceability_head = nn.Linear(prev_dim, 1)
        self.acousticness_head = nn.Linear(prev_dim, 1)
        self.instrumentalness_head = nn.Linear(prev_dim, 1)
        self.liveness_head = nn.Linear(prev_dim, 1)
        self.speechiness_head = nn.Linear(prev_dim, 1)
        self.tempo_head = nn.Linear(prev_dim, 1)  # Will be normalized to 0-1
        
        # Activation for bounded outputs (0-1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
    
    def forward(
        self,
        title_tokens: torch.Tensor,
        artist_tokens: torch.Tensor,
        genre_ids: torch.Tensor,
        mood_ids: torch.Tensor,
        metadata: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            title_tokens: Tokenized track titles [batch_size, seq_len]
            artist_tokens: Tokenized artist names [batch_size, seq_len]
            genre_ids: Genre IDs [batch_size, max_genres]
            mood_ids: Mood IDs [batch_size, max_moods]
            metadata: Numerical metadata [batch_size, metadata_dim]
            
        Returns:
            Dictionary of predicted audio features
        """
        batch_size = title_tokens.size(0)
        
        # Text embeddings (average over sequence length)
        title_emb = self.text_embedding(title_tokens).mean(dim=1)  # [batch_size, embedding_dim]
        artist_emb = self.text_embedding(artist_tokens).mean(dim=1)  # [batch_size, embedding_dim]
        text_features = (title_emb + artist_emb) / 2  # Average title and artist
        
        # Genre embeddings (average over genres)
        genre_mask = (genre_ids != 0).float()  # Mask for padding
        genre_emb = self.genre_embedding(genre_ids)  # [batch_size, max_genres, 64]
        genre_features = (genre_emb * genre_mask.unsqueeze(-1)).sum(dim=1)  # [batch_size, 64]
        genre_count = genre_mask.sum(dim=1, keepdim=True).clamp(min=1)
        genre_features = genre_features / genre_count
        
        # Mood embeddings (average over moods)
        mood_mask = (mood_ids != 0).float()  # Mask for padding
        mood_emb = self.mood_embedding(mood_ids)  # [batch_size, max_moods, 32]
        mood_features = (mood_emb * mood_mask.unsqueeze(-1)).sum(dim=1)  # [batch_size, 32]
        mood_count = mood_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mood_features = mood_features / mood_count
        
        # Metadata features
        metadata_features = self.metadata_linear(metadata)  # [batch_size, 32]
        
        # Concatenate all features
        combined_features = torch.cat([
            text_features,
            genre_features,
            mood_features,
            metadata_features
        ], dim=1)  # [batch_size, input_dim]
        
        # Pass through hidden layers
        hidden = self.hidden_layers(combined_features)
        
        # Predict audio features
        predictions = {
            'energy': self.sigmoid(self.energy_head(hidden)),
            'valence': self.sigmoid(self.valence_head(hidden)),
            'danceability': self.sigmoid(self.danceability_head(hidden)),
            'acousticness': self.sigmoid(self.acousticness_head(hidden)),
            'instrumentalness': self.sigmoid(self.instrumentalness_head(hidden)),
            'liveness': self.sigmoid(self.liveness_head(hidden)),
            'speechiness': self.sigmoid(self.speechiness_head(hidden)),
            'tempo': self.sigmoid(self.tempo_head(hidden))  # Normalized tempo
        }
        
        return predictions


class AudioFeatureTrainer:
    """Trainer for the audio feature prediction model."""
    
    def __init__(
        self,
        model: AudioFeaturePredictor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: AudioFeaturePredictor model
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Preprocessing objects
        self.text_tokenizer = None
        self.genre_encoder = None
        self.mood_encoder = None
        self.metadata_scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Prepare training data from DataFrame.
        
        Args:
            df: DataFrame with tracks and audio features
            
        Returns:
            Tuple of (inputs_dict, targets_dict)
        """
        # Build vocabulary from titles and artists
        all_text = []
        for text in df['title'].fillna('') + ' ' + df['artist'].fillna(''):
            all_text.extend(text.lower().split())
        
        vocab = ['<PAD>', '<UNK>'] + list(set(all_text))[:19998]  # Top 20k words
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        def tokenize_text(text, max_len=20):
            tokens = text.lower().split()[:max_len]
            token_ids = [word_to_idx.get(token, 1) for token in tokens]  # 1 = UNK
            # Pad to max_len
            token_ids.extend([0] * (max_len - len(token_ids)))  # 0 = PAD
            return token_ids
        
        # Tokenize text
        title_tokens = np.array([tokenize_text(title) for title in df['title'].fillna('')])
        artist_tokens = np.array([tokenize_text(artist) for artist in df['artist'].fillna('')])
        
        # Encode genres and moods
        self.genre_encoder = LabelEncoder()
        self.mood_encoder = LabelEncoder()
        
        # Flatten genres and moods for encoding
        all_genres = []
        all_moods = []
        for genres in df['genres'].fillna('[]'):
            if isinstance(genres, str):
                import ast
                try:
                    genres = ast.literal_eval(genres)
                except:
                    genres = []
            all_genres.extend(genres)
        
        for moods in df['moods'].fillna('[]'):
            if isinstance(moods, str):
                import ast
                try:
                    moods = ast.literal_eval(moods)
                except:
                    moods = []
            all_moods.extend(moods)
        
        # Fit encoders
        unique_genres = ['<PAD>'] + list(set(all_genres))
        unique_moods = ['<PAD>'] + list(set(all_moods))
        
        genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        mood_to_idx = {mood: idx for idx, mood in enumerate(unique_moods)}
        
        def encode_categories(categories, cat_to_idx, max_len=10):
            if isinstance(categories, str):
                import ast
                try:
                    categories = ast.literal_eval(categories)
                except:
                    categories = []
            
            encoded = [cat_to_idx.get(cat, 0) for cat in categories[:max_len]]
            encoded.extend([0] * (max_len - len(encoded)))  # Pad
            return encoded
        
        genre_ids = np.array([
            encode_categories(genres, genre_to_idx) 
            for genres in df['genres'].fillna('[]')
        ])
        
        mood_ids = np.array([
            encode_categories(moods, mood_to_idx)
            for moods in df['moods'].fillna('[]')
        ])
        
        # Prepare metadata features (NO popularity, ratings, or user-specific data)
        metadata_features = []
        feature_cols = [
            'duration_ms', 'release_year', 'explicit',
            'track_number', 'album_total_tracks', 'decade'  # decade derived from year
        ]
        
        for col in feature_cols:
            if col == 'decade':
                # Derive decade from release_year
                if 'release_year' in df.columns:
                    values = (df['release_year'].fillna(2000) // 10) * 10
                else:
                    values = pd.Series([2000] * len(df))
            elif col in df.columns:
                values = df[col].fillna(0).astype(float)
            else:
                values = pd.Series([0] * len(df))
            metadata_features.append(values)
        
        metadata = np.column_stack(metadata_features)
        metadata = self.metadata_scaler.fit_transform(metadata)
        
        # Prepare inputs
        inputs = {
            'title_tokens': torch.LongTensor(title_tokens),
            'artist_tokens': torch.LongTensor(artist_tokens),
            'genre_ids': torch.LongTensor(genre_ids),
            'mood_ids': torch.LongTensor(mood_ids),
            'metadata': torch.FloatTensor(metadata)
        }
        
        # Prepare targets (audio features)
        audio_features = [
            'energy', 'valence', 'danceability', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo'
        ]
        
        targets = {}
        for feature in audio_features:
            if feature in df.columns:
                values = df[feature].fillna(0.5).astype(float)
                # Normalize tempo to 0-1 range (assuming 60-200 BPM range)
                if feature == 'tempo':
                    values = (values - 60) / (200 - 60)
                    values = np.clip(values, 0, 1)
                targets[feature] = torch.FloatTensor(values.values)
            else:
                # Default values if feature not available
                targets[feature] = torch.FloatTensor([0.5] * len(df))
        
        # Store encoders for later use
        self.word_to_idx = word_to_idx
        self.genre_to_idx = genre_to_idx
        self.mood_to_idx = mood_to_idx
        
        return inputs, targets
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(**inputs)
            
            # Calculate loss (sum of MSE for each audio feature)
            loss = 0
            for feature in predictions:
                feature_loss = self.criterion(
                    predictions[feature].squeeze(),
                    targets[feature]
                )
                loss += feature_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(**inputs)
                
                loss = 0
                for feature in predictions:
                    feature_loss = self.criterion(
                        predictions[feature].squeeze(),
                        targets[feature]
                    )
                    loss += feature_loss
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict_audio_features(
        self,
        title: str,
        artist: str,
        genres: List[str],
        moods: List[str],
        metadata: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict audio features for a single track.
        
        Args:
            title: Track title
            artist: Artist name
            genres: List of genres
            moods: List of moods
            metadata: Dictionary of metadata features
            
        Returns:
            Dictionary of predicted audio features
        """
        self.model.eval()
        
        # Tokenize text
        def tokenize_text(text, max_len=20):
            tokens = text.lower().split()[:max_len]
            token_ids = [self.word_to_idx.get(token, 1) for token in tokens]
            token_ids.extend([0] * (max_len - len(token_ids)))
            return token_ids
        
        title_tokens = torch.LongTensor([tokenize_text(title)]).to(self.device)
        artist_tokens = torch.LongTensor([tokenize_text(artist)]).to(self.device)
        
        # Encode genres and moods
        def encode_categories(categories, cat_to_idx, max_len=10):
            encoded = [cat_to_idx.get(cat, 0) for cat in categories[:max_len]]
            encoded.extend([0] * (max_len - len(encoded)))
            return encoded
        
        genre_ids = torch.LongTensor([
            encode_categories(genres, self.genre_to_idx)
        ]).to(self.device)
        
        mood_ids = torch.LongTensor([
            encode_categories(moods, self.mood_to_idx)
        ]).to(self.device)
        
        # Prepare metadata (NO popularity or rating data to avoid leakage)
        feature_names = [
            'duration_ms', 'release_year', 'explicit',
            'track_number', 'album_total_tracks', 'decade'
        ]
        
        metadata_values = []
        for name in feature_names:
            if name == 'decade':
                year = metadata.get('release_year', 2000)
                metadata_values.append((year // 10) * 10)
            else:
                metadata_values.append(metadata.get(name, 0))
        
        metadata_array = np.array([metadata_values])
        metadata_array = self.metadata_scaler.transform(metadata_array)
        metadata_tensor = torch.FloatTensor(metadata_array).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(
                title_tokens=title_tokens,
                artist_tokens=artist_tokens,
                genre_ids=genre_ids,
                mood_ids=mood_ids,
                metadata=metadata_tensor
            )
        
        # Convert to dictionary
        result = {}
        for feature, tensor in predictions.items():
            value = tensor.cpu().item()
            # Denormalize tempo
            if feature == 'tempo':
                value = value * (200 - 60) + 60  # Convert back to BPM
            result[feature] = value
        
        return result
    
    def save(self, filepath: str):
        """Save model and preprocessors."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'word_to_idx': self.word_to_idx,
            'genre_to_idx': self.genre_to_idx,
            'mood_to_idx': self.mood_to_idx,
            'metadata_scaler': self.metadata_scaler,
            'model_config': {
                'vocab_size': len(self.word_to_idx),
                'num_genres': len(self.genre_to_idx),
                'num_moods': len(self.mood_to_idx)
            }
        }, filepath)
        logger.info(f"Audio feature predictor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and preprocessors."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load preprocessors
        self.word_to_idx = checkpoint['word_to_idx']
        self.genre_to_idx = checkpoint['genre_to_idx']
        self.mood_to_idx = checkpoint['mood_to_idx']
        self.metadata_scaler = checkpoint['metadata_scaler']
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Audio feature predictor loaded from {filepath}")
