"""
Two-Tower Neural Network for Content-Based Music Recommendations.

This module implements a two-tower architecture where user preferences
and item content are encoded separately, then combined via dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel
import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Text encoder using pre-trained BERT for music metadata."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 384,
        freeze_bert: bool = True
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: Pre-trained model name
            output_dim: Output embedding dimension
            freeze_bert: Whether to freeze BERT parameters
        """
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Project BERT output to desired dimension
        bert_dim = self.bert.config.hidden_size
        if bert_dim != output_dim:
            self.projection = nn.Linear(bert_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, text_batch: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encode batch of text strings.
        
        Args:
            text_batch: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT embeddings
        with torch.no_grad() if self.training is False else torch.enable_grad():
            outputs = self.bert(**encoded)
            # Use [CLS] token or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Project to output dimension
        embeddings = self.projection(embeddings)
        
        return embeddings


class UserTower(nn.Module):
    """User tower for encoding user preferences and behavior."""
    
    def __init__(
        self,
        num_users: int,
        num_genres: int,
        num_moods: int,
        embedding_dim: int = 128,
        output_dim: int = 64
    ):
        """
        Initialize user tower.
        
        Args:
            num_users: Number of unique users
            num_genres: Number of unique genres
            num_moods: Number of unique moods
            embedding_dim: Dimension of embeddings
            output_dim: Final output dimension
        """
        super().__init__()
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Genre and mood preference embeddings
        self.genre_embedding = nn.Embedding(num_genres, 64)
        self.mood_embedding = nn.Embedding(num_moods, 64)
        
        # Activity and behavior features
        self.activity_linear = nn.Linear(10, 32)  # Play counts, session length, etc.
        
        # Combine all user features
        total_dim = embedding_dim + 64 + 64 + 32  # user + genre + mood + activity
        
        self.user_mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.genre_embedding.weight, 0, 0.1)
        nn.init.normal_(self.mood_embedding.weight, 0, 0.1)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        genre_preferences: torch.Tensor,
        mood_preferences: torch.Tensor,
        activity_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through user tower.
        
        Args:
            user_ids: User IDs [batch_size]
            genre_preferences: Genre preference weights [batch_size, max_genres]
            mood_preferences: Mood preference weights [batch_size, max_moods]
            activity_features: User activity features [batch_size, activity_dim]
            
        Returns:
            User embeddings [batch_size, output_dim]
        """
        # User embedding
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        
        # Genre preferences (weighted average)
        genre_emb = self.genre_embedding.weight.unsqueeze(0)  # [1, num_genres, 64]
        genre_pref = torch.matmul(
            genre_preferences.unsqueeze(1),  # [batch_size, 1, num_genres]
            genre_emb  # [1, num_genres, 64]
        ).squeeze(1)  # [batch_size, 64]
        
        # Mood preferences (weighted average)
        mood_emb = self.mood_embedding.weight.unsqueeze(0)  # [1, num_moods, 64]
        mood_pref = torch.matmul(
            mood_preferences.unsqueeze(1),  # [batch_size, 1, num_moods]
            mood_emb  # [1, num_moods, 64]
        ).squeeze(1)  # [batch_size, 64]
        
        # Activity features
        activity_emb = self.activity_linear(activity_features)  # [batch_size, 32]
        
        # Concatenate all features
        user_features = torch.cat([
            user_emb,
            genre_pref,
            mood_pref,
            activity_emb
        ], dim=1)  # [batch_size, total_dim]
        
        # Pass through MLP
        user_vector = self.user_mlp(user_features)  # [batch_size, output_dim]
        
        # L2 normalize for dot product similarity
        user_vector = F.normalize(user_vector, p=2, dim=1)
        
        return user_vector


class ItemTower(nn.Module):
    """Item tower for encoding track content and metadata."""
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        num_artists: int,
        num_genres: int,
        num_moods: int,
        text_dim: int = 384,
        output_dim: int = 64
    ):
        """
        Initialize item tower.
        
        Args:
            text_encoder: Pre-trained text encoder
            num_artists: Number of unique artists
            num_genres: Number of unique genres
            num_moods: Number of unique moods
            text_dim: Text encoding dimension
            output_dim: Final output dimension
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        
        # Artist embedding
        self.artist_embedding = nn.Embedding(num_artists, 128)
        
        # Genre and mood embeddings
        self.genre_embedding = nn.Embedding(num_genres, 64)
        self.mood_embedding = nn.Embedding(num_moods, 64)
        
        # Audio features (predicted) + confidence scores
        self.audio_linear = nn.Linear(16, 32)  # 8 audio features + 8 confidence scores
        
        # Rating and popularity features
        self.rating_linear = nn.Linear(5, 16)  # ratings, popularity, etc.
        
        # Combine all item features
        total_dim = text_dim + 128 + 64 + 64 + 32 + 16
        
        self.item_mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.artist_embedding.weight, 0, 0.1)
        nn.init.normal_(self.genre_embedding.weight, 0, 0.1)
        nn.init.normal_(self.mood_embedding.weight, 0, 0.1)
    
    def forward(
        self,
        track_texts: List[str],
        artist_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        mood_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_confidence: torch.Tensor,
        rating_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through item tower.
        
        Args:
            track_texts: List of track text descriptions
            artist_ids: Artist IDs [batch_size]
            genre_ids: Genre IDs [batch_size, max_genres]
            mood_ids: Mood IDs [batch_size, max_moods]
            audio_features: Predicted audio features [batch_size, 8]
            audio_confidence: Confidence scores for audio features [batch_size, 8]
            rating_features: Rating/popularity features [batch_size, 5]
            
        Returns:
            Item embeddings [batch_size, output_dim]
        """
        batch_size = len(track_texts)
        
        # Text encoding
        text_emb = self.text_encoder(track_texts)  # [batch_size, text_dim]
        
        # Artist embedding
        artist_emb = self.artist_embedding(artist_ids)  # [batch_size, 128]
        
        # Genre embeddings (average over genres)
        genre_mask = (genre_ids != 0).float()  # Mask for padding
        genre_emb = self.genre_embedding(genre_ids)  # [batch_size, max_genres, 64]
        genre_features = (genre_emb * genre_mask.unsqueeze(-1)).sum(dim=1)  # [batch_size, 64]
        genre_count = genre_mask.sum(dim=1, keepdim=True).clamp(min=1)
        genre_features = genre_features / genre_count
        
        # Mood embeddings (average over moods)
        mood_mask = (mood_ids != 0).float()  # Mask for padding
        mood_emb = self.mood_embedding(mood_ids)  # [batch_size, max_moods, 64]
        mood_features = (mood_emb * mood_mask.unsqueeze(-1)).sum(dim=1)  # [batch_size, 64]
        mood_count = mood_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mood_features = mood_features / mood_count
        
        # Audio features combined with confidence scores
        audio_with_confidence = torch.cat([audio_features, audio_confidence], dim=1)  # [batch_size, 16]
        audio_emb = self.audio_linear(audio_with_confidence)  # [batch_size, 32]
        
        # Rating features
        rating_emb = self.rating_linear(rating_features)  # [batch_size, 16]
        
        # Concatenate all features
        item_features = torch.cat([
            text_emb,
            artist_emb,
            genre_features,
            mood_features,
            audio_emb,
            rating_emb
        ], dim=1)  # [batch_size, total_dim]
        
        # Pass through MLP
        item_vector = self.item_mlp(item_features)  # [batch_size, output_dim]
        
        # L2 normalize for dot product similarity
        item_vector = F.normalize(item_vector, p=2, dim=1)
        
        return item_vector


class TwoTowerModel(nn.Module):
    """Two-tower model for content-based music recommendations."""
    
    def __init__(
        self,
        num_users: int,
        num_artists: int,
        num_genres: int,
        num_moods: int,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 64,
        text_dim: int = 384
    ):
        """
        Initialize two-tower model.
        
        Args:
            num_users: Number of unique users
            num_artists: Number of unique artists
            num_genres: Number of unique genres
            num_moods: Number of unique moods
            text_model_name: Pre-trained text model name
            embedding_dim: Final embedding dimension for both towers
            text_dim: Text encoding dimension
        """
        super().__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=text_dim
        )
        
        # User and item towers
        self.user_tower = UserTower(
            num_users=num_users,
            num_genres=num_genres,
            num_moods=num_moods,
            output_dim=embedding_dim
        )
        
        self.item_tower = ItemTower(
            text_encoder=self.text_encoder,
            num_artists=num_artists,
            num_genres=num_genres,
            num_moods=num_moods,
            text_dim=text_dim,
            output_dim=embedding_dim
        )
        
        # Optional: add auxiliary tasks
        self.use_auxiliary = True
        if self.use_auxiliary:
            # User genre prediction (multi-label)
            self.user_genre_head = nn.Linear(embedding_dim, num_genres)
            
            # Item rating prediction
            self.item_rating_head = nn.Linear(embedding_dim, 1)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        user_genre_prefs: torch.Tensor,
        user_mood_prefs: torch.Tensor,
        user_activity: torch.Tensor,
        track_texts: List[str],
        artist_ids: torch.Tensor,
        item_genre_ids: torch.Tensor,
        item_mood_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_confidence: torch.Tensor,
        rating_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through two-tower model.
        
        Returns:
            Dictionary containing:
            - user_embeddings: User tower outputs
            - item_embeddings: Item tower outputs
            - scores: Dot product similarity scores
            - aux_outputs: Auxiliary task outputs (if enabled)
        """
        # Get embeddings from both towers
        user_embeddings = self.user_tower(
            user_ids=user_ids,
            genre_preferences=user_genre_prefs,
            mood_preferences=user_mood_prefs,
            activity_features=user_activity
        )
        
        item_embeddings = self.item_tower(
            track_texts=track_texts,
            artist_ids=artist_ids,
            genre_ids=item_genre_ids,
            mood_ids=item_mood_ids,
            audio_features=audio_features,
            audio_confidence=audio_confidence,
            rating_features=rating_features
        )
        
        # Compute similarity scores (dot product)
        scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        outputs = {
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'scores': scores
        }
        
        # Auxiliary tasks
        if self.use_auxiliary:
            aux_outputs = {}
            
            # User genre prediction
            user_genre_logits = self.user_genre_head(user_embeddings)
            aux_outputs['user_genre_logits'] = user_genre_logits
            
            # Item rating prediction
            item_rating_pred = self.item_rating_head(item_embeddings)
            aux_outputs['item_rating_pred'] = item_rating_pred
            
            outputs['aux_outputs'] = aux_outputs
        
        return outputs
    
    def get_user_embedding(
        self,
        user_id: int,
        genre_prefs: torch.Tensor,
        mood_prefs: torch.Tensor,
        activity: torch.Tensor
    ) -> torch.Tensor:
        """Get embedding for a single user."""
        self.eval()
        with torch.no_grad():
            user_ids = torch.tensor([user_id], device=next(self.parameters()).device)
            
            user_emb = self.user_tower(
                user_ids=user_ids,
                genre_preferences=genre_prefs.unsqueeze(0),
                mood_preferences=mood_prefs.unsqueeze(0),
                activity_features=activity.unsqueeze(0)
            )
            
            return user_emb.squeeze(0)
    
    def get_item_embedding(
        self,
        track_text: str,
        artist_id: int,
        genre_ids: torch.Tensor,
        mood_ids: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_conf: torch.Tensor,
        rating_feats: torch.Tensor
    ) -> torch.Tensor:
        """Get embedding for a single item."""
        self.eval()
        with torch.no_grad():
            artist_ids = torch.tensor([artist_id], device=next(self.parameters()).device)
            
            item_emb = self.item_tower(
                track_texts=[track_text],
                artist_ids=artist_ids,
                genre_ids=genre_ids.unsqueeze(0),
                mood_ids=mood_ids.unsqueeze(0),
                audio_features=audio_feats.unsqueeze(0),
                audio_confidence=audio_conf.unsqueeze(0),
                rating_features=rating_feats.unsqueeze(0)
            )
            
            return item_emb.squeeze(0)
    
    def predict_similarity(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor
    ) -> float:
        """Predict similarity between user and item embeddings."""
        similarity = torch.dot(user_embedding, item_embedding).item()
        return similarity
    
    def recommend_items(
        self,
        user_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recommend top-k items for a user.
        
        Args:
            user_embedding: User embedding [embedding_dim]
            candidate_embeddings: Candidate item embeddings [num_items, embedding_dim]
            top_k: Number of recommendations
            
        Returns:
            Tuple of (item_indices, similarity_scores)
        """
        # Compute similarities
        similarities = torch.matmul(candidate_embeddings, user_embedding)
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        return top_indices, top_scores


class TwoTowerTrainer:
    """Trainer for the two-tower model."""
    
    def __init__(
        self,
        model: TwoTowerModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.text_encoder.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for pre-trained
            {'params': self.model.user_tower.parameters(), 'lr': learning_rate},
            {'params': self.model.item_tower.parameters(), 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Loss functions
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        user_genre_labels: Optional[torch.Tensor] = None,
        item_rating_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs
            positive_scores: Scores for positive pairs
            negative_scores: Scores for negative pairs
            user_genre_labels: User genre preference labels
            item_rating_labels: Item rating labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Main ranking loss
        target = torch.ones_like(positive_scores)
        ranking_loss = self.ranking_loss(positive_scores, negative_scores, target)
        losses['ranking'] = ranking_loss
        
        total_loss = ranking_loss
        
        # Auxiliary losses
        if self.model.use_auxiliary and 'aux_outputs' in outputs:
            aux_outputs = outputs['aux_outputs']
            
            # User genre prediction loss
            if user_genre_labels is not None and 'user_genre_logits' in aux_outputs:
                genre_loss = self.bce_loss(
                    aux_outputs['user_genre_logits'],
                    user_genre_labels
                )
                losses['user_genre'] = genre_loss
                total_loss += 0.1 * genre_loss  # Weight auxiliary loss
            
            # Item rating prediction loss
            if item_rating_labels is not None and 'item_rating_pred' in aux_outputs:
                rating_loss = self.mse_loss(
                    aux_outputs['item_rating_pred'].squeeze(),
                    item_rating_labels
                )
                losses['item_rating'] = rating_loss
                total_loss += 0.1 * rating_loss  # Weight auxiliary loss
        
        losses['total'] = total_loss
        return losses
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, filepath)
        logger.info(f"Two-tower model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Two-tower model loaded from {filepath}")
