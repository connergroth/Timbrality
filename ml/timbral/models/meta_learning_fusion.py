"""
Meta-Learning Fusion Architecture for Adaptive Uncertainty Weighting.

This module implements a meta-learning approach where the model learns to 
adaptively weight uncertain item features based on session context, framing
confidence + session fusion as a higher-order learning problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetaLearningFusion(nn.Module):
    """
    Meta-learning module that learns how to weight predictions based on context.
    
    Key insight: Rather than fixed fusion weights, learn context-dependent
    weighting strategies that adapt to:
    - User session patterns (exploratory vs routine)
    - Item uncertainty levels (predicted vs actual features)
    - Historical prediction accuracy for similar contexts
    
    This transforms the hybrid model into a meta-learner that learns to learn
    optimal weighting strategies for different recommendation scenarios.
    """
    
    def __init__(
        self,
        user_context_dim: int = 64,     # User session features
        item_context_dim: int = 32,     # Item uncertainty features  
        cf_embedding_dim: int = 64,     # Collaborative filtering embedding
        content_embedding_dim: int = 64, # Content-based embedding
        meta_hidden_dim: int = 128,
        num_meta_layers: int = 3
    ):
        """
        Initialize meta-learning fusion module.
        
        Args:
            user_context_dim: Dimension of user session context
            item_context_dim: Dimension of item uncertainty context
            cf_embedding_dim: CF model embedding dimension
            content_embedding_dim: Content model embedding dimension
            meta_hidden_dim: Meta-network hidden dimension
            num_meta_layers: Number of meta-network layers
        """
        super().__init__()
        
        # Context encoders
        self.user_context_encoder = nn.Sequential(
            nn.Linear(user_context_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.item_context_encoder = nn.Sequential(
            nn.Linear(item_context_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Meta-network that generates fusion weights
        meta_input_dim = meta_hidden_dim + cf_embedding_dim + content_embedding_dim
        
        meta_layers = []
        for i in range(num_meta_layers):
            if i == 0:
                meta_layers.extend([
                    nn.Linear(meta_input_dim, meta_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            elif i == num_meta_layers - 1:
                # Output layer: generate weights + confidence adjustment
                meta_layers.append(nn.Linear(meta_hidden_dim, 4))  # [cf_weight, content_weight, cf_conf, content_conf]
            else:
                meta_layers.extend([
                    nn.Linear(meta_hidden_dim, meta_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
        
        self.meta_network = nn.Sequential(*meta_layers)
        
        # Attention mechanism for temporal session context
        self.session_attention = nn.MultiheadAttention(
            embed_dim=meta_hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Uncertainty calibration network
        self.uncertainty_calibrator = nn.Sequential(
            nn.Linear(item_context_dim + 2, 32),  # +2 for base cf/content scores
            nn.ReLU(),
            nn.Linear(32, 2),  # Output uncertainty estimates
            nn.Sigmoid()
        )
        
    def forward(
        self,
        cf_score: torch.Tensor,                    # [batch_size]
        content_score: torch.Tensor,               # [batch_size]
        cf_embedding: torch.Tensor,                # [batch_size, cf_embedding_dim]
        content_embedding: torch.Tensor,           # [batch_size, content_embedding_dim]
        user_session_context: torch.Tensor,       # [batch_size, user_context_dim]
        item_uncertainty_context: torch.Tensor,   # [batch_size, item_context_dim]
        session_history: Optional[torch.Tensor] = None  # [batch_size, seq_len, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Meta-learning forward pass that adapts fusion weights to context.
        
        Returns:
            Dict containing:
            - fused_score: Final recommendation score
            - cf_weight: Learned weight for collaborative filtering
            - content_weight: Learned weight for content-based
            - uncertainty_estimate: Estimated prediction uncertainty
            - attention_weights: Session attention weights (if history provided)
        """
        batch_size = cf_score.size(0)
        
        # 1. Encode contexts
        user_context_emb = self.user_context_encoder(user_session_context)
        item_context_emb = self.item_context_encoder(item_uncertainty_context)
        
        # 2. Session attention (if history available)
        attention_weights = None
        if session_history is not None:
            # Apply attention to incorporate temporal session patterns
            attended_context, attention_weights = self.session_attention(
                user_context_emb.unsqueeze(1),  # Query: current context
                session_history,                # Key/Value: session history
                session_history
            )
            user_context_emb = attended_context.squeeze(1)
        
        # 3. Combine all contexts
        combined_context = torch.cat([
            user_context_emb,
            item_context_emb
        ], dim=1)
        
        # 4. Meta-network generates adaptive weights
        meta_input = torch.cat([
            combined_context,
            cf_embedding,
            content_embedding
        ], dim=1)
        
        meta_output = self.meta_network(meta_input)  # [batch_size, 4]
        
        # 5. Extract weights and confidence adjustments
        raw_weights = meta_output[:, :2]  # [cf_weight, content_weight]
        confidence_adjustments = torch.sigmoid(meta_output[:, 2:])  # [cf_conf_adj, content_conf_adj]
        
        # Apply softmax to ensure weights sum to 1
        fusion_weights = F.softmax(raw_weights, dim=1)
        cf_weight = fusion_weights[:, 0]
        content_weight = fusion_weights[:, 1]
        
        # 6. Uncertainty estimation
        uncertainty_input = torch.cat([
            item_uncertainty_context,
            cf_score.unsqueeze(1),
            content_score.unsqueeze(1)
        ], dim=1)
        
        uncertainty_estimate = self.uncertainty_calibrator(uncertainty_input)
        
        # 7. Apply confidence-aware fusion
        # Weight scores by both learned weights and uncertainty
        cf_confidence = confidence_adjustments[:, 0]
        content_confidence = confidence_adjustments[:, 1]
        
        weighted_cf = cf_weight * cf_score * cf_confidence
        weighted_content = content_weight * content_score * content_confidence
        
        # Final fusion with uncertainty normalization
        total_confidence = cf_confidence + content_confidence + 1e-8  # Avoid division by zero
        fused_score = (weighted_cf + weighted_content) / total_confidence
        
        return {
            'fused_score': fused_score,
            'cf_weight': cf_weight,
            'content_weight': content_weight,
            'cf_confidence': cf_confidence,
            'content_confidence': content_confidence,
            'uncertainty_estimate': uncertainty_estimate,
            'attention_weights': attention_weights
        }


class SessionContextExtractor(nn.Module):
    """Extracts relevant session context for meta-learning fusion."""
    
    def __init__(self, session_feature_dim: int = 18, output_dim: int = 64):
        """Initialize session context extractor."""
        super().__init__()
        
        self.context_network = nn.Sequential(
            nn.Linear(session_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        
        # Temporal pattern encoder for session sequences
        self.temporal_encoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
    def forward(
        self, 
        session_features: torch.Tensor,
        temporal_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract session context.
        
        Args:
            session_features: Current session features [batch_size, session_feature_dim]
            temporal_sequence: Historical session sequence [batch_size, seq_len, session_feature_dim]
            
        Returns:
            Tuple of (current_context, temporal_context)
        """
        # Current session context
        current_context = self.context_network(session_features)
        
        # Temporal context (if available)
        temporal_context = None
        if temporal_sequence is not None:
            # Encode each timestep
            seq_len = temporal_sequence.size(1)
            batch_size = temporal_sequence.size(0)
            
            # Reshape for processing
            flat_sequence = temporal_sequence.view(-1, temporal_sequence.size(-1))
            encoded_sequence = self.context_network(flat_sequence)
            encoded_sequence = encoded_sequence.view(batch_size, seq_len, -1)
            
            # Apply LSTM
            temporal_output, _ = self.temporal_encoder(encoded_sequence)
            temporal_context = temporal_output  # [batch_size, seq_len, hidden_dim]
        
        return current_context, temporal_context


class UncertaintyContextExtractor(nn.Module):
    """Extracts item uncertainty context for meta-learning."""
    
    def __init__(self, audio_feature_dim: int = 8, metadata_dim: int = 10, output_dim: int = 32):
        """Initialize uncertainty context extractor."""
        super().__init__()
        
        self.uncertainty_network = nn.Sequential(
            nn.Linear(audio_feature_dim + metadata_dim + 3, output_dim),  # +3 for prediction source indicators
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_confidence: torch.Tensor,
        metadata_completeness: torch.Tensor,
        is_predicted: torch.Tensor,
        prediction_variance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract uncertainty context from item features.
        
        Args:
            audio_features: Audio feature predictions [batch_size, 8]
            audio_confidence: Confidence scores [batch_size, 8]
            metadata_completeness: Metadata completeness score [batch_size, 1]
            is_predicted: Whether features are predicted vs actual [batch_size, 1]
            prediction_variance: Optional prediction variance [batch_size, 8]
            
        Returns:
            Uncertainty context embedding [batch_size, output_dim]
        """
        # Aggregate confidence and uncertainty indicators
        avg_confidence = audio_confidence.mean(dim=1, keepdim=True)
        confidence_std = audio_confidence.std(dim=1, keepdim=True)
        
        # Include prediction variance if available
        if prediction_variance is not None:
            avg_variance = prediction_variance.mean(dim=1, keepdim=True)
        else:
            avg_variance = torch.zeros_like(avg_confidence)
        
        # Combine uncertainty indicators
        uncertainty_context = torch.cat([
            audio_features.mean(dim=1, keepdim=True),  # Feature magnitude
            avg_confidence,
            confidence_std,
            metadata_completeness,
            is_predicted,
            avg_variance
        ], dim=1)
        
        return self.uncertainty_network(uncertainty_context)


# Integration example
class MetaHybridRecommender(nn.Module):
    """
    Hybrid recommender with meta-learning fusion.
    
    This represents the complete integration of the meta-learning approach
    into the recommendation pipeline.
    """
    
    def __init__(
        self,
        base_cf_model,
        base_content_model,
        session_feature_dim: int = 18,
        meta_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.cf_model = base_cf_model
        self.content_model = base_content_model
        
        self.session_extractor = SessionContextExtractor(session_feature_dim)
        self.uncertainty_extractor = UncertaintyContextExtractor()
        
        self.meta_fusion = MetaLearningFusion(
            user_context_dim=64,
            item_context_dim=32,
            cf_embedding_dim=64,
            content_embedding_dim=64,
            meta_hidden_dim=meta_hidden_dim
        )
        
    def forward(self, user_data, item_data, session_features):
        """
        Meta-learning hybrid recommendation.
        
        The model learns to adaptively weight CF vs content predictions
        based on user session context and item uncertainty.
        """
        # Get base model predictions and embeddings
        cf_output = self.cf_model(user_data, item_data)
        content_output = self.content_model(user_data, item_data)
        
        # Extract contexts
        user_context, session_history = self.session_extractor(session_features)
        item_uncertainty_context = self.uncertainty_extractor(
            item_data['audio_features'],
            item_data['audio_confidence'],
            item_data['metadata_completeness'],
            item_data['is_predicted']
        )
        
        # Meta-learning fusion
        fusion_output = self.meta_fusion(
            cf_score=cf_output['score'],
            content_score=content_output['score'],
            cf_embedding=cf_output['embedding'],
            content_embedding=content_output['embedding'],
            user_session_context=user_context,
            item_uncertainty_context=item_uncertainty_context,
            session_history=session_history
        )
        
        return fusion_output





