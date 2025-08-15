"""
Non-negative Matrix Factorization (NMF) model for collaborative filtering.

This module implements NMF-based collaborative filtering for music recommendations,
handling user-item interaction matrices and generating latent factor representations.
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, Optional, Dict, Any
import joblib
import os

from ..config.settings import settings, Constants


class NMFModel:
    """
    Non-negative Matrix Factorization model for music recommendations.
    
    This class implements NMF-based collaborative filtering to learn
    latent representations of users and items from interaction data.
    """
    
    def __init__(
        self,
        n_components: int = None,
        random_state: int = None,
        max_iter: int = None,
        tol: float = None
    ):
        """
        Initialize NMF model with specified parameters.
        
        Args:
            n_components: Number of latent factors
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for optimization
            tol: Tolerance for convergence
        """
        self.n_components = n_components or settings.NMF_N_COMPONENTS
        self.random_state = random_state or settings.NMF_RANDOM_STATE
        self.max_iter = max_iter or Constants.NMF_MAX_ITER
        self.tol = tol or Constants.NMF_TOL
        
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        self.user_factors = None
        self.item_factors = None
        self.is_fitted = False
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, user_item_matrix: np.ndarray) -> 'NMFModel':
        """
        Fit NMF model to user-item interaction matrix.
        
        Args:
            user_item_matrix: 2D array of user-item interactions
            
        Returns:
            Self for method chaining
        """
        try:
            # Handle sparse matrices
            if hasattr(user_item_matrix, 'toarray'):
                user_item_matrix = user_item_matrix.toarray()
            
            # Apply log transformation to reduce effect of outliers
            processed_matrix = np.log1p(user_item_matrix)
            
            # Fit NMF model
            self.user_factors = self.model.fit_transform(processed_matrix)
            self.item_factors = self.model.components_
            
            # Store mapping information
            self.n_users, self.n_items = processed_matrix.shape
            self.is_fitted = True
            
            print(f"NMF model fitted with {self.n_components} components")
            print(f"User factors shape: {self.user_factors.shape}")
            print(f"Item factors shape: {self.item_factors.shape}")
            
            return self
            
        except Exception as e:
            print(f"Error fitting NMF model: {e}")
            raise
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict interaction scores for user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Predicted interaction scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Ensure inputs are numpy arrays
            user_ids = np.asarray(user_ids)
            item_ids = np.asarray(item_ids)
            
            # Validate indices are within bounds
            if np.any(user_ids >= self.n_users) or np.any(user_ids < 0):
                raise ValueError(f"User IDs must be in range [0, {self.n_users})")
            if np.any(item_ids >= self.n_items) or np.any(item_ids < 0):
                raise ValueError(f"Item IDs must be in range [0, {self.n_items})")
            
            # Compute predictions as dot product of user and item factors
            user_features = self.user_factors[user_ids]
            item_features = self.item_factors[:, item_ids].T
            
            # For batch prediction
            if user_features.ndim == 2 and item_features.ndim == 2:
                predictions = np.sum(user_features * item_features, axis=1)
            else:
                predictions = np.dot(user_features, item_features)
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            raise
    
    def get_user_embeddings(self, user_ids: np.ndarray) -> np.ndarray:
        """
        Get latent factor embeddings for users.
        
        Args:
            user_ids: Array of user IDs
            
        Returns:
            User embeddings matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting embeddings")
        
        user_ids = np.asarray(user_ids)
        if np.any(user_ids >= self.n_users) or np.any(user_ids < 0):
            raise ValueError(f"User IDs must be in range [0, {self.n_users})")
        
        return self.user_factors[user_ids]
    
    def get_item_embeddings(self, item_ids: np.ndarray) -> np.ndarray:
        """
        Get latent factor embeddings for items.
        
        Args:
            item_ids: Array of item IDs
            
        Returns:
            Item embeddings matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting embeddings")
        
        item_ids = np.asarray(item_ids)
        if np.any(item_ids >= self.n_items) or np.any(item_ids < 0):
            raise ValueError(f"Item IDs must be in range [0, {self.n_items})")
        
        return self.item_factors[:, item_ids].T
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'NMFModel':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.n_users = model_data['n_users']
        self.n_items = model_data['n_items']
        self.n_components = model_data['n_components']
        self.random_state = model_data['random_state']
        self.max_iter = model_data['max_iter']
        self.tol = model_data['tol']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        return self
    
    def get_top_recommendations(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already interacted with
            
        Returns:
            Tuple of (item_ids, scores) for top recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id >= self.n_users or user_id < 0:
            raise ValueError(f"User ID must be in range [0, {self.n_users})")
        
        # Get user's latent factors
        user_factors = self.user_factors[user_id:user_id+1]
        
        # Compute scores for all items
        scores = np.dot(user_factors, self.item_factors).flatten()
        
        # Get top-k items
        top_items = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_items]
        
        return top_items, top_scores 