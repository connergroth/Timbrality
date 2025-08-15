#!/usr/bin/env python3
"""
Simple test script for collaborative filtering implementation.

This script tests the NMF-based collaborative filtering without requiring Jupyter.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add the project to Python path
sys.path.append(os.path.abspath('.'))

# Import our custom modules
from timbral.models.nmf_model import NMFModel
from timbral.utils.data_loader import DataLoader
from timbral.logic.trainer import ModelTrainer
from timbral.utils.redis_connector import RedisConnector


def create_sample_data():
    """Create synthetic user-item interaction data."""
    print("Creating sample data...")
    
    np.random.seed(42)
    
    # Parameters
    n_users = 50
    n_items = 100
    n_interactions = 1000
    
    # Generate random interactions
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.uniform(1, 5, n_interactions)
    
    # Create DataFrame
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Remove duplicates and keep highest rating
    interactions_df = interactions_df.groupby(['user_id', 'item_id'])['rating'].max().reset_index()
    
    print(f"Generated {len(interactions_df)} unique interactions")
    print(f"Users: {interactions_df['user_id'].nunique()}")
    print(f"Items: {interactions_df['item_id'].nunique()}")
    
    return interactions_df


def test_data_loading(interactions_df):
    """Test data loading functionality."""
    print("\n=== Testing Data Loading ===")
    
    data_loader = DataLoader()
    user_item_matrix = data_loader.create_user_item_matrix(interactions_df)
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Matrix sparsity: {(user_item_matrix == 0).sum().sum() / user_item_matrix.size:.3f}")
    
    return data_loader, user_item_matrix


def test_model_training(user_item_matrix):
    """Test NMF model training."""
    print("\n=== Testing Model Training ===")
    
    trainer = ModelTrainer()
    
    # Train model
    nmf_model = trainer.train_nmf_model(
        user_item_matrix=user_item_matrix,
        n_components=10,
        random_state=42
    )
    
    print(f"Model trained successfully!")
    print(f"User factors shape: {nmf_model.user_factors.shape}")
    print(f"Item factors shape: {nmf_model.item_factors.shape}")
    
    return nmf_model


def test_predictions(nmf_model, user_item_matrix):
    """Test model predictions."""
    print("\n=== Testing Predictions ===")
    
    # Test predictions for a few user-item pairs
    test_users = np.array([0, 1, 2])
    test_items = np.array([0, 5, 10])
    
    predictions = nmf_model.predict(test_users, test_items)
    
    print("Sample predictions:")
    for user, item, pred in zip(test_users, test_items, predictions):
        actual = user_item_matrix.iloc[user, item]
        print(f"User {user}, Item {item}: Predicted={pred:.3f}, Actual={actual:.3f}")


def test_recommendations(nmf_model):
    """Test recommendation generation."""
    print("\n=== Testing Recommendations ===")
    
    user_id = 0
    top_k = 5
    
    recommended_items, scores = nmf_model.get_top_recommendations(user_id, top_k)
    
    print(f"Top {top_k} recommendations for User {user_id}:")
    for i, (item, score) in enumerate(zip(recommended_items, scores)):
        print(f"{i+1}. Item {item}: Score={score:.3f}")


def test_model_persistence(nmf_model):
    """Test model saving and loading."""
    print("\n=== Testing Model Persistence ===")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    model_path = "models/test_nmf_model.pkl"
    
    # Save model
    nmf_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model
    loaded_model = NMFModel()
    loaded_model.load(model_path)
    
    print("Model loaded successfully!")
    
    # Test predictions are the same
    original_pred = nmf_model.predict([0], [0])
    loaded_pred = loaded_model.predict([0], [0])
    
    print(f"Prediction consistency check:")
    print(f"Original: {original_pred[0]:.6f}")
    print(f"Loaded: {loaded_pred[0]:.6f}")
    print(f"Difference: {abs(original_pred[0] - loaded_pred[0]):.10f}")
    
    return loaded_model


def test_evaluation(interactions_df, data_loader):
    """Test model evaluation."""
    print("\n=== Testing Model Evaluation ===")
    
    # Split data
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42
    )
    
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    # Create train matrix
    train_matrix = data_loader.create_user_item_matrix(train_interactions)
    
    # Train model
    train_model = NMFModel(n_components=10, random_state=42)
    train_model.fit(train_matrix.values)
    
    # Evaluate on test set
    test_users = test_interactions['user_id'].values
    test_items = test_interactions['item_id'].values
    test_ratings = test_interactions['rating'].values
    
    # Map to matrix indices
    test_user_indices = [data_loader.user_to_idx.get(user, -1) for user in test_users]
    test_item_indices = [data_loader.item_to_idx.get(item, -1) for item in test_items]
    
    # Filter valid indices
    valid_indices = [(i, u, it) for i, (u, it) in enumerate(zip(test_user_indices, test_item_indices)) 
                     if u >= 0 and it >= 0 and u < train_model.n_users and it < train_model.n_items]
    
    if valid_indices:
        original_indices, valid_users, valid_items = zip(*valid_indices)
        valid_ratings = test_ratings[list(original_indices)]
        
        predictions = train_model.predict(np.array(valid_users), np.array(valid_items))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(valid_ratings, predictions))
        mae = np.mean(np.abs(valid_ratings - predictions))
        
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test MAE: {mae:.3f}")
    else:
        print("No valid test samples found")


def test_redis_integration():
    """Test Redis caching (optional)."""
    print("\n=== Testing Redis Integration ===")
    
    redis_connector = RedisConnector()
    
    if redis_connector.redis_client:
        print("Redis connection successful!")
        
        # Test recommendations caching
        sample_recs = [
            {"item_id": 1, "score": 0.95},
            {"item_id": 2, "score": 0.87}
        ]
        
        success = redis_connector.set_recommendations(123, sample_recs)
        print(f"Caching recommendations: {'Success' if success else 'Failed'}")
        
        cached_recs = redis_connector.get_recommendations(123)
        print(f"Retrieved recommendations: {cached_recs is not None}")
        
    else:
        print("Redis not available - will fall back gracefully")


def main():
    """Run all tests."""
    print("Testing Collaborative Filtering Implementation\n")
    
    try:
        # Create sample data
        interactions_df = create_sample_data()
        
        # Test data loading
        data_loader, user_item_matrix = test_data_loading(interactions_df)
        
        # Test model training
        nmf_model = test_model_training(user_item_matrix)
        
        # Test predictions
        test_predictions(nmf_model, user_item_matrix)
        
        # Test recommendations
        test_recommendations(nmf_model)
        
        # Test model persistence
        loaded_model = test_model_persistence(nmf_model)
        
        # Test evaluation
        test_evaluation(interactions_df, data_loader)
        
        # Test Redis integration
        test_redis_integration()
        
        print("\nAll tests completed successfully!")
        print("\nThe collaborative filtering implementation is ready for integration with your ingestion pipeline!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()