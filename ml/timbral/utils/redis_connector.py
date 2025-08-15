"""
Redis connector for Upstash integration.

This module provides Redis connectivity for caching recommendations,
embeddings, and other data in the Timbral recommendation system.
"""

import redis
import json
import pickle
from typing import Any, Optional, Dict, List
import logging
from datetime import timedelta

from ..config.settings import settings

logger = logging.getLogger(__name__)


class RedisConnector:
    """
    Redis connector for Upstash integration.
    
    This class handles all Redis operations including caching
    recommendations, embeddings, and other system data.
    """
    
    def __init__(self):
        """
        Initialize Redis connection.
        """
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """
        Establish connection to Redis (Upstash).
        """
        try:
            # Connect to Redis using URL from settings
            if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            else:
                # Fallback for local Redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            # Don't raise - allow graceful degradation
    
    def set_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        ttl: int = 3600
    ) -> bool:
        """
        Cache user recommendations in Redis.
        
        Args:
            user_id: User ID
            recommendations: List of recommendations
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.redis_client:
            return False
            
        try:
            key = f"recommendations:user:{user_id}"
            value = json.dumps(recommendations)
            
            self.redis_client.setex(key, ttl, value)
            logger.debug(f"Cached recommendations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache recommendations for user {user_id}: {e}")
            return False
    
    def get_recommendations(
        self,
        user_id: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached recommendations for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Cached recommendations or None
        """
        if not self.redis_client:
            return None
            
        try:
            key = f"recommendations:user:{user_id}"
            value = self.redis_client.get(key)
            
            if value:
                recommendations = json.loads(value)
                logger.debug(f"Retrieved cached recommendations for user {user_id}")
                return recommendations
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve recommendations for user {user_id}: {e}")
            return None
    
    def set_embeddings(
        self,
        key: str,
        embeddings: Dict[str, Any],
        ttl: int = 86400
    ) -> bool:
        """
        Cache embeddings in Redis.
        
        Args:
            key: Cache key
            embeddings: Embeddings data
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.redis_client:
            return False
            
        try:
            cache_key = f"embeddings:{key}"
            # Use pickle for numpy arrays
            value = pickle.dumps(embeddings)
            
            self.redis_client.setex(cache_key, ttl, value)
            logger.debug(f"Cached embeddings with key {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings with key {key}: {e}")
            return False
    
    def get_embeddings(
        self,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached embeddings.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embeddings or None
        """
        if not self.redis_client:
            return None
            
        try:
            cache_key = f"embeddings:{key}"
            value = self.redis_client.get(cache_key)
            
            if value:
                embeddings = pickle.loads(value)
                logger.debug(f"Retrieved cached embeddings with key {key}")
                return embeddings
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings with key {key}: {e}")
            return None
    
    def set_model_cache(
        self,
        model_name: str,
        model_data: Any,
        ttl: int = 86400
    ) -> bool:
        """
        Cache model data in Redis.
        
        Args:
            model_name: Name of the model
            model_data: Model data to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            # TODO: Implement model caching
            # - Serialize model data
            # - Set with TTL
            # - Handle large models
            pass
            
        except Exception as e:
            logger.error(f"Failed to cache model {model_name}: {e}")
            return False
    
    def get_model_cache(
        self,
        model_name: str
    ) -> Optional[Any]:
        """
        Retrieve cached model data.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Cached model data or None
        """
        try:
            # TODO: Implement model retrieval
            # - Get from Redis
            # - Deserialize data
            # - Handle missing models
            pass
            
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_name}: {e}")
            return None
    
    def delete_key(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Key to delete
            
        Returns:
            Success status
        """
        try:
            # TODO: Implement key deletion
            # - Delete key from Redis
            # - Handle missing keys gracefully
            pass
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.
        
        Returns:
            Health status dictionary
        """
        try:
            # TODO: Implement health check
            # - Test connection
            # - Check basic operations
            # - Return status information
            pass
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self):
        """
        Close Redis connection.
        """
        if self.redis_client:
            self.redis_client.close() 