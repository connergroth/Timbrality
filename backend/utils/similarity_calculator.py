"""
Utility for efficiently calculating and managing user similarities
"""
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy import text, and_, func
from sqlalchemy.orm import Session
from collections import defaultdict
import math

from models.database import get_db
from models.collaborative_filtering import LastfmUser, UserTrackInteraction, UserSimilarity

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Optimized similarity calculation with batch processing and caching"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.user_profiles_cache = {}
    
    async def calculate_all_similarities(self, 
                                       recalculate_existing: bool = False,
                                       min_shared_tracks: int = 3,
                                       min_similarity: float = 0.01) -> Dict[str, Any]:
        """
        Calculate similarities for all user pairs efficiently
        
        Args:
            recalculate_existing: Whether to recalculate existing similarities
            min_shared_tracks: Minimum shared tracks required for similarity calculation
            min_similarity: Minimum similarity to store in database
            
        Returns:
            Dictionary with calculation statistics
        """
        start_time = datetime.now()
        
        try:
            db = next(get_db())
            
            # Get all active users
            users = db.query(LastfmUser).filter(LastfmUser.is_active == True).all()
            user_ids = [str(user.id) for user in users]
            
            logger.info(f"Calculating similarities for {len(users)} users")
            
            # Load user profiles into memory for efficiency
            user_profiles = await self._load_user_profiles(user_ids)
            
            stats = {
                "total_users": len(users),
                "total_pairs": len(user_ids) * (len(user_ids) - 1) // 2,
                "similarities_calculated": 0,
                "similarities_stored": 0,
                "pairs_skipped": 0,
                "processing_time": 0
            }
            
            # Process in batches to avoid memory issues
            batch_count = 0
            similarities_to_store = []
            
            for i, user1_id in enumerate(user_ids):
                for j, user2_id in enumerate(user_ids[i+1:], i+1):
                    
                    # Check if similarity already exists and we're not recalculating
                    if not recalculate_existing:
                        existing = db.query(UserSimilarity).filter(
                            ((UserSimilarity.user_id_1 == user1_id) & (UserSimilarity.user_id_2 == user2_id)) |
                            ((UserSimilarity.user_id_1 == user2_id) & (UserSimilarity.user_id_2 == user1_id))
                        ).first()
                        
                        if existing:
                            stats["pairs_skipped"] += 1
                            continue
                    
                    # Calculate similarity
                    similarity_data = self._calculate_pair_similarity_fast(
                        user1_id, user2_id, user_profiles, min_shared_tracks
                    )
                    
                    stats["similarities_calculated"] += 1
                    
                    if similarity_data and similarity_data["similarity"] >= min_similarity:
                        similarities_to_store.append(similarity_data)
                        stats["similarities_stored"] += 1
                    
                    # Store batch when it reaches batch_size
                    if len(similarities_to_store) >= self.batch_size:
                        await self._store_similarity_batch(similarities_to_store, recalculate_existing)
                        similarities_to_store = []
                        batch_count += 1
                        logger.info(f"Stored batch {batch_count} ({stats['similarities_calculated']} calculated so far)")
            
            # Store remaining similarities
            if similarities_to_store:
                await self._store_similarity_batch(similarities_to_store, recalculate_existing)
            
            stats["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Similarity calculation completed: {stats['similarities_stored']} stored, "
                       f"{stats['similarities_calculated']} calculated in {stats['processing_time']:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {str(e)}")
            return {"error": str(e)}
        finally:
            db.close()
    
    async def _load_user_profiles(self, user_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Load all user track interaction profiles into memory for fast similarity calculation
        
        Returns user profiles as {user_id: {track_id: normalized_score}}
        """
        try:
            db = next(get_db())
            
            # Load all interactions for these users
            query = text("""
                SELECT lastfm_user_id, track_id, play_count
                FROM user_track_interactions 
                WHERE lastfm_user_id = ANY(:user_ids)
                AND play_count > 0
            """)
            
            result = db.execute(query, {"user_ids": user_ids}).fetchall()
            
            # Build user profiles with TF-IDF style normalization
            user_profiles = defaultdict(dict)
            user_track_counts = defaultdict(int)
            track_user_counts = defaultdict(int)
            
            # First pass: collect raw data
            for user_id, track_id, play_count in result:
                user_profiles[user_id][track_id] = float(play_count)
                user_track_counts[user_id] += 1
                track_user_counts[track_id] += 1
            
            # Second pass: normalize with TF-IDF
            total_users = len(user_ids)
            
            for user_id in user_profiles:
                user_total_plays = sum(user_profiles[user_id].values())
                
                for track_id in user_profiles[user_id]:
                    # TF: normalized play count
                    tf = user_profiles[user_id][track_id] / user_total_plays
                    
                    # IDF: inverse document frequency
                    idf = math.log(total_users / track_user_counts[track_id])
                    
                    # TF-IDF score
                    user_profiles[user_id][track_id] = tf * idf
            
            logger.info(f"Loaded profiles for {len(user_profiles)} users")
            return dict(user_profiles)
            
        except Exception as e:
            logger.error(f"Error loading user profiles: {str(e)}")
            return {}
        finally:
            db.close()
    
    def _calculate_pair_similarity_fast(self, 
                                      user1_id: str, 
                                      user2_id: str, 
                                      user_profiles: Dict[str, Dict[str, float]],
                                      min_shared_tracks: int = 3) -> Optional[Dict[str, Any]]:
        """
        Fast cosine similarity calculation using pre-loaded profiles
        """
        try:
            profile1 = user_profiles.get(user1_id, {})
            profile2 = user_profiles.get(user2_id, {})
            
            if not profile1 or not profile2:
                return None
            
            # Find shared tracks
            shared_tracks = set(profile1.keys()) & set(profile2.keys())
            
            if len(shared_tracks) < min_shared_tracks:
                return None
            
            # Calculate cosine similarity
            dot_product = sum(profile1[track] * profile2[track] for track in shared_tracks)
            
            norm1 = math.sqrt(sum(profile1[track] ** 2 for track in shared_tracks))
            norm2 = math.sqrt(sum(profile2[track] ** 2 for track in shared_tracks))
            
            if norm1 == 0 or norm2 == 0:
                return None
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # Calculate additional metadata
            total_tracks1 = len(profile1)
            total_tracks2 = len(profile2)
            jaccard_similarity = len(shared_tracks) / len(set(profile1.keys()) | set(profile2.keys()))
            
            return {
                "user_id_1": user1_id,
                "user_id_2": user2_id,
                "similarity": float(cosine_similarity),
                "similarity_type": "cosine_tfidf",
                "shared_tracks_count": len(shared_tracks),
                "jaccard_similarity": float(jaccard_similarity),
                "total_tracks_user1": total_tracks1,
                "total_tracks_user2": total_tracks2
            }
            
        except Exception as e:
            logger.error(f"Error calculating pair similarity: {str(e)}")
            return None
    
    async def _store_similarity_batch(self, similarities: List[Dict[str, Any]], recalculate_existing: bool):
        """Store a batch of similarities to database"""
        try:
            db = next(get_db())
            
            for sim_data in similarities:
                if recalculate_existing:
                    # Delete existing similarity first
                    db.query(UserSimilarity).filter(
                        ((UserSimilarity.user_id_1 == sim_data["user_id_1"]) & 
                         (UserSimilarity.user_id_2 == sim_data["user_id_2"])) |
                        ((UserSimilarity.user_id_1 == sim_data["user_id_2"]) & 
                         (UserSimilarity.user_id_2 == sim_data["user_id_1"]))
                    ).delete()
                
                # Create new similarity record
                similarity = UserSimilarity(
                    user_id_1=sim_data["user_id_1"],
                    user_id_2=sim_data["user_id_2"],
                    similarity_score=sim_data["similarity"],
                    similarity_type=sim_data["similarity_type"],
                    shared_tracks_count=sim_data["shared_tracks_count"],
                    calculated_at=datetime.now()
                )
                
                db.add(similarity)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing similarity batch: {str(e)}")
            db.rollback()
        finally:
            db.close()
    
    async def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored similarities"""
        try:
            db = next(get_db())
            
            # Basic statistics
            total_similarities = db.query(UserSimilarity).count()
            
            # Similarity distribution
            similarity_stats = db.query(
                func.min(UserSimilarity.similarity_score).label('min_sim'),
                func.max(UserSimilarity.similarity_score).label('max_sim'), 
                func.avg(UserSimilarity.similarity_score).label('avg_sim'),
                func.stddev(UserSimilarity.similarity_score).label('std_sim')
            ).first()
            
            # Shared tracks statistics
            shared_tracks_stats = db.query(
                func.min(UserSimilarity.shared_tracks_count).label('min_shared'),
                func.max(UserSimilarity.shared_tracks_count).label('max_shared'),
                func.avg(UserSimilarity.shared_tracks_count).label('avg_shared')
            ).first()
            
            # Users with most neighbors
            user_neighbor_counts = db.execute(text("""
                SELECT user_id, neighbor_count FROM (
                    SELECT user_id_1 as user_id, COUNT(*) as neighbor_count
                    FROM user_similarities
                    GROUP BY user_id_1
                    UNION ALL
                    SELECT user_id_2 as user_id, COUNT(*) as neighbor_count  
                    FROM user_similarities
                    GROUP BY user_id_2
                ) combined
                ORDER BY neighbor_count DESC
                LIMIT 10
            """)).fetchall()
            
            return {
                "total_similarities": total_similarities,
                "similarity_distribution": {
                    "min": float(similarity_stats.min_sim) if similarity_stats.min_sim else 0,
                    "max": float(similarity_stats.max_sim) if similarity_stats.max_sim else 0,
                    "mean": float(similarity_stats.avg_sim) if similarity_stats.avg_sim else 0,
                    "std": float(similarity_stats.std_sim) if similarity_stats.std_sim else 0
                },
                "shared_tracks_distribution": {
                    "min": shared_tracks_stats.min_shared if shared_tracks_stats.min_shared else 0,
                    "max": shared_tracks_stats.max_shared if shared_tracks_stats.max_shared else 0,  
                    "mean": float(shared_tracks_stats.avg_shared) if shared_tracks_stats.avg_shared else 0
                },
                "top_connected_users": [
                    {"user_id": row[0], "neighbor_count": row[1]} 
                    for row in user_neighbor_counts
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting similarity statistics: {str(e)}")
            return {"error": str(e)}
        finally:
            db.close()
    
    async def cleanup_old_similarities(self, days_old: int = 7) -> Dict[str, int]:
        """Remove similarities older than specified days"""
        try:
            db = next(get_db())
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Delete old similarities
            deleted_count = db.query(UserSimilarity).filter(
                UserSimilarity.calculated_at < cutoff_date
            ).delete()
            
            db.commit()
            
            logger.info(f"Cleaned up {deleted_count} old similarities")
            
            return {"deleted_count": deleted_count}
            
        except Exception as e:
            logger.error(f"Error cleaning up similarities: {str(e)}")
            return {"error": str(e), "deleted_count": 0}
        finally:
            db.close()
    
    async def recalculate_user_similarities(self, user_id: str) -> Dict[str, Any]:
        """Recalculate similarities for a specific user"""
        try:
            db = next(get_db())
            
            # Get all other users
            all_users = db.query(LastfmUser).filter(
                and_(LastfmUser.is_active == True, LastfmUser.id != user_id)
            ).all()
            
            other_user_ids = [str(user.id) for user in all_users]
            user_ids = [user_id] + other_user_ids
            
            # Load profiles
            user_profiles = await self._load_user_profiles(user_ids)
            
            # Delete existing similarities for this user
            db.query(UserSimilarity).filter(
                (UserSimilarity.user_id_1 == user_id) | (UserSimilarity.user_id_2 == user_id)
            ).delete()
            
            # Calculate new similarities
            similarities_to_store = []
            
            for other_user_id in other_user_ids:
                similarity_data = self._calculate_pair_similarity_fast(
                    user_id, other_user_id, user_profiles, min_shared_tracks=3
                )
                
                if similarity_data and similarity_data["similarity"] >= 0.01:
                    similarities_to_store.append(similarity_data)
            
            # Store new similarities
            await self._store_similarity_batch(similarities_to_store, recalculate_existing=True)
            
            return {
                "user_id": user_id,
                "similarities_calculated": len(similarities_to_store),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error recalculating similarities for user {user_id}: {str(e)}")
            return {"user_id": user_id, "status": "error", "error": str(e)}
        finally:
            db.close()