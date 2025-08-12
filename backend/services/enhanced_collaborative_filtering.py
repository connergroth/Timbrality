"""
Enhanced Collaborative Filtering Service with 2-hop neighbors, diversity, and advanced scoring
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, text
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from collections import defaultdict, Counter
import math

from models.collaborative_filtering import (
    LastfmUser, UserTrackInteraction, UserAlbumInteraction, 
    UserArtistInteraction, UserSimilarity, CollaborativeRecommendation
)
from models.database import get_db
from services.supabase_collaborative_filtering import SupabaseCollaborativeFilteringService

logger = logging.getLogger(__name__)


class EnhancedCollaborativeFilteringService(SupabaseCollaborativeFilteringService):
    """Enhanced collaborative filtering with 2-hop neighbors and diversity optimization"""
    
    def __init__(self, 
                 min_neighbors: int = 25,
                 max_neighbors: int = 125,
                 similarity_threshold: float = 0.05,
                 cumulative_similarity_target: float = 0.85,
                 diversity_lambda: float = 0.7,
                 content_blend_alpha: float = 0.6):
        super().__init__()
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors  
        self.similarity_threshold = similarity_threshold
        self.cumulative_similarity_target = cumulative_similarity_target
        self.diversity_lambda = diversity_lambda
        self.content_blend_alpha = content_blend_alpha
        
        # Cache for expensive calculations
        self._user_similarity_cache = {}
        self._item_popularity_cache = {}
        self._content_vectors_cache = {}
    
    async def generate_recommendations(self, 
                                     target_user_id: str, 
                                     num_recommendations: int = 20,
                                     include_explanations: bool = True) -> Dict[str, Any]:
        """
        Generate enhanced collaborative filtering recommendations with 2-hop neighbors and diversity
        
        Args:
            target_user_id: ID of user to generate recommendations for
            num_recommendations: Number of final recommendations to return
            include_explanations: Whether to include recommendation explanations
            
        Returns:
            Dictionary with recommendations, scores, explanations, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Find 2-hop neighbors with adaptive k
            neighbors = await self._find_two_hop_neighbors(target_user_id)
            logger.info(f"Found {len(neighbors)} neighbors for user {target_user_id}")
            
            # Step 2: Generate candidate items with base scoring
            candidates = await self._generate_candidates(target_user_id, neighbors)
            logger.info(f"Generated {len(candidates)} candidate items")
            
            # Step 3: Apply popularity de-biasing
            candidates = await self._apply_popularity_debiasing(candidates)
            
            # Step 4: Re-rank with MMR for diversity
            diverse_candidates = await self._apply_mmr_reranking(
                candidates, 
                num_candidates=min(500, len(candidates))
            )
            
            # Step 5: Apply hard caps and gentle boosts
            final_recommendations = await self._apply_caps_and_boosts(
                diverse_candidates, 
                num_recommendations
            )
            
            # Step 6: Blend with content-based recommendations
            if self.content_blend_alpha < 1.0:
                final_recommendations = await self._blend_with_content_based(
                    target_user_id,
                    final_recommendations,
                    num_recommendations
                )
            
            # Step 7: Generate explanations
            explanations = []
            if include_explanations:
                explanations = await self._generate_explanations(
                    target_user_id, 
                    final_recommendations, 
                    neighbors
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store recommendations in database
            await self._store_recommendations(target_user_id, final_recommendations)
            
            return {
                "user_id": target_user_id,
                "recommendations": final_recommendations,
                "explanations": explanations,
                "metadata": {
                    "neighbors_used": len(neighbors),
                    "candidates_generated": len(candidates),
                    "processing_time_seconds": processing_time,
                    "algorithm": "enhanced_2hop_collaborative_filtering",
                    "diversity_lambda": self.diversity_lambda,
                    "content_blend_alpha": self.content_blend_alpha,
                    "quality_metrics": await self._calculate_quality_metrics(final_recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {target_user_id}: {str(e)}")
            return {
                "error": str(e),
                "user_id": target_user_id,
                "recommendations": [],
                "explanations": []
            }
    
    async def _find_two_hop_neighbors(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Find neighbors using 2-hop approach with adaptive k selection
        
        Returns list of neighbors with their similarity scores
        """
        try:
            # Step 1: Find direct neighbors (1-hop)
            direct_neighbors = await self._find_direct_neighbors(user_id)
            
            # Step 2: Find 2-hop neighbors through direct neighbors
            two_hop_neighbors = {}
            
            for neighbor in direct_neighbors:
                neighbor_id = neighbor['user_id']
                sim_to_neighbor = neighbor['similarity']
                
                # Get this neighbor's neighbors
                neighbor_neighbors = await self._find_direct_neighbors(neighbor_id)
                
                for nn in neighbor_neighbors:
                    nn_id = nn['user_id']
                    
                    # Skip if it's the original user or already a direct neighbor
                    if nn_id == user_id or any(n['user_id'] == nn_id for n in direct_neighbors):
                        continue
                    
                    # Calculate 2-hop similarity with decay
                    sim_2hop = sim_to_neighbor * nn['similarity']
                    
                    # Only include if above threshold
                    if sim_2hop >= self.similarity_threshold:
                        if nn_id not in two_hop_neighbors:
                            two_hop_neighbors[nn_id] = {
                                'user_id': nn_id,
                                'similarity': sim_2hop,
                                'hop': 2,
                                'path': f"{user_id} -> {neighbor_id} -> {nn_id}"
                            }
                        else:
                            # Keep the highest similarity path
                            if sim_2hop > two_hop_neighbors[nn_id]['similarity']:
                                two_hop_neighbors[nn_id]['similarity'] = sim_2hop
                                two_hop_neighbors[nn_id]['path'] = f"{user_id} -> {neighbor_id} -> {nn_id}"
            
            # Combine direct and 2-hop neighbors
            all_neighbors = direct_neighbors + list(two_hop_neighbors.values())
            
            # Sort by similarity
            all_neighbors.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply adaptive k selection
            selected_neighbors = self._apply_adaptive_k_selection(all_neighbors)
            
            logger.info(f"Selected {len(selected_neighbors)} neighbors "
                       f"({len([n for n in selected_neighbors if n['hop'] == 1])} direct, "
                       f"{len([n for n in selected_neighbors if n['hop'] == 2])} 2-hop)")
            
            return selected_neighbors
            
        except Exception as e:
            logger.error(f"Error finding 2-hop neighbors for user {user_id}: {str(e)}")
            return []
    
    async def _find_direct_neighbors(self, user_id: str) -> List[Dict[str, Any]]:
        """Find direct neighbors (1-hop) for a user"""
        try:
            db = next(get_db())
            
            # Get cached similarities or calculate them
            similarities = db.query(UserSimilarity).filter(
                func.or_(
                    UserSimilarity.user_id_1 == user_id,
                    UserSimilarity.user_id_2 == user_id
                )
            ).all()
            
            neighbors = []
            for sim in similarities:
                other_user_id = sim.user_id_2 if sim.user_id_1 == user_id else sim.user_id_1
                neighbors.append({
                    'user_id': other_user_id,
                    'similarity': sim.similarity_score,
                    'hop': 1,
                    'shared_tracks': sim.shared_tracks_count,
                    'shared_albums': sim.shared_albums_count,
                    'shared_artists': sim.shared_artists_count
                })
            
            return sorted(neighbors, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding direct neighbors for user {user_id}: {str(e)}")
            return []
        finally:
            db.close()
    
    def _apply_adaptive_k_selection(self, neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply adaptive k selection based on cumulative similarity threshold
        
        Selects neighbors until cumulative similarity >= target (min 25, max 125)
        """
        if not neighbors:
            return []
        
        selected = []
        cumulative_sim = 0.0
        
        for neighbor in neighbors:
            selected.append(neighbor)
            cumulative_sim += neighbor['similarity']
            
            # Check if we've met our criteria
            if (len(selected) >= self.min_neighbors and 
                cumulative_sim >= self.cumulative_similarity_target):
                break
            
            # Hard limit
            if len(selected) >= self.max_neighbors:
                break
        
        # Ensure we have at least min_neighbors if available
        if len(selected) < self.min_neighbors and len(neighbors) > len(selected):
            additional_needed = min(self.min_neighbors - len(selected), 
                                  len(neighbors) - len(selected))
            selected.extend(neighbors[len(selected):len(selected) + additional_needed])
        
        logger.info(f"Adaptive k selected {len(selected)} neighbors "
                   f"(cumulative similarity: {cumulative_sim:.3f})")
        
        return selected
    
    async def _generate_candidates(self, 
                                 user_id: str, 
                                 neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate candidate items from neighbors with base scoring
        
        Base score: score(item) = Σ_n sim(u,n) * log(1 + plays_{n,item})
        """
        try:
            db = next(get_db())
            
            # Get user's existing interactions to exclude
            user_interactions = db.query(UserTrackInteraction).filter(
                UserTrackInteraction.lastfm_user_id == user_id
            ).all()
            user_track_ids = {interaction.track_id for interaction in user_interactions}
            
            # Aggregate scores from all neighbors
            candidate_scores = defaultdict(float)
            candidate_metadata = defaultdict(lambda: {
                'contributing_neighbors': [],
                'total_neighbor_plays': 0,
                'neighbor_count': 0
            })
            
            for neighbor in neighbors:
                neighbor_id = neighbor['user_id']
                neighbor_sim = neighbor['similarity']
                
                # Get neighbor's track interactions
                neighbor_interactions = db.query(UserTrackInteraction).filter(
                    UserTrackInteraction.lastfm_user_id == neighbor_id
                ).all()
                
                for interaction in neighbor_interactions:
                    track_id = interaction.track_id
                    
                    # Skip if user already has this track
                    if track_id in user_track_ids:
                        continue
                    
                    # Calculate weighted score with log dampening
                    play_count = max(1, interaction.play_count)  # Avoid log(0)
                    weighted_score = neighbor_sim * math.log(1 + play_count)
                    
                    candidate_scores[track_id] += weighted_score
                    
                    # Store metadata for explanations
                    candidate_metadata[track_id]['contributing_neighbors'].append({
                        'neighbor_id': neighbor_id,
                        'similarity': neighbor_sim,
                        'play_count': play_count,
                        'contribution': weighted_score
                    })
                    candidate_metadata[track_id]['total_neighbor_plays'] += play_count
                    candidate_metadata[track_id]['neighbor_count'] += 1
            
            # Convert to list format with metadata
            candidates = []
            for track_id, score in candidate_scores.items():
                candidates.append({
                    'track_id': track_id,
                    'base_score': score,
                    'final_score': score,  # Will be modified by later steps
                    'metadata': dict(candidate_metadata[track_id])
                })
            
            # Sort by score
            candidates.sort(key=lambda x: x['base_score'], reverse=True)
            
            logger.info(f"Generated {len(candidates)} candidates with base scoring")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error generating candidates for user {user_id}: {str(e)}")
            return []
        finally:
            db.close()
    
    async def _apply_popularity_debiasing(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply popularity de-biasing by dividing by log(1 + global_popularity)
        """
        try:
            if not candidates:
                return candidates
            
            # Get global popularity for all candidate tracks
            track_ids = [c['track_id'] for c in candidates]
            popularity_scores = await self._get_global_popularity(track_ids)
            
            # Apply de-biasing
            for candidate in candidates:
                track_id = candidate['track_id']
                global_pop = popularity_scores.get(track_id, 1)
                
                # De-bias the score
                popularity_penalty = math.log(1 + global_pop)
                candidate['debiased_score'] = candidate['base_score'] / popularity_penalty
                candidate['final_score'] = candidate['debiased_score']
                candidate['metadata']['global_popularity'] = global_pop
                candidate['metadata']['popularity_penalty'] = popularity_penalty
            
            # Re-sort by debiased score
            candidates.sort(key=lambda x: x['debiased_score'], reverse=True)
            
            logger.info("Applied popularity de-biasing to candidates")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error applying popularity de-biasing: {str(e)}")
            return candidates
    
    async def _get_global_popularity(self, track_ids: List[str]) -> Dict[str, int]:
        """Get global popularity (total plays across all users) for tracks"""
        try:
            if track_ids in self._item_popularity_cache:
                return self._item_popularity_cache[track_ids]
            
            db = next(get_db())
            
            # Query total play counts across all users
            popularity_query = text("""
                SELECT track_id, SUM(play_count) as total_plays
                FROM user_track_interactions 
                WHERE track_id = ANY(:track_ids)
                GROUP BY track_id
            """)
            
            result = db.execute(popularity_query, {"track_ids": track_ids}).fetchall()
            
            popularity_scores = {}
            for row in result:
                popularity_scores[row[0]] = row[1]
            
            # Cache the results
            self._item_popularity_cache[tuple(track_ids)] = popularity_scores
            
            return popularity_scores
            
        except Exception as e:
            logger.error(f"Error getting global popularity: {str(e)}")
            return {}
        finally:
            db.close()
    
    async def _apply_mmr_reranking(self, 
                                 candidates: List[Dict[str, Any]], 
                                 num_candidates: int = 500) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity re-ranking
        
        MMR = λ * relevance - (1-λ) * max_sim_to_selected
        """
        try:
            if len(candidates) <= num_candidates:
                return candidates
            
            # Take top candidates by score for MMR
            top_candidates = candidates[:num_candidates]
            
            # Get content vectors for similarity calculation
            track_ids = [c['track_id'] for c in top_candidates]
            content_vectors = await self._get_content_vectors(track_ids)
            
            if not content_vectors:
                logger.warning("No content vectors available, skipping MMR reranking")
                return top_candidates
            
            selected = []
            remaining = list(top_candidates)
            
            # Select first item (highest relevance)
            if remaining:
                best = max(remaining, key=lambda x: x['final_score'])
                selected.append(best)
                remaining.remove(best)
            
            # Iteratively select items using MMR
            while remaining and len(selected) < num_candidates:
                best_mmr_score = -float('inf')
                best_candidate = None
                
                for candidate in remaining:
                    relevance = candidate['final_score']
                    
                    # Calculate max similarity to already selected items
                    max_sim_to_selected = 0.0
                    candidate_vector = content_vectors.get(candidate['track_id'])
                    
                    if candidate_vector is not None:
                        for selected_item in selected:
                            selected_vector = content_vectors.get(selected_item['track_id'])
                            if selected_vector is not None:
                                # Calculate cosine similarity
                                similarity = 1 - cosine(candidate_vector, selected_vector)
                                max_sim_to_selected = max(max_sim_to_selected, similarity)
                    
                    # Calculate MMR score
                    mmr_score = (self.diversity_lambda * relevance - 
                               (1 - self.diversity_lambda) * max_sim_to_selected)
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_candidate = candidate
                
                if best_candidate:
                    best_candidate['mmr_score'] = best_mmr_score
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            logger.info(f"MMR reranking selected {len(selected)} diverse candidates")
            
            return selected
            
        except Exception as e:
            logger.error(f"Error applying MMR reranking: {str(e)}")
            return candidates
    
    async def _get_content_vectors(self, track_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Get content-based feature vectors for tracks
        
        This should use your existing content features (tags, genres, audio features, etc.)
        For now, implementing a placeholder that you can replace with your actual vectors
        """
        try:
            # Check cache first
            cache_key = tuple(sorted(track_ids))
            if cache_key in self._content_vectors_cache:
                return self._content_vectors_cache[cache_key]
            
            # Placeholder implementation - replace with your actual content vectors
            # This should query your tracks table and build vectors from:
            # - Genre tags
            # - Audio features (if available)
            # - AOTY tags
            # - Any other content features
            
            vectors = {}
            
            # For now, create random vectors as placeholder
            # REPLACE THIS with your actual content vector generation
            np.random.seed(42)  # For reproducibility
            for track_id in track_ids:
                # Placeholder: random 50-dimensional vector
                vectors[track_id] = np.random.random(50)
            
            # Cache the results
            self._content_vectors_cache[cache_key] = vectors
            
            return vectors
            
        except Exception as e:
            logger.error(f"Error getting content vectors: {str(e)}")
            return {}
    
    async def _apply_caps_and_boosts(self, 
                                   candidates: List[Dict[str, Any]], 
                                   num_recommendations: int) -> List[Dict[str, Any]]:
        """
        Apply hard caps and gentle boosts:
        - Artist cap: ≤3 tracks per artist in top 20
        - Long-tail boost: +ε for items below 70th percentile popularity
        - Temporal mix: ≥30% from last 24 months
        """
        try:
            # Get track metadata for caps and boosts
            track_metadata = await self._get_track_metadata([c['track_id'] for c in candidates])
            
            # Apply boosts first
            boosted_candidates = []
            for candidate in candidates:
                track_id = candidate['track_id']
                metadata = track_metadata.get(track_id, {})
                
                boosted_candidate = candidate.copy()
                boost_multiplier = 1.0
                
                # Long-tail boost (for tracks below 70th percentile popularity)
                if self._is_long_tail_track(candidate, candidates):
                    boost_multiplier *= 1.2  # 20% boost
                    boosted_candidate['metadata']['long_tail_boost'] = True
                
                # Temporal boost (for recent tracks)
                if self._is_recent_track(metadata):
                    boost_multiplier *= 1.1  # 10% boost
                    boosted_candidate['metadata']['temporal_boost'] = True
                
                boosted_candidate['final_score'] *= boost_multiplier
                boosted_candidate['metadata']['boost_multiplier'] = boost_multiplier
                
                boosted_candidates.append(boosted_candidate)
            
            # Re-sort after boosts
            boosted_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Apply caps while selecting final recommendations
            selected = []
            artist_counts = Counter()
            temporal_count = 0
            
            for candidate in boosted_candidates:
                if len(selected) >= num_recommendations:
                    break
                
                track_id = candidate['track_id']
                metadata = track_metadata.get(track_id, {})
                artist = metadata.get('artist', 'Unknown')
                
                # Artist cap check (≤3 per artist in top recommendations)
                if len(selected) < num_recommendations and artist_counts[artist] >= 3:
                    continue  # Skip this track, artist cap reached
                
                selected.append(candidate)
                artist_counts[artist] += 1
                
                # Count recent tracks for temporal mix requirement
                if self._is_recent_track(metadata):
                    temporal_count += 1
            
            # Ensure temporal mix (≥30% from last 24 months)
            target_recent = max(1, int(0.3 * num_recommendations))
            if temporal_count < target_recent:
                # Need to add more recent tracks, potentially violating artist cap
                logger.info(f"Temporal mix adjustment needed: {temporal_count}/{target_recent} recent tracks")
                
                # Find recent tracks not yet selected
                remaining_recent = []
                selected_track_ids = {c['track_id'] for c in selected}
                
                for candidate in boosted_candidates:
                    track_id = candidate['track_id']
                    if (track_id not in selected_track_ids and 
                        self._is_recent_track(track_metadata.get(track_id, {}))):
                        remaining_recent.append(candidate)
                
                # Replace oldest non-recent tracks with recent ones
                needed = target_recent - temporal_count
                replaced = 0
                
                for i in range(len(selected) - 1, -1, -1):
                    if replaced >= needed or not remaining_recent:
                        break
                    
                    track_metadata_sel = track_metadata.get(selected[i]['track_id'], {})
                    if not self._is_recent_track(track_metadata_sel):
                        # Replace with a recent track
                        selected[i] = remaining_recent.pop(0)
                        replaced += 1
            
            logger.info(f"Applied caps and boosts, selected {len(selected)} recommendations")
            
            return selected
            
        except Exception as e:
            logger.error(f"Error applying caps and boosts: {str(e)}")
            return candidates[:num_recommendations]
    
    def _is_long_tail_track(self, candidate: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> bool:
        """Check if track is in the long tail (below 70th percentile popularity)"""
        try:
            global_pop = candidate['metadata'].get('global_popularity', 0)
            
            # Get 70th percentile of popularity among candidates
            popularities = [c['metadata'].get('global_popularity', 0) for c in all_candidates]
            percentile_70 = np.percentile(popularities, 70)
            
            return global_pop <= percentile_70
        except:
            return False
    
    def _is_recent_track(self, metadata: Dict[str, Any]) -> bool:
        """Check if track is from the last 24 months"""
        try:
            release_date_str = metadata.get('release_date')
            if not release_date_str:
                return False
            
            # Parse release date (assuming YYYY-MM-DD format)
            release_date = datetime.strptime(release_date_str.split()[0], '%Y-%m-%d')
            cutoff_date = datetime.now() - timedelta(days=24 * 30)  # ~24 months
            
            return release_date >= cutoff_date
        except:
            return False
    
    async def _get_track_metadata(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get track metadata (artist, release_date, etc.) from database"""
        try:
            db = next(get_db())
            
            # Query track metadata - adjust this based on your tracks table schema
            # This is a placeholder query
            query = text("""
                SELECT id, artist, release_date, title, genres, popularity
                FROM tracks 
                WHERE id = ANY(:track_ids)
            """)
            
            result = db.execute(query, {"track_ids": track_ids}).fetchall()
            
            metadata = {}
            for row in result:
                metadata[row[0]] = {
                    'artist': row[1],
                    'release_date': row[2],
                    'title': row[3],
                    'genres': row[4],
                    'popularity': row[5]
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting track metadata: {str(e)}")
            return {}
        finally:
            db.close()
    
    async def _blend_with_content_based(self, 
                                      user_id: str,
                                      cf_recommendations: List[Dict[str, Any]], 
                                      num_recommendations: int) -> List[Dict[str, Any]]:
        """
        Blend collaborative filtering with content-based recommendations
        Final score = α * CF + (1-α) * Content
        """
        try:
            if self.content_blend_alpha >= 1.0:
                return cf_recommendations
            
            # Generate content-based recommendations (placeholder implementation)
            content_recommendations = await self._generate_content_based_recommendations(
                user_id, 
                num_recommendations * 2  # Generate more for blending
            )
            
            # Create blended scores
            cf_track_ids = {rec['track_id']: rec for rec in cf_recommendations}
            content_track_ids = {rec['track_id']: rec for rec in content_recommendations}
            
            all_track_ids = set(cf_track_ids.keys()) | set(content_track_ids.keys())
            
            blended_recommendations = []
            for track_id in all_track_ids:
                cf_score = cf_track_ids.get(track_id, {}).get('final_score', 0.0)
                content_score = content_track_ids.get(track_id, {}).get('content_score', 0.0)
                
                # Normalize scores to [0,1] range for blending
                cf_score_norm = min(1.0, cf_score / max(1.0, max(rec['final_score'] for rec in cf_recommendations)))
                content_score_norm = min(1.0, content_score / max(1.0, max(rec.get('content_score', 0) for rec in content_recommendations)))
                
                blended_score = (self.content_blend_alpha * cf_score_norm + 
                               (1 - self.content_blend_alpha) * content_score_norm)
                
                # Use CF recommendation as base, update score
                base_rec = cf_track_ids.get(track_id) or content_track_ids.get(track_id)
                if base_rec:
                    blended_rec = base_rec.copy()
                    blended_rec['final_score'] = blended_score
                    blended_rec['cf_score'] = cf_score_norm
                    blended_rec['content_score'] = content_score_norm
                    blended_rec['metadata']['blended'] = True
                    
                    blended_recommendations.append(blended_rec)
            
            # Sort by blended score and return top N
            blended_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"Blended CF and content-based recommendations "
                       f"(α={self.content_blend_alpha})")
            
            return blended_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error blending with content-based recommendations: {str(e)}")
            return cf_recommendations
    
    async def _generate_content_based_recommendations(self, 
                                                    user_id: str, 
                                                    num_recommendations: int) -> List[Dict[str, Any]]:
        """
        Generate content-based recommendations (placeholder implementation)
        
        This should use your metadata (tags, AOTY, audio features) to surface adjacent items
        """
        # Placeholder implementation - replace with your actual content-based logic
        try:
            # For now, return empty list - implement based on your content features
            return []
        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {str(e)}")
            return []
    
    async def _generate_explanations(self, 
                                   user_id: str,
                                   recommendations: List[Dict[str, Any]], 
                                   neighbors: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for recommendations"""
        explanations = []
        
        for rec in recommendations:
            explanation_parts = []
            metadata = rec.get('metadata', {})
            
            # Neighbor-based explanations
            contributing_neighbors = metadata.get('contributing_neighbors', [])
            if contributing_neighbors:
                top_contributor = max(contributing_neighbors, key=lambda x: x['contribution'])
                explanation_parts.append(
                    f"Similar users who listen to this track include user {top_contributor['neighbor_id']}"
                )
            
            # Diversity explanations
            if rec.get('mmr_score') is not None:
                explanation_parts.append("Selected for musical diversity")
            
            # Boost explanations
            if metadata.get('long_tail_boost'):
                explanation_parts.append("Hidden gem with strong appeal to similar users")
            
            if metadata.get('temporal_boost'):
                explanation_parts.append("Recent release matching your taste")
            
            explanation = ". ".join(explanation_parts) if explanation_parts else "Recommended based on collaborative filtering"
            explanations.append(explanation)
        
        return explanations
    
    async def _calculate_quality_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate recommendation quality metrics"""
        if not recommendations:
            return {}
        
        try:
            # Get content vectors for diversity calculation
            track_ids = [rec['track_id'] for rec in recommendations]
            content_vectors = await self._get_content_vectors(track_ids)
            
            # Calculate Intra-List Diversity (ILD)
            diversities = []
            if len(content_vectors) > 1:
                for i, rec1 in enumerate(recommendations):
                    for j, rec2 in enumerate(recommendations[i+1:], i+1):
                        vec1 = content_vectors.get(rec1['track_id'])
                        vec2 = content_vectors.get(rec2['track_id'])
                        
                        if vec1 is not None and vec2 is not None:
                            similarity = 1 - cosine(vec1, vec2)
                            diversity = 1 - similarity
                            diversities.append(diversity)
            
            ild = np.mean(diversities) if diversities else 0.0
            
            # Get track metadata for other metrics
            track_metadata = await self._get_track_metadata(track_ids)
            
            # Artist/genre entropy
            artists = [track_metadata.get(tid, {}).get('artist', 'Unknown') for tid in track_ids]
            artist_counts = Counter(artists)
            artist_entropy = entropy(list(artist_counts.values()))
            
            # Long-tail fraction
            popularities = [rec['metadata'].get('global_popularity', 0) for rec in recommendations]
            median_popularity = np.median([p for p in popularities if p > 0]) if popularities else 0
            long_tail_fraction = len([p for p in popularities if p <= median_popularity]) / len(popularities)
            
            # Repeat artist rate
            total_artists = len(artists)
            unique_artists = len(set(artists))
            repeat_artist_rate = (total_artists - unique_artists) / total_artists if total_artists > 0 else 0
            
            return {
                'intra_list_diversity': float(ild),
                'artist_entropy': float(artist_entropy),
                'long_tail_fraction': float(long_tail_fraction),
                'repeat_artist_rate': float(repeat_artist_rate),
                'unique_artists': unique_artists,
                'total_recommendations': len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}
    
    async def _store_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]]):
        """Store generated recommendations in database"""
        try:
            db = next(get_db())
            
            # Clear existing recommendations for this user
            db.query(CollaborativeRecommendation).filter(
                CollaborativeRecommendation.target_user_id == user_id
            ).delete()
            
            # Store new recommendations
            for i, rec in enumerate(recommendations):
                recommendation = CollaborativeRecommendation(
                    target_user_id=user_id,
                    track_id=rec['track_id'],
                    recommendation_score=rec['final_score'],
                    algorithm_type='enhanced_2hop_collaborative_filtering',
                    confidence_score=min(1.0, rec['final_score'] / max(1.0, recommendations[0]['final_score'])),
                    reason=f"Rank {i+1}: Enhanced CF with 2-hop neighbors and diversity"
                )
                db.add(recommendation)
            
            db.commit()
            logger.info(f"Stored {len(recommendations)} recommendations for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {str(e)}")
            db.rollback()
        finally:
            db.close()