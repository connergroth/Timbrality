"""
Personal Library Ingestion Script
Pulls in your own library, finds similar users (neighbors), and ingests their data
"""
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Set
from datetime import datetime
import json
import os

from services.enhanced_collaborative_filtering import EnhancedCollaborativeFilteringService
from services.supabase_collaborative_filtering import SupabaseCollaborativeFilteringService
from services.lastfm_service import LastFMService
from services.spotify_service import SpotifyService
from supabase import create_client
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LibraryIngestionScript:
    """Script to ingest personal library and expand training data with neighbors"""
    
    def __init__(self):
        self.cf_service = SupabaseCollaborativeFilteringService()
        self.enhanced_cf_service = EnhancedCollaborativeFilteringService()
        self.lastfm_service = LastFMService()
        self.spotify_service = SpotifyService()
        self.ingested_users = set()
        
    async def ingest_personal_library(self, 
                                    lastfm_username: str = None,
                                    spotify_username: str = None,
                                    expand_neighbors: bool = True,
                                    max_neighbors: int = 50,
                                    similarity_threshold: float = 0.1,
                                    use_following: bool = False) -> Dict[str, Any]:
        """
        Main ingestion function - pulls your library and expands with neighbors
        
        Args:
            lastfm_username: Your Last.fm username
            spotify_username: Your Spotify username  
            expand_neighbors: Whether to find and ingest neighbors
            max_neighbors: Maximum number of neighbors to ingest
            similarity_threshold: Minimum similarity for neighbor inclusion
            use_following: Use your Last.fm following list instead of similarity discovery
        """
        start_time = datetime.now()
        results = {
            "personal_data": {},
            "neighbors_found": [],
            "neighbors_ingested": [],
            "total_users": 0,
            "total_tracks": 0,
            "processing_time": 0,
            "errors": []
        }
        
        try:
            logger.info("Starting personal library ingestion...")
            
            # Step 1: Ingest your own data
            if lastfm_username:
                logger.info(f"Ingesting Last.fm data for {lastfm_username}")
                personal_lastfm = await self._ingest_personal_lastfm(lastfm_username)
                results["personal_data"]["lastfm"] = personal_lastfm
                self.ingested_users.add(lastfm_username)
            
            if spotify_username:
                logger.info(f"Ingesting Spotify data for {spotify_username}")
                personal_spotify = await self._ingest_personal_spotify(spotify_username)
                results["personal_data"]["spotify"] = personal_spotify
            
            # Step 2: Find similar users (your neighbors)
            if expand_neighbors and lastfm_username:
                if use_following:
                    logger.info("Finding users from your following list...")
                    neighbors = await self._find_following_users(
                        lastfm_username, 
                        max_neighbors
                    )
                else:
                    logger.info("Finding similar users...")
                    neighbors = await self._find_similar_users(
                        lastfm_username, 
                        max_neighbors, 
                        similarity_threshold
                    )
                results["neighbors_found"] = neighbors
                
                # Step 3: Ingest neighbor data
                logger.info(f"Ingesting data for {len(neighbors)} neighbors...")
                ingested_neighbors = await self._ingest_neighbors_data(neighbors)
                results["neighbors_ingested"] = ingested_neighbors
            
            # Step 4: Create unique albums and artists
            logger.info("Creating unique albums and artists...")
            albums_artists_stats = await self.cf_service.create_unique_albums_and_artists()
            results.update(albums_artists_stats)
            
            # Step 5: Calculate user similarities
            logger.info("Calculating user similarities...")
            await self._calculate_user_similarities()
            
            # Step 6: Generate summary statistics
            stats = await self._generate_ingestion_stats()
            results.update(stats)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time
            
            logger.info(f"Library ingestion completed in {processing_time:.2f} seconds")
            logger.info(f"Total users: {results['total_users']}, Total unique tracks: {results['total_tracks']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in library ingestion: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
    
    async def _ingest_personal_lastfm(self, username: str) -> Dict[str, Any]:
        """Ingest your personal Last.fm data"""
        try:
            # Add user to database
            add_result = await self.cf_service.add_lastfm_user(username, f"Personal Account ({username})")
            
            # Fetch comprehensive data
            fetch_result = await self.cf_service.fetch_user_data(username)
            
            return {
                "username": username,
                "add_result": add_result,
                "fetch_result": fetch_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting personal Last.fm data: {str(e)}")
            return {"username": username, "status": "error", "error": str(e)}
    
    async def _ingest_personal_spotify(self, username: str) -> Dict[str, Any]:
        """Ingest your personal Spotify data"""
        try:
            # Note: This requires Spotify OAuth setup for accessing personal data
            # For now, we'll focus on Last.fm which is more accessible for CF
            
            logger.info(f"Spotify ingestion for {username} - requires OAuth setup")
            
            # Placeholder for Spotify personal data ingestion
            # You would need to:
            # 1. Set up Spotify OAuth
            # 2. Get user's saved tracks, playlists, top tracks
            # 3. Store in database similar to Last.fm approach
            
            return {
                "username": username,
                "status": "not_implemented",
                "message": "Spotify personal data ingestion requires OAuth setup"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting personal Spotify data: {str(e)}")
            return {"username": username, "status": "error", "error": str(e)}
    
    async def _find_similar_users(self, 
                                username: str, 
                                max_neighbors: int, 
                                similarity_threshold: float) -> List[Dict[str, Any]]:
        """Find similar Last.fm users to expand training data"""
        try:
            # Get user's top artists for similarity matching
            user_artists = await self.lastfm_service.get_user_top_artists(username, limit=100)
            
            if not user_artists:
                logger.warning(f"No artists found for user {username}")
                return []
            
            # Extract artist names
            user_artist_names = [artist['name'] for artist in user_artists]
            
            # Find users with similar artists using Last.fm's user.getSimilar if available
            # Or use a different approach to find similar users
            similar_users = []
            
            try:
                # Method 1: Use Last.fm's getSimilar API if available
                similar_users_raw = await self.lastfm_service.get_similar_users(username, limit=max_neighbors)
                
                for similar_user in similar_users_raw:
                    similar_username = similar_user.get('name')
                    similarity_score = float(similar_user.get('match', 0))
                    
                    if similarity_score >= similarity_threshold and similar_username != username:
                        similar_users.append({
                            "username": similar_username,
                            "similarity": similarity_score,
                            "method": "lastfm_api"
                        })
                        
            except Exception as e:
                logger.warning(f"Last.fm getSimilar API not available: {e}")
                
                # Method 2: Use shared artists approach via Supabase
                similar_users = await self.cf_service.find_similar_users_by_shared_artists(
                    username,
                    max_neighbors
                )
                
                # Filter by threshold
                similar_users = [u for u in similar_users if u['similarity'] >= similarity_threshold]
            
            # Sort by similarity and limit
            similar_users.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_users[:max_neighbors]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []
    
    async def _find_following_users(self, username: str, max_neighbors: int) -> List[Dict[str, Any]]:
        """Find users from your Last.fm following list"""
        try:
            following_users = await self.lastfm_service.get_user_following(username, limit=max_neighbors)
            
            if not following_users:
                logger.warning(f"No following users found for {username}")
                return []
            
            # Convert to consistent format
            neighbors = []
            for user in following_users:
                if user.get('name') and user['name'] != username:
                    neighbors.append({
                        "username": user['name'],
                        "similarity": 1.0,  # High trust score for manually followed users
                        "playcount": user.get('playcount', 0),
                        "country": user.get('country', ''),
                        "method": "following"
                    })
            
            logger.info(f"Found {len(neighbors)} users from following list")
            return neighbors[:max_neighbors]
            
        except Exception as e:
            logger.error(f"Error finding following users: {str(e)}")
            return []
    
    async def _ingest_neighbors_data(self, neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ingest data for discovered neighbors"""
        ingested = []
        
        for neighbor in neighbors:
            username = neighbor['username']
            
            if username in self.ingested_users:
                logger.info(f"Skipping {username} - already ingested")
                continue
            
            try:
                logger.info(f"Ingesting data for neighbor: {username}")
                
                # Add user to database
                add_result = await self.cf_service.add_lastfm_user(
                    username, 
                    f"Neighbor of personal account (sim: {neighbor['similarity']:.3f})"
                )
                
                if add_result['success']:
                    # Fetch user data
                    fetch_result = await self.cf_service.fetch_user_data(username)
                    
                    ingested.append({
                        "username": username,
                        "similarity": neighbor['similarity'],
                        "add_result": add_result,
                        "fetch_result": fetch_result,
                        "status": "success"
                    })
                    
                    self.ingested_users.add(username)
                    
                    # Add delay to respect API rate limits
                    await asyncio.sleep(1)
                else:
                    logger.warning(f"Failed to add user {username}: {add_result['message']}")
                    
            except Exception as e:
                logger.error(f"Error ingesting neighbor {username}: {str(e)}")
                ingested.append({
                    "username": username,
                    "status": "error",
                    "error": str(e)
                })
        
        return ingested
    
    async def _calculate_user_similarities(self):
        """Calculate pairwise user similarities for collaborative filtering (simplified version)"""
        try:
            logger.info("Calculating user similarities (simplified)...")
            
            # Get all active users
            active_users = await self.cf_service.get_active_users()
            
            logger.info(f"Found {len(active_users)} active users for similarity calculation")
            
            # For now, we'll skip the computationally expensive similarity calculation
            # This would typically be done as a background job
            logger.info("Similarity calculation skipped - would be done as background job")
            
        except Exception as e:
            logger.error(f"Error calculating user similarities: {str(e)}")
    
    async def _generate_ingestion_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for the ingestion using Supabase"""
        try:
            supabase = self.cf_service.supabase
            
            # Count users
            users_result = supabase.table('lastfm_users').select('id', count='exact').eq('is_active', True).execute()
            total_users = users_result.count if users_result.count else 0
            
            # Count unique tracks with interactions
            interactions_result = supabase.table('user_track_interactions').select('track_id', count='exact').execute()
            total_interactions = interactions_result.count if interactions_result.count else 0
            
            # Get unique tracks count (approximation)
            unique_tracks_result = supabase.table('tracks').select('id', count='exact').execute()
            total_tracks = unique_tracks_result.count if unique_tracks_result.count else 0
            
            return {
                "total_users": total_users,
                "total_tracks": total_tracks,  
                "total_interactions": total_interactions,
                "avg_tracks_per_user": total_interactions / total_users if total_users > 0 else 0,
                "min_tracks_per_user": 0,
                "max_tracks_per_user": 0
            }
            
        except Exception as e:
            logger.error(f"Error generating stats: {str(e)}")
            return {}
    
    async def run_sample_recommendations(self, user_id: str, num_recommendations: int = 10):
        """Generate sample recommendations to test the system"""
        try:
            logger.info(f"Generating sample recommendations for user {user_id}")
            
            recommendations = await self.enhanced_cf_service.generate_recommendations(
                user_id, 
                num_recommendations
            )
            
            logger.info(f"Generated {len(recommendations.get('recommendations', []))} recommendations")
            logger.info(f"Quality metrics: {recommendations.get('metadata', {}).get('quality_metrics', {})}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating sample recommendations: {str(e)}")
            return {}


async def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Personal Library Ingestion Script")
    parser.add_argument("--lastfm-username", help="Your Last.fm username")
    parser.add_argument("--spotify-username", help="Your Spotify username")  
    parser.add_argument("--max-neighbors", type=int, default=50, help="Maximum neighbors to ingest")
    parser.add_argument("--similarity-threshold", type=float, default=0.1, help="Minimum similarity for neighbors")
    parser.add_argument("--skip-neighbors", action="store_true", help="Skip neighbor discovery and ingestion")
    parser.add_argument("--use-following", action="store_true", help="Use your Last.fm following list instead of similarity discovery")
    parser.add_argument("--test-recommendations", help="User ID to test recommendations for")
    parser.add_argument("--output-file", help="File to save results to")
    
    args = parser.parse_args()
    
    if not args.lastfm_username and not args.spotify_username:
        print("Please provide at least one of --lastfm-username or --spotify-username")
        return
    
    script = LibraryIngestionScript()
    
    # Run ingestion
    results = await script.ingest_personal_library(
        lastfm_username=args.lastfm_username,
        spotify_username=args.spotify_username,
        expand_neighbors=not args.skip_neighbors,
        max_neighbors=args.max_neighbors,
        similarity_threshold=args.similarity_threshold,
        use_following=args.use_following
    )
    
    # Test recommendations if requested
    if args.test_recommendations:
        recommendations = await script.run_sample_recommendations(args.test_recommendations)
        results["sample_recommendations"] = recommendations
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Total users ingested: {results.get('total_users', 0)}")
    print(f"Total unique tracks: {results.get('total_tracks', 0)}")
    print(f"Processing time: {results.get('processing_time', 0):.2f} seconds")
    print(f"Neighbors found: {len(results.get('neighbors_found', []))}")
    print(f"Neighbors ingested: {len(results.get('neighbors_ingested', []))}")
    
    if results.get('errors'):
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())