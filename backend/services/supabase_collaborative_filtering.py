"""
Collaborative Filtering Service using Supabase (compatible with existing Timbre architecture)
"""
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from supabase import create_client, Client
import os
from dotenv import load_dotenv

from services.lastfm_service import LastFMService

load_dotenv()
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')


def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class SupabaseCollaborativeFilteringService:
    """Collaborative filtering service using Supabase (same as existing Timbre architecture)"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.lastfm_service = LastFMService()
    
    async def setup_collaborative_filtering_tables(self):
        """
        Setup collaborative filtering tables in Supabase
        Note: These would typically be created via Supabase SQL editor or migrations
        """
        logger.info("Collaborative filtering tables should be created via Supabase SQL editor:")
        
        sql_statements = """
        -- LastFM Users table
        CREATE TABLE IF NOT EXISTS lastfm_users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lastfm_username TEXT UNIQUE NOT NULL,
            display_name TEXT,
            real_name TEXT,
            country TEXT,
            age INTEGER,
            gender TEXT,
            subscriber BOOLEAN DEFAULT FALSE,
            playcount_total INTEGER DEFAULT 0,
            playlists_count INTEGER DEFAULT 0,
            registered_date TIMESTAMP,
            last_updated TIMESTAMP DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            data_fetch_enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- User Track Interactions table
        CREATE TABLE IF NOT EXISTS user_track_interactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE,
            interaction_type TEXT DEFAULT 'play',
            play_count INTEGER DEFAULT 0,
            user_loved BOOLEAN DEFAULT FALSE,
            last_played TIMESTAMP,
            tags JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(lastfm_user_id, track_id)
        );
        
        -- User Album Interactions table  
        CREATE TABLE IF NOT EXISTS user_album_interactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            album_title TEXT NOT NULL,
            album_artist TEXT NOT NULL,
            play_count INTEGER DEFAULT 0,
            user_loved BOOLEAN DEFAULT FALSE,
            last_played TIMESTAMP,
            tags JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(lastfm_user_id, album_title, album_artist)
        );
        
        -- User Artist Interactions table
        CREATE TABLE IF NOT EXISTS user_artist_interactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            artist_name TEXT NOT NULL,
            play_count INTEGER DEFAULT 0,
            user_loved BOOLEAN DEFAULT FALSE,
            last_played TIMESTAMP,
            tags JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(lastfm_user_id, artist_name)
        );
        
        -- User Similarities table
        CREATE TABLE IF NOT EXISTS user_similarities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id_1 UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            user_id_2 UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            similarity_score FLOAT NOT NULL CHECK (similarity_score >= -1 AND similarity_score <= 1),
            similarity_type TEXT DEFAULT 'cosine',
            shared_tracks_count INTEGER DEFAULT 0,
            shared_albums_count INTEGER DEFAULT 0,
            shared_artists_count INTEGER DEFAULT 0,
            calculated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id_1, user_id_2)
        );
        
        -- Collaborative Recommendations table
        CREATE TABLE IF NOT EXISTS collaborative_recommendations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            target_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
            track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE,
            recommendation_score FLOAT NOT NULL CHECK (recommendation_score >= 0 AND recommendation_score <= 1),
            algorithm_type TEXT DEFAULT 'user_based',
            confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
            reason TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(target_user_id, track_id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_user_track_interactions_user ON user_track_interactions(lastfm_user_id);
        CREATE INDEX IF NOT EXISTS idx_user_track_interactions_track ON user_track_interactions(track_id);
        CREATE INDEX IF NOT EXISTS idx_user_similarities_user1 ON user_similarities(user_id_1);
        CREATE INDEX IF NOT EXISTS idx_user_similarities_user2 ON user_similarities(user_id_2);
        CREATE INDEX IF NOT EXISTS idx_collaborative_recommendations_user ON collaborative_recommendations(target_user_id);
        """
        
        logger.info("Run these SQL statements in Supabase SQL editor to create tables")
        return sql_statements
    
    async def add_lastfm_user(self, username: str, display_name: str = None) -> Dict[str, Any]:
        """Add a new Last.fm user to track for collaborative filtering"""
        try:
            # Check if user already exists
            existing_user = self.supabase.table('lastfm_users').select('*').eq('lastfm_username', username).execute()
            
            if existing_user.data:
                return {
                    "success": False,
                    "message": f"User {username} already exists",
                    "user_id": existing_user.data[0]['id']
                }
            
            # Create new user
            new_user = {
                'lastfm_username': username,
                'display_name': display_name or username,
                'data_fetch_enabled': True,
                'is_active': True,
                'created_at': datetime.now().isoformat()
            }
            
            result = self.supabase.table('lastfm_users').insert(new_user).execute()
            
            if result.data:
                logger.info(f"Added new Last.fm user: {username}")
                return {
                    "success": True,
                    "message": f"Successfully added user {username}",
                    "user_id": result.data[0]['id']
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to insert user"
                }
            
        except Exception as e:
            logger.error(f"Error adding Last.fm user {username}: {str(e)}")
            return {
                "success": False,
                "message": f"Error adding user: {str(e)}"
            }
    
    async def fetch_user_data(self, username: str, fetch_types: List[str] = None) -> Dict[str, Any]:
        """Fetch all data for a specific Last.fm user"""
        if fetch_types is None:
            fetch_types = ['profile', 'albums', 'artists', 'tracks']
        
        try:
            results = {
                "username": username,
                "fetch_types": fetch_types,
                "results": {},
                "errors": []
            }
            
            for fetch_type in fetch_types:
                try:
                    if fetch_type == 'profile':
                        profile_data = await self._fetch_user_profile(username)
                        results["results"]["profile"] = profile_data
                        
                    elif fetch_type == 'tracks':
                        tracks_data = await self._fetch_user_tracks(username)
                        results["results"]["tracks"] = tracks_data
                        
                        # Also fetch loved tracks separately for proper loved status
                        loved_data = await self._fetch_user_loved_tracks(username)
                        results["results"]["loved_tracks"] = loved_data
                        
                    elif fetch_type == 'albums':
                        albums_data = await self._fetch_user_albums(username)
                        results["results"]["albums"] = albums_data
                        
                    elif fetch_type == 'artists':
                        artists_data = await self._fetch_user_artists(username)
                        results["results"]["artists"] = artists_data
                        
                except Exception as e:
                    error_msg = f"Error fetching {fetch_type} for {username}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in fetch_user_data for {username}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "username": username
            }
    
    async def _fetch_user_profile(self, username: str) -> Dict[str, Any]:
        """Fetch user profile information from Last.fm"""
        try:
            profile = await self.lastfm_service.get_user_info(username)
            
            if profile:
                # Update user profile in Supabase
                update_data = {
                    'display_name': profile.get('name', username),
                    'real_name': profile.get('realname'),
                    'country': profile.get('country'),
                    'age': profile.get('age'),
                    'gender': profile.get('gender'),
                    'subscriber': profile.get('subscriber') == '1',
                    'playcount_total': int(profile.get('playcount', 0)),
                    'playlists_count': int(profile.get('playlists', 0)),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Try to parse registered date
                if profile.get('registered'):
                    try:
                        reg_timestamp = int(profile.get('registered', 0))
                        update_data['registered_date'] = datetime.fromtimestamp(reg_timestamp).isoformat()
                    except:
                        pass
                
                # Update the user profile
                self.supabase.table('lastfm_users').update(update_data).eq('lastfm_username', username).execute()
            
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching profile for {username}: {str(e)}")
            raise
    
    async def _fetch_user_tracks(self, username: str) -> Dict[str, Any]:
        """Fetch user's top tracks from Last.fm"""
        try:
            tracks = await self.lastfm_service.get_user_top_tracks(username, limit=1000)
            
            # Store track interactions
            await self._store_track_interactions(username, tracks)
            
            return {
                "tracks_fetched": len(tracks),
                "tracks": tracks[:10]  # Return first 10 for preview
            }
            
        except Exception as e:
            logger.error(f"Error fetching tracks for {username}: {str(e)}")
            raise
    
    async def _fetch_user_loved_tracks(self, username: str) -> Dict[str, Any]:
        """Fetch user's loved tracks from Last.fm and update loved status"""
        try:
            loved_tracks = await self.lastfm_service.get_user_loved_tracks(username, limit=500)
            
            if not loved_tracks:
                return {"loved_tracks_fetched": 0, "loved_tracks": []}
            
            # Update loved status for existing track interactions
            await self._update_loved_track_status(username, loved_tracks)
            
            return {
                "loved_tracks_fetched": len(loved_tracks),
                "loved_tracks": loved_tracks[:10]  # Return first 10 for preview
            }
            
        except Exception as e:
            logger.error(f"Error fetching loved tracks for {username}: {str(e)}")
            raise
    
    async def _fetch_user_albums(self, username: str) -> Dict[str, Any]:
        """Fetch user's top albums from Last.fm"""
        try:
            albums = await self.lastfm_service.get_user_top_albums(username, limit=500)
            
            # Store album interactions
            await self._store_album_interactions(username, albums)
            
            return {
                "albums_fetched": len(albums),
                "albums": albums[:10]  # Return first 10 for preview
            }
            
        except Exception as e:
            logger.error(f"Error fetching albums for {username}: {str(e)}")
            raise
    
    async def _fetch_user_artists(self, username: str) -> Dict[str, Any]:
        """Fetch user's top artists from Last.fm"""
        try:
            artists = await self.lastfm_service.get_user_top_artists(username, limit=500)
            
            # Store artist interactions
            await self._store_artist_interactions(username, artists)
            
            return {
                "artists_fetched": len(artists),
                "artists": artists[:10]  # Return first 10 for preview
            }
            
        except Exception as e:
            logger.error(f"Error fetching artists for {username}: {str(e)}")
            raise
    
    async def _store_track_interactions(self, username: str, tracks: List[Dict[str, Any]]):
        """Store track interactions in Supabase, creating tracks if needed"""
        import hashlib
        
        try:
            # Get user ID
            user_result = self.supabase.table('lastfm_users').select('id').eq('lastfm_username', username).execute()
            
            if not user_result.data:
                logger.error(f"User {username} not found in database")
                return
            
            user_id = user_result.data[0]['id']
            
            interactions_to_upsert = []
            tracks_to_create = []
            
            for track in tracks:
                track_title = track.get('name', '').strip()
                track_artist = track.get('artist', '').strip()
                
                if not track_title or not track_artist:
                    continue
                
                # Create track ID directly (skip lookup to avoid Supabase API issues)
                track_id = f"lastfm_{hashlib.md5(f'{track_title}_{track_artist}'.encode()).hexdigest()[:16]}"
                
                # Always try to create track record (upsert will handle duplicates)
                # Extract album info from Last.fm track data
                album_name = track.get('album', '').strip() if track.get('album') else ''
                
                track_record = {
                    'id': track_id,
                    'title': track_title,
                    'artist': track_artist,
                    'album': album_name,
                    'popularity': 0,  # Will be updated by content expansion
                    'spotify_id': None,  # Will be populated later
                    'aoty_score': None,  # Will be populated later
                    'explicit': None,  # Will be populated from Spotify (reliable source)
                    # Note: genres and moods will be populated by content expansion script
                    # Last.fm track data doesn't include genre/mood info per track
                }
                
                tracks_to_create.append(track_record)
                
                # Fetch track tags/moods from Last.fm (optional - adds API calls)
                # We'll do this in the content expansion script to avoid slowing down user ingestion
                
                # Create user interaction
                interaction = {
                    'lastfm_user_id': user_id,
                    'track_id': track_id,
                    'play_count': int(track.get('playcount', 0)),
                    'user_loved': False,  # Will be updated by loved tracks method
                    'last_played': datetime.now().isoformat(),
                    'tags': track.get('tags', {}),
                    'updated_at': datetime.now().isoformat()
                }
                
                interactions_to_upsert.append(interaction)
            
            # First, create any new tracks
            if tracks_to_create:
                try:
                    tracks_result = self.supabase.table('tracks').upsert(tracks_to_create).execute()
                    logger.info(f"Created {len(tracks_to_create)} new track records for {username}")
                except Exception as e:
                    logger.error(f"Error creating track records: {str(e)}")
                    # Continue with interactions even if track creation fails
            
            # Then, create user interactions
            if interactions_to_upsert:
                try:
                    result = self.supabase.table('user_track_interactions').upsert(
                        interactions_to_upsert,
                        on_conflict='lastfm_user_id,track_id'
                    ).execute()
                    logger.info(f"Stored {len(interactions_to_upsert)} track interactions for {username}")
                except Exception as e:
                    logger.error(f"Error storing track interactions: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error storing track interactions for {username}: {str(e)}")
    
    async def _update_loved_track_status(self, username: str, loved_tracks: List[Dict[str, Any]]):
        """Update loved status for tracks that user has loved"""
        import hashlib
        
        try:
            # Get user ID
            user_result = self.supabase.table('lastfm_users').select('id').eq('lastfm_username', username).execute()
            
            if not user_result.data:
                logger.error(f"User {username} not found in database")
                return
            
            user_id = user_result.data[0]['id']
            
            updates_made = 0
            
            for loved_track in loved_tracks:
                track_title = loved_track.get('name', '').strip()
                track_artist = loved_track.get('artist', '').strip()
                
                if not track_title or not track_artist:
                    continue
                
                # Generate the same track ID we use for creation
                track_id = f"lastfm_{hashlib.md5(f'{track_title}_{track_artist}'.encode()).hexdigest()[:16]}"
                
                # Update the track interaction to mark as loved
                try:
                    result = self.supabase.table('user_track_interactions').update({
                        'user_loved': True,
                        'updated_at': datetime.now().isoformat()
                    }).eq('lastfm_user_id', user_id).eq('track_id', track_id).execute()
                    
                    if result.data:
                        updates_made += 1
                        
                except Exception as e:
                    logger.warning(f"Could not update loved status for {track_title} by {track_artist}: {str(e)}")
            
            logger.info(f"Updated loved status for {updates_made} tracks for {username}")
            
        except Exception as e:
            logger.error(f"Error updating loved track status for {username}: {str(e)}")
    
    async def create_unique_albums_and_artists(self):
        """Create unique album and artist records from interaction data"""
        import hashlib
        
        try:
            logger.info("Creating unique albums and artists from interaction data...")
            
            # Get unique albums from album interactions
            album_result = self.supabase.table('user_album_interactions').select('album_title, album_artist').execute()
            
            unique_albums = {}
            for row in album_result.data:
                album_key = f"{row['album_title']}|{row['album_artist']}"
                if album_key not in unique_albums and row['album_title'] and row['album_artist']:
                    unique_albums[album_key] = {
                        'id': f"album_{hashlib.md5(album_key.encode()).hexdigest()[:16]}",
                        'title': row['album_title'],
                        'artist': row['album_artist']
                    }
            
            # Get unique artists from artist interactions  
            artist_result = self.supabase.table('user_artist_interactions').select('artist_name').execute()
            
            unique_artists = {}
            for row in artist_result.data:
                artist_name = row['artist_name']
                if artist_name not in unique_artists and artist_name:
                    unique_artists[artist_name] = {
                        'id': f"artist_{hashlib.md5(artist_name.encode()).hexdigest()[:16]}",
                        'name': artist_name
                    }
            
            # Insert unique albums
            if unique_albums:
                albums_to_create = list(unique_albums.values())
                try:
                    albums_result = self.supabase.table('albums').upsert(albums_to_create).execute()
                    logger.info(f"Created {len(albums_to_create)} unique album records")
                except Exception as e:
                    logger.error(f"Error creating album records: {str(e)}")
            
            # Insert unique artists
            if unique_artists:
                artists_to_create = list(unique_artists.values())
                try:
                    artists_result = self.supabase.table('artists').upsert(artists_to_create).execute()
                    logger.info(f"Created {len(artists_to_create)} unique artist records")
                except Exception as e:
                    logger.error(f"Error creating artist records: {str(e)}")
            
            return {
                "unique_albums": len(unique_albums),
                "unique_artists": len(unique_artists)
            }
            
        except Exception as e:
            logger.error(f"Error creating unique albums and artists: {str(e)}")
            return {"unique_albums": 0, "unique_artists": 0}
    
    async def _store_album_interactions(self, username: str, albums: List[Dict[str, Any]]):
        """Store album interactions in Supabase"""
        try:
            # Get user ID
            user_result = self.supabase.table('lastfm_users').select('id').eq('lastfm_username', username).execute()
            
            if not user_result.data:
                logger.error(f"User {username} not found in database")
                return
            
            user_id = user_result.data[0]['id']
            
            interactions_to_upsert = []
            
            for album in albums:
                interaction = {
                    'lastfm_user_id': user_id,
                    'album_title': album.get('name', ''),
                    'album_artist': album.get('artist', ''),
                    'play_count': int(album.get('playcount', 0)),
                    'user_loved': False,  # Albums don't have loved status in Last.fm
                    'last_played': datetime.now().isoformat(),
                    'tags': album.get('tags', {}),
                    'updated_at': datetime.now().isoformat()
                }
                
                interactions_to_upsert.append(interaction)
            
            # Batch upsert interactions (will update existing records)
            if interactions_to_upsert:
                result = self.supabase.table('user_album_interactions').upsert(
                    interactions_to_upsert,
                    on_conflict='lastfm_user_id,album_title,album_artist'
                ).execute()
                logger.info(f"Stored {len(interactions_to_upsert)} album interactions for {username}")
            
        except Exception as e:
            logger.error(f"Error storing album interactions for {username}: {str(e)}")
    
    async def _store_artist_interactions(self, username: str, artists: List[Dict[str, Any]]):
        """Store artist interactions in Supabase"""
        try:
            # Get user ID
            user_result = self.supabase.table('lastfm_users').select('id').eq('lastfm_username', username).execute()
            
            if not user_result.data:
                logger.error(f"User {username} not found in database")
                return
            
            user_id = user_result.data[0]['id']
            
            interactions_to_upsert = []
            
            for artist in artists:
                interaction = {
                    'lastfm_user_id': user_id,
                    'artist_name': artist.get('name', ''),
                    'play_count': int(artist.get('playcount', 0)),
                    'user_loved': False,  # Artists don't have loved status in Last.fm
                    'last_played': datetime.now().isoformat(),
                    'tags': artist.get('tags', {}),
                    'updated_at': datetime.now().isoformat()
                }
                
                interactions_to_upsert.append(interaction)
            
            # Batch upsert interactions
            if interactions_to_upsert:
                result = self.supabase.table('user_artist_interactions').upsert(interactions_to_upsert).execute()
                logger.info(f"Stored {len(interactions_to_upsert)} artist interactions for {username}")
            
        except Exception as e:
            logger.error(f"Error storing artist interactions for {username}: {str(e)}")
    
    async def _find_track_id(self, title: str, artist: str) -> Optional[str]:
        """Find track ID in tracks table by title and artist"""
        try:
            if not title or not artist:
                return None
            
            # Clean the search terms
            title_clean = title.strip()
            artist_clean = artist.strip()
            
            # Try exact match first
            result = self.supabase.table('tracks').select('id').eq('title', title_clean).eq('artist', artist_clean).limit(1).execute()
            
            if result.data:
                return result.data[0]['id']
            
            # Try case-insensitive match
            result = self.supabase.table('tracks').select('id').ilike('title', title_clean).ilike('artist', artist_clean).limit(1).execute()
            
            if result.data:
                return result.data[0]['id']
            
            # Try partial match with wildcards
            result = self.supabase.table('tracks').select('id').ilike('title', f'%{title_clean}%').ilike('artist', f'%{artist_clean}%').limit(1).execute()
            
            if result.data:
                return result.data[0]['id']
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding track ID for '{title}' by '{artist}': {str(e)}")
            return None
    
    async def get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active Last.fm users being tracked"""
        try:
            result = self.supabase.table('lastfm_users').select('*').eq('is_active', True).eq('data_fetch_enabled', True).execute()
            
            user_list = []
            if result.data:
                for user in result.data:
                    user_list.append({
                        "id": user['id'],
                        "username": user['lastfm_username'],
                        "display_name": user['display_name'],
                        "last_updated": user['last_updated'],
                        "is_active": user['is_active']
                    })
            
            return user_list
            
        except Exception as e:
            logger.error(f"Error getting active users: {str(e)}")
            return []
    
    async def find_similar_users_by_shared_artists(self, 
                                                  target_username: str, 
                                                  max_neighbors: int = 50) -> List[Dict[str, Any]]:
        """
        Find similar users based on shared artists with the target user
        This is an alternative approach when Last.fm's getSimilar API isn't available
        """
        try:
            # Get target user's top artists
            target_artists = await self.lastfm_service.get_user_top_artists(target_username, limit=50)
            
            if not target_artists:
                return []
            
            target_artist_names = set(artist['name'].lower() for artist in target_artists)
            
            # Get all other users and their artist interactions
            all_users = await self.get_active_users()
            similar_users = []
            
            for user in all_users:
                if user['username'] == target_username:
                    continue
                
                # Get this user's artist interactions
                user_artists_result = self.supabase.table('user_artist_interactions').select('artist_name, play_count').eq('lastfm_user_id', user['id']).execute()
                
                if not user_artists_result.data:
                    continue
                
                user_artist_names = set(interaction['artist_name'].lower() for interaction in user_artists_result.data)
                
                # Calculate Jaccard similarity based on shared artists
                shared_artists = target_artist_names & user_artist_names
                total_artists = target_artist_names | user_artist_names
                
                if len(shared_artists) >= 2:  # Minimum shared artists
                    jaccard_similarity = len(shared_artists) / len(total_artists) if total_artists else 0
                    
                    similar_users.append({
                        "username": user['username'],
                        "similarity": jaccard_similarity,
                        "shared_artists": list(shared_artists),
                        "shared_count": len(shared_artists),
                        "method": "shared_artists_jaccard"
                    })
            
            # Sort by similarity and return top neighbors
            similar_users.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_users[:max_neighbors]
            
        except Exception as e:
            logger.error(f"Error finding similar users for {target_username}: {str(e)}")
            return []