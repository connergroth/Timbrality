"""
Service for managing collaborative filtering data from multiple Last.fm users
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from models.collaborative_filtering import (
    LastfmUser, UserTrackInteraction, UserAlbumInteraction, 
    UserArtistInteraction, DataFetchLog
)
from services.lastfm_service import LastFMService
from models.database import get_db

logger = logging.getLogger(__name__)


class CollaborativeFilteringService:
    """Service for managing collaborative filtering data collection"""
    
    def __init__(self):
        self.lastfm_service = LastFMService()
    
    async def add_lastfm_user(self, username: str, display_name: str = None) -> Dict[str, Any]:
        """Add a new Last.fm user to track for collaborative filtering"""
        try:
            db = next(get_db())
            
            # Check if user already exists
            existing_user = db.query(LastfmUser).filter(
                LastfmUser.lastfm_username == username
            ).first()
            
            if existing_user:
                return {
                    "success": False,
                    "message": f"User {username} already exists",
                    "user_id": str(existing_user.id)
                }
            
            # Create new user
            new_user = LastfmUser(
                lastfm_username=username,
                display_name=display_name or username,
                data_fetch_enabled=True,
                is_active=True
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            logger.info(f"Added new Last.fm user: {username}")
            
            return {
                "success": True,
                "message": f"Successfully added user {username}",
                "user_id": str(new_user.id)
            }
            
        except Exception as e:
            logger.error(f"Error adding Last.fm user {username}: {str(e)}")
            return {
                "success": False,
                "message": f"Error adding user: {str(e)}"
            }
        finally:
            db.close()
    
    async def fetch_user_data(self, username: str, fetch_types: List[str] = None) -> Dict[str, Any]:
        """Fetch all data for a specific Last.fm user"""
        if fetch_types is None:
            fetch_types = ['profile', 'tracks', 'albums', 'artists']
        
        try:
            # Start fetch log
            fetch_log = await self._create_fetch_log(username, 'comprehensive')
            
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
            
            # Update fetch log
            await self._update_fetch_log(fetch_log["id"], "success", results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in fetch_user_data for {username}: {str(e)}"
            logger.error(error_msg)
            await self._update_fetch_log(fetch_log["id"], "failed", {"error": error_msg})
            return {
                "success": False,
                "message": error_msg,
                "username": username
            }
    
    async def fetch_multiple_users_data(self, usernames: List[str]) -> Dict[str, Any]:
        """Fetch data for multiple Last.fm users concurrently"""
        tasks = []
        for username in usernames:
            task = self.fetch_user_data(username)
            tasks.append(task)
        
        # Run all fetches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        summary = {
            "total_users": len(usernames),
            "successful_fetches": 0,
            "failed_fetches": 0,
            "results": {},
            "errors": []
        }
        
        for i, result in enumerate(results):
            username = usernames[i]
            if isinstance(result, Exception):
                summary["failed_fetches"] += 1
                summary["errors"].append(f"User {username}: {str(result)}")
            else:
                summary["successful_fetches"] += 1
                summary["results"][username] = result
        
        return summary
    
    async def get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active Last.fm users being tracked"""
        try:
            db = next(get_db())
            
            users = db.query(LastfmUser).filter(
                LastfmUser.is_active == True,
                LastfmUser.data_fetch_enabled == True
            ).all()
            
            user_list = []
            for user in users:
                user_list.append({
                    "id": str(user.id),
                    "username": user.lastfm_username,
                    "display_name": user.display_name,
                    "last_updated": user.last_updated.isoformat() if user.last_updated else None,
                    "is_active": user.is_active
                })
            
            return user_list
            
        except Exception as e:
            logger.error(f"Error getting active users: {str(e)}")
            return []
        finally:
            db.close()
    
    async def _fetch_user_profile(self, username: str) -> Dict[str, Any]:
        """Fetch user profile information from Last.fm"""
        try:
            profile = await self.lastfm_service.get_user_info(username)
            
            # Update user profile in database
            db = next(get_db())
            user = db.query(LastfmUser).filter(
                LastfmUser.lastfm_username == username
            ).first()
            
            if user and profile:
                user.display_name = profile.get('name', username)
                user.real_name = profile.get('realname')
                user.country = profile.get('country')
                user.age = profile.get('age')
                user.gender = profile.get('gender')
                user.subscriber = profile.get('subscriber') == '1'
                user.playcount_total = int(profile.get('playcount', 0))
                user.playlists_count = int(profile.get('playlists', 0))
                user.registered_date = datetime.fromtimestamp(int(profile.get('registered', 0)))
                user.last_updated = datetime.now()
                
                db.commit()
            
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
        """Store track interactions in database"""
        try:
            db = next(get_db())
            user = db.query(LastfmUser).filter(
                LastfmUser.lastfm_username == username
            ).first()
            
            if not user:
                logger.error(f"User {username} not found in database")
                return
            
            for track in tracks:
                # Find track in database by title and artist
                track_id = self._find_track_id(track['name'], track['artist']['name'])
                
                if track_id:
                    interaction = UserTrackInteraction(
                        lastfm_user_id=user.id,
                        track_id=track_id,
                        play_count=int(track.get('playcount', 0)),
                        user_loved=track.get('loved') == '1',
                        last_played=datetime.now(),  # Last.fm doesn't provide this
                        tags=track.get('tags', {})
                    )
                    
                    # Use upsert to avoid duplicates
                    existing = db.query(UserTrackInteraction).filter(
                        and_(
                            UserTrackInteraction.lastfm_user_id == user.id,
                            UserTrackInteraction.track_id == track_id
                        )
                    ).first()
                    
                    if existing:
                        existing.play_count = interaction.play_count
                        existing.user_loved = interaction.user_loved
                        existing.updated_at = datetime.now()
                    else:
                        db.add(interaction)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing track interactions for {username}: {str(e)}")
            db.rollback()
        finally:
            db.close()
    
    async def _store_album_interactions(self, username: str, albums: List[Dict[str, Any]]):
        """Store album interactions in database"""
        try:
            db = next(get_db())
            user = db.query(LastfmUser).filter(
                LastfmUser.lastfm_username == username
            ).first()
            
            if not user:
                return
            
            for album in albums:
                interaction = UserAlbumInteraction(
                    lastfm_user_id=user.id,
                    album_title=album['name'],
                    album_artist=album['artist']['name'],
                    play_count=int(album.get('playcount', 0)),
                    user_loved=album.get('loved') == '1',
                    last_played=datetime.now(),
                    tags=album.get('tags', {})
                )
                
                # Use upsert
                existing = db.query(UserAlbumInteraction).filter(
                    and_(
                        UserAlbumInteraction.lastfm_user_id == user.id,
                        UserAlbumInteraction.album_title == album['name'],
                        UserAlbumInteraction.album_artist == album['artist']['name']
                    )
                ).first()
                
                if existing:
                    existing.play_count = interaction.play_count
                    existing.user_loved = interaction.user_loved
                    existing.updated_at = datetime.now()
                else:
                    db.add(interaction)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing album interactions for {username}: {str(e)}")
            db.rollback()
        finally:
            db.close()
    
    async def _store_artist_interactions(self, username: str, artists: List[Dict[str, Any]]):
        """Store artist interactions in database"""
        try:
            db = next(get_db())
            user = db.query(LastfmUser).filter(
                LastfmUser.lastfm_username == username
            ).first()
            
            if not user:
                return
            
            for artist in artists:
                interaction = UserArtistInteraction(
                    lastfm_user_id=user.id,
                    artist_name=artist['name'],
                    play_count=int(artist.get('playcount', 0)),
                    user_loved=artist.get('loved') == '1',
                    last_played=datetime.now(),
                    tags=artist.get('tags', {})
                )
                
                # Use upsert
                existing = db.query(UserArtistInteraction).filter(
                    and_(
                        UserArtistInteraction.lastfm_user_id == user.id,
                        UserArtistInteraction.artist_name == artist['name']
                    )
                ).first()
                
                if existing:
                    existing.play_count = interaction.play_count
                    existing.user_loved = interaction.user_loved
                    existing.updated_at = datetime.now()
                else:
                    db.add(interaction)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing artist interactions for {username}: {str(e)}")
            db.rollback()
        finally:
            db.close()
    
    def _find_track_id(self, title: str, artist: str) -> Optional[str]:
        """Find track ID in database by title and artist"""
        try:
            db = next(get_db())
            
            track = db.query(func.min(LastfmUser.id)).filter(
                and_(
                    func.lower(LastfmUser.title) == title.lower(),
                    func.lower(LastfmUser.artist) == artist.lower()
                )
            ).first()
            
            return track[0] if track and track[0] else None
            
        except Exception as e:
            logger.error(f"Error finding track ID: {str(e)}")
            return None
        finally:
            db.close()
    
    async def _create_fetch_log(self, username: str, fetch_type: str) -> Dict[str, Any]:
        """Create a new fetch log entry"""
        try:
            db = next(get_db())
            
            fetch_log = DataFetchLog(
                lastfm_username=username,
                fetch_type=fetch_type,
                status="pending",
                started_at=datetime.now()
            )
            
            db.add(fetch_log)
            db.commit()
            db.refresh(fetch_log)
            
            return {
                "id": str(fetch_log.id),
                "username": username,
                "fetch_type": fetch_type
            }
            
        except Exception as e:
            logger.error(f"Error creating fetch log: {str(e)}")
            return {"id": None, "username": username, "fetch_type": fetch_type}
        finally:
            db.close()
    
    async def _update_fetch_log(self, log_id: str, status: str, results: Dict[str, Any]):
        """Update fetch log with results"""
        if not log_id:
            return
            
        try:
            db = next(get_db())
            
            fetch_log = db.query(DataFetchLog).filter(
                DataFetchLog.id == log_id
            ).first()
            
            if fetch_log:
                fetch_log.status = status
                fetch_log.completed_at = datetime.now()
                
                if status == "success":
                    fetch_log.tracks_fetched = results.get("results", {}).get("tracks", {}).get("tracks_fetched", 0)
                    fetch_log.albums_fetched = results.get("results", {}).get("albums", {}).get("albums_fetched", 0)
                    fetch_log.artists_fetched = results.get("results", {}).get("artists", {}).get("artists_fetched", 0)
                else:
                    fetch_log.error_message = str(results.get("error", "Unknown error"))
                
                fetch_log.duration_ms = int((fetch_log.completed_at - fetch_log.started_at).total_seconds() * 1000)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Error updating fetch log: {str(e)}")
        finally:
            db.close()





