#!/usr/bin/env python3
"""
Script to populate the collaborative filtering database with multiple users' Last.fm data.
This script fetches data from multiple Last.fm users and stores it in the new tables.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from dotenv import load_dotenv

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import get_settings
from services.lastfm_service import LastFMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeDataPopulator:
    def __init__(self):
        self.settings = get_settings()
        self.db_conn = None
        self.lastfm_service = LastFMService()
        
        # List of Last.fm usernames to fetch data from
        # You can modify this list to include users you want to analyze
        self.target_users = [
            'your_username',  # Replace with your actual Last.fm username
            'sample_user_1',  # Add real Last.fm usernames here
            'sample_user_2',
            # Add more usernames as needed
        ]
        
        # Limit the number of tracks/albums to fetch per user to avoid rate limiting
        self.max_tracks_per_user = 100
        self.max_albums_per_user = 50
        self.max_artists_per_user = 30

    async def connect_db(self):
        """Connect to the PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.settings.database_host,
                port=self.settings.database_port,
                database=self.settings.database_name,
                user=self.settings.database_user,
                password=self.settings.database_password
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close_db(self):
        """Close the database connection"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")

    def get_or_create_lastfm_user(self, username: str) -> str:
        """Get or create a Last.fm user record and return the ID"""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Check if user exists
            cursor.execute(
                "SELECT id FROM lastfm_users WHERE lastfm_username = %s",
                (username,)
            )
            result = cursor.fetchone()
            
            if result:
                user_id = result['id']
                logger.info(f"Found existing user: {username} (ID: {user_id})")
                return user_id
            
            # Create new user
            cursor.execute("""
                INSERT INTO lastfm_users (lastfm_username, display_name, data_fetch_enabled)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (username, username, True))
            
            user_id = cursor.fetchone()['id']
            self.db_conn.commit()
            logger.info(f"Created new user: {username} (ID: {user_id})")
            return user_id

    def log_data_fetch(self, username: str, fetch_type: str, status: str, 
                       tracks_fetched: int = 0, albums_fetched: int = 0, 
                       artists_fetched: int = 0, error_message: str = None):
        """Log data fetch operations"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO data_fetch_logs 
                (lastfm_username, fetch_type, status, tracks_fetched, 
                 albums_fetched, artists_fetched, error_message, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (username, fetch_type, status, tracks_fetched, 
                  albums_fetched, artists_fetched, error_message, datetime.now()))
            self.db_conn.commit()

    async def fetch_user_top_tracks(self, username: str) -> List[Dict]:
        """Fetch top tracks for a specific user"""
        try:
            logger.info(f"Fetching top tracks for user: {username}")
            
            # Use the existing LastFM service to fetch data
            tracks_data = await self.lastfm_service.get_user_top_tracks(
                username, 
                limit=self.max_tracks_per_user
            )
            
            if not tracks_data:
                logger.warning(f"No tracks found for user: {username}")
                return []
            
            logger.info(f"Fetched {len(tracks_data)} tracks for user: {username}")
            return tracks_data
            
        except Exception as e:
            logger.error(f"Error fetching tracks for {username}: {e}")
            self.log_data_fetch(username, 'tracks', 'failed', error_message=str(e))
            return []

    async def fetch_user_top_albums(self, username: str) -> List[Dict]:
        """Fetch top albums for a specific user"""
        try:
            logger.info(f"Fetching top albums for user: {username}")
            
            albums_data = await self.lastfm_service.get_user_top_albums(
                username, 
                limit=self.max_albums_per_user
            )
            
            if not albums_data:
                logger.warning(f"No albums found for user: {username}")
                return []
            
            logger.info(f"Fetched {len(albums_data)} albums for user: {username}")
            return albums_data
            
        except Exception as e:
            logger.error(f"Error fetching albums for {username}: {e}")
            self.log_data_fetch(username, 'albums', 'failed', error_message=str(e))
            return []

    async def fetch_user_top_artists(self, username: str) -> List[Dict]:
        """Fetch top artists for a specific user"""
        try:
            logger.info(f"Fetching top artists for user: {username}")
            
            artists_data = await self.lastfm_service.get_user_top_artists(
                username, 
                limit=self.max_artists_per_user
            )
            
            if not artists_data:
                logger.warning(f"No artists found for user: {username}")
                return []
            
            logger.info(f"Fetched {len(artists_data)} artists for user: {username}")
            return artists_data
            
        except Exception as e:
            logger.error(f"Error fetching artists for {username}: {e}")
            self.log_data_fetch(username, 'artists', 'failed', error_message=str(e))
            return []

    def find_or_create_track(self, track_data: Dict) -> Optional[str]:
        """Find or create a track record and return the ID"""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Try to find by title and artist first
            cursor.execute("""
                SELECT id FROM tracks 
                WHERE LOWER(title) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
                LIMIT 1
            """, (track_data.get('name', ''), track_data.get('artist', {}).get('name', '')))
            
            result = cursor.fetchone()
            if result:
                return result['id']
            
            # If not found, create a new track record
            cursor.execute("""
                INSERT INTO tracks (title, artist, album, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                track_data.get('name', ''),
                track_data.get('artist', {}).get('name', ''),
                track_data.get('album', {}).get('name', ''),
                datetime.now(),
                datetime.now()
            ))
            
            track_id = cursor.fetchone()['id']
            self.db_conn.commit()
            logger.debug(f"Created new track: {track_data.get('name', '')}")
            return track_id

    def store_user_track_interactions(self, user_id: str, tracks_data: List[Dict]):
        """Store user-track interactions in the database"""
        if not tracks_data:
            return 0
        
        stored_count = 0
        with self.db_conn.cursor() as cursor:
            for track_data in tracks_data:
                try:
                    track_id = self.find_or_create_track(track_data)
                    if not track_id:
                        continue
                    
                    # Store the interaction
                    cursor.execute("""
                        INSERT INTO user_track_interactions 
                        (lastfm_user_id, track_id, play_count, user_loved, tags, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (lastfm_user_id, track_id) 
                        DO UPDATE SET 
                            play_count = EXCLUDED.play_count,
                            user_loved = EXCLUDED.user_loved,
                            tags = EXCLUDED.tags,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        user_id,
                        track_id,
                        track_data.get('playcount', 0),
                        track_data.get('loved', False),
                        json.dumps(track_data.get('tags', [])),
                        datetime.now(),
                        datetime.now()
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing track interaction: {e}")
                    continue
            
            self.db_conn.commit()
        
        logger.info(f"Stored {stored_count} track interactions for user {user_id}")
        return stored_count

    def store_user_album_interactions(self, user_id: str, albums_data: List[Dict]):
        """Store user-album interactions in the database"""
        if not albums_data:
            return 0
        
        stored_count = 0
        with self.db_conn.cursor() as cursor:
            for album_data in albums_data:
                try:
                    cursor.execute("""
                        INSERT INTO user_album_interactions 
                        (lastfm_user_id, album_title, album_artist, play_count, 
                         user_loved, tags, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (lastfm_user_id, album_title, album_artist) 
                        DO UPDATE SET 
                            play_count = EXCLUDED.play_count,
                            user_loved = EXCLUDED.user_loved,
                            tags = EXCLUDED.tags,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        user_id,
                        album_data.get('name', ''),
                        album_data.get('artist', {}).get('name', ''),
                        album_data.get('playcount', 0),
                        album_data.get('loved', False),
                        json.dumps(album_data.get('tags', [])),
                        datetime.now(),
                        datetime.now()
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing album interaction: {e}")
                    continue
            
            self.db_conn.commit()
        
        logger.info(f"Stored {stored_count} album interactions for user {user_id}")
        return stored_count

    def store_user_artist_interactions(self, user_id: str, artists_data: List[Dict]):
        """Store user-artist interactions in the database"""
        if not artists_data:
            return 0
        
        stored_count = 0
        with self.db_conn.cursor() as cursor:
            for artist_data in artists_data:
                try:
                    cursor.execute("""
                        INSERT INTO user_artist_interactions 
                        (lastfm_user_id, artist_name, play_count, 
                         user_loved, tags, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (lastfm_user_id, artist_name) 
                        DO UPDATE SET 
                            play_count = EXCLUDED.play_count,
                            user_loved = EXCLUDED.user_loved,
                            tags = EXCLUDED.tags,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        user_id,
                        artist_data.get('name', ''),
                        artist_data.get('playcount', 0),
                        artist_data.get('loved', False),
                        json.dumps(artist_data.get('tags', [])),
                        datetime.now(),
                        datetime.now()
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing artist interaction: {e}")
                    continue
            
            self.db_conn.commit()
        
        logger.info(f"Stored {stored_count} artist interactions for user {user_id}")
        return stored_count

    async def process_user_data(self, username: str):
        """Process all data for a single user"""
        logger.info(f"Processing data for user: {username}")
        
        # Get or create user record
        user_id = self.get_or_create_lastfm_user(username)
        
        # Fetch user data
        tracks_data = await self.fetch_user_top_tracks(username)
        albums_data = await self.fetch_user_top_albums(username)
        artists_data = await self.fetch_user_top_artists(username)
        
        # Store interactions
        tracks_stored = self.store_user_track_interactions(user_id, tracks_data)
        albums_stored = self.store_user_album_interactions(user_id, albums_data)
        artists_stored = self.store_user_artist_interactions(user_id, artists_data)
        
        # Log successful fetch
        self.log_data_fetch(
            username, 'all', 'success', 
            tracks_stored, albums_stored, artists_stored
        )
        
        # Update user's last_updated timestamp
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                UPDATE lastfm_users 
                SET last_updated = %s 
                WHERE id = %s
            """, (datetime.now(), user_id))
            self.db_conn.commit()
        
        logger.info(f"Completed processing for user: {username}")
        return {
            'username': username,
            'tracks': tracks_stored,
            'albums': albums_stored,
            'artists': artists_stored
        }

    async def run(self):
        """Main method to run the data population process"""
        try:
            await self.connect_db()
            
            results = []
            for username in self.target_users:
                try:
                    result = await self.process_user_data(username)
                    results.append(result)
                    
                    # Add a small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing user {username}: {e}")
                    self.log_data_fetch(username, 'all', 'failed', error_message=str(e))
                    continue
            
            # Print summary
            logger.info("=== Data Population Summary ===")
            for result in results:
                logger.info(f"{result['username']}: {result['tracks']} tracks, "
                           f"{result['albums']} albums, {result['artists']} artists")
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            self.close_db()

async def main():
    """Main entry point"""
    populator = CollaborativeDataPopulator()
    await populator.run()

if __name__ == "__main__":
    asyncio.run(main())





