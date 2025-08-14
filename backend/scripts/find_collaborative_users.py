#!/usr/bin/env python3
"""
Script to help find good Last.fm users for collaborative filtering.
This script analyzes user profiles to find users with substantial listening history
and diverse music taste that would be good candidates for collaborative filtering.
"""

import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv
import os

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LastFMUserFinder:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.lastfm_api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        
        # Criteria for good collaborative filtering users
        self.min_playcount = 1000  # Minimum total plays
        self.min_tracks = 50        # Minimum unique tracks
        self.min_albums = 20        # Minimum unique albums
        self.min_artists = 15       # Minimum unique artists

    async def find_users_from_artist(self, artist_name: str, limit: int = 20) -> List[Dict]:
        """Find users who listen to a specific artist"""
        try:
            params = {
                'method': 'artist.gettopfans',
                'artist': artist_name,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'topfans' in data and 'user' in data['topfans']:
                return data['topfans']['user']
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding users for artist {artist_name}: {e}")
            return []

    async def find_users_from_track(self, track_name: str, artist_name: str, limit: int = 20) -> List[Dict]:
        """Find users who listen to a specific track"""
        try:
            params = {
                'method': 'track.gettopfans',
                'track': track_name,
                'artist': artist_name,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'topfans' in data and 'user' in data['topfans']:
                return data['topfans']['user']
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding users for track {track_name}: {e}")
            return []

    async def find_users_from_tag(self, tag_name: str, limit: int = 20) -> List[Dict]:
        """Find users who listen to music with a specific tag"""
        try:
            params = {
                'method': 'tag.gettopfans',
                'tag': tag_name,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'topfans' in data and 'user' in data['topfans']:
                return data['topfans']['user']
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding users for tag {tag_name}: {e}")
            return []

    async def analyze_user_profile(self, username: str) -> Optional[Dict]:
        """Analyze a user's profile to determine if they're a good candidate"""
        try:
            # Get user info
            params = {
                'method': 'user.getinfo',
                'user': username,
                'api_key': self.api_key,
                'format': 'json'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'user' not in data:
                return None
            
            user_info = data['user']
            
            # Check if profile is public
            if user_info.get('subscriber') == '0' and int(user_info.get('playcount', 0)) < self.min_playcount:
                return None
            
            # Get user's top tracks to assess diversity
            top_tracks = await self.get_user_top_tracks(username, limit=50)
            if not top_tracks or len(top_tracks) < self.min_tracks:
                return None
            
            # Get user's top albums
            top_albums = await self.get_user_top_albums(username, limit=30)
            if not top_albums or len(top_albums) < self.min_albums:
                return None
            
            # Get user's top artists
            top_artists = await self.get_user_top_artists(username, limit=20)
            if not top_artists or len(top_artists) < self.min_artists:
                return None
            
            # Calculate diversity score
            diversity_score = self.calculate_diversity_score(top_tracks, top_albums, top_artists)
            
            return {
                'username': username,
                'display_name': user_info.get('realname', username),
                'playcount': int(user_info.get('playcount', 0)),
                'registered': user_info.get('registered', {}).get('unixtime', 0),
                'country': user_info.get('country', 'Unknown'),
                'tracks_count': len(top_tracks),
                'albums_count': len(top_albums),
                'artists_count': len(top_artists),
                'diversity_score': diversity_score,
                'top_tracks': top_tracks[:10],  # Top 10 for preview
                'top_albums': top_albums[:10],
                'top_artists': top_artists[:10]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user {username}: {e}")
            return None

    async def get_user_top_tracks(self, username: str, limit: int = 50) -> List[Dict]:
        """Get user's top tracks"""
        try:
            params = {
                'method': 'user.gettoptracks',
                'user': username,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'toptracks' in data and 'track' in data['toptracks']:
                return data['toptracks']['track']
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting top tracks for {username}: {e}")
            return []

    async def get_user_top_albums(self, username: str, limit: int = 30) -> List[Dict]:
        """Get user's top albums"""
        try:
            params = {
                'method': 'user.gettopalbums',
                'user': username,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'topalbums' in data and 'album' in data['topalbums']:
                return data['topalbums']['album']
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting top albums for {username}: {e}")
            return []

    async def get_user_top_artists(self, username: str, limit: int = 20) -> List[Dict]:
        """Get user's top artists"""
        try:
            params = {
                'method': 'user.gettopartists',
                'user': username,
                'api_key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'topartists' in data and 'artist' in data['topartists']:
                return data['topartists']['artist']
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting top artists for {username}: {e}")
            return []

    def calculate_diversity_score(self, tracks: List[Dict], albums: List[Dict], artists: List[Dict]) -> float:
        """Calculate a diversity score based on variety of music taste"""
        if not tracks or not albums or not artists:
            return 0.0
        
        # Count unique artists in tracks
        unique_artists = set()
        for track in tracks:
            if 'artist' in track and 'name' in track['artist']:
                unique_artists.add(track['artist']['name'].lower())
        
        # Count unique genres/tags (if available)
        unique_tags = set()
        for track in tracks:
            if 'tags' in track and 'tag' in track['tags']:
                for tag in track['tags']['tag']:
                    if isinstance(tag, dict) and 'name' in tag:
                        unique_tags.add(tag['name'].lower())
                    elif isinstance(tag, str):
                        unique_tags.add(tag.lower())
        
        # Calculate diversity based on variety
        artist_diversity = min(len(unique_artists) / len(artists), 1.0) if artists else 0
        tag_diversity = min(len(unique_tags) / max(len(tracks), 1), 1.0)
        
        # Weighted diversity score
        diversity_score = (artist_diversity * 0.6) + (tag_diversity * 0.4)
        
        return round(diversity_score, 3)

    async def find_good_collaborative_users(self, search_terms: List[str], max_users_per_term: int = 10) -> List[Dict]:
        """Find good collaborative filtering users based on search terms"""
        all_candidates = []
        
        for term in search_terms:
            logger.info(f"Searching for users based on: {term}")
            
            # Try different search methods
            users = []
            
            # Search by artist
            users.extend(await self.find_users_from_artist(term, max_users_per_term))
            
            # Search by tag
            users.extend(await self.find_users_from_tag(term, max_users_per_term))
            
            # Remove duplicates
            seen_usernames = set()
            unique_users = []
            for user in users:
                username = user.get('name', '')
                if username and username not in seen_usernames:
                    seen_usernames.add(username)
                    unique_users.append(user)
            
            # Analyze each user
            for user in unique_users[:max_users_per_term]:
                username = user.get('name', '')
                if username:
                    profile = await self.analyze_user_profile(username)
                    if profile:
                        all_candidates.append(profile)
                        logger.info(f"Found candidate: {username} (Score: {profile['diversity_score']})")
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Sort by diversity score and playcount
        all_candidates.sort(key=lambda x: (x['diversity_score'], x['playcount']), reverse=True)
        
        return all_candidates

    def save_candidates_to_file(self, candidates: List[Dict], filename: str = 'collaborative_user_candidates.json'):
        """Save candidate users to a JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(candidates, f, indent=2, default=str)
            
            logger.info(f"Saved {len(candidates)} candidates to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving candidates to file: {e}")

    def print_candidates_summary(self, candidates: List[Dict]):
        """Print a summary of candidate users"""
        if not candidates:
            logger.info("No suitable candidates found")
            return
        
        logger.info(f"\n=== Found {len(candidates)} Candidate Users ===\n")
        
        for i, candidate in enumerate(candidates[:20], 1):  # Show top 20
            logger.info(f"{i:2d}. {candidate['username']} ({candidate['display_name']})")
            logger.info(f"     Plays: {candidate['playcount']:,} | "
                       f"Tracks: {candidate['tracks_count']} | "
                       f"Albums: {candidate['albums_count']} | "
                       f"Artists: {candidate['artists_count']}")
            logger.info(f"     Diversity Score: {candidate['diversity_score']} | "
                       f"Country: {candidate['country']}")
            
            # Show top artists
            top_artists = [artist.get('name', '') for artist in candidate['top_artists'][:5]]
            logger.info(f"     Top Artists: {', '.join(top_artists)}")
            logger.info("")

async def main():
    """Main entry point"""
    finder = LastFMUserFinder()
    
    # Search terms - you can modify these based on your music taste
    search_terms = [
        # Genres/styles
        'indie rock',
        'jazz',
        'electronic',
        'hip hop',
        'folk',
        
        # Specific artists you like
        'Radiohead',
        'Kendrick Lamar',
        'Daft Punk',
        'The Beatles',
        'Miles Davis',
        
        # Specific tags
        'experimental',
        'underground',
        'classic',
        'modern',
        'alternative'
    ]
    
    logger.info("Starting user discovery process...")
    logger.info(f"Search terms: {', '.join(search_terms)}")
    
    # Find candidates
    candidates = await finder.find_good_collaborative_users(search_terms, max_users_per_term=15)
    
    # Print summary
    finder.print_candidates_summary(candidates)
    
    # Save to file
    finder.save_candidates_to_file(candidates)
    
    logger.info(f"\nDiscovery complete! Found {len(candidates)} candidate users.")
    logger.info("Review the candidates and add the best ones to your collaborative_users.py config file.")

if __name__ == "__main__":
    asyncio.run(main())





