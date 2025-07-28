from typing import Dict, Any, List
import statistics
from .base import BaseTool, ToolResult
from services.spotify_service import SpotifyService
from ..embedding.engine import EmbeddingEngine


class PlaylistAnalyzerTool(BaseTool):
    """Tool for analyzing Spotify playlists and extracting characteristics."""
    
    def __init__(self):
        super().__init__(
            name="playlist_analyzer",
            description="Analyze Spotify playlist to extract musical characteristics and preferences"
        )
        self.spotify_service = SpotifyService()
        self.embedding_engine = EmbeddingEngine()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute playlist analysis based on intent results."""
        try:
            entities = intent_result.entities
            playlist_id = entities.get("playlist_id")
            playlist_url = entities.get("playlist_url")
            
            # Extract playlist ID from URL if provided
            if playlist_url and not playlist_id:
                playlist_id = self._extract_playlist_id(playlist_url)
            
            if not playlist_id:
                return self._create_error_result("No playlist ID or URL provided")
            
            # Get playlist data from Spotify
            playlist_data = await self.spotify_service.get_playlist(playlist_id)
            if not playlist_data:
                return self._create_error_result("Could not fetch playlist data")
            
            # Analyze playlist characteristics
            analysis = await self._analyze_playlist(playlist_data)
            
            return self._create_success_result({
                "playlist_analysis": analysis,
                "tracks": analysis["tracks"][:20],  # Return sample tracks
                "explanation": f"Analyzed playlist '{playlist_data.get('name', 'Unknown')}' with {len(analysis['tracks'])} tracks"
            }, confidence=0.9)
            
        except Exception as e:
            return self._create_error_result(f"Playlist analysis failed: {str(e)}")
    
    def _extract_playlist_id(self, playlist_url: str) -> str:
        """Extract playlist ID from Spotify URL."""
        # Handle various Spotify URL formats
        if "playlist/" in playlist_url:
            return playlist_url.split("playlist/")[1].split("?")[0]
        elif "playlist:" in playlist_url:
            return playlist_url.split("playlist:")[1]
        return ""
    
    async def _analyze_playlist(self, playlist_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive playlist analysis."""
        tracks = []
        audio_features_list = []
        
        # Extract track information
        for item in playlist_data.get("tracks", {}).get("items", []):
            track = item.get("track")
            if not track:
                continue
            
            track_info = {
                "id": track["id"],
                "name": track["name"],
                "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                "album": track["album"]["name"],
                "popularity": track.get("popularity", 0),
                "duration_ms": track.get("duration_ms", 0),
                "explicit": track.get("explicit", False)
            }
            
            # Add album artwork
            if track["album"].get("images"):
                track_info["artwork_url"] = track["album"]["images"][0]["url"]
            
            tracks.append(track_info)
        
        # Get audio features for all tracks
        track_ids = [track["id"] for track in tracks if track["id"]]
        if track_ids:
            audio_features = await self.spotify_service.get_audio_features(track_ids)
            audio_features_list = [f for f in audio_features if f]
        
        # Calculate playlist characteristics
        characteristics = self._calculate_characteristics(audio_features_list)
        
        # Extract genres and artists
        genres, artists = self._extract_genres_and_artists(playlist_data)
        
        # Generate playlist embedding
        playlist_embedding = await self._generate_playlist_embedding(tracks)
        
        return {
            "playlist_id": playlist_data.get("id"),
            "playlist_name": playlist_data.get("name"),
            "description": playlist_data.get("description"),
            "tracks": tracks,
            "track_count": len(tracks),
            "characteristics": characteristics,
            "top_genres": genres[:10],
            "top_artists": artists[:10],
            "playlist_embedding": playlist_embedding.tolist() if playlist_embedding is not None else None,
            "analysis_summary": self._generate_analysis_summary(characteristics, genres, artists)
        }
    
    def _calculate_characteristics(self, audio_features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate musical characteristics from audio features."""
        if not audio_features_list:
            return {}
        
        # Define feature extractors
        features = [
            "danceability", "energy", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "loudness"
        ]
        
        characteristics = {}
        
        for feature in features:
            values = [f.get(feature, 0) for f in audio_features_list if f.get(feature) is not None]
            if values:
                characteristics[feature] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
        
        # Calculate derived characteristics
        characteristics["mood_profile"] = self._determine_mood_profile(characteristics)
        characteristics["energy_level"] = self._determine_energy_level(characteristics)
        characteristics["musical_style"] = self._determine_musical_style(characteristics)
        
        return characteristics
    
    def _extract_genres_and_artists(self, playlist_data: Dict[str, Any]) -> tuple:
        """Extract and count genres and artists from playlist."""
        artist_count = {}
        genre_count = {}
        
        for item in playlist_data.get("tracks", {}).get("items", []):
            track = item.get("track")
            if not track:
                continue
            
            # Count artists
            for artist in track.get("artists", []):
                artist_name = artist["name"]
                artist_count[artist_name] = artist_count.get(artist_name, 0) + 1
                
                # Note: Getting genres would require additional Spotify API calls
                # This is a simplified version
        
        # Sort by frequency
        top_artists = sorted(artist_count.items(), key=lambda x: x[1], reverse=True)
        top_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        
        return [artist for artist, _ in top_artists], [genre for genre, _ in top_genres]
    
    async def _generate_playlist_embedding(self, tracks: List[Dict[str, Any]]) -> Any:
        """Generate embedding vector for the entire playlist."""
        if not tracks:
            return None
        
        # Generate embeddings for each track
        track_embeddings = []
        for track in tracks[:50]:  # Limit to first 50 tracks for performance
            embedding = await self.embedding_engine.embed_track_metadata(track)
            track_embeddings.append(embedding.vector)
        
        if not track_embeddings:
            return None
        
        # Average embeddings to represent playlist
        import numpy as np
        return np.mean(track_embeddings, axis=0)
    
    def _determine_mood_profile(self, characteristics: Dict[str, Any]) -> str:
        """Determine overall mood profile of the playlist."""
        if not characteristics:
            return "unknown"
        
        energy = characteristics.get("energy", {}).get("mean", 0.5)
        valence = characteristics.get("valence", {}).get("mean", 0.5)
        danceability = characteristics.get("danceability", {}).get("mean", 0.5)
        
        if energy > 0.7 and valence > 0.7:
            return "energetic_positive"
        elif energy > 0.7 and valence < 0.3:
            return "energetic_aggressive"
        elif energy < 0.3 and valence > 0.7:
            return "calm_positive"
        elif energy < 0.3 and valence < 0.3:
            return "calm_melancholic"
        elif danceability > 0.7:
            return "danceable"
        else:
            return "balanced"
    
    def _determine_energy_level(self, characteristics: Dict[str, Any]) -> str:
        """Determine energy level of the playlist."""
        if not characteristics:
            return "unknown"
        
        energy = characteristics.get("energy", {}).get("mean", 0.5)
        tempo = characteristics.get("tempo", {}).get("mean", 120)
        
        if energy > 0.8 or tempo > 140:
            return "very_high"
        elif energy > 0.6 or tempo > 120:
            return "high"
        elif energy > 0.4 or tempo > 100:
            return "medium"
        elif energy > 0.2 or tempo > 80:
            return "low"
        else:
            return "very_low"
    
    def _determine_musical_style(self, characteristics: Dict[str, Any]) -> List[str]:
        """Determine musical style tags for the playlist."""
        if not characteristics:
            return []
        
        styles = []
        
        acousticness = characteristics.get("acousticness", {}).get("mean", 0)
        instrumentalness = characteristics.get("instrumentalness", {}).get("mean", 0)
        speechiness = characteristics.get("speechiness", {}).get("mean", 0)
        liveness = characteristics.get("liveness", {}).get("mean", 0)
        
        if acousticness > 0.7:
            styles.append("acoustic")
        elif acousticness < 0.3:
            styles.append("electronic")
        
        if instrumentalness > 0.5:
            styles.append("instrumental")
        
        if speechiness > 0.33:
            styles.append("spoken_word")
        elif speechiness > 0.1:
            styles.append("rap_hip_hop")
        
        if liveness > 0.3:
            styles.append("live_recording")
        
        return styles
    
    def _generate_analysis_summary(
        self, 
        characteristics: Dict[str, Any], 
        genres: List[str], 
        artists: List[str]
    ) -> str:
        """Generate human-readable analysis summary."""
        if not characteristics:
            return "Analysis could not be completed due to insufficient data."
        
        mood = characteristics.get("mood_profile", "balanced")
        energy = characteristics.get("energy_level", "medium")
        styles = characteristics.get("musical_style", [])
        
        summary_parts = []
        
        # Mood and energy
        mood_descriptions = {
            "energetic_positive": "upbeat and energetic",
            "energetic_aggressive": "intense and powerful",
            "calm_positive": "relaxed and uplifting",
            "calm_melancholic": "mellow and contemplative",
            "danceable": "groove-oriented and danceable",
            "balanced": "well-balanced"
        }
        
        summary_parts.append(f"This playlist has a {mood_descriptions.get(mood, mood)} vibe")
        summary_parts.append(f"with {energy} energy levels")
        
        # Musical style
        if styles:
            style_text = ", ".join(styles)
            summary_parts.append(f"featuring {style_text} elements")
        
        # Top artists
        if artists:
            if len(artists) >= 3:
                summary_parts.append(f"with frequent appearances by {artists[0]}, {artists[1]}, and {artists[2]}")
            elif len(artists) >= 2:
                summary_parts.append(f"with frequent appearances by {artists[0]} and {artists[1]}")
            else:
                summary_parts.append(f"with frequent appearances by {artists[0]}")
        
        return ". ".join(summary_parts) + "."