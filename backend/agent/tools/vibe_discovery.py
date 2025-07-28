from typing import Dict, Any, List, Optional
import httpx
import json
import os
from .base import BaseTool, ToolResult


class VibeDiscoveryTool(BaseTool):
    """Tool for discovering song vibes and characteristics using Perplexity AI."""
    
    def __init__(self):
        super().__init__(
            name="vibe_discovery",
            description="Analyze songs to discover their vibe, mood, and characteristics using AI when tags/metadata are missing"
        )
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute vibe discovery for a specific song - REQUIRES song_name and artist_name."""
        try:
            # Extract song information from different intent formats
            song_name = None
            artist_name = None
            query = None
            
            if hasattr(intent_result, 'song_name') and hasattr(intent_result, 'artist_name'):
                # New ToolInput format
                song_name = intent_result.song_name
                artist_name = intent_result.artist_name
                query = intent_result.raw_text if hasattr(intent_result, 'raw_text') else ""
            elif isinstance(intent_result, dict):
                song_name = intent_result.get("song", "")
                artist_name = intent_result.get("artist", "")
                query = intent_result.get("query", "")
            elif hasattr(intent_result, 'raw_text'):
                # Legacy format - try to extract from raw text
                query = intent_result.raw_text
                # Only allow this for backward compatibility, prefer structured input
                is_song_analysis = self._is_song_analysis_request(query)
                if not is_song_analysis:
                    return self._create_error_result(
                        "Vibe discovery requires specific song_name and artist_name, not general mood queries. Use web_feel tool for mood-based discovery."
                    )
            
            # ENFORCE specific song analysis - no fallback to mood discovery
            if not song_name or not artist_name:
                return self._create_error_result(
                    "Vibe discovery requires specific song_name and artist_name. "
                    "For mood-based discovery, use the web_feel tool instead."
                )
            
            # Construct query for analysis
            analysis_query = f"{song_name} by {artist_name}"
            
            # Analyze the specific song
            vibe_data = await self._analyze_song_vibe(analysis_query)
            
            if not vibe_data:
                return self._create_error_result(f"Could not analyze vibe for '{analysis_query}'")
            
            return self._create_success_result({
                "query": query or analysis_query,
                "song_analyzed": analysis_query,
                "vibe_analysis": vibe_data,
                "is_song_analysis": True,
                "search_keywords": self._extract_search_keywords(vibe_data),
                "mood_tags": self._extract_mood_tags(vibe_data),
                "musical_characteristics": self._extract_musical_characteristics(vibe_data)
            }, confidence=0.8)
            
        except Exception as e:
            return self._create_error_result(f"Vibe discovery failed: {str(e)}")
    
    def _is_song_analysis_request(self, query: str) -> bool:
        """Determine if the query is asking to analyze a specific song."""
        song_indicators = ["by ", " - ", "song ", "track ", "'", '"']
        return any(indicator in query.lower() for indicator in song_indicators)
    
    async def _analyze_song_vibe(self, song_query: str) -> Optional[Dict[str, Any]]:
        """Analyze the vibe and characteristics of a specific song."""
        if not self.perplexity_api_key:
            return await self._fallback_song_analysis(song_query)
        
        prompt = f"""Analyze the song "{song_query}" and provide detailed information about:

1. Musical vibe and mood (happy, sad, energetic, calm, etc.)
2. Genre and subgenre
3. Tempo and energy level
4. Emotional characteristics
5. Musical elements (instruments, production style)
6. Similar artists or songs
7. What situations/activities this song is good for
8. Key descriptive tags and keywords

Provide a comprehensive analysis focusing on the song's emotional and musical characteristics that would help with music recommendation."""
        
        try:
            response = await self._query_perplexity(prompt)
            if response:
                return {
                    "song": song_query,
                    "analysis": response,
                    "type": "song_analysis"
                }
        except Exception as e:
            print(f"Perplexity analysis failed: {e}")
            return await self._fallback_song_analysis(song_query)
        
        return None
    
    async def _discover_mood_characteristics(self, mood_query: str) -> Optional[Dict[str, Any]]:
        """Discover characteristics of a mood/vibe for music recommendation."""
        if not self.perplexity_api_key:
            return self._fallback_mood_analysis(mood_query)
        
        prompt = f"""Describe music that matches the mood/vibe: "{mood_query}"

Provide information about:
1. What musical genres and styles fit this mood
2. Tempo and energy characteristics (BPM range, high/low energy)
3. Instrumental and vocal characteristics
4. Emotional qualities to look for
5. Similar moods and related descriptors
6. Example artists and songs that embody this vibe
7. Musical elements (major/minor keys, instruments, production styles)

Focus on actionable characteristics that would help find music matching this specific mood."""
        
        try:
            response = await self._query_perplexity(prompt)
            if response:
                return {
                    "mood": mood_query,
                    "characteristics": response,
                    "type": "mood_discovery"
                }
        except Exception as e:
            print(f"Perplexity mood discovery failed: {e}")
            return self._fallback_mood_analysis(mood_query)
        
        return None
    
    async def _query_perplexity(self, prompt: str) -> Optional[str]:
        """Query Perplexity AI API."""
        if not self.perplexity_api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.perplexity_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"Perplexity API error: {response.status_code} - {response.text}")
                return None
    
    async def _fallback_song_analysis(self, song_query: str) -> Dict[str, Any]:
        """Fallback analysis when Perplexity API is unavailable - LIMITED functionality."""
        # Very basic pattern matching - much more limited than before
        song_lower = song_query.lower()
        
        characteristics = ["unknown"]
        analysis_note = f"Limited fallback analysis for {song_query}. Full analysis requires external API."
        
        # Only provide very basic characteristics based on obvious keywords
        if any(word in song_lower for word in ['sad', 'melancholy', 'blue']):
            characteristics = ['melancholic']
        elif any(word in song_lower for word in ['happy', 'joy']):
            characteristics = ['upbeat']
        elif any(word in song_lower for word in ['love', 'heart']):
            characteristics = ['romantic']
        
        return {
            "song": song_query,
            "analysis": analysis_note,
            "type": "limited_fallback_analysis",
            "characteristics": characteristics,
            "warning": "This is a limited fallback analysis. For full vibe discovery, external API is required."
        }
    
    def _fallback_mood_analysis(self, mood_query: str) -> Dict[str, Any]:
        """DEPRECATED: Vibe discovery no longer supports mood queries."""
        return {
            "error": "Vibe discovery tool no longer supports general mood queries.",
            "suggestion": "Use the web_feel tool for mood-based music discovery.",
            "type": "deprecated_functionality"
        }
        
        return {
            "mood": mood_query,
            "characteristics": f"Fallback characteristics for {mood_query}: {', '.join(characteristics)}",
            "type": "fallback_mood",
            "inferred_genres": genres,
            "inferred_characteristics": characteristics
        }
    
    def _extract_search_keywords(self, vibe_data: Dict[str, Any]) -> List[str]:
        """Extract keywords that can be used for music search."""
        keywords = []
        
        analysis_text = vibe_data.get("analysis", "") + " " + vibe_data.get("characteristics", "")
        
        # Common music descriptors to extract
        descriptors = [
            'electronic', 'acoustic', 'rock', 'pop', 'hip-hop', 'jazz', 'classical',
            'energetic', 'calm', 'upbeat', 'melancholic', 'atmospheric', 'ambient',
            'fast', 'slow', 'moderate', 'heavy', 'light', 'dark', 'bright',
            'emotional', 'introspective', 'celebratory', 'romantic', 'aggressive'
        ]
        
        for descriptor in descriptors:
            if descriptor in analysis_text.lower():
                keywords.append(descriptor)
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _extract_mood_tags(self, vibe_data: Dict[str, Any]) -> List[str]:
        """Extract mood tags from the analysis."""
        mood_tags = []
        
        analysis_text = vibe_data.get("analysis", "") + " " + vibe_data.get("characteristics", "")
        
        mood_keywords = [
            'happy', 'sad', 'energetic', 'calm', 'excited', 'melancholic',
            'uplifting', 'dark', 'bright', 'mysterious', 'nostalgic', 'dreamy',
            'aggressive', 'peaceful', 'intense', 'gentle', 'powerful', 'delicate'
        ]
        
        for mood in mood_keywords:
            if mood in analysis_text.lower():
                mood_tags.append(mood)
        
        return mood_tags[:8]  # Limit to top 8 mood tags
    
    def _extract_musical_characteristics(self, vibe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract musical characteristics for ML model features."""
        analysis_text = vibe_data.get("analysis", "") + " " + vibe_data.get("characteristics", "")
        text_lower = analysis_text.lower()
        
        characteristics = {}
        
        # Energy level
        if any(word in text_lower for word in ['high energy', 'energetic', 'intense', 'powerful']):
            characteristics['energy'] = 'high'
        elif any(word in text_lower for word in ['low energy', 'calm', 'peaceful', 'gentle']):
            characteristics['energy'] = 'low'
        else:
            characteristics['energy'] = 'medium'
        
        # Valence (positivity)
        if any(word in text_lower for word in ['happy', 'uplifting', 'positive', 'joyful']):
            characteristics['valence'] = 'high'
        elif any(word in text_lower for word in ['sad', 'melancholic', 'dark', 'negative']):
            characteristics['valence'] = 'low'
        else:
            characteristics['valence'] = 'medium'
        
        # Tempo
        if any(word in text_lower for word in ['fast', 'quick', 'rapid', 'uptempo']):
            characteristics['tempo'] = 'fast'
        elif any(word in text_lower for word in ['slow', 'ballad', 'downtempo']):
            characteristics['tempo'] = 'slow'
        else:
            characteristics['tempo'] = 'medium'
        
        # Acousticness
        if any(word in text_lower for word in ['acoustic', 'organic', 'natural', 'unplugged']):
            characteristics['acousticness'] = 'high'
        elif any(word in text_lower for word in ['electronic', 'synthetic', 'digital']):
            characteristics['acousticness'] = 'low'
        else:
            characteristics['acousticness'] = 'medium'
        
        return characteristics