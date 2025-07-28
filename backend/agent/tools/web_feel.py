from typing import Dict, Any, List
import httpx
import json
from .base import BaseTool, ToolResult


class WebFeelTool(BaseTool):
    """Tool for extracting mood/vibe descriptions using web search APIs."""
    
    def __init__(self):
        super().__init__(
            name="web_feel",
            description="Extract mood and vibe descriptions from web search"
        )
        self.perplexity_api_key = None  # Would be set from config
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute web mood extraction based on intent results."""
        try:
            # Handle different intent formats
            if hasattr(intent_result, 'entities'):
                entities = intent_result.entities
                mood_query = entities.get("mood", entities.get("search_query", ""))
            elif hasattr(intent_result, 'raw_text'):
                mood_query = intent_result.raw_text
            elif isinstance(intent_result, dict):
                mood_query = intent_result.get("mood", intent_result.get("query", ""))
            else:
                mood_query = str(intent_result)
            
            if not mood_query:
                return self._create_error_result("No mood query provided")
            
            # Try different mood extraction methods
            mood_description = await self._extract_mood_description(mood_query)
            
            if not mood_description:
                return self._create_error_result("Could not extract mood description")
            
            # Generate music-related keywords from mood
            music_keywords = self._generate_music_keywords(mood_description)
            
            return self._create_success_result({
                "mood_description": mood_description,
                "music_keywords": music_keywords,
                "search_terms": self._create_search_terms(mood_description),
                "explanation": f"Extracted mood characteristics for '{mood_query}'"
            }, confidence=0.7)
            
        except Exception as e:
            return self._create_error_result(f"Web feel extraction failed: {str(e)}")
    
    async def _extract_mood_description(self, mood_query: str) -> str:
        """Extract mood description using various methods."""
        # Try Perplexity API first
        perplexity_result = await self._query_perplexity(mood_query)
        if perplexity_result:
            return perplexity_result
        
        # Fallback to built-in mood mapping
        return self._get_builtin_mood_description(mood_query)
    
    async def _query_perplexity(self, mood_query: str) -> str:
        """Query Perplexity API for mood description."""
        if not self.perplexity_api_key:
            return ""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a music expert. Describe the musical characteristics, mood, and vibe of the given phrase in 2-3 sentences. Focus on tempo, energy, instruments, and emotional qualities that would help in music recommendation."
                            },
                            {
                                "role": "user",
                                "content": f"What kind of music would match this mood or feeling: '{mood_query}'"
                            }
                        ],
                        "max_tokens": 200,
                        "temperature": 0.3
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except Exception:
            # Silently fall back to builtin descriptions
            pass
        
        return ""
    
    def _get_builtin_mood_description(self, mood_query: str) -> str:
        """Get mood description from built-in mapping."""
        mood_descriptions = {
            "summer night": "Warm, atmospheric tracks with a relaxed tempo and dreamy quality. Think indie folk, chillwave, or acoustic guitar-driven songs with a nostalgic, romantic feel.",
            
            "rainy day": "Contemplative, melancholic music with gentle rhythms. Perfect for introspective indie rock, ambient electronic, or singer-songwriter tracks with emotional depth.",
            
            "workout": "High-energy tracks with driving beats and motivational lyrics. Electronic dance music, hip-hop, or rock with strong rhythmic elements and uplifting energy.",
            
            "studying": "Instrumental or minimal vocal tracks that aid concentration. Lo-fi hip-hop, ambient music, or classical pieces with steady, non-distracting rhythms.",
            
            "party": "Upbeat, danceable tracks with infectious energy. Pop, dance, hip-hop, or electronic music with strong beats and crowd-pleasing melodies.",
            
            "chill": "Laid-back, relaxed music with smooth production. Jazz, neo-soul, chillwave, or acoustic tracks with mellow vibes and easy-going rhythms.",
            
            "energetic": "Fast-paced, dynamic music with high energy levels. Rock, electronic, pop-punk, or any genre with driving rhythms and exciting arrangements.",
            
            "melancholic": "Emotionally resonant music with minor keys and introspective themes. Indie rock, folk, or alternative tracks that evoke deep feelings.",
            
            "nostalgic": "Music that evokes memories and past experiences. Classic rock, vintage pop, or contemporary tracks with retro influences and emotional connection.",
            
            "focus": "Minimal, repetitive music that enhances concentration. Ambient, minimal techno, or instrumental tracks with subtle variations and hypnotic qualities."
        }
        
        query_lower = mood_query.lower()
        
        # Direct match
        if query_lower in mood_descriptions:
            return mood_descriptions[query_lower]
        
        # Partial match
        for mood, description in mood_descriptions.items():
            if any(word in query_lower for word in mood.split()):
                return description
        
        # Generic fallback
        return f"Music that captures the essence and feeling of '{mood_query}', with appropriate tempo, energy, and emotional qualities."
    
    def _generate_music_keywords(self, mood_description: str) -> List[str]:
        """Generate music-related keywords from mood description."""
        keyword_mapping = {
            "warm": ["ambient", "dreamy", "soft"],
            "atmospheric": ["reverb", "ethereal", "spacious"],
            "relaxed": ["chill", "mellow", "laid-back"],
            "dreamy": ["shoegaze", "ambient", "ethereal"],
            "nostalgic": ["vintage", "retro", "classic"],
            "romantic": ["intimate", "soft", "acoustic"],
            "contemplative": ["introspective", "thoughtful", "minimal"],
            "melancholic": ["sad", "minor key", "emotional"],
            "gentle": ["soft", "acoustic", "tender"],
            "high-energy": ["energetic", "fast", "driving"],
            "driving": ["rhythmic", "powerful", "intense"],
            "motivational": ["uplifting", "positive", "inspiring"],
            "upbeat": ["happy", "energetic", "danceable"],
            "danceable": ["groove", "rhythm", "beat"],
            "infectious": ["catchy", "memorable", "hooky"],
            "laid-back": ["relaxed", "easy", "smooth"],
            "smooth": ["jazz", "soul", "r&b"],
            "mellow": ["soft", "gentle", "calm"],
            "easy-going": ["casual", "comfortable", "relaxed"],
            "fast-paced": ["uptempo", "energetic", "quick"],
            "dynamic": ["varied", "exciting", "changing"],
            "introspective": ["thoughtful", "deep", "personal"],
            "emotional": ["feeling", "expressive", "moving"],
            "hypnotic": ["repetitive", "trance", "mesmerizing"],
            "minimal": ["simple", "clean", "sparse"]
        }
        
        keywords = set()
        description_lower = mood_description.lower()
        
        for word, related_keywords in keyword_mapping.items():
            if word in description_lower:
                keywords.update(related_keywords)
        
        return list(keywords)[:10]  # Limit to top 10 keywords
    
    def _create_search_terms(self, mood_description: str) -> List[str]:
        """Create search terms for music discovery."""
        # Extract key musical terms from description
        musical_terms = [
            "tempo", "energy", "rhythm", "beat", "melody", "harmony",
            "acoustic", "electronic", "rock", "pop", "jazz", "folk",
            "indie", "alternative", "classical", "ambient", "dance",
            "hip-hop", "r&b", "soul", "funk", "blues", "country"
        ]
        
        search_terms = []
        description_lower = mood_description.lower()
        
        for term in musical_terms:
            if term in description_lower:
                search_terms.append(term)
        
        # Add genre keywords based on description
        if "electronic" in description_lower or "dance" in description_lower:
            search_terms.extend(["electronic", "edm", "house", "techno"])
        
        if "acoustic" in description_lower or "folk" in description_lower:
            search_terms.extend(["acoustic", "folk", "singer-songwriter"])
        
        if "ambient" in description_lower or "atmospheric" in description_lower:
            search_terms.extend(["ambient", "chillout", "downtempo"])
        
        return list(set(search_terms))[:8]  # Limit and deduplicate