from typing import Dict, List, Any, Optional
import json
import os
from openai import AsyncOpenAI
from dataclasses import dataclass


@dataclass
class ToolSelection:
    tools: List[str]
    reasoning: str
    confidence: float


@dataclass 
class LLMResponse:
    content: str
    reasoning: str
    confidence: float


class LLMService:
    """
    LLM service for Timbre agent decision making and natural conversation.
    Handles tool selection and response generation using OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                self.client = AsyncOpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        self.available_tools = {
            "track_search": "Find specific songs, tracks, or artists by name",
            "similarity": "Find music similar to given tracks or artists", 
            "playlist_analyzer": "Analyze Spotify playlists and extract insights",
            "web_feel": "Extract mood and vibe descriptions for music discovery",
            "vibe_discovery": "Analyze songs to discover their vibe and characteristics using AI",
            "hybrid_recommender": "Generate personalized music recommendations",
            "explainability": "Explain why certain music was recommended",
            "memory_embedder": "Learn and update user music preferences",
            "spotify_search": "Search Spotify for tracks, albums, and artists",
            "spotify_playlist": "Access user's Spotify playlists and tracks",
            "song_recommendation": "Recommend specific songs with full Spotify metadata",
            "none": "No tools needed - pure conversational response"
        }
    
    async def select_tools(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolSelection:
        """
        Use LLM to intelligently select which tools to use for a user request.
        """
        tools_description = "\n".join([
            f"- {name}: {desc}" for name, desc in self.available_tools.items()
        ])
        
        context_info = ""
        if context:
            context_info = f"\nContext: {json.dumps(context, indent=2)}"
        
        system_prompt = f"""You are Timbre's intelligent tool selection system. Your job is to analyze user requests and select the most appropriate tools to fulfill them.

Available tools:
{tools_description}

Rules:
1. Select tools that best match the user's intent
2. You can select multiple tools if needed (e.g., spotify_search + similarity)
3. Use "spotify_search" for:
   - Finding tracks by album name (e.g., "songs from album Blonde")
   - Finding tracks by artist name
   - Finding specific tracks when you know the name (e.g., "find song X by artist Y")
   - ANY query that mentions a specific song name and artist
   - General music search queries
4. Use "track_search" ONLY for general music discovery when you don't have specific song/artist names
5. IMPORTANT: For queries like "find the song [NAME] by [ARTIST]", ALWAYS use "spotify_search" not "track_search"
5. Use "song_recommendation" ONLY when:
   - You want to suggest a specific song you already know (song name + artist)
   - You need to fetch metadata for a song you're specifically recommending
6. Use "vibe_discovery" for mood-based or vibe-based music exploration
7. Use "none" only for pure greetings, thanks, or casual conversation
8. For album-based requests like "recommend from album X", use "spotify_search" to find tracks from that album
9. Always provide reasoning for your selection
10. Return confidence score (0.0-1.0) based on how certain you are

Respond in JSON format:
{{
    "tools": ["tool1", "tool2"],
    "reasoning": "explanation of why these tools were selected",
    "confidence": 0.85
}}"""

        user_prompt = f"""User request: "{user_input}"{context_info}

Select the appropriate tools to handle this request."""

        # If no OpenAI client available, use fallback logic
        if not self.client:
            print("No OpenAI API key available, using fallback tool selection")
            if any(word in user_input.lower() for word in ['hey', 'hi', 'hello', 'thanks', 'bye', 'good morning', 'good afternoon', 'good evening']):
                return ToolSelection(tools=["none"], reasoning="Greeting detected (fallback)", confidence=0.9)
            elif any(word in user_input.lower() for word in ['similar', 'like', 'sounds like']):
                return ToolSelection(tools=["similarity"], reasoning="Similarity request detected (fallback)", confidence=0.8)
            elif 'playlist' in user_input.lower():
                return ToolSelection(tools=["playlist_analyzer"], reasoning="Playlist analysis request (fallback)", confidence=0.8)
            elif any(word in user_input.lower() for word in ['album', 'from the album', 'song from']):
                return ToolSelection(tools=["spotify_search"], reasoning="Album-based search request (fallback)", confidence=0.8)
            elif any(word in user_input.lower() for word in ['summer', 'winter', 'chill', 'energetic', 'sad', 'happy', 'mood', 'vibe', 'feeling']):
                return ToolSelection(tools=["vibe_discovery", "hybrid_recommender"], reasoning="Mood/vibe-based request detected (fallback)", confidence=0.8)
            elif any(word in user_input.lower() for word in ['sad', 'kendrick', 'song', 'track', 'artist']):
                return ToolSelection(tools=["spotify_search"], reasoning="Specific song/artist search request (fallback)", confidence=0.7)
            else:
                return ToolSelection(tools=["hybrid_recommender"], reasoning="General recommendation request (fallback)", confidence=0.6)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            return ToolSelection(
                tools=result["tools"],
                reasoning=result["reasoning"], 
                confidence=result["confidence"]
            )
            
        except Exception as e:
            # Fallback to safe defaults
            print(f"LLM tool selection error: {e}")
            if any(word in user_input.lower() for word in ['hey', 'hi', 'hello', 'thanks', 'bye']):
                return ToolSelection(tools=["none"], reasoning="Greeting detected (error fallback)", confidence=0.9)
            else:
                return ToolSelection(tools=["hybrid_recommender"], reasoning="Fallback recommendation (error)", confidence=0.5)
    
    async def generate_response(
        self,
        user_input: str,
        tools_used: List[str],
        tool_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate natural language response based on user input and tool results.
        """
        context_info = ""
        if context:
            context_info = f"\nUser context: {json.dumps(context, indent=2)}"
        
        tool_results_info = ""
        if tool_results:
            # Convert tool results to serializable format
            serializable_results = {}
            for tool_name, result in tool_results.items():
                if hasattr(result, 'data'):
                    serializable_results[tool_name] = {
                        'success': result.success,
                        'data': result.data,
                        'error': result.error
                    }
                else:
                    serializable_results[tool_name] = str(result)
            tool_results_info = f"\nTool results: {json.dumps(serializable_results, indent=2)}"
        
        system_prompt = """You are Timbre, an AI music discovery companion. You are helpful, knowledgeable, and passionate about music.

Your personality:
- Friendly and conversational
- Knowledgeable about music across all genres
- Enthusiastic about helping users discover new music
- Natural and human-like in responses
- Brief but informative

Background Information:
- Timbrality uses Hybrid Fusion of Collaborative Filtering and Content-Based Filtering to recommend music.
- Timbrality was created by Conner Groth, a Machine Learning Engineer Intern & CS student at CU Boulder.
- Timbrality combines metadata from Spotify, Last.fm, and albumoftheyear.org to recommend music.
- You have memory of all the user's past chats and can use this to recommend music.
- You can also use the user's Spotify/Last.fm listening history to recommend music.

Guidelines:
1. If tools were used, incorporate their results naturally into your response
2. For pure conversation (no tools), be friendly and guide toward music discovery
3. Always stay in character as Timbre
4. Keep responses concise but helpful
5. When recommending music, be enthusiastic but not overwhelming
6. When spotify_search returns multiple tracks from an album, pick ONE specific track to recommend and mention it by name
7. Format your recommendation as: "I recommend '[song name]' by [artist] from the album [album name]"

CRITICAL DATA ACCURACY RULES:
- ONLY use song names, artist names, and album names that appear EXACTLY in the tool results
- NEVER substitute or guess alternative names, even if you think you know the song
- If tool results show "Maroon by Kevin Abstract from album Blush", you MUST use "Blush" as the album name
- DO NOT use your training knowledge to "correct" or change any metadata provided by tools
- When in doubt, quote the tool result data verbatim

Respond in JSON format:
{
    "content": "your response to the user",
    "reasoning": "brief explanation of your response approach",
    "confidence": 0.85
}"""

        user_prompt = f"""User said: "{user_input}"
Tools used: {tools_used}{tool_results_info}{context_info}

Generate an appropriate response as Timbre."""

        # If no OpenAI client available, use simple fallback responses
        if not self.client:
            print("No OpenAI API key available, using fallback response generation")
            if tools_used == ["none"] or not tool_results:
                # Conversational responses for greetings
                if any(word in user_input.lower() for word in ['hey', 'hi', 'hello']):
                    return LLMResponse(
                        content="Hey! I'm Timbre, your AI music companion. I can help you discover new music, find similar tracks, analyze playlists, or explore different vibes. What kind of music are you in the mood for?",
                        reasoning="Greeting response (fallback)",
                        confidence=0.8
                    )
                elif any(word in user_input.lower() for word in ['thanks', 'thank you']):
                    return LLMResponse(
                        content="You're welcome! Happy to help you discover amazing music. Anything else you'd like to explore?",
                        reasoning="Thank you response (fallback)",
                        confidence=0.8
                    )
                elif any(word in user_input.lower() for word in ['bye', 'goodbye']):
                    return LLMResponse(
                        content="See you later! Thanks for exploring music with me. Come back anytime you want to discover something new!",
                        reasoning="Goodbye response (fallback)",
                        confidence=0.8
                    )
                else:
                    return LLMResponse(
                        content="I'm Timbre, your music discovery companion! I can help you find specific tracks, discover similar music, analyze playlists, or explore music based on your mood. What would you like to try?",
                        reasoning="General introduction (fallback)",
                        confidence=0.7
                    )
            else:
                return LLMResponse(
                    content="I found some music recommendations for you! Let me know what you think or if you'd like to explore something different.",
                    reasoning="Results response (fallback)",
                    confidence=0.6
                )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            return LLMResponse(
                content=result["content"],
                reasoning=result["reasoning"],
                confidence=result["confidence"]
            )
            
        except Exception as e:
            # Fallback response
            print(f"LLM response generation error: {e}")
            if not tool_results or tools_used == ["none"]:
                return LLMResponse(
                    content="Hey! I'm Timbre, your AI music companion. I can help you discover new music, find similar tracks, or explore different vibes. What kind of music are you in the mood for?",
                    reasoning="Fallback conversational response",
                    confidence=0.7
                )
            else:
                return LLMResponse(
                    content="I found some music recommendations for you! Let me know what you think or if you'd like to explore something different.",
                    reasoning="Fallback with results",
                    confidence=0.6
                )
    
    async def extract_music_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Extract structured music intent from natural language query.
        
        Returns:
            Dict with song_name, artist_name, album_name, search_query, intent_type, etc.
        """
        if not self.client:
            # Fallback to regex parsing if no LLM available
            return self._fallback_intent_extraction(user_query)
        
        system_prompt = """You are a music intent extraction system. Your job is to parse natural language music queries and extract structured information.

Extract the following information from user queries:
- song_name: The specific song title mentioned
- artist_name: The specific artist name mentioned  
- album_name: The specific album name mentioned (if any)
- search_query: Clean search terms for music services (song + artist)
- intent_type: The user's intent (search, analyze, recommend, etc.)
- confidence: How confident you are in the extraction (0.0-1.0)

Examples:
Input: "find the song maroon by kevin abstract and explain its vibe"
Output: {
  "song_name": "Maroon",
  "artist_name": "Kevin Abstract", 
  "album_name": null,
  "search_query": "Maroon Kevin Abstract",
  "intent_type": "search_and_analyze",
  "confidence": 0.95
}

Input: "play something from the album blonde by frank ocean"
Output: {
  "song_name": null,
  "artist_name": "Frank Ocean",
  "album_name": "Blonde", 
  "search_query": "Frank Ocean Blonde",
  "intent_type": "album_search",
  "confidence": 0.9
}

Rules:
1. Extract exact names as mentioned (don't correct or change them)
2. If no specific song/artist mentioned, set to null
3. Create clean search_query by combining relevant terms
4. Be conservative with confidence scores
5. Handle typos and variations gracefully
6. Return valid JSON only

Respond with JSON only, no other text."""

        user_prompt = f"Extract music intent from: \"{user_query}\""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            intent_data = json.loads(content)
            
            # Validate required fields
            required_fields = ["song_name", "artist_name", "search_query", "intent_type", "confidence"]
            for field in required_fields:
                if field not in intent_data:
                    intent_data[field] = None
            
            return intent_data
            
        except Exception as e:
            print(f"LLM intent extraction failed: {e}")
            return self._fallback_intent_extraction(user_query)
    
    def _fallback_intent_extraction(self, user_query: str) -> Dict[str, Any]:
        """Fallback regex-based intent extraction when LLM fails."""
        import re
        
        query_lower = user_query.lower().strip()
        
        # Remove common command phrases
        prefixes_to_remove = [
            r'^find\s+(the\s+)?song\s+',
            r'^search\s+for\s+',
            r'^look\s+for\s+',
            r'^play\s+',
            r'^get\s+me\s+',
        ]
        
        for prefix in prefixes_to_remove:
            query_lower = re.sub(prefix, '', query_lower)
        
        # Try to extract "SONG by ARTIST" pattern
        by_pattern = r'^(.+?)\s+by\s+(.+?)(?:\s+and\s+.*)?$'
        match = re.match(by_pattern, query_lower)
        
        if match:
            song_name = match.group(1).strip()
            artist_name = match.group(2).strip()
            # Remove trailing phrases
            artist_name = re.sub(r'\s+(and\s+explain.*|on\s+spotify.*|from.*)', '', artist_name)
            
            return {
                "song_name": song_name,
                "artist_name": artist_name,
                "album_name": None,
                "search_query": f"{song_name} {artist_name}",
                "intent_type": "search",
                "confidence": 0.7
            }
        
        # Fallback: return original query
        return {
            "song_name": None,
            "artist_name": None,
            "album_name": None,
            "search_query": query_lower,
            "intent_type": "general_search",
            "confidence": 0.3
        }