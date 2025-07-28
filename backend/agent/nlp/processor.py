from typing import Dict, Any, List, Optional
import re
from dataclasses import dataclass
from datetime import datetime

from ..types import IntentType


@dataclass
class IntentResult:
    intent: IntentType
    entities: Dict[str, Any]
    confidence: float
    raw_text: str
    processed_text: str


class NaturalLanguageProcessor:
    """
    Processes natural language input for intent detection and entity extraction.
    Handles music-specific queries and generates human-readable responses.
    """
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns()
    
    async def parse_intent(self, user_input: str, context) -> IntentResult:
        """Parse user input to detect intent and extract entities."""
        processed_text = self._preprocess_text(user_input)
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(processed_text, context)
        
        # Extract entities
        entities = self._extract_entities(processed_text, intent)
        
        # Add context-based entities
        entities.update(self._extract_context_entities(context))
        
        return IntentResult(
            intent=intent,
            entities=entities,
            confidence=intent_confidence,
            raw_text=user_input,
            processed_text=processed_text
        )
    
    async def generate_response(
        self,
        user_input: str,
        intent_result: IntentResult,
        tracks: List[Dict[str, Any]],
        explanations: List[str],
        context
    ) -> str:
        """Generate natural language response based on results."""
        if not tracks:
            return self._generate_no_results_response(user_input, intent_result)
        
        # Generate appropriate response based on intent
        if intent_result.intent == IntentType.SEARCH_TRACK:
            return self._generate_search_response(tracks, user_input)
        elif intent_result.intent == IntentType.FIND_SIMILAR:
            return self._generate_similarity_response(tracks, intent_result.entities)
        elif intent_result.intent == IntentType.VIBE_DISCOVERY:
            return self._generate_vibe_response(tracks, intent_result.entities)
        elif intent_result.intent == IntentType.ANALYZE_PLAYLIST:
            return self._generate_playlist_analysis_response(tracks, explanations)
        elif intent_result.intent == IntentType.CONVERSATIONAL:
            return self._generate_conversational_response(user_input, intent_result)
        else:
            return self._generate_general_response(tracks, explanations)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _detect_intent(self, text: str, context) -> tuple[IntentType, float]:
        """Detect user intent from processed text."""
        best_intent = IntentType.SEARCH_TRACK
        best_confidence = 0.5
        
        for intent, patterns in self.intent_patterns.items():
            confidence = self._calculate_pattern_match(text, patterns)
            
            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence
        
        # Context-based intent adjustment
        if context and hasattr(context, 'conversation_history'):
            best_intent, best_confidence = self._adjust_intent_with_context(
                best_intent, best_confidence, context
            )
        
        return best_intent, best_confidence
    
    def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract entities from text based on intent."""
        entities = {}
        
        # Extract common entities
        entities.update(self._extract_track_entities(text))
        entities.update(self._extract_mood_entities(text))
        entities.update(self._extract_search_entities(text))
        entities.update(self._extract_count_entities(text))
        
        # Intent-specific entity extraction
        if intent == IntentType.ANALYZE_PLAYLIST:
            entities.update(self._extract_playlist_entities(text))
        elif intent == IntentType.FIND_SIMILAR:
            entities.update(self._extract_similarity_entities(text))
        
        return entities
    
    def _extract_context_entities(self, context) -> Dict[str, Any]:
        """Extract entities from conversation context."""
        entities = {}
        
        if hasattr(context, 'current_mood') and context.current_mood:
            entities['context_mood'] = context.current_mood
        
        if hasattr(context, 'user_preferences') and context.user_preferences:
            entities['user_preferences'] = context.user_preferences
        
        return entities
    
    def _build_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Build patterns for intent detection."""
        return {
            IntentType.SEARCH_TRACK: [
                r'\b(find|search|look for|show me)\b.*\b(song|track|music)\b',
                r'\b(play|playing)\b.*\b(song|track|music)\b',
                r'\bby\s+\w+',  # "by artist"
                r'\btrack\s+called\b',
                r'\bsong\s+called\b'
            ],
            
            IntentType.FIND_SIMILAR: [
                r'\b(similar|like|sounds like)\b',
                r'\b(more|other)\s+(songs?|tracks?|music)\s+(like|similar)',
                r'\brecommend\s+.*(similar|like)',
                r'\bfind\s+.*(similar|like)',
                r'\bsomething\s+like\b'
            ],
            
            IntentType.VIBE_DISCOVERY: [
                r'\b(vibe|mood|feeling|atmosphere)\b',
                r'\bfeels?\s+like\b',
                r'\bmood\s+for\b',
                r'\bsomething\s+(chill|energetic|sad|happy|upbeat)\b',
                r'\b(summer|winter|night|morning)\s+(music|vibe)\b',
                r'\bi\s+(want|need)\s+something\b'
            ],
            
            IntentType.ANALYZE_PLAYLIST: [
                r'\bplaylist\b',
                r'\banalyze\s+.*playlist\b',
                r'\btell\s+me\s+about\s+.*playlist\b',
                r'spotify\.com.*playlist',
                r'\bplaylist\s+id\b'
            ],
            
            IntentType.EXPLAIN_RECOMMENDATION: [
                r'\bwhy\b.*\brecommend\b',
                r'\bexplain\b.*\brecommendation\b',
                r'\bhow\s+did\s+you\b',
                r'\bwhat\s+makes\b.*\bsimilar\b'
            ],
            
            IntentType.UPDATE_PREFERENCES: [
                r'\bi\s+(like|love|hate|dislike)\b',
                r'\bnot\s+my\s+style\b',
                r'\bmore\s+like\s+this\b',
                r'\b(good|great|perfect|terrible|awful)\s+(recommendation|suggestion)\b'
            ],
            
            IntentType.CONVERSATIONAL: [
                r'(?:^|\s)(hey|hi|hello|hiya|sup|yo)(?:\s|$)',
                r'(?:^|\s)(good morning|good afternoon|good evening)(?:\s|$)',
                r'(?:^|\s)(how are you|whats up|how\'s it going)(?:\s|$)',
                r'(?:^|\s)(thanks|thank you|thx)(?:\s|$)',
                r'(?:^|\s)(bye|goodbye|see ya|later|cya)(?:\s|$)',
                r'(?:^|\s)(what can you do|help|what are you)(?:\s|$)'
            ]
        }
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for entity extraction."""
        return {
            'track_name': [
                r'(?:song|track|music)\s+["\']([^"\']+)["\']',
                r'["\']([^"\']+)["\']',
                r'(?:called|titled)\s+["\']?([^"\']+?)["\']?(?:\s|$)'
            ],
            
            'artist': [
                r'by\s+([A-Za-z\s&\-]+?)(?:\s|$|,|\.|;)',
                r'artist\s+([A-Za-z\s&\-]+?)(?:\s|$|,|\.|;)',
                r'from\s+([A-Za-z\s&\-]+?)(?:\s|$|,|\.|;)'
            ],
            
            'mood': [
                r'\b(chill|relaxed|calm|mellow|peaceful)\b',
                r'\b(energetic|upbeat|exciting|intense|powerful)\b',
                r'\b(sad|melancholic|depressing|emotional|heartbreak)\b',
                r'\b(happy|joyful|cheerful|positive|uplifting)\b',
                r'\b(dark|moody|atmospheric|ambient|mysterious)\b',
                r'\b(romantic|love|intimate|sensual)\b',
                r'\b(party|dance|club|dancing|groove)\b',
                r'\b(workout|gym|running|exercise|motivation)\b',
                r'\b(summer|tropical|sunny|beach|vacation)\b',
                r'\b(winter|cold|cozy|fireplace)\b',
                r'\b(night|evening|late|midnight|3am)\b',
                r'\b(morning|dawn|sunrise|coffee|wake up)\b'
            ],
            
            'genre': [
                r'\b(rock|pop|jazz|classical|electronic|hip.hop|rap|country|folk|blues)\b',
                r'\b(indie|alternative|metal|punk|reggae|ska|funk|soul|r&b)\b',
                r'\b(ambient|techno|house|dubstep|trance|drum.and.bass)\b',
                r'\b(acoustic|lo.fi|chillwave|synthwave|vaporwave)\b'
            ],
            
            'playlist_url': [
                r'(https?://open\.spotify\.com/playlist/[a-zA-Z0-9]+)',
                r'(spotify:playlist:[a-zA-Z0-9]+)'
            ],
            
            'count': [
                r'\b(\d+)\s+(?:songs?|tracks?|recommendations?)\b',
                r'\b(\d+)\s+(?:summer|winter|chill|energetic|sad|happy|mood|vibe)\s+(?:songs?|tracks?|music)\b',
                r'\b(few|several|many|lots?|bunch)\b',
                r'\btop\s+(\d+)\b',
                r'\b(\d+)\b'  # Fallback: any number
            ]
        }
    
    def _calculate_pattern_match(self, text: str, patterns: List[str]) -> float:
        """Calculate how well text matches a set of patterns."""
        if not patterns:
            return 0.0
        
        matches = 0
        total_weight = 0
        
        for pattern in patterns:
            weight = 1.0
            if re.search(pattern, text, re.IGNORECASE):
                matches += weight
            total_weight += weight
        
        return matches / total_weight if total_weight > 0 else 0.0
    
    def _adjust_intent_with_context(self, intent: IntentType, confidence: float, context) -> tuple:
        """Adjust intent based on conversation context."""
        # If previous query was about a specific track/playlist, 
        # "similar" might refer to that
        history = getattr(context, 'conversation_history', [])
        if history:
            last_interaction = history[-1]
            if 'tracks' in last_interaction and intent == IntentType.SEARCH_TRACK:
                # Might be asking for similar tracks
                intent = IntentType.FIND_SIMILAR
                confidence = min(confidence + 0.2, 1.0)
        
        return intent, confidence
    
    def _extract_track_entities(self, text: str) -> Dict[str, Any]:
        """Extract track-related entities."""
        entities = {}
        
        # Track name
        for pattern in self.entity_patterns['track_name']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['track_name'] = match.group(1).strip()
                break
        
        # Artist
        for pattern in self.entity_patterns['artist']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['artist'] = match.group(1).strip()
                break
        
        return entities
    
    def _extract_mood_entities(self, text: str) -> Dict[str, Any]:
        """Extract mood and vibe entities."""
        entities = {}
        moods = []
        
        for pattern in self.entity_patterns['mood']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                moods.append(match.group())
        
        if moods:
            entities['mood'] = ' '.join(moods)
        
        return entities
    
    def _extract_search_entities(self, text: str) -> Dict[str, Any]:
        """Extract general search query."""
        entities = {}
        
        # If no specific track/artist found, use the whole query
        if 'track_name' not in entities and 'artist' not in entities:
            # Remove common stopwords and extract meaningful terms
            stopwords = {'i', 'want', 'need', 'find', 'search', 'for', 'some', 'music', 'song', 'track'}
            words = [w for w in text.split() if w not in stopwords and len(w) > 2]
            
            if words:
                entities['search_query'] = ' '.join(words)
        
        return entities
    
    def _extract_count_entities(self, text: str) -> Dict[str, Any]:
        """Extract number of recommendations requested."""
        entities = {}
        
        for pattern in self.entity_patterns['count']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count_str = match.group(1)
                if count_str.isdigit():
                    count = int(count_str)
                    # Only use fallback numbers if they're reasonable (1-20)
                    if 1 <= count <= 20:
                        entities['count'] = count
                        break
                else:
                    # Convert word to number
                    word_to_num = {
                        'few': 5, 'several': 7, 'many': 15, 
                        'lot': 20, 'lots': 20, 'bunch': 10
                    }
                    entities['count'] = word_to_num.get(count_str, 10)
                    break
        
        return entities
    
    def _extract_playlist_entities(self, text: str) -> Dict[str, Any]:
        """Extract playlist-related entities."""
        entities = {}
        
        for pattern in self.entity_patterns['playlist_url']:
            match = re.search(pattern, text)
            if match:
                entities['playlist_url'] = match.group(1)
                # Extract playlist ID
                if 'playlist/' in match.group(1):
                    playlist_id = match.group(1).split('playlist/')[1].split('?')[0]
                    entities['playlist_id'] = playlist_id
                break
        
        return entities
    
    def _extract_similarity_entities(self, text: str) -> Dict[str, Any]:
        """Extract similarity-related entities."""
        entities = {}
        
        # Look for references to "this", "that", etc. which might refer to context
        if re.search(r'\b(this|that|it)\b', text):
            entities['similarity_reference'] = 'context'
        
        return entities
    
    def _generate_search_response(self, tracks: List[Dict[str, Any]], user_input: str) -> str:
        """Generate response for search results."""
        if not tracks:
            return f"I couldn't find any tracks matching '{user_input}'. Try a different search term?"
        
        count = len(tracks)
        first_track = tracks[0]
        
        if count == 1:
            return f"I found '{first_track.get('name')}' by {first_track.get('artist')}."
        else:
            return f"I found {count} tracks. Here's '{first_track.get('name')}' by {first_track.get('artist')} and {count-1} others."
    
    def _generate_similarity_response(self, tracks: List[Dict[str, Any]], entities: Dict[str, Any]) -> str:
        """Generate response for similarity recommendations."""
        if not tracks:
            return "I couldn't find similar tracks. Try being more specific about what you're looking for."
        
        seed_info = ""
        if entities.get('track_name'):
            seed_info = f" similar to '{entities['track_name']}'"
        elif entities.get('artist'):
            seed_info = f" in the style of {entities['artist']}"
        
        count = len(tracks)
        return f"I found {count} tracks{seed_info}. Here are some recommendations that match that vibe."
    
    def _generate_vibe_response(self, tracks: List[Dict[str, Any]], entities: Dict[str, Any]) -> str:
        """Generate response for vibe/mood recommendations."""
        mood = entities.get('mood', 'your request')
        count = len(tracks)
        
        mood_descriptions = {
            'chill': 'relaxing and laid-back',
            'energetic': 'high-energy and exciting',
            'sad': 'emotional and contemplative',
            'happy': 'uplifting and positive',
            'summer': 'perfect for summer vibes',
            'night': 'atmospheric and nighttime-appropriate'
        }
        
        description = mood_descriptions.get(mood, f'matching the "{mood}" vibe')
        
        return f"I found {count} tracks that are {description}. These should capture the feeling you're looking for."
    
    def _generate_playlist_analysis_response(
        self, 
        tracks: List[Dict[str, Any]], 
        explanations: List[str]
    ) -> str:
        """Generate response for playlist analysis."""
        if explanations:
            return f"I analyzed your playlist. {explanations[0]} Here are some tracks from it:"
        else:
            return f"I analyzed your playlist and found {len(tracks)} tracks. Here's what I discovered:"
    
    def _generate_general_response(
        self, 
        tracks: List[Dict[str, Any]], 
        explanations: List[str]
    ) -> str:
        """Generate general response."""
        count = len(tracks)
        
        if explanations:
            explanation = explanations[0]
            return f"I found {count} recommendations. {explanation}"
        else:
            return f"Here are {count} personalized recommendations based on your preferences."
    
    def _generate_conversational_response(self, user_input: str, intent_result: IntentResult) -> str:
        """Generate conversational response for greetings and general chat."""
        user_input_lower = user_input.lower().strip()
        
        # Greetings
        if any(greeting in user_input_lower for greeting in ['hey', 'hi', 'hello', 'hiya', 'sup', 'yo']):
            return "Hey! I'm Timbre, your AI music companion. I can help you discover new music, find similar tracks, analyze playlists, or explore different vibes. What kind of music are you in the mood for?"
        
        # Morning/evening greetings
        elif any(greeting in user_input_lower for greeting in ['good morning', 'good afternoon', 'good evening']):
            return "Hello! Ready to discover some great music today? I can help you find tracks based on your mood, search for specific songs, or recommend similar music to what you love."
        
        # How are you / what's up
        elif any(phrase in user_input_lower for phrase in ['how are you', 'whats up', "what's up", "how's it going"]):
            return "I'm doing great, thanks for asking! I'm here and ready to help you explore music. What can I help you discover today?"
        
        # Thanks
        elif any(thanks in user_input_lower for thanks in ['thanks', 'thank you', 'thx']):
            return "You're welcome! Happy to help you discover amazing music. Anything else you'd like to explore?"
        
        # Goodbye
        elif any(bye in user_input_lower for bye in ['bye', 'goodbye', 'see ya', 'later', 'cya']):
            return "See you later! Thanks for exploring music with me. Come back anytime you want to discover something new!"
        
        # Help/capabilities
        elif any(phrase in user_input_lower for phrase in ['what can you do', 'help', 'what are you']):
            return "I'm Timbre, your AI music companion! I can help you:\n\n• Search for specific tracks or artists\n• Find music similar to songs you love\n• Discover music based on moods and vibes\n• Analyze your Spotify playlists\n• Get personalized recommendations\n\nJust tell me what you're looking for, like 'find me some chill indie music' or 'something similar to Radiohead'!"
        
        # Default friendly response
        else:
            return "I'm Timbre, your music discovery companion! I'm here to help you explore and discover music. Try asking me to find similar songs, recommend music for a specific mood, or search for tracks by your favorite artists!"
    
    def _generate_no_results_response(self, user_input: str, intent_result: IntentResult) -> str:
        """Generate response when no results are found."""
        responses = {
            IntentType.SEARCH_TRACK: "I couldn't find any tracks matching your search. Try different keywords or artist names.",
            IntentType.FIND_SIMILAR: "I couldn't find similar tracks. Try providing more specific information about what you're looking for.",
            IntentType.VIBE_DISCOVERY: "I couldn't find tracks matching that vibe. Try describing the mood or feeling differently.",
            IntentType.ANALYZE_PLAYLIST: "I couldn't analyze that playlist. Make sure the playlist URL is correct and public.",
        }
        
        return responses.get(
            intent_result.intent, 
            "I couldn't find any recommendations. Try rephrasing your request or being more specific."
        )