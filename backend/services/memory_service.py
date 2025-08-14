"""
Dual-Store Memory Service for Timbrality

Implements the Redis (fast working memory) + PostgreSQL+pgvector (durable long-term memory) architecture.
"""

import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import httpx
import numpy as np
from dataclasses import dataclass

from config.settings import settings
from cache.redis import set_cache, get_cache, delete_cache
from agent.embedding.engine import get_embedding


@dataclass
class MemoryEntry:
    """Structured memory entry."""
    id: Optional[int] = None
    user_id: str = ""
    chat_id: Optional[str] = None
    kind: str = "fact"  # summary, fact, preference, tool_output, conversation
    content: str = ""
    embedding: Optional[List[float]] = None
    importance: int = 1  # 1-5 scale
    created_at: Optional[datetime] = None
    similarity: Optional[float] = None
    final_score: Optional[float] = None


@dataclass
class UserPreferences:
    """User preference profile."""
    user_id: str
    top_genres: List[str] = None
    top_moods: List[str] = None
    artist_affinities: Dict[str, float] = None
    depth_weight: float = 0.5
    novelty_weight: float = 0.5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class RecommendationEvent:
    """Recommendation tracking entry."""
    user_id: str
    item_id: str
    item_type: str = "track"  # track, album
    reason: Optional[str] = None
    score: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class ChatTurn:
    """Individual chat turn for working memory."""
    message_type: str  # user, agent, system, tool
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime


class MemoryService:
    """
    Dual-store memory service implementing Redis for working memory 
    and PostgreSQL+pgvector for long-term memory.
    """
    
    def __init__(self):
        self.supabase_client = None
        self._init_supabase()
    
    def _init_supabase(self):
        """Initialize Supabase client."""
        try:
            from supabase import create_client
            if settings.supabase_url and settings.supabase_anon_key:
                self.supabase_client = create_client(
                    settings.supabase_url, 
                    settings.supabase_anon_key
                )
        except ImportError:
            print("Warning: supabase-py not installed. Long-term memory will not work.")
        except Exception as e:
            print(f"Warning: Failed to initialize Supabase client: {e}")
    
    # === REDIS WORKING MEMORY ===
    
    async def add_chat_turn(
        self, 
        user_id: str, 
        chat_id: str, 
        turn: ChatTurn,
        max_turns: int = 200
    ) -> None:
        """Add a chat turn to Redis working memory."""
        key = f"session:{user_id}:{chat_id}:recent"
        
        # Serialize turn
        turn_data = {
            "message_type": turn.message_type,
            "content": turn.content,
            "metadata": turn.metadata,
            "timestamp": turn.timestamp.isoformat()
        }
        
        try:
            # Get existing turns
            existing = await get_cache(key) or []
            if not isinstance(existing, list):
                existing = []
            
            # Add new turn
            existing.append(turn_data)
            
            # Keep only recent turns
            if len(existing) > max_turns:
                existing = existing[-max_turns:]
            
            # Store back with TTL
            await set_cache(key, existing, expire_seconds=60*60*72)  # 72 hours
            
            # Also store in PostgreSQL for durability
            await self._store_chat_message_db(user_id, chat_id, turn)
            
        except Exception as e:
            print(f"Error adding chat turn to Redis: {e}")
    
    async def get_recent_turns(
        self, 
        user_id: str, 
        chat_id: str, 
        limit: int = 50
    ) -> List[ChatTurn]:
        """Get recent chat turns from Redis."""
        key = f"session:{user_id}:{chat_id}:recent"
        
        try:
            cached_turns = await get_cache(key)
            if not cached_turns:
                return []
            
            # Convert back to ChatTurn objects
            turns = []
            for turn_data in cached_turns[-limit:]:
                turns.append(ChatTurn(
                    message_type=turn_data["message_type"],
                    content=turn_data["content"],
                    metadata=turn_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(turn_data["timestamp"])
                ))
            
            return turns
            
        except Exception as e:
            print(f"Error getting recent turns from Redis: {e}")
            return []
    
    async def update_scratch_state(
        self, 
        user_id: str, 
        chat_id: str, 
        state_data: Dict[str, Any]
    ) -> None:
        """Update ephemeral scratch state in Redis."""
        key = f"session:{user_id}:{chat_id}:scratch"
        
        try:
            # Get existing state
            current_state = await get_cache(key) or {}
            
            # Update with new data
            current_state.update(state_data)
            current_state["last_updated"] = datetime.now().isoformat()
            
            # Store with TTL
            await set_cache(key, current_state, expire_seconds=60*60*24)  # 24 hours
            
        except Exception as e:
            print(f"Error updating scratch state: {e}")
    
    async def get_scratch_state(
        self, 
        user_id: str, 
        chat_id: str
    ) -> Dict[str, Any]:
        """Get ephemeral scratch state from Redis."""
        key = f"session:{user_id}:{chat_id}:scratch"
        
        try:
            return await get_cache(key) or {}
        except Exception as e:
            print(f"Error getting scratch state: {e}")
            return {}
    
    async def update_recent_topics(
        self, 
        user_id: str, 
        topic: str, 
        score: float = None
    ) -> None:
        """Update user's recent topics in Redis."""
        key = f"user:{user_id}:recent_topics"
        
        try:
            topics = await get_cache(key) or {}
            
            # Add/update topic with current timestamp as score
            topics[topic] = score or datetime.now().timestamp()
            
            # Keep only recent topics (last 20)
            if len(topics) > 20:
                # Sort by score and keep top 20
                sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
                topics = dict(sorted_topics[:20])
            
            await set_cache(key, topics, expire_seconds=60*60*24*7)  # 1 week
            
        except Exception as e:
            print(f"Error updating recent topics: {e}")
    
    async def get_recent_topics(self, user_id: str) -> List[str]:
        """Get user's recent topics from Redis."""
        key = f"user:{user_id}:recent_topics"
        
        try:
            topics = await get_cache(key) or {}
            # Return topics sorted by recency
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            return [topic for topic, _ in sorted_topics]
        except Exception as e:
            print(f"Error getting recent topics: {e}")
            return []
    
    # === POSTGRESQL LONG-TERM MEMORY ===
    
    async def store_memory(self, memory: MemoryEntry) -> Optional[int]:
        """Store a memory entry in PostgreSQL with embedding."""
        if not self.supabase_client:
            print("Warning: Supabase client not available")
            return None
        
        try:
            # Generate embedding if not provided
            if not memory.embedding and memory.content:
                memory.embedding = await get_embedding(memory.content)
            
            # Prepare data for insertion
            data = {
                "user_id": memory.user_id,
                "chat_id": memory.chat_id,
                "kind": memory.kind,
                "content": memory.content,
                "embedding": memory.embedding,
                "importance": memory.importance
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            # Insert into enhanced agent_memory table
            result = self.supabase_client.table("agent_memory").insert(data).execute()
            
            if result.data:
                return result.data[0]["id"]
            
        except Exception as e:
            print(f"Error storing memory in PostgreSQL: {e}")
        
        return None
    
    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        chat_id: Optional[str] = None,
        match_count: int = 12,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
        similarity_threshold: float = 0.7
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories using enhanced semantic search."""
        if not self.supabase_client:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await get_embedding(query)
            if not query_embedding:
                return []
            
            # Call the enhanced match_agent_memories function
            result = self.supabase_client.rpc(
                "match_agent_memories",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                    "filter_user_id": user_id,
                    "recency_weight": recency_weight,
                    "importance_weight": importance_weight,
                    "similarity_threshold": similarity_threshold
                }
            ).execute()
            
            # Convert to MemoryEntry objects
            memories = []
            for row in result.data:
                memories.append(MemoryEntry(
                    id=row["id"],
                    user_id=row["user_id"],
                    chat_id=None,  # agent_memory doesn't store chat_id
                    kind=row["kind"],
                    content=row["content"],
                    importance=int(row["weight"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    similarity=row["similarity"],
                    final_score=row["final_score"]
                ))
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []
    
    async def create_conversation_summary(
        self,
        user_id: str,
        chat_id: str,
        window_size: int = 20
    ) -> Optional[MemoryEntry]:
        """Create a summary of recent conversation turns."""
        if not self.supabase_client:
            return None
        
        try:
            # Get conversation data from PostgreSQL
            result = self.supabase_client.rpc(
                "summarize_conversation_window",
                {
                    "filter_user_id": user_id,
                    "filter_chat_id": chat_id,
                    "window_size": window_size
                }
            ).execute()
            
            if not result.data or not result.data[0]["summary_content"]:
                return None
            
            conversation_data = result.data[0]
            
            # Create summary using LLM (you'd implement this)
            summary_content = await self._summarize_with_llm(
                conversation_data["summary_content"]
            )
            
            if summary_content:
                # Store the summary as a memory
                summary_memory = MemoryEntry(
                    user_id=user_id,
                    chat_id=chat_id,
                    kind="summary",
                    content=summary_content,
                    importance=3  # Medium importance for summaries
                )
                
                memory_id = await self.store_memory(summary_memory)
                summary_memory.id = memory_id
                
                return summary_memory
            
        except Exception as e:
            print(f"Error creating conversation summary: {e}")
        
        return None
    
    # === CONTEXT ASSEMBLY ===
    
    async def assemble_context(
        self,
        user_id: str,
        chat_id: str,
        query: str,
        max_recent_turns: int = 20,
        max_memories: int = 10
    ) -> Dict[str, Any]:
        """Assemble complete context from both Redis and PostgreSQL."""
        context = {}
        
        try:
            # Get recent turns from Redis (fast working memory)
            recent_turns = await self.get_recent_turns(
                user_id, chat_id, limit=max_recent_turns
            )
            
            # Get relevant memories from PostgreSQL (long-term memory)
            relevant_memories = await self.retrieve_memories(
                query=query,
                user_id=user_id,
                chat_id=chat_id,
                match_count=max_memories
            )
            
            # Get scratch state
            scratch_state = await self.get_scratch_state(user_id, chat_id)
            
            # Get recent topics
            recent_topics = await self.get_recent_topics(user_id)
            
            context = {
                "recent_turns": [
                    {
                        "type": turn.message_type,
                        "content": turn.content,
                        "metadata": turn.metadata,
                        "timestamp": turn.timestamp.isoformat()
                    }
                    for turn in recent_turns
                ],
                "relevant_memories": [
                    {
                        "kind": memory.kind,
                        "content": memory.content,
                        "importance": memory.importance,
                        "similarity": memory.similarity,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None
                    }
                    for memory in relevant_memories
                ],
                "scratch_state": scratch_state,
                "recent_topics": recent_topics[:10],  # Top 10 recent topics
                "context_stats": {
                    "recent_turn_count": len(recent_turns),
                    "memory_count": len(relevant_memories),
                    "avg_memory_similarity": (
                        sum(m.similarity for m in relevant_memories if m.similarity) / len(relevant_memories)
                        if relevant_memories else 0
                    )
                }
            }
            
        except Exception as e:
            print(f"Error assembling context: {e}")
        
        return context
    
    # === BACKGROUND PROCESSING ===
    
    async def process_conversation_batch(
        self,
        user_id: str,
        chat_id: str,
        batch_size: int = 20
    ) -> None:
        """Background processing of conversation turns into memories."""
        try:
            # Get recent turns that haven't been processed
            recent_turns = await self.get_recent_turns(user_id, chat_id, limit=batch_size)
            
            if len(recent_turns) >= batch_size:
                # Create conversation summary
                summary = await self.create_conversation_summary(
                    user_id, chat_id, window_size=batch_size
                )
                
                # Extract and store user preferences/facts
                await self._extract_user_facts(user_id, recent_turns)
                
                # Update recent topics
                await self._update_topics_from_turns(user_id, recent_turns)
        
        except Exception as e:
            print(f"Error in background conversation processing: {e}")
    
    # === UTILITY METHODS ===
    
    async def clear_session_memory(self, user_id: str, chat_id: str) -> None:
        """Clear Redis session memory for a chat."""
        keys = [
            f"session:{user_id}:{chat_id}:recent",
            f"session:{user_id}:{chat_id}:scratch"
        ]
        
        for key in keys:
            await delete_cache(key)
    
    async def _store_chat_message_db(
        self, 
        user_id: str, 
        chat_id: str, 
        turn: ChatTurn
    ) -> None:
        """Store raw chat message in PostgreSQL."""
        if not self.supabase_client:
            return
        
        try:
            data = {
                "user_id": user_id,
                "chat_id": chat_id,
                "message_type": turn.message_type,
                "content": turn.content,
                "metadata": turn.metadata
            }
            
            self.supabase_client.table("chat_messages").insert(data).execute()
            
        except Exception as e:
            print(f"Error storing chat message: {e}")
    
    async def _summarize_with_llm(self, conversation_text: str) -> Optional[str]:
        """Summarize conversation using LLM (implement based on your LLM service)."""
        # This would integrate with your existing LLM service
        # For now, return a simple truncation
        if len(conversation_text) > 500:
            return conversation_text[:500] + "..."
        return conversation_text
    
    async def _extract_user_facts(self, user_id: str, turns: List[ChatTurn]) -> None:
        """Extract user facts and preferences from conversation turns."""
        # This would use your LLM to extract facts like:
        # "User likes dream pop", "User dislikes shouty vocals", etc.
        # For now, this is a placeholder
        pass
    
    async def _update_topics_from_turns(self, user_id: str, turns: List[ChatTurn]) -> None:
        """Extract and update topics from conversation turns."""
        # This would extract topics mentioned in the conversation
        # For now, this is a placeholder
        pass
    
    # === USER PREFERENCES MANAGEMENT ===
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences from database."""
        if not self.supabase_client:
            return None
        
        try:
            result = self.supabase_client.table("user_prefs").select("*").eq("user_id", user_id).execute()
            
            if result.data:
                row = result.data[0]
                return UserPreferences(
                    user_id=row["user_id"],
                    top_genres=row.get("top_genres", []),
                    top_moods=row.get("top_moods", []),
                    artist_affinities=row.get("artist_affinities", {}),
                    depth_weight=row.get("depth_weight", 0.5),
                    novelty_weight=row.get("novelty_weight", 0.5),
                    created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else None
                )
            
            return None
            
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return None
    
    async def update_user_preferences(
        self, 
        user_id: str,
        top_genres: Optional[List[str]] = None,
        top_moods: Optional[List[str]] = None,
        artist_affinities: Optional[Dict[str, float]] = None,
        depth_weight: Optional[float] = None,
        novelty_weight: Optional[float] = None
    ) -> bool:
        """Update user preferences using database function."""
        if not self.supabase_client:
            return False
        
        try:
            result = self.supabase_client.rpc(
                "update_user_preferences",
                {
                    "target_user_id": user_id,
                    "new_top_genres": top_genres,
                    "new_top_moods": top_moods,
                    "new_artist_affinities": artist_affinities,
                    "new_depth_weight": depth_weight,
                    "new_novelty_weight": novelty_weight
                }
            ).execute()
            
            # Invalidate cache
            await self._invalidate_user_prefs_cache(user_id)
            
            return True
            
        except Exception as e:
            print(f"Error updating user preferences: {e}")
            return False
    
    async def get_user_taste_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user taste profile."""
        if not self.supabase_client:
            return {}
        
        try:
            result = self.supabase_client.rpc(
                "get_user_taste_profile",
                {"target_user_id": user_id}
            ).execute()
            
            if result.data:
                profile = result.data[0]
                return {
                    "genres": profile.get("genres", []),
                    "moods": profile.get("moods", []),
                    "artist_affinities": profile.get("artist_affinities", {}),
                    "depth_weight": profile.get("depth_weight", 0.5),
                    "novelty_weight": profile.get("novelty_weight", 0.5),
                    "recent_tracks": profile.get("recent_tracks", []),
                    "blocked_tracks": profile.get("blocked_tracks", [])
                }
            
            return {}
            
        except Exception as e:
            print(f"Error getting taste profile: {e}")
            return {}
    
    # === RECOMMENDATION TRACKING ===
    
    async def record_recommendation(
        self,
        user_id: str,
        item_id: str,
        item_type: str = "track",
        reason: Optional[str] = None,
        score: Optional[float] = None
    ) -> bool:
        """Record a recommendation event for deduplication."""
        if not self.supabase_client:
            return False
        
        try:
            result = self.supabase_client.rpc(
                "record_recommendation_event",
                {
                    "target_user_id": user_id,
                    "target_item_id": item_id,
                    "target_item_type": item_type,
                    "rec_reason": reason,
                    "rec_score": score
                }
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"Error recording recommendation: {e}")
            return False
    
    async def get_recommendations_with_dedup(
        self,
        user_id: str,
        exclude_days: int = 21,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recommendations while avoiding recently recommended items."""
        if not self.supabase_client:
            return []
        
        try:
            result = self.supabase_client.rpc(
                "get_recommendations_with_dedup",
                {
                    "target_user_id": user_id,
                    "exclude_days": exclude_days,
                    "max_results": max_results
                }
            ).execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"Error getting deduplicated recommendations: {e}")
            return []
    
    async def get_recent_recommendations(
        self,
        user_id: str,
        days: int = 7,
        limit: int = 20
    ) -> List[RecommendationEvent]:
        """Get recent recommendation events for a user."""
        if not self.supabase_client:
            return []
        
        try:
            result = self.supabase_client.table("rec_events").select("*").eq(
                "user_id", user_id
            ).gte(
                "created_at", (datetime.now() - timedelta(days=days)).isoformat()
            ).order("created_at", desc=True).limit(limit).execute()
            
            events = []
            for row in result.data:
                events.append(RecommendationEvent(
                    user_id=row["user_id"],
                    item_id=row["item_id"],
                    item_type=row["item_type"],
                    reason=row.get("reason"),
                    score=row.get("score"),
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
            
            return events
            
        except Exception as e:
            print(f"Error getting recent recommendations: {e}")
            return []
    
    # === CACHING UTILITIES ===
    
    async def _invalidate_user_prefs_cache(self, user_id: str) -> None:
        """Invalidate cached user preferences."""
        cache_key = f"user:{user_id}:prefs"
        await delete_cache(cache_key)
    
    async def get_cached_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences with caching."""
        cache_key = f"user:{user_id}:prefs"
        
        try:
            # Try cache first
            cached_prefs = await get_cache(cache_key)
            if cached_prefs:
                return UserPreferences(
                    user_id=cached_prefs["user_id"],
                    top_genres=cached_prefs.get("top_genres", []),
                    top_moods=cached_prefs.get("top_moods", []),
                    artist_affinities=cached_prefs.get("artist_affinities", {}),
                    depth_weight=cached_prefs.get("depth_weight", 0.5),
                    novelty_weight=cached_prefs.get("novelty_weight", 0.5)
                )
            
            # Get from database
            prefs = await self.get_user_preferences(user_id)
            if prefs:
                # Cache for 1 hour
                cache_data = {
                    "user_id": prefs.user_id,
                    "top_genres": prefs.top_genres or [],
                    "top_moods": prefs.top_moods or [],
                    "artist_affinities": prefs.artist_affinities or {},
                    "depth_weight": prefs.depth_weight,
                    "novelty_weight": prefs.novelty_weight
                }
                await set_cache(cache_key, cache_data, expire_seconds=3600)
            
            return prefs
            
        except Exception as e:
            print(f"Error getting cached preferences: {e}")
            return None


# Global instance
memory_service = MemoryService()