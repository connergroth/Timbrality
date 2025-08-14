"""
Background Memory Processing Service

Handles async processing of conversations into long-term memories,
fact extraction, and topic analysis.
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

from services.memory_service import memory_service, MemoryEntry, ChatTurn
from agent.embedding.engine import get_embedding


@dataclass
class ProcessingJob:
    """Background processing job."""
    job_id: str
    user_id: str
    chat_id: str
    job_type: str  # summarize, extract_facts, update_topics
    parameters: Dict[str, Any]
    created_at: datetime
    priority: int = 1  # 1-5, higher is more urgent


class MemoryProcessor:
    """
    Background processor for converting conversations into structured memories.
    """
    
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        self.worker_tasks = []
    
    async def start_processing(self, num_workers: int = 2):
        """Start background processing workers."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_tasks = []
        
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        print(f"Started {num_workers} memory processing workers")
    
    async def stop_processing(self):
        """Stop background processing workers."""
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks = []
        
        print("Stopped memory processing workers")
    
    async def queue_conversation_summary(
        self,
        user_id: str,
        chat_id: str,
        window_size: int = 20,
        priority: int = 2
    ):
        """Queue a conversation summarization job."""
        job = ProcessingJob(
            job_id=f"summary_{user_id}_{chat_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            chat_id=chat_id,
            job_type="summarize",
            parameters={"window_size": window_size},
            created_at=datetime.now(),
            priority=priority
        )
        
        await self.processing_queue.put(job)
    
    async def queue_fact_extraction(
        self,
        user_id: str,
        chat_id: str,
        turn_count: int = 10,
        priority: int = 3
    ):
        """Queue a fact extraction job."""
        job = ProcessingJob(
            job_id=f"facts_{user_id}_{chat_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            chat_id=chat_id,
            job_type="extract_facts",
            parameters={"turn_count": turn_count},
            created_at=datetime.now(),
            priority=priority
        )
        
        await self.processing_queue.put(job)
    
    async def queue_topic_analysis(
        self,
        user_id: str,
        chat_id: str,
        turn_count: int = 10,
        priority: int = 1
    ):
        """Queue a topic analysis job."""
        job = ProcessingJob(
            job_id=f"topics_{user_id}_{chat_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            chat_id=chat_id,
            job_type="update_topics",
            parameters={"turn_count": turn_count},
            created_at=datetime.now(),
            priority=priority
        )
        
        await self.processing_queue.put(job)
    
    async def _worker(self, worker_name: str):
        """Background worker for processing memory jobs."""
        print(f"Memory processor {worker_name} started")
        
        while self.is_running:
            try:
                # Get job from queue (wait up to 5 seconds)
                job = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=5.0
                )
                
                print(f"{worker_name} processing job: {job.job_id}")
                
                # Process the job based on type
                if job.job_type == "summarize":
                    await self._process_summarization(job)
                elif job.job_type == "extract_facts":
                    await self._process_fact_extraction(job)
                elif job.job_type == "update_topics":
                    await self._process_topic_analysis(job)
                elif job.job_type == "learn_preferences":
                    await self._process_preference_learning(job)
                
                # Mark job as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No jobs in queue, continue
                continue
            except Exception as e:
                print(f"Error in {worker_name}: {e}")
                continue
        
        print(f"Memory processor {worker_name} stopped")
    
    async def _process_summarization(self, job: ProcessingJob):
        """Process conversation summarization job."""
        try:
            window_size = job.parameters.get("window_size", 20)
            
            # Get recent conversation turns
            recent_turns = await memory_service.get_recent_turns(
                job.user_id, 
                job.chat_id, 
                limit=window_size
            )
            
            if len(recent_turns) < 5:  # Need minimum turns for summary
                return
            
            # Create conversation text
            conversation_text = self._format_turns_for_summary(recent_turns)
            
            # Generate summary using LLM
            summary = await self._generate_summary_llm(conversation_text)
            
            if summary:
                # Store summary as memory
                summary_memory = MemoryEntry(
                    user_id=job.user_id,
                    chat_id=job.chat_id,
                    kind="summary",
                    content=summary,
                    importance=3  # Medium importance
                )
                
                await memory_service.store_memory(summary_memory)
                print(f"Stored conversation summary for {job.user_id}/{job.chat_id}")
        
        except Exception as e:
            print(f"Error processing summarization job: {e}")
    
    async def _process_fact_extraction(self, job: ProcessingJob):
        """Process fact extraction job."""
        try:
            turn_count = job.parameters.get("turn_count", 10)
            
            # Get recent turns
            recent_turns = await memory_service.get_recent_turns(
                job.user_id, 
                job.chat_id, 
                limit=turn_count
            )
            
            if not recent_turns:
                return
            
            # Extract facts from user messages
            user_turns = [
                turn for turn in recent_turns 
                if turn.message_type == "user"
            ]
            
            facts = await self._extract_facts_llm(user_turns)
            
            # Store extracted facts
            for fact in facts:
                fact_memory = MemoryEntry(
                    user_id=job.user_id,
                    chat_id=job.chat_id,
                    kind="fact",
                    content=fact["content"],
                    importance=fact.get("importance", 2)
                )
                
                await memory_service.store_memory(fact_memory)
            
            if facts:
                print(f"Extracted {len(facts)} facts for {job.user_id}")
        
        except Exception as e:
            print(f"Error processing fact extraction job: {e}")
    
    async def _process_topic_analysis(self, job: ProcessingJob):
        """Process topic analysis job."""
        try:
            turn_count = job.parameters.get("turn_count", 10)
            
            # Get recent turns
            recent_turns = await memory_service.get_recent_turns(
                job.user_id, 
                job.chat_id, 
                limit=turn_count
            )
            
            if not recent_turns:
                return
            
            # Extract topics from conversation
            topics = await self._extract_topics_llm(recent_turns)
            
            # Update recent topics in Redis
            for topic, score in topics:
                await memory_service.update_recent_topics(
                    job.user_id, 
                    topic, 
                    score
                )
            
            if topics:
                print(f"Updated {len(topics)} topics for {job.user_id}")
        
        except Exception as e:
            print(f"Error processing topic analysis job: {e}")
    
    async def _process_preference_learning(self, job: ProcessingJob):
        """Process preference learning from user interactions."""
        try:
            # Get recent recommendation events
            recent_recs = await memory_service.get_recent_recommendations(
                job.user_id, days=7, limit=50
            )
            
            if not recent_recs:
                return
            
            # Get current preferences
            current_prefs = await memory_service.get_user_preferences(job.user_id)
            
            # Simple preference learning: extract genres/moods from recent interactions
            genre_freq = {}
            mood_freq = {}
            
            # This would ideally integrate with your track database to get genre/mood info
            # For now, just demonstrate the structure
            
            # Update preferences if we have enough data
            if len(recent_recs) >= 5:
                # Extract top genres and moods
                top_genres = sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                top_moods = sorted(mood_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                await memory_service.update_user_preferences(
                    user_id=job.user_id,
                    top_genres=[g[0] for g in top_genres],
                    top_moods=[m[0] for m in top_moods]
                )
                
                print(f"Updated preferences for {job.user_id} based on {len(recent_recs)} interactions")
        
        except Exception as e:
            print(f"Error processing preference learning job: {e}")
    
    async def queue_preference_learning(
        self,
        user_id: str,
        priority: int = 1
    ):
        """Queue a preference learning job."""
        job = ProcessingJob(
            job_id=f"prefs_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            chat_id="",  # Not chat-specific
            job_type="learn_preferences",
            parameters={},
            created_at=datetime.now(),
            priority=priority
        )
        
        await self.processing_queue.put(job)
    
    def _format_turns_for_summary(self, turns: List[ChatTurn]) -> str:
        """Format conversation turns for LLM summarization."""
        formatted_turns = []
        
        for turn in turns:
            role = "User" if turn.message_type == "user" else "Assistant"
            formatted_turns.append(f"{role}: {turn.content}")
        
        return "\n".join(formatted_turns)
    
    async def _generate_summary_llm(self, conversation_text: str) -> Optional[str]:
        """Generate conversation summary using LLM."""
        # This would integrate with your existing LLM service
        # For now, provide a simple implementation
        
        if len(conversation_text) < 100:
            return None
        
        # Simple extractive summary - take first and last parts
        lines = conversation_text.split("\n")
        if len(lines) > 10:
            summary_lines = lines[:3] + ["..."] + lines[-3:]
            return "\n".join(summary_lines)
        
        return conversation_text[:200] + "..." if len(conversation_text) > 200 else conversation_text
    
    async def _extract_facts_llm(self, user_turns: List[ChatTurn]) -> List[Dict[str, Any]]:
        """Extract user facts and preferences using LLM."""
        # This would use your LLM to extract structured facts
        # For now, provide a simple implementation
        
        facts = []
        
        for turn in user_turns:
            content = turn.content.lower()
            
            # Simple pattern matching for music preferences
            if "love" in content or "like" in content:
                if any(genre in content for genre in ["pop", "rock", "jazz", "classical", "hip hop", "electronic"]):
                    facts.append({
                        "content": f"User expressed preference: {turn.content[:100]}",
                        "importance": 3
                    })
            
            if "hate" in content or "dislike" in content or "don't like" in content:
                facts.append({
                    "content": f"User expressed dislike: {turn.content[:100]}",
                    "importance": 3
                })
        
        return facts
    
    async def _extract_topics_llm(self, turns: List[ChatTurn]) -> List[tuple[str, float]]:
        """Extract topics from conversation using LLM."""
        # This would use your LLM or NLP service to extract topics
        # For now, provide a simple implementation
        
        topics = []
        current_time = datetime.now().timestamp()
        
        # Simple keyword extraction
        music_keywords = [
            "dream pop", "shoegaze", "ambient", "electronic", "indie rock",
            "jazz", "classical", "hip hop", "r&b", "soul", "funk",
            "playlist", "album", "artist", "song", "track", "music"
        ]
        
        all_text = " ".join(turn.content.lower() for turn in turns)
        
        for keyword in music_keywords:
            if keyword in all_text:
                # Score based on frequency and recency
                frequency = all_text.count(keyword)
                score = current_time - (frequency * 0.1)  # Higher frequency = higher score
                topics.append((keyword, score))
        
        # Sort by score and return top topics
        topics.sort(key=lambda x: x[1], reverse=True)
        return topics[:5]


class RecommendationDiversifier:
    """Handles recommendation deduplication and diversity using MMR algorithm."""
    
    @staticmethod
    def calculate_similarity(track1: Dict[str, Any], track2: Dict[str, Any]) -> float:
        """Calculate similarity between two tracks."""
        similarity_score = 0.0
        
        # Artist similarity (highest weight)
        if track1.get("artist") == track2.get("artist"):
            similarity_score += 0.4
        
        # Genre similarity
        genres1 = set(track1.get("genres", []))
        genres2 = set(track2.get("genres", []))
        if genres1 and genres2:
            genre_overlap = len(genres1.intersection(genres2)) / len(genres1.union(genres2))
            similarity_score += genre_overlap * 0.3
        
        # Mood similarity
        moods1 = set(track1.get("moods", []))
        moods2 = set(track2.get("moods", []))
        if moods1 and moods2:
            mood_overlap = len(moods1.intersection(moods2)) / len(moods1.union(moods2))
            similarity_score += mood_overlap * 0.2
        
        # Audio feature similarity (if available)
        if "pred_energy" in track1 and "pred_energy" in track2:
            energy_diff = abs(track1["pred_energy"] - track2["pred_energy"])
            similarity_score += max(0, (1 - energy_diff)) * 0.1
        
        return min(1.0, similarity_score)
    
    @staticmethod
    def mmr_diversify(
        candidates: List[Dict[str, Any]],
        diversity_lambda: float = 0.3,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance for diversity."""
        if not candidates:
            return []
        
        if len(candidates) <= max_results:
            return candidates
        
        # Start with highest scored item
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        while len(selected) < max_results and remaining:
            best_mmr_score = -1
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Original relevance score
                relevance = candidate.get("recommendation_score", 0.0)
                
                # Calculate max similarity to already selected items
                max_similarity = 0.0
                for selected_item in selected:
                    similarity = RecommendationDiversifier.calculate_similarity(
                        candidate, selected_item
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR formula: relevance - lambda * max_similarity
                mmr_score = relevance - diversity_lambda * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    @staticmethod
    def apply_artist_diversity(
        tracks: List[Dict[str, Any]],
        max_per_artist: int = 2
    ) -> List[Dict[str, Any]]:
        """Limit tracks per artist for diversity."""
        artist_counts = {}
        filtered_tracks = []
        
        for track in tracks:
            artist = track.get("artist", "Unknown")
            current_count = artist_counts.get(artist, 0)
            
            if current_count < max_per_artist:
                filtered_tracks.append(track)
                artist_counts[artist] = current_count + 1
        
        return filtered_tracks
    
    @staticmethod
    async def process_recommendations(
        user_id: str,
        raw_recommendations: List[Dict[str, Any]],
        diversity_lambda: float = 0.3,
        max_results: int = 10,
        max_per_artist: int = 2,
        exclude_days: int = 21
    ) -> List[Dict[str, Any]]:
        """Complete recommendation processing pipeline."""
        
        # Step 1: Get user taste profile for personalized weights
        taste_profile = await memory_service.get_user_taste_profile(user_id)
        
        # Step 2: Apply seen penalty using database deduplication
        deduped_recs = await memory_service.get_recommendations_with_dedup(
            user_id=user_id,
            exclude_days=exclude_days,
            max_results=len(raw_recommendations) * 2  # Get more for filtering
        )
        
        # Merge with raw recommendations
        rec_lookup = {rec["track_id"]: rec for rec in deduped_recs}
        enhanced_recs = []
        
        for raw_rec in raw_recommendations:
            track_id = raw_rec.get("id") or raw_rec.get("track_id")
            if track_id in rec_lookup:
                db_rec = rec_lookup[track_id]
                # Apply seen penalty
                penalty = db_rec.get("seen_penalty", 0.0)
                original_score = raw_rec.get("recommendation_score", 0.0)
                adjusted_score = original_score - (penalty * 0.3)
                
                enhanced_rec = raw_rec.copy()
                enhanced_rec["recommendation_score"] = max(0.0, adjusted_score)
                enhanced_rec["days_since_last_rec"] = db_rec.get("days_since_last_rec")
                enhanced_rec["seen_penalty"] = penalty
                enhanced_recs.append(enhanced_rec)
            else:
                # New recommendation, no penalty
                enhanced_recs.append(raw_rec)
        
        # Step 3: Sort by adjusted score
        enhanced_recs.sort(key=lambda x: x.get("recommendation_score", 0.0), reverse=True)
        
        # Step 4: Apply artist diversity filter
        artist_filtered = RecommendationDiversifier.apply_artist_diversity(
            enhanced_recs, max_per_artist
        )
        
        # Step 5: Apply MMR for content diversity
        final_recs = RecommendationDiversifier.mmr_diversify(
            artist_filtered, diversity_lambda, max_results
        )
        
        # Step 6: Record recommendations for future deduplication
        for rec in final_recs:
            await memory_service.record_recommendation(
                user_id=user_id,
                item_id=rec.get("id") or rec.get("track_id"),
                item_type="track",
                reason=rec.get("reason", "processed_recommendation"),
                score=rec.get("recommendation_score")
            )
        
        return final_recs


# Global processor instance
memory_processor = MemoryProcessor()


# Convenience functions for triggering background processing
async def trigger_conversation_processing(user_id: str, chat_id: str):
    """Trigger background processing for a conversation."""
    # Queue different types of processing
    await memory_processor.queue_conversation_summary(user_id, chat_id)
    await memory_processor.queue_fact_extraction(user_id, chat_id)
    await memory_processor.queue_topic_analysis(user_id, chat_id)


async def process_and_diversify_recommendations(
    user_id: str,
    raw_recommendations: List[Dict[str, Any]],
    diversity_lambda: float = 0.3,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """Process raw recommendations with deduplication and diversification."""
    return await RecommendationDiversifier.process_recommendations(
        user_id=user_id,
        raw_recommendations=raw_recommendations,
        diversity_lambda=diversity_lambda,
        max_results=max_results
    )


async def start_memory_processing():
    """Start the memory processing service."""
    await memory_processor.start_processing(num_workers=2)


async def stop_memory_processing():
    """Stop the memory processing service."""
    await memory_processor.stop_processing()