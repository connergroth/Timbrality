#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Matching Service

This service provides intelligent matching between music data from different platforms
(Last.fm, Spotify, AOTY) using fuzzy matching techniques. It helps resolve inconsistencies
in song titles, artist names, and album names across different music services.

Features:
- Fuzzy matching for artists, albums, and tracks
- AOTY integration with CloudScraper
- Caching to avoid repeated lookups
- Batch processing for efficiency
- Confidence scoring and match validation
"""

import asyncio
import time
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from ..utils.fuzzy_matcher import (
        MusicFuzzyMatcher, MatchResult, match_artist, match_album, 
        match_track, find_best_album_match, find_best_track_matches
    )
    from ..scraper.utils.scraper import get_album_url, scrape_album
    from ..scraper.models import Album, Track
except ImportError:
    # Fallback for direct execution
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, backend_dir)
    
    from utils.fuzzy_matcher import (
        MusicFuzzyMatcher, MatchResult, match_artist, match_album, 
        match_track, find_best_album_match, find_best_track_matches
    )
    from scraper.utils.scraper import get_album_url, scrape_album
    from scraper.models import Album, Track


@dataclass
class MusicMatchResult:
    """Result of matching music data between platforms"""
    source_platform: str
    target_platform: str
    match_type: str  # "exact", "fuzzy", "not_found"
    confidence: str  # "high", "medium", "low"
    score: float
    source_data: Dict[str, Any]
    target_data: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    cached: bool = False


@dataclass
class LastfmTrack:
    """Representation of a Last.fm track for matching"""
    name: str
    artist: str
    album: str
    playcount: int = 0
    loved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlbumMatchCandidate:
    """Candidate album from AOTY for matching"""
    artist: str
    title: str
    url: str
    year: Optional[int] = None
    score: Optional[float] = None
    cover_image: Optional[str] = None


class MusicMatchingService:
    """Service for matching music data across platforms using fuzzy matching"""
    
    def __init__(self, cache_dir: str = "music_matching_cache"):
        self.fuzzy_matcher = MusicFuzzyMatcher()
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "music_matches.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cache
        self.match_cache = self.load_cache()
        
        # Performance tracking
        self.stats = {
            "total_matches": 0,
            "cache_hits": 0,
            "aoty_requests": 0,
            "successful_matches": 0,
            "failed_matches": 0
        }
    
    def load_cache(self) -> Dict[str, Any]:
        """Load match cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}")
        return {}
    
    def save_cache(self):
        """Save match cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.match_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CACHE] Failed to save cache: {e}")
    
    def get_cache_key(self, platform: str, artist: str, album: str = "", track: str = "") -> str:
        """Generate cache key for a match query"""
        key_parts = [platform, artist.lower(), album.lower(), track.lower()]
        return "|".join(key_parts)
    
    async def find_aoty_album(self, artist: str, album: str) -> Optional[Album]:
        """Find and scrape album data from AOTY"""
        cache_key = self.get_cache_key("aoty", artist, album)
        
        # Check cache first
        if cache_key in self.match_cache:
            cached_data = self.match_cache[cache_key]
            if cached_data and cached_data.get("target_data"):
                print(f"[CACHE] Using cached AOTY data for {artist} - {album}")
                self.stats["cache_hits"] += 1
                return Album(**cached_data["target_data"])
        
        # Search AOTY
        try:
            self.stats["aoty_requests"] += 1
            print(f"[AOTY] Searching for: {artist} - {album}")
            
            url_result = await get_album_url(artist, album)
            if not url_result:
                print(f"[AOTY] No results found for {artist} - {album}")
                # Cache negative result
                self.match_cache[cache_key] = {
                    "match_type": "not_found",
                    "target_data": None,
                    "timestamp": datetime.now().isoformat()
                }
                self.save_cache()
                return None
            
            url, found_artist, found_title = url_result
            print(f"[AOTY] Found: {found_artist} - {found_title}")
            
            # Scrape full album data
            album_data = await scrape_album(url, found_artist, found_title)
            
            # Cache successful result
            self.match_cache[cache_key] = {
                "match_type": "found",
                "target_data": album_data.model_dump(),
                "timestamp": datetime.now().isoformat()
            }
            self.save_cache()
            
            return album_data
            
        except Exception as e:
            print(f"[AOTY] Error searching for {artist} - {album}: {str(e)}")
            return None
    
    async def match_lastfm_track_to_aoty(self, lastfm_track: LastfmTrack) -> MusicMatchResult:
        """Match a Last.fm track to AOTY data"""
        start_time = time.time()
        self.stats["total_matches"] += 1
        
        source_data = lastfm_track.to_dict()
        
        # Try to find the album on AOTY
        aoty_album = await self.find_aoty_album(lastfm_track.artist, lastfm_track.album)
        
        if not aoty_album:
            self.stats["failed_matches"] += 1
            return MusicMatchResult(
                source_platform="lastfm",
                target_platform="aoty",
                match_type="not_found",
                confidence="low",
                score=0.0,
                source_data=source_data,
                target_data=None,
                processing_time=time.time() - start_time
            )
        
        # Now try to match the specific track
        aoty_track_names = [track.title for track in aoty_album.tracks]
        
        if not aoty_track_names:
            # Album found but no tracks
            self.stats["failed_matches"] += 1
            return MusicMatchResult(
                source_platform="lastfm",
                target_platform="aoty",
                match_type="album_only",
                confidence="medium",
                score=0.5,
                source_data=source_data,
                target_data={
                    "album": aoty_album.model_dump(),
                    "matched_track": None
                },
                processing_time=time.time() - start_time
            )
        
        # Find best track match using fuzzy matching
        track_match = None
        best_score = 0.0
        matched_track = None
        
        for i, aoty_track_name in enumerate(aoty_track_names):
            match_result = match_track(lastfm_track.name, aoty_track_name)
            if match_result and match_result.score > best_score:
                best_score = match_result.score
                track_match = match_result
                matched_track = aoty_album.tracks[i]
        
        if track_match and best_score >= 0.75:
            # Successful track match
            self.stats["successful_matches"] += 1
            return MusicMatchResult(
                source_platform="lastfm",
                target_platform="aoty",
                match_type="track_match",
                confidence=track_match.confidence,
                score=best_score,
                source_data=source_data,
                target_data={
                    "album": aoty_album.model_dump(),
                    "matched_track": matched_track.model_dump(),
                    "match_details": {
                        "normalized_query": track_match.normalized_query,
                        "normalized_target": track_match.normalized_target,
                        "match_type": track_match.match_type
                    }
                },
                processing_time=time.time() - start_time
            )
        else:
            # Album found but track not matched well enough
            self.stats["failed_matches"] += 1
            return MusicMatchResult(
                source_platform="lastfm",
                target_platform="aoty",
                match_type="album_only",
                confidence="low",
                score=0.3,
                source_data=source_data,
                target_data={
                    "album": aoty_album.model_dump(),
                    "matched_track": None,
                    "best_track_match_score": best_score
                },
                processing_time=time.time() - start_time
            )
    
    async def batch_match_lastfm_tracks(self, lastfm_tracks: List[LastfmTrack], 
                                      max_concurrent: int = 3,
                                      progress_callback: Optional[callable] = None) -> List[MusicMatchResult]:
        """Batch process multiple Last.fm tracks for AOTY matching"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_track(track: LastfmTrack, index: int) -> MusicMatchResult:
            async with semaphore:
                try:
                    result = await self.match_lastfm_track_to_aoty(track)
                    if progress_callback:
                        progress_callback(index + 1, len(lastfm_tracks), track.name, result.match_type)
                    return result
                except Exception as e:
                    print(f"[ERROR] Failed to process track {track.artist} - {track.name}: {e}")
                    return MusicMatchResult(
                        source_platform="lastfm",
                        target_platform="aoty",
                        match_type="error",
                        confidence="low",
                        score=0.0,
                        source_data=track.to_dict(),
                        target_data={"error": str(e)},
                        processing_time=0.0
                    )
        
        print(f"[BATCH] Processing {len(lastfm_tracks)} tracks with max {max_concurrent} concurrent requests")
        
        tasks = [process_single_track(track, i) for i, track in enumerate(lastfm_tracks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                track = lastfm_tracks[i]
                processed_results.append(MusicMatchResult(
                    source_platform="lastfm",
                    target_platform="aoty",
                    match_type="error",
                    confidence="low",
                    score=0.0,
                    source_data=track.to_dict(),
                    target_data={"error": str(result)},
                    processing_time=0.0
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """Get matching statistics and performance metrics"""
        total = self.stats["total_matches"]
        if total == 0:
            success_rate = 0.0
            cache_hit_rate = 0.0
        else:
            success_rate = (self.stats["successful_matches"] / total) * 100
            cache_hit_rate = (self.stats["cache_hits"] / total) * 100
        
        return {
            "total_matches_attempted": total,
            "successful_matches": self.stats["successful_matches"],
            "failed_matches": self.stats["failed_matches"],
            "success_rate_percent": round(success_rate, 2),
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "aoty_requests_made": self.stats["aoty_requests"],
            "cache_entries": len(self.match_cache)
        }
    
    def clear_cache(self):
        """Clear the match cache"""
        self.match_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("[CACHE] Cache cleared")
    
    def export_results(self, results: List[MusicMatchResult], output_file: str):
        """Export match results to JSON file"""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "statistics": self.get_match_statistics(),
                "results": [asdict(result) for result in results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"[EXPORT] Results exported to {output_file}")
        except Exception as e:
            print(f"[EXPORT] Failed to export results: {e}")


# Convenience functions
async def match_lastfm_to_aoty(lastfm_tracks: List[Dict[str, Any]]) -> List[MusicMatchResult]:
    """Convenience function to match Last.fm data to AOTY"""
    service = MusicMatchingService()
    
    # Convert dictionaries to LastfmTrack objects
    tracks = []
    for track_data in lastfm_tracks:
        track = LastfmTrack(
            name=track_data.get('name', ''),
            artist=track_data.get('artist', ''),
            album=track_data.get('album', ''),
            playcount=track_data.get('playcount', 0),
            loved=track_data.get('loved', False)
        )
        tracks.append(track)
    
    def progress_callback(current, total, track_name, match_type):
        print(f"[PROGRESS] {current}/{total}: {track_name} -> {match_type}")
    
    return await service.batch_match_lastfm_tracks(tracks, progress_callback=progress_callback)


if __name__ == "__main__":
    async def test_music_matching_service():
        print("Music Matching Service Test")
        print("=" * 60)
        
        # Create test service
        service = MusicMatchingService()
        
        # Test single track matching
        print("\n[TEST] Single track matching...")
        test_track = LastfmTrack(
            name="Paranoid Android",
            artist="Radiohead",
            album="OK Computer",
            playcount=42,
            loved=True
        )
        
        result = await service.match_lastfm_track_to_aoty(test_track)
        print(f"Match result: {result.match_type} (confidence: {result.confidence}, score: {result.score:.3f})")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.target_data and result.target_data.get("matched_track"):
            matched = result.target_data["matched_track"]
            print(f"Matched AOTY track: {matched['title']} (#{matched['number']})")
        
        # Test batch processing
        print("\n[TEST] Batch processing...")
        test_tracks = [
            LastfmTrack("Karma Police", "Radiohead", "OK Computer", 30, False),
            LastfmTrack("No Surprises", "Radiohead", "OK Computer", 25, True),
            LastfmTrack("Let Down", "Radiohead", "OK Computer", 20, False),
        ]
        
        def simple_progress(current, total, track, match_type):
            print(f"  [{current}/{total}] {track} -> {match_type}")
        
        batch_results = await service.batch_match_lastfm_tracks(test_tracks, progress_callback=simple_progress)
        
        print(f"\nBatch results: {len(batch_results)} processed")
        successful = sum(1 for r in batch_results if r.match_type == "track_match")
        print(f"Successful matches: {successful}/{len(batch_results)}")
        
        # Show statistics
        print("\n[STATISTICS]")
        stats = service.get_match_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n[SUCCESS] Music matching service test completed!")
    
    asyncio.run(test_music_matching_service())