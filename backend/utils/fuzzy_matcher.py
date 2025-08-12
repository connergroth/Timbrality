#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy Matching Utilities for Music Metadata

This module provides sophisticated fuzzy matching for artist names, album titles,
and track titles to handle inconsistencies between different music platforms
like Last.fm, Spotify, AOTY, etc.

Common variations handled:
- "Remastered", "Deluxe Edition", "Expanded", etc.
- Feature credits: "feat.", "ft.", "featuring"
- Punctuation differences
- Case sensitivity
- Extra whitespace
- Release year suffixes
- Different bracket styles
"""

import re
import string
from typing import List, Tuple, Optional, Dict, Set
from difflib import SequenceMatcher
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of a fuzzy match operation"""
    score: float
    normalized_query: str
    normalized_target: str
    match_type: str
    confidence: str  # "high", "medium", "low"


class MusicFuzzyMatcher:
    """Advanced fuzzy matcher for music metadata"""
    
    def __init__(self):
        # Common variations to normalize
        self.remix_variations = [
            r'\s*-\s*remix\b', r'\s*\(remix\)', r'\s*remix\b',
            r'\s*-\s*radio edit\b', r'\s*\(radio edit\)', 
            r'\s*-\s*extended\b', r'\s*\(extended\)',
            r'\s*-\s*instrumental\b', r'\s*\(instrumental\)',
            r'\s*-\s*acoustic\b', r'\s*\(acoustic\)',
            r'\s*-\s*live\b', r'\s*\(live\)', r'\s*live version\b'
        ]
        
        self.remaster_variations = [
            r'\s*-\s*remaster(?:ed)?\b', r'\s*\(remaster(?:ed)?\)',
            r'\s*remaster(?:ed)?\b', r'\s*-\s*\d{4}\s*remaster(?:ed)?\b',
            r'\s*\(\d{4}\s*remaster(?:ed)?\)', r'\s*\d{4}\s*remaster(?:ed)?\b'
        ]
        
        self.edition_variations = [
            r'\s*-\s*deluxe edition\b', r'\s*\(deluxe edition\)',
            r'\s*deluxe edition\b', r'\s*-\s*deluxe\b', r'\s*\(deluxe\)',
            r'\s*-\s*expanded edition\b', r'\s*\(expanded edition\)',
            r'\s*expanded edition\b', r'\s*-\s*special edition\b',
            r'\s*\(special edition\)', r'\s*special edition\b',
            r'\s*-\s*collector\'?s edition\b', r'\s*\(collector\'?s edition\)',
            r'\s*-\s*anniversary edition\b', r'\s*\(anniversary edition\)',
            r'\s*-\s*bonus track version\b', r'\s*\(bonus track version\)'
        ]
        
        self.feature_variations = [
            r'\s*\(feat\.?\s+[^)]+\)', r'\s*\(ft\.?\s+[^)]+\)',
            r'\s*\(featuring\s+[^)]+\)', r'\s*feat\.?\s+.*$',
            r'\s*ft\.?\s+.*$', r'\s*featuring\s+.*$',
            r'\s*\[feat\.?\s+[^\]]+\]', r'\s*\[ft\.?\s+[^\]]+\]',
            r'\s*\[featuring\s+[^\]]+\]', r'\s*with\s+.*$'
        ]
        
        self.year_suffix_pattern = r'\s*\(\d{4}\)$'
        
        # Common article words to handle specially
        self.articles = {'the', 'a', 'an', 'la', 'le', 'les', 'el', 'los', 'las'}
        
        # Common punctuation to normalize
        self.punct_mapping = str.maketrans(
            '''""–—…''',
            '''""-..'''
        )
    
    def normalize_text(self, text: str, is_track: bool = False) -> str:
        """Normalize text for matching"""
        if not text:
            return ""
        
        # Apply punctuation mapping
        text = text.translate(self.punct_mapping)
        
        # Remove variations (more aggressive for tracks)
        variations_to_remove = []
        
        if is_track:
            variations_to_remove.extend(self.remix_variations)
            variations_to_remove.extend(self.feature_variations)
        
        variations_to_remove.extend(self.remaster_variations)
        variations_to_remove.extend(self.edition_variations)
        
        for pattern in variations_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove year suffixes
        text = re.sub(self.year_suffix_pattern, '', text)
        
        # Normalize whitespace and case
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Remove extra punctuation
        text = re.sub(r'[^\w\s\-\'&]', '', text)
        
        return text.strip()
    
    def normalize_artist(self, artist: str) -> str:
        """Normalize artist name for matching"""
        normalized = self.normalize_text(artist)
        
        # Handle "The" prefix specially
        if normalized.startswith('the '):
            # Try both with and without "the"
            return normalized[4:]  # Remove "the "
        elif not normalized.startswith('the '):
            # This will be used for comparison with "the" variants
            pass
        
        return normalized
    
    def normalize_album(self, album: str) -> str:
        """Normalize album title for matching"""
        return self.normalize_text(album, is_track=False)
    
    def normalize_track(self, track: str) -> str:
        """Normalize track title for matching"""
        return self.normalize_text(track, is_track=True)
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using multiple methods"""
        if not str1 or not str2:
            return 0.0
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Sequence matcher (longest common subsequence)
        seq_score = SequenceMatcher(None, str1, str2).ratio()
        
        # Word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            word_score = 0.0
        else:
            word_score = len(words1 & words2) / len(words1 | words2)
        
        # Character-based similarity (for typos)
        char_score = 0.0
        if len(str1) > 3 and len(str2) > 3:  # Only for longer strings
            # Calculate character-level similarity
            char_score = sum(c1 == c2 for c1, c2 in zip(str1, str2)) / max(len(str1), len(str2))
        
        # Weighted combination
        combined_score = (
            seq_score * 0.5 +      # Sequence similarity (primary)
            word_score * 0.35 +    # Word overlap
            char_score * 0.15      # Character similarity
        )
        
        return min(1.0, combined_score)
    
    def match_artist(self, query_artist: str, target_artist: str, threshold: float = 0.8) -> Optional[MatchResult]:
        """Match artist names with fuzzy logic"""
        norm_query = self.normalize_artist(query_artist)
        norm_target = self.normalize_artist(target_artist)
        
        # Try exact match first
        if norm_query == norm_target:
            return MatchResult(
                score=1.0,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="exact",
                confidence="high"
            )
        
        # Try without "The" prefix variations
        query_no_the = norm_query[4:] if norm_query.startswith('the ') else f"the {norm_query}"
        target_no_the = norm_target[4:] if norm_target.startswith('the ') else f"the {norm_target}"
        
        if norm_query == target_no_the or query_no_the == norm_target:
            return MatchResult(
                score=0.95,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="article_variation",
                confidence="high"
            )
        
        # Calculate fuzzy similarity
        score = self.calculate_similarity(norm_query, norm_target)
        
        if score >= threshold:
            confidence = "high" if score >= 0.9 else "medium" if score >= 0.8 else "low"
            return MatchResult(
                score=score,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="fuzzy",
                confidence=confidence
            )
        
        return None
    
    def match_album(self, query_album: str, target_album: str, threshold: float = 0.8) -> Optional[MatchResult]:
        """Match album titles with fuzzy logic"""
        norm_query = self.normalize_album(query_album)
        norm_target = self.normalize_album(target_album)
        
        # Exact match
        if norm_query == norm_target:
            return MatchResult(
                score=1.0,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="exact",
                confidence="high"
            )
        
        # Calculate similarity
        score = self.calculate_similarity(norm_query, norm_target)
        
        if score >= threshold:
            confidence = "high" if score >= 0.9 else "medium" if score >= 0.8 else "low"
            return MatchResult(
                score=score,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="fuzzy",
                confidence=confidence
            )
        
        return None
    
    def match_track(self, query_track: str, target_track: str, threshold: float = 0.75) -> Optional[MatchResult]:
        """Match track titles with fuzzy logic (lower threshold for tracks)"""
        norm_query = self.normalize_track(query_track)
        norm_target = self.normalize_track(target_track)
        
        # Exact match
        if norm_query == norm_target:
            return MatchResult(
                score=1.0,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="exact",
                confidence="high"
            )
        
        # Calculate similarity
        score = self.calculate_similarity(norm_query, norm_target)
        
        if score >= threshold:
            confidence = "high" if score >= 0.9 else "medium" if score >= 0.8 else "low"
            return MatchResult(
                score=score,
                normalized_query=norm_query,
                normalized_target=norm_target,
                match_type="fuzzy",
                confidence=confidence
            )
        
        return None
    
    def match_album_comprehensive(self, query_artist: str, query_album: str,
                                target_artist: str, target_album: str,
                                artist_threshold: float = 0.8,
                                album_threshold: float = 0.8) -> Optional[Dict]:
        """Match both artist and album with comprehensive scoring"""
        
        artist_match = self.match_artist(query_artist, target_artist, artist_threshold)
        album_match = self.match_album(query_album, target_album, album_threshold)
        
        if not artist_match or not album_match:
            return None
        
        # Combined score weighted more heavily toward artist match
        combined_score = (artist_match.score * 0.6) + (album_match.score * 0.4)
        
        # Determine overall confidence
        if artist_match.confidence == "high" and album_match.confidence == "high":
            overall_confidence = "high"
        elif artist_match.confidence == "low" or album_match.confidence == "low":
            overall_confidence = "low"
        else:
            overall_confidence = "medium"
        
        return {
            "combined_score": combined_score,
            "artist_match": artist_match,
            "album_match": album_match,
            "confidence": overall_confidence,
            "match_strength": "strong" if combined_score >= 0.9 else "medium" if combined_score >= 0.8 else "weak"
        }
    
    def find_best_album_match(self, query_artist: str, query_album: str,
                            candidates: List[Tuple[str, str]]) -> Optional[Tuple[int, Dict]]:
        """Find the best matching album from a list of candidates"""
        best_match = None
        best_index = -1
        best_score = 0.0
        
        for i, (target_artist, target_album) in enumerate(candidates):
            match = self.match_album_comprehensive(
                query_artist, query_album,
                target_artist, target_album
            )
            
            if match and match["combined_score"] > best_score:
                best_score = match["combined_score"]
                best_match = match
                best_index = i
        
        if best_match:
            return (best_index, best_match)
        
        return None
    
    def find_best_track_matches(self, query_tracks: List[str], target_tracks: List[str],
                              threshold: float = 0.75) -> Dict[int, Tuple[int, MatchResult]]:
        """Find best matching tracks between two track lists"""
        matches = {}
        used_targets = set()
        
        # Sort query tracks by length (longer titles often more specific)
        query_indexed = [(i, track) for i, track in enumerate(query_tracks)]
        query_indexed.sort(key=lambda x: len(x[1]), reverse=True)
        
        for query_idx, query_track in query_indexed:
            best_match = None
            best_target_idx = -1
            best_score = 0.0
            
            for target_idx, target_track in enumerate(target_tracks):
                if target_idx in used_targets:
                    continue
                
                match = self.match_track(query_track, target_track, threshold)
                if match and match.score > best_score:
                    best_score = match.score
                    best_match = match
                    best_target_idx = target_idx
            
            if best_match:
                matches[query_idx] = (best_target_idx, best_match)
                used_targets.add(best_target_idx)
        
        return matches


# Global instance for easy access
_fuzzy_matcher = MusicFuzzyMatcher()

def match_artist(query_artist: str, target_artist: str, threshold: float = 0.8) -> Optional[MatchResult]:
    """Convenience function for artist matching"""
    return _fuzzy_matcher.match_artist(query_artist, target_artist, threshold)

def match_album(query_album: str, target_album: str, threshold: float = 0.8) -> Optional[MatchResult]:
    """Convenience function for album matching"""
    return _fuzzy_matcher.match_album(query_album, target_album, threshold)

def match_track(query_track: str, target_track: str, threshold: float = 0.75) -> Optional[MatchResult]:
    """Convenience function for track matching"""
    return _fuzzy_matcher.match_track(query_track, target_track, threshold)

def find_best_album_match(query_artist: str, query_album: str, candidates: List[Tuple[str, str]]) -> Optional[Tuple[int, Dict]]:
    """Convenience function for finding best album match"""
    return _fuzzy_matcher.find_best_album_match(query_artist, query_album, candidates)

def find_best_track_matches(query_tracks: List[str], target_tracks: List[str], threshold: float = 0.75) -> Dict[int, Tuple[int, MatchResult]]:
    """Convenience function for finding best track matches"""
    return _fuzzy_matcher.find_best_track_matches(query_tracks, target_tracks, threshold)


if __name__ == "__main__":
    # Test the fuzzy matcher
    matcher = MusicFuzzyMatcher()
    
    print("Music Fuzzy Matcher Test Suite")
    print("=" * 50)
    
    # Test artist matching
    print("\n[ARTIST MATCHING TESTS]")
    artist_tests = [
        ("The Beatles", "Beatles"),
        ("Radiohead", "radiohead"),
        ("Pink Floyd", "Pink Floyd"),
        ("Led Zeppelin", "Led Zep"),  # Should not match well
        ("The Strokes", "Strokes"),
    ]
    
    for query, target in artist_tests:
        result = matcher.match_artist(query, target)
        if result:
            print(f"  [SUCCESS] '{query}' -> '{target}': {result.score:.3f} ({result.match_type}, {result.confidence})")
        else:
            print(f"  [NO MATCH] '{query}' -> '{target}': No match")
    
    # Test album matching
    print("\n[ALBUM MATCHING TESTS]")
    album_tests = [
        ("OK Computer", "OK Computer"),
        ("The Dark Side of the Moon", "Dark Side of the Moon"),
        ("Abbey Road (Remastered)", "Abbey Road"),
        ("In Rainbows - Deluxe Edition", "In Rainbows"),
        ("Nevermind (2011 Remaster)", "Nevermind"),
    ]
    
    for query, target in album_tests:
        result = matcher.match_album(query, target)
        if result:
            print(f"  [SUCCESS] '{query}' -> '{target}': {result.score:.3f} ({result.match_type}, {result.confidence})")
        else:
            print(f"  [NO MATCH] '{query}' -> '{target}': No match")
    
    # Test track matching
    print("\n[TRACK MATCHING TESTS]")
    track_tests = [
        ("Paranoid Android", "Paranoid Android"),
        ("Karma Police (feat. Someone)", "Karma Police"),
        ("Creep - Remastered", "Creep"),
        ("Everything In Its Right Place", "Everything in Its Right Place"),
        ("15 Step", "15 Step"),
        ("No Surprises (Live)", "No Surprises"),
    ]
    
    for query, target in track_tests:
        result = matcher.match_track(query, target)
        if result:
            print(f"  [SUCCESS] '{query}' -> '{target}': {result.score:.3f} ({result.match_type}, {result.confidence})")
        else:
            print(f"  [NO MATCH] '{query}' -> '{target}': No match")
    
    # Test comprehensive album matching
    print("\n[COMPREHENSIVE ALBUM MATCHING TESTS]")
    comp_tests = [
        ("Radiohead", "OK Computer (Remastered)", "Radiohead", "OK Computer"),
        ("The Beatles", "Abbey Road", "Beatles", "Abbey Road (2019 Remaster)"),
        ("Pink Floyd", "The Wall", "Pink Floyd", "The Wall (Deluxe Edition)"),
    ]
    
    for qa, qal, ta, tal in comp_tests:
        result = matcher.match_album_comprehensive(qa, qal, ta, tal)
        if result:
            print(f"  [SUCCESS] '{qa} - {qal}' -> '{ta} - {tal}':")
            print(f"    Combined: {result['combined_score']:.3f} ({result['match_strength']}, {result['confidence']})")
            print(f"    Artist: {result['artist_match'].score:.3f}, Album: {result['album_match'].score:.3f}")
        else:
            print(f"  [NO MATCH] '{qa} - {qal}' -> '{ta} - {tal}': No match")
    
    print("\n[SUCCESS] Fuzzy matcher tests completed!")