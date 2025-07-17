import re
from typing import Tuple, Optional


def clean_search_query(query: str) -> str:
    """
    Clean a search query by removing platform-specific suffixes and unnecessary text.
    
    Args:
        query: Raw search query that may contain platform suffixes like "(Spotify)", "(Apple Music)", etc.
        
    Returns:
        Cleaned search query with just the artist and album name
    """
    if not query:
        return ""
    
    # Remove platform-specific suffixes
    platform_patterns = [
        r'\s*\(Spotify\)$',
        r'\s*\(Apple Music\)$',
        r'\s*\(YouTube Music\)$',
        r'\s*\(Amazon Music\)$',
        r'\s*\(Tidal\)$',
        r'\s*\(Deezer\)$',
        r'\s*\(Pandora\)$',
        r'\s*\[Spotify\]$',
        r'\s*\[Apple Music\]$',
        r'\s*\[YouTube Music\]$',
        r'\s*\[Amazon Music\]$',
        r'\s*\[Tidal\]$',
        r'\s*\[Deezer\]$',
        r'\s*\[Pandora\]$',
    ]
    
    cleaned_query = query.strip()
    
    # Remove platform suffixes
    for pattern in platform_patterns:
        cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
    
    # Remove other common suffixes that don't help with search
    other_patterns = [
        r'\s*\(Explicit\)$',
        r'\s*\(Clean\)$',
        r'\s*\(Remastered\)$',
        r'\s*\(Deluxe Edition\)$',
        r'\s*\(Deluxe\)$',
        r'\s*\(Extended\)$',
        r'\s*\(Expanded Edition\)$',
        r'\s*\(Anniversary Edition\)$',
        r'\s*\[Explicit\]$',
        r'\s*\[Clean\]$',
        r'\s*\[Remastered\]$',
        r'\s*\[Deluxe Edition\]$',
        r'\s*\[Deluxe\]$',
        r'\s*\[Extended\]$',
        r'\s*\[Expanded Edition\]$',
        r'\s*\[Anniversary Edition\]$',
    ]
    
    for pattern in other_patterns:
        cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    return cleaned_query


def extract_artist_and_album(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract separate artist and album names from a search query.
    
    This function tries to intelligently split a query like "Artist Name Album Name"
    into separate artist and album components.
    
    Args:
        query: Search query containing artist and album name
        
    Returns:
        Tuple of (artist_name, album_name) or (None, None) if extraction fails
    """
    cleaned_query = clean_search_query(query)
    
    if not cleaned_query:
        return None, None
    
    # Common patterns to split artist and album
    # This is a heuristic approach and may not work for all cases
    
    # Try to detect if there are quotes around album name
    quote_match = re.search(r'^(.+?)\s*["\'""](.+?)["\'""]', cleaned_query)
    if quote_match:
        artist = quote_match.group(1).strip()
        album = quote_match.group(2).strip()
        return artist, album
    
    # Try to detect common separators
    separators = [' - ', ' by ', ' from ']
    for separator in separators:
        if separator in cleaned_query:
            parts = cleaned_query.split(separator, 1)
            if len(parts) == 2:
                # Assume format is "Album - Artist" or "Artist - Album"
                # Check which is more likely the artist (shorter, more common words)
                part1, part2 = parts[0].strip(), parts[1].strip()
                
                # Simple heuristic: if part1 looks like an artist name (shorter, common words)
                if len(part1.split()) <= 3 and len(part2.split()) >= 2:
                    return part1, part2  # Artist - Album
                else:
                    return part2, part1  # Album - Artist
    
    # If no clear separators, try to split based on common patterns
    # This is very heuristic and may not work well for all cases
    words = cleaned_query.split()
    if len(words) >= 3:
        # Try different split points
        for i in range(1, len(words)):
            potential_artist = ' '.join(words[:i])
            potential_album = ' '.join(words[i:])
            
            # Simple heuristic: prefer shorter artist names
            if len(potential_artist.split()) <= 3:
                return potential_artist, potential_album
    
    # If we can't split intelligently, return the whole query as album with no artist
    return None, cleaned_query


def build_search_query(artist: str = None, album: str = None) -> str:
    """
    Build a clean search query from artist and album components.
    
    Args:
        artist: Artist name
        album: Album name
        
    Returns:
        Clean search query string
    """
    parts = []
    
    if artist and artist.strip():
        parts.append(artist.strip())
    
    if album and album.strip():
        parts.append(album.strip())
    
    return ' '.join(parts) 