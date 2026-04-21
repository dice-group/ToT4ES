#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URI Utilities for Linked Data Entity Semantic Analysis

Provides pattern matching and type extraction for LMDB LinkedMDB URIs.
"""

import re
from typing import Optional, Dict, Tuple


# LinkedMDB URI patterns and their semantic types
LMDB_URI_PATTERNS = {
    r'/film/': 'Film',
    r'/actor/': 'Actor',
    r'/director/': 'Director',
    r'/writer/': 'Writer',
    r'/producer/': 'Producer',
    r'/character/': 'Character',
    r'/genre/': 'Genre',
    r'/performance/': 'Performance',
    r'/film_art_director/': 'FilmArtDirector',
    r'/film_story_contributor/': 'FilmStoryContributor',
    r'/film_subject/': 'FilmSubject',
    r'/film_genre/': 'FilmGenre',
}

# DBpedia patterns (for future support)
DBPEDIA_URI_PATTERNS = {
    r'/resource/': 'Entity',
    r'/ontology/': 'Class',
    r'/property/': 'Property',
}

# Generic patterns
GENERIC_PATTERNS = {
    r'rdf-schema': 'RDFSProperty',
    r'rdf-syntax': 'RDFProperty',
    r'foaf': 'FOAFEntity',
}


def extract_uri_type(uri: str) -> Optional[str]:
    """
    Extract entity type from URI pattern.
    
    Args:
        uri: Full URI string
        
    Returns:
        Type string if pattern matches, None otherwise
        
    Example:
        'http://data.linkedmdb.org/resource/actor/29977' -> 'Actor'
        'http://data.linkedmdb.org/resource/film/12398' -> 'Film'
    """
    # Try LMDB patterns first
    for pattern, entity_type in LMDB_URI_PATTERNS.items():
        if re.search(pattern, uri):
            return entity_type
    
    # Try generic patterns
    for pattern, entity_type in GENERIC_PATTERNS.items():
        if re.search(pattern, uri):
            return entity_type
    
    return None


def extract_uri_id(uri: str) -> Optional[str]:
    """
    Extract numeric/string ID from URI.
    
    Args:
        uri: Full URI string
        
    Returns:
        ID component (last part after '/'), or None
        
    Example:
        'http://data.linkedmdb.org/resource/actor/29977' -> '29977'
        'http://www.w3.org/2000/01/rdf-schema#label' -> 'label'
    """
    # Remove trailing '>'
    uri = uri.rstrip('>')
    
    # Extract last component
    parts = uri.split('/')
    if parts:
        last_part = parts[-1]
        # Handle fragment identifiers (#)
        if '#' in last_part:
            return last_part.split('#')[-1]
        return last_part
    
    return None


def extract_uri_namespace(uri: str) -> Optional[str]:
    """
    Extract namespace/schema from URI.
    
    Args:
        uri: Full URI string
        
    Returns:
        Namespace prefix
        
    Example:
        'http://data.linkedmdb.org/resource/actor/29977' -> 'linkedmdb'
        'http://www.w3.org/2000/01/rdf-schema#label' -> 'rdf-schema'
    """
    # Remove protocol and www
    uri = uri.replace('http://', '').replace('https://', '')
    uri = uri.replace('www.', '')
    
    # Get domain or first meaningful part
    parts = uri.split('/')
    if parts:
        # Try to use second meaningful part
        if len(parts) > 1 and parts[0]:
            first = parts[0].split('.')[0]  # Get main domain
            if first:
                return first
    
    return None


def is_linkedmdb_uri(uri: str) -> bool:
    """Check if URI is from LinkedMDB."""
    return 'linkedmdb.org' in uri


def is_dbpedia_uri(uri: str) -> bool:
    """Check if URI is from DBpedia."""
    return 'dbpedia.org' in uri


def categorize_uri(uri: str) -> Dict[str, str]:
    """
    Fully categorize a URI with all available information.
    
    Returns:
        Dict with 'type', 'id', 'namespace', 'source'
    """
    result = {
        'type': extract_uri_type(uri) or 'Unknown',
        'id': extract_uri_id(uri) or None,
        'namespace': extract_uri_namespace(uri) or None,
        'source': None,
    }
    
    if is_linkedmdb_uri(uri):
        result['source'] = 'LinkedMDB'
    elif is_dbpedia_uri(uri):
        result['source'] = 'DBpedia'
    else:
        result['source'] = 'Other'
    
    return result


def get_localname(uri: str) -> str:
    """
    Get the local name (last component) of a URI.
    
    Args:
        uri: Full URI string
        
    Returns:
        Local name without namespace
        
    Example:
        'http://www.w3.org/2000/01/rdf-schema#label' -> 'label'
        'http://data.linkedmdb.org/resource/film/12398' -> '12398'
    """
    uri = uri.rstrip('>')
    
    # Split by # first (fragment identifier takes precedence)
    if '#' in uri:
        return uri.split('#')[-1]
    
    # Then split by /
    parts = uri.split('/')
    if parts:
        return parts[-1]
    
    return uri


def uri_to_debug_string(uri: str) -> str:
    """
    Convert URI to human-readable debug string.
    
    Example:
        'http://data.linkedmdb.org/resource/film/12398' -> 'Film(12398)'
    """
    cat = categorize_uri(uri)
    if cat['id']:
        return f"{cat['type']}({cat['id']})"
    return cat['type']


if __name__ == "__main__":
    # Test cases
    test_uris = [
        'http://data.linkedmdb.org/resource/film/12398',
        'http://data.linkedmdb.org/resource/actor/29977',
        'http://data.linkedmdb.org/resource/director/82',
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://xmlns.com/foaf/0.1/name',
    ]
    
    print("URI Type Extraction Tests:")
    print("=" * 70)
    for uri in test_uris:
        cat = categorize_uri(uri)
        debug_str = uri_to_debug_string(uri)
        print(f"\nURI: {uri}")
        print(f"  Type: {cat['type']}")
        print(f"  ID: {cat['id']}")
        print(f"  Namespace: {cat['namespace']}")
        print(f"  Source: {cat['source']}")
        print(f"  Debug: {debug_str}")
