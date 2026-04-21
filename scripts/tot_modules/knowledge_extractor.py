#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Extractor for LinkedMDB Entity Semantic Information

Scans LMDB dataset files to build a cache mapping entity URIs to their
semantic information (type, label, etc.).
"""

import os
import re
import glob
from typing import Dict, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

from .uri_utils import extract_uri_type, extract_uri_id, uri_to_debug_string


class EntityKnowledgeExtractor:
    """
    Builds and maintains a cache of entity semantic information from LMDB data.
    
    Cache structure:
    {
        'http://data.linkedmdb.org/resource/film/12398': {
            'type': 'Film',
            'id': '12398',
            'label': 'Men in White',
            'predicates': {'actor': 3, 'director': 1, ...},
            'is_target_entity': True  # if it's the main entity we're analyzing
        },
        'http://data.linkedmdb.org/resource/actor/29977': {
            'type': 'Actor',
            'id': '29977',
            'label': None,  # Not extracted from description files
            'predicates': {},
            'is_target_entity': False
        }
    }
    """
    
    def __init__(self, lmdb_data_path: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            lmdb_data_path: Path to LMDB data directory. Can be set later.
        """
        self.lmdb_data_path = lmdb_data_path
        self.entity_cache: Dict = {}
        self.predicate_stats: Dict[str, int] = defaultdict(int)
        self.extracted_entities: Set[str] = set()
        self.referenced_entities: Set[str] = set()
        
        if lmdb_data_path:
            self.build_cache()
    
    def build_cache(self, lmdb_data_path: Optional[str] = None) -> Dict:
        """
        Scan LMDB data directory and build entity cache.
        
        Args:
            lmdb_data_path: Path to LMDB data directory
            
        Returns:
            Entity cache dictionary
        """
        if lmdb_data_path:
            self.lmdb_data_path = lmdb_data_path
        
        if not self.lmdb_data_path:
            raise ValueError("lmdb_data_path not set")
        
        print(f"Building entity cache from {self.lmdb_data_path}...")
        
        # Find all entity description files
        entity_files = glob.glob(os.path.join(self.lmdb_data_path, "*", "*_desc.nt"))
        
        if not entity_files:
            print(f"Warning: No entity files found in {self.lmdb_data_path}")
            return self.entity_cache
        
        print(f"Found {len(entity_files)} entity description files")
        
        # Process each entity file
        for entity_file in entity_files:
            self._process_entity_file(entity_file)
        
        print(f"Cache built: {len(self.entity_cache)} entities, "
              f"{len(self.referenced_entities)} referenced entities")
        
        return self.entity_cache
    
    def _process_entity_file(self, filepath: str):
        """
        Process a single entity description file.
        
        Args:
            filepath: Path to .nt file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                triples = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return
        
        subject_uri = None
        label = None
        predicates = defaultdict(int)
        
        for triple_line in triples:
            triple_line = triple_line.strip()
            if not triple_line:
                continue
            
            # Parse triple
            subject, predicate, obj = self._parse_triple_line(triple_line)
            
            if not subject:
                continue
            
            # Track subject (should be same for all triples in file)
            if subject_uri is None:
                subject_uri = subject
                self.extracted_entities.add(subject_uri)
            
            # Extract label from common predicates
            if label is None and obj:
                if self._is_label_predicate(predicate):
                    label = self._extract_literal_value(obj)
            
            # Track predicates
            if predicate:
                pred_local = predicate.split('/')[-1].split('#')[-1]
                predicates[pred_local] += 1
                self.predicate_stats[pred_local] += 1
            
            # Track referenced entities
            if obj and obj.startswith('http://data.linkedmdb.org'):
                self.referenced_entities.add(obj)
        
        # Store entity info
        if subject_uri:
            entity_type = extract_uri_type(subject_uri) or 'Unknown'
            entity_id = extract_uri_id(subject_uri)
            
            self.entity_cache[subject_uri] = {
                'type': entity_type,
                'id': entity_id,
                'label': label,
                'predicates': dict(predicates),
                'is_target_entity': True,
                'source_file': filepath
            }
    
    def _parse_triple_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse an N-Triples line.
        
        Args:
            line: N-Triples format line
            
        Returns:
            Tuple of (subject, predicate, object) URIs or None
        """
        try:
            # N-Triples format: <subject> <predicate> <object> .
            # Object can be a URI or literal
            
            parts = line.split(None, 2)
            if len(parts) < 3:
                return None, None, None
            
            subject = parts[0].strip('<>')
            predicate = parts[1].strip('<>')
            obj_part = parts[2].rstrip(' .').strip()
            
            # Handle both URIs and literals
            if obj_part.startswith('<'):
                obj = obj_part.strip('<>')
            else:
                obj = obj_part
            
            return subject, predicate, obj
        
        except Exception:
            return None, None, None
    
    def _is_label_predicate(self, predicate: str) -> bool:
        """Check if predicate is a label/name predicate."""
        label_predicates = {
            'label', 'name', 'title',
            'rdf-schema#label',
            'foaf/0.1/name',
            'dcterms/title',
        }
        
        pred_lower = predicate.lower()
        for lp in label_predicates:
            if lp in pred_lower:
                return True
        
        return False
    
    def _extract_literal_value(self, obj_part: str) -> Optional[str]:
        """
        Extract literal value from N-Triples object.
        
        Args:
            obj_part: Object part from N-Triple
            
        Returns:
            Extracted literal string or None
            
        Examples:
            '"Men in White"@en' -> 'Men in White'
            '"Men in White"' -> 'Men in White'
            '"12398"^^<http://www.w3.org/2001/XMLSchema#int>' -> '12398'
        """
        if not obj_part.startswith('"'):
            return None
        
        # Extract between quotes
        match = re.match(r'"([^"]*)"', obj_part)
        if match:
            return match.group(1)
        
        return None
    
    def enrich_cache_with_references(self):
        """
        Fill in missing information for referenced entities.
        
        If entity A references entity B (but B's description wasn't processed),
        we can still infer B's type from its URI pattern.
        """
        print("Enriching cache with referenced entities...")
        
        for entity_uri in self.referenced_entities:
            if entity_uri not in self.entity_cache:
                entity_type = extract_uri_type(entity_uri) or 'Unknown'
                entity_id = extract_uri_id(entity_uri)
                
                self.entity_cache[entity_uri] = {
                    'type': entity_type,
                    'id': entity_id,
                    'label': None,
                    'predicates': {},
                    'is_target_entity': False,
                    'source_file': None
                }
    
    def get_entity_info(self, entity_uri: str) -> Optional[Dict]:
        """
        Get semantic information for an entity URI.
        
        Args:
            entity_uri: Full URI of entity
            
        Returns:
            Dict with entity info, or None if not found
        """
        return self.entity_cache.get(entity_uri)
    
    def get_entity_label(self, entity_uri: str) -> Optional[str]:
        """Get label for entity, with fallback to URI local name."""
        info = self.get_entity_info(entity_uri)
        if info and info['label']:
            return info['label']
        
        # Fallback to local name
        entity_id = extract_uri_id(entity_uri)
        if entity_id:
            entity_type = extract_uri_type(entity_uri) or 'Entity'
            return f"{entity_type}({entity_id})"
        
        return uri_to_debug_string(entity_uri)
    
    def get_entity_type(self, entity_uri: str) -> str:
        """Get type for entity."""
        info = self.get_entity_info(entity_uri)
        if info:
            return info['type']
        return 'Unknown'
    
    def get_predicate_frequency(self, predicate_local: str) -> int:
        """Get how many times a predicate appears in the entire dataset."""
        return self.predicate_stats.get(predicate_local, 0)
    
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        return {
            'total_entities': len(self.entity_cache),
            'extracted_entities': len(self.extracted_entities),
            'referenced_entities': len(self.referenced_entities),
            'unique_predicates': len(self.predicate_stats),
            'top_predicates': dict(
                sorted(self.predicate_stats.items(), 
                       key=lambda x: x[1], reverse=True)[:10]
            )
        }
    
    def print_statistics(self):
        """Print cache statistics."""
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("ENTITY KNOWLEDGE CACHE STATISTICS")
        print("="*70)
        print(f"Total entities in cache: {stats['total_entities']}")
        print(f"  - Extracted from files: {stats['extracted_entities']}")
        print(f"  - Referenced only: {stats['referenced_entities']}")
        print(f"Unique predicates: {stats['unique_predicates']}")
        print(f"\nTop 10 predicates:")
        for pred, count in list(stats['top_predicates'].items())[:10]:
            print(f"  {pred}: {count}")


def create_lmdb_knowledge_extractor(lmdb_data_path: str) -> EntityKnowledgeExtractor:
    """
    Factory function to create and build an entity knowledge extractor.
    
    Args:
        lmdb_data_path: Path to LMDB data directory
        
    Returns:
        EntityKnowledgeExtractor with cache built
    """
    extractor = EntityKnowledgeExtractor()
    extractor.build_cache(lmdb_data_path)
    extractor.enrich_cache_with_references()
    return extractor


if __name__ == "__main__":
    # Test cases
    import sys
    
    if len(sys.argv) > 1:
        lmdb_path = sys.argv[1]
    else:
        # Default to common locations
        possible_paths = [
            "/home/asepff/Documents/Github/dice/ToT4ES/datasets/ESBM_benchmark_v1.2/lmdb_data",
            "./datasets/ESBM_benchmark_v1.2/lmdb_data",
        ]
        lmdb_path = None
        for path in possible_paths:
            if os.path.exists(path):
                lmdb_path = path
                break
    
    if not lmdb_path:
        print("LMDB data path not found. Usage: python knowledge_extractor.py <path>")
        sys.exit(1)
    
    # Build cache
    extractor = create_lmdb_knowledge_extractor(lmdb_path)
    
    # Print stats
    extractor.print_statistics()
    
    # Test lookups
    print("\n" + "="*70)
    print("SAMPLE ENTITY LOOKUPS")
    print("="*70)
    
    test_uris = [
        "http://data.linkedmdb.org/resource/film/12398",
        "http://data.linkedmdb.org/resource/actor/29977",
        "http://data.linkedmdb.org/resource/director/82",
    ]
    
    for uri in test_uris:
        info = extractor.get_entity_info(uri)
        label = extractor.get_entity_label(uri)
        if info:
            print(f"\n{uri}")
            print(f"  Type: {info['type']}")
            print(f"  Label: {label}")
            print(f"  Predicates: {len(info['predicates'])}")
        else:
            print(f"\n{uri}: Not found in cache")
