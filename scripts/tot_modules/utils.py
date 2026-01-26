#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Functions
"""

import re
import os
from typing import List, Optional, Tuple


def extract_first_int(text: str) -> Optional[int]:
    """
    Extract the first integer occurrence in text.
    
    Args:
        text: Input string
        
    Returns:
        First integer found, or None if no integer exists
    """
    m = re.search(r"\d+", text)
    if m:
        return int(m.group(0))
    return None


def decode_state_to_triples(state: str, all_triples: List[str]) -> List[str]:
    """
    Convert a state string (e.g., "2\n5\n9") into corresponding triples.
    
    Args:
        state: Newline-separated triple indices
        all_triples: Complete list of available triples
        
    Returns:
        List of selected triple strings
    """
    if not state.strip():
        return []
    ids = [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]
    return [all_triples[i - 1] for i in ids if 1 <= i <= len(all_triples)]


def load_entity_description_from_nt(nt_path: str) -> Tuple[str, List[str]]:
    """
    Load an entity description from an N-Triples file.
    
    Args:
        nt_path: Path to .nt file
        
    Returns:
        Tuple of (entity_label, all_triples)
        - entity_label: Human-readable label (from rdfs:label/foaf:name) or subject URI
        - all_triples: List of raw triple strings
        
    Raises:
        ValueError: If no triples found in file
    """
    triples: List[str] = []
    entity_label: Optional[str] = None
    subject_uri: Optional[str] = None

    # Simple literal pattern: "label"@en . or "label" .
    literal_pattern = re.compile(r'"(.*?)"(?:@[a-zA-Z\-]+|\^\^<[^>]+>)?\s*\.\s*$')

    with open(nt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(line)

            # Extract subject URI if not set
            if subject_uri is None:
                parts = line.split()
                if parts:
                    subject_uri = parts[0].strip("<>")

            # Try to get a label from rdfs:label or foaf:name
            if entity_label is None:
                if ("rdf-schema#label" in line) or ("/label>" in line) or ("foaf/0.1/name" in line):
                    m = literal_pattern.search(line)
                    if m:
                        entity_label = m.group(1)

    # Fallback: use subject URI as label
    if entity_label is None:
        if subject_uri is not None:
            entity_label = subject_uri
        else:
            entity_label = os.path.basename(nt_path)

    if not triples:
        raise ValueError(f"No triples found in {nt_path}")

    return entity_label, triples
