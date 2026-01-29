#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Analyzer for Entity Summarization
Uses Description Logic principles to analyze triple informativeness and relatedness
"""

from typing import List, Dict, Set, Tuple
from collections import Counter
import re


class SemanticAnalyzer:
    """
    Analyzes triples to extract semantic information for better summarization.
    
    Uses DL-inspired heuristics:
    - Identifies defining predicates (rdf:type, rdfs:label)
    - Calculates predicate specificity (rare = more informative)
    - Detects functional predicates (birthDate, birthPlace)
    - Analyzes value types (literals vs entities)
    """
    
    def __init__(self, all_triples: List[str]):
        """
        Initialize analyzer with entity's triples.
        
        Args:
            all_triples: List of N-Triple strings
        """
        self.all_triples = all_triples
        self.parsed_triples = [self._parse_triple(t) for t in all_triples]
        
        # Semantic categories
        self.defining_predicates = {
            'rdf:type', 'type', 'a',
            'rdfs:label', 'label', 'name',
            'foaf:name', 'dcterms:title'
        }
        
        self.functional_predicates = {
            'birthDate', 'deathDate', 'birthPlace', 'deathPlace',
            'foundingDate', 'dissolutionDate', 'foundingYear',
            'isbn', 'issn', 'doi', 'orcid',
            'latitude', 'longitude', 'elevation',
            'population', 'area', 'capital'
        }
        
        self.general_predicates = {
            'sameAs', 'seeAlso', 'primaryTopic',
            'wikiPageID', 'wikiPageRevisionID', 'wikiPageWikiLink'
        }
    
    def _parse_triple(self, triple: str) -> Dict[str, str]:
        """
        Parse N-Triple into components.
        
        Args:
            triple: N-Triple string
            
        Returns:
            Dict with subject, predicate, object
        """
        parts = triple.strip().split(None, 2)
        if len(parts) < 3:
            return {'subject': '', 'predicate': '', 'object': '', 'is_literal': False}
        
        subject = parts[0].strip('<>')
        predicate = parts[1].strip('<>')
        obj = parts[2].rstrip(' .').strip()
        
        is_literal = obj.startswith('"')
        
        # Extract predicate local name
        pred_local = predicate.split('/')[-1].split('#')[-1]
        
        return {
            'subject': subject,
            'predicate': predicate,
            'predicate_local': pred_local,
            'object': obj,
            'is_literal': is_literal
        }
    
    def get_predicate_specificity(self) -> Dict[str, float]:
        """
        Calculate predicate specificity scores.
        
        Rare predicates = more informative
        
        Returns:
            Dict mapping predicate to specificity score (0-1)
        """
        pred_counts = Counter([t['predicate_local'] for t in self.parsed_triples])
        total = len(self.parsed_triples)
        
        # Inverse frequency: rare predicates get high scores
        specificity = {}
        for pred, count in pred_counts.items():
            freq = count / total
            specificity[pred] = 1.0 - freq  # Rare = high score
        
        return specificity
    
    def get_triple_categories(self) -> Dict[int, Dict[str, bool]]:
        """
        Categorize each triple by semantic role.
        
        Returns:
            Dict mapping triple index to category flags
        """
        categories = {}
        
        for idx, parsed in enumerate(self.parsed_triples, start=1):
            pred_local = parsed['predicate_local']
            
            categories[idx] = {
                'is_defining': any(dp in pred_local.lower() for dp in self.defining_predicates),
                'is_functional': any(fp in pred_local for fp in self.functional_predicates),
                'is_general': any(gp in pred_local for gp in self.general_predicates),
                'is_literal': parsed['is_literal'],
                'is_entity_link': not parsed['is_literal'] and parsed['object'].startswith('<'),
            }
        
        return categories
    
    def get_informativeness_scores(self) -> Dict[int, float]:
        """
        Calculate informativeness score for each triple.
        
        Based on:
        - Predicate specificity (rare = informative)
        - Functional predicates (unique values = informative)
        - Non-general predicates (specific info = informative)
        
        Returns:
            Dict mapping triple index to informativeness score (0-1)
        """
        specificity = self.get_predicate_specificity()
        categories = self.get_triple_categories()
        
        scores = {}
        for idx in range(1, len(self.all_triples) + 1):
            parsed = self.parsed_triples[idx - 1]
            pred_local = parsed['predicate_local']
            cat = categories[idx]
            
            score = 0.0
            
            # Base score from specificity
            score += specificity.get(pred_local, 0.5) * 0.5
            
            # Bonus for functional predicates (unique values)
            if cat['is_functional']:
                score += 0.3
            
            # Penalty for general predicates
            if cat['is_general']:
                score -= 0.3
            
            # Bonus for entity links (more informative than literals)
            if cat['is_entity_link']:
                score += 0.2
            
            # Normalize to [0, 1]
            scores[idx] = max(0.0, min(1.0, score))
        
        return scores
    
    def get_relatedness_scores(self) -> Dict[int, float]:
        """
        Calculate relatedness score for each triple.
        
        Based on:
        - Defining predicates (type, label = core)
        - Common predicates (frequent = central)
        
        Returns:
            Dict mapping triple index to relatedness score (0-1)
        """
        specificity = self.get_predicate_specificity()
        categories = self.get_triple_categories()
        
        scores = {}
        for idx in range(1, len(self.all_triples) + 1):
            parsed = self.parsed_triples[idx - 1]
            pred_local = parsed['predicate_local']
            cat = categories[idx]
            
            score = 0.0
            
            # High score for defining predicates
            if cat['is_defining']:
                score += 0.6
            
            # Common predicates are central (inverse of specificity)
            frequency = 1.0 - specificity.get(pred_local, 0.5)
            score += frequency * 0.4
            
            scores[idx] = max(0.0, min(1.0, score))
        
        return scores
    
    def get_diversity_hints(self, selected_ids: List[int]) -> Dict[int, float]:
        """
        Get diversity scores for remaining triples.
        
        Args:
            selected_ids: Already selected triple indices
            
        Returns:
            Dict mapping triple index to diversity score (0-1)
        """
        if not selected_ids:
            # No selection yet, all equally diverse
            return {i: 0.5 for i in range(1, len(self.all_triples) + 1)}
        
        selected_preds = [self.parsed_triples[i-1]['predicate_local'] for i in selected_ids]
        selected_types = [self.get_triple_categories()[i] for i in selected_ids]
        
        scores = {}
        for idx in range(1, len(self.all_triples) + 1):
            if idx in selected_ids:
                scores[idx] = 0.0
                continue
            
            parsed = self.parsed_triples[idx - 1]
            pred_local = parsed['predicate_local']
            cat = self.get_triple_categories()[idx]
            
            score = 0.0
            
            # Different predicate = diverse
            if pred_local not in selected_preds:
                score += 0.5
            
            # Different type (literal vs entity) = diverse
            has_literal = any(t['is_literal'] for t in selected_types)
            has_entity = any(t['is_entity_link'] for t in selected_types)
            
            if cat['is_literal'] and not has_literal:
                score += 0.25
            if cat['is_entity_link'] and not has_entity:
                score += 0.25
            
            scores[idx] = max(0.0, min(1.0, score))
        
        return scores
    
    def get_enriched_triple_info(self, idx: int) -> str:
        """
        Get semantic annotations for a triple.
        
        Args:
            idx: Triple index (1-based)
            
        Returns:
            Human-readable annotation string
        """
        categories = self.get_triple_categories()
        informativeness = self.get_informativeness_scores()
        relatedness = self.get_relatedness_scores()
        
        cat = categories.get(idx, {})
        info_score = informativeness.get(idx, 0.5)
        rel_score = relatedness.get(idx, 0.5)
        
        annotations = []
        
        if cat.get('is_defining'):
            annotations.append("CORE")
        if cat.get('is_functional'):
            annotations.append("UNIQUE")
        if info_score > 0.7:
            annotations.append("HIGHLY_INFORMATIVE")
        if rel_score > 0.7:
            annotations.append("CENTRAL")
        if cat.get('is_entity_link'):
            annotations.append("ENTITY_LINK")
        
        return " | ".join(annotations) if annotations else "STANDARD"
    
    def get_summary_statistics(self) -> Dict:
        """
        Get overall statistics about the entity's triples.
        
        Returns:
            Dict with summary statistics
        """
        categories = self.get_triple_categories()
        
        return {
            'total_triples': len(self.all_triples),
            'unique_predicates': len(set(t['predicate_local'] for t in self.parsed_triples)),
            'defining_count': sum(1 for c in categories.values() if c['is_defining']),
            'functional_count': sum(1 for c in categories.values() if c['is_functional']),
            'literal_count': sum(1 for c in categories.values() if c['is_literal']),
            'entity_link_count': sum(1 for c in categories.values() if c['is_entity_link']),
        }
