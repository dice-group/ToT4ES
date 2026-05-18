#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heuristic Scorers for Entity Summarization

Provides alternative scoring methods when LLM-based evaluation is disabled.
Used for ablation study: measuring contribution of LLM scoring vs heuristics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


class HeuristicScorer(ABC):
    """
    Base class for heuristic-based scoring methods.
    
    Alternative to LLM-based evaluation for measuring:
    - Relatedness (R): Core entity identity relevance
    - Informativeness (I): Specificity and information gain
    - Coverage (C): Thematic diversity
    """
    
    def __init__(self, all_triples: List[str]):
        """
        Initialize scorer with entity's all triples.
        
        Args:
            all_triples: List of RDF triple strings from entity description
        """
        self.all_triples = all_triples
        self.n_triples = len(all_triples)
        self._precompute_statistics()
    
    def _precompute_statistics(self):
        """Precompute statistics about triples for scoring."""
        # Extract predicates and count frequencies
        self.predicates = []
        self.predicate_freqs = Counter()
        
        for triple in self.all_triples:
            parts = triple.split(maxsplit=2)
            if len(parts) >= 2:
                predicate = parts[1]
                self.predicates.append(predicate)
                self.predicate_freqs[predicate] += 1
        
        # Extract subjects (objects)
        self.subjects = []
        self.subject_freqs = Counter()
        
        for triple in self.all_triples:
            parts = triple.split(maxsplit=1)
            if len(parts) >= 1:
                subject = parts[0]
                self.subjects.append(subject)
                self.subject_freqs[subject] += 1
        
        # Compute semantic roles for triples
        self.semantic_roles = self._categorize_semantic_roles()
        self.role_freqs = Counter(self.semantic_roles.values())
    
    def _categorize_semantic_roles(self) -> Dict[str, str]:
        """
        Categorize predicates by semantic roles.
        Used for informativeness and coverage analysis.
        """
        role_patterns = {
            'location': ['place', 'location', 'birthplace', 'birthPlace', 
                        'deathplace', 'deathPlace', 'hometown'],
            'time': ['date', 'born', 'died', 'founded', 'year', 'birthdate', 
                    'birthDate', 'deathdate', 'deathDate'],
            'relationship': ['knows', 'friend', 'spouse', 'parent', 'child', 
                           'sibling', 'related', 'ownedBy', 'owns', 'associate'],
            'attribute': ['name', 'label', 'title', 'description', 'comment', 
                         'type', 'class', 'category', 'abstract'],
            'work': ['work', 'wrote', 'created', 'author', 'composed', 'directed', 
                    'produced', 'album', 'book', 'film'],
            'organization': ['member', 'organization', 'company', 'institution', 
                           'team', 'group', 'affiliation'],
        }
        
        semantic_roles = {}
        for triple in self.all_triples:
            parts = triple.split(maxsplit=2)
            if len(parts) >= 2:
                predicate = parts[1]
                if predicate not in semantic_roles:
                    pred_name = predicate.split('/')[-1].split(':')[-1].lower()
                    assigned_role = 'other'
                    for role, patterns in role_patterns.items():
                        if any(pattern in pred_name for pattern in patterns):
                            assigned_role = role
                            break
                    semantic_roles[predicate] = assigned_role
        
        return semantic_roles
    
    @abstractmethod
    def score_relatedness(self, triple: str) -> float:
        """
        Score relatedness: how central/core to entity identity.
        
        Args:
            triple: RDF triple string
            
        Returns:
            Score in [0, 1]
        """
        pass
    
    @abstractmethod
    def score_informativeness(self, triple: str) -> float:
        """
        Score informativeness: specificity and novelty.
        
        Args:
            triple: RDF triple string
            
        Returns:
            Score in [0, 1]
        """
        pass
    
    @abstractmethod
    def score_coverage(self, triple: str, selected_triples: List[str]) -> float:
        """
        Score coverage/diversity: how much new semantic space this adds.
        
        Args:
            triple: RDF triple string
            selected_triples: Triples already selected in summary
            
        Returns:
            Score in [0, 1]
        """
        pass


class TFIDFScorer(HeuristicScorer):
    """
    TF-IDF + Graph Centrality based scoring.
    
    Baseline heuristic approach:
    - Relatedness: Subject frequency (centrality)
    - Informativeness: IDF of predicate (rarity)
    - Coverage: Predicate diversity in selected
    """
    
    def score_relatedness(self, triple: str) -> float:
        """
        Relatedness via subject frequency.
        More frequent subjects = more central to entity.
        """
        parts = triple.split(maxsplit=1)
        if len(parts) < 1:
            return 0.0
        
        subject = parts[0]
        freq = self.subject_freqs.get(subject, 0)
        return min(1.0, freq / max(1, max(self.subject_freqs.values())))
    
    def score_informativeness(self, triple: str) -> float:
        """
        Informativeness via predicate IDF.
        Rare predicates = more informative.
        """
        parts = triple.split(maxsplit=2)
        if len(parts) < 2:
            return 0.5
        
        predicate = parts[1]
        freq = self.predicate_freqs.get(predicate, 1)
        
        # IDF-like: 1 - (frequency / max_frequency)
        idf = 1.0 - (freq / max(1, max(self.predicate_freqs.values())))
        return min(1.0, idf)
    
    def score_coverage(self, triple: str, selected_triples: List[str]) -> float:
        """
        Coverage via semantic role diversity.
        Penalize selecting multiple triples with same role.
        """
        parts = triple.split(maxsplit=2)
        if len(parts) < 2:
            return 0.5
        
        predicate = parts[1]
        triple_role = self.semantic_roles.get(predicate, 'other')
        
        # Count roles already selected
        selected_roles = Counter()
        for s_triple in selected_triples:
            s_parts = s_triple.split(maxsplit=2)
            if len(s_parts) >= 2:
                s_pred = s_parts[1]
                s_role = self.semantic_roles.get(s_pred, 'other')
                selected_roles[s_role] += 1
        
        # Penalize if role already well-represented
        role_count = selected_roles.get(triple_role, 0)
        return max(0.0, 1.0 - (role_count / max(1, len(self.semantic_roles))))


class RandomScorer(HeuristicScorer):
    """
    Random baseline scorer.
    
    All scores are random to test:
    - Contribution of scoring vs random selection
    - Lower bound of heuristic approaches
    """
    
    def score_relatedness(self, triple: str) -> float:
        """Random score."""
        return np.random.uniform(0.0, 1.0)
    
    def score_informativeness(self, triple: str) -> float:
        """Random score."""
        return np.random.uniform(0.0, 1.0)
    
    def score_coverage(self, triple: str, selected_triples: List[str]) -> float:
        """Random score."""
        return np.random.uniform(0.0, 1.0)


def create_heuristic_scorer(method: str, all_triples: List[str]) -> HeuristicScorer:
    """
    Factory function to create appropriate heuristic scorer.
    
    Args:
        method: Scorer type ('tfidf', 'random', 'fca')
        all_triples: List of RDF triples
        
    Returns:
        HeuristicScorer instance
    """
    if method == 'tfidf':
        return TFIDFScorer(all_triples)
    elif method == 'random':
        return RandomScorer(all_triples)
    elif method == 'fca':
        from tot_modules.fca_heuristic_scorer import FCAScorerImpl
        return FCAScorerImpl(all_triples)
    else:
        raise ValueError(f"Unknown heuristic method: {method}")
