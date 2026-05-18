#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FCA (Formal Concept Analysis) Based Heuristic Scorer

Implements lattice-based scoring using formal concepts derived from RDF triples.
Provides semantic-aware scoring without LLM evaluation.
"""

from typing import List, Dict, Set, Tuple
from collections import Counter
import numpy as np
from tot_modules.heuristic_scorers import HeuristicScorer


class FCAContext:
    """
    Formal Context: objects (triples) × attributes (predicate types, roles).
    
    Represents a binary relation between triples and their properties.
    This forms the basis for concept generation.
    """
    
    def __init__(self, all_triples: List[str]):
        """
        Build formal context from RDF triples.
        
        Args:
            all_triples: List of RDF triple strings
        """
        self.triples = all_triples
        self.attributes = self._extract_attributes()
        self.context_matrix = self._build_context_matrix()
    
    def _extract_attributes(self) -> List[str]:
        """
        Extract all unique attributes from triples.
        Attributes include: predicates, semantic roles, subject types.
        """
        attributes = set()
        
        for triple in self.triples:
            parts = triple.split(maxsplit=2)
            if len(parts) >= 2:
                predicate = parts[1]
                # Add predicate as attribute
                attributes.add(f"pred:{predicate}")
                
                # Add semantic role
                role = self._get_semantic_role(predicate)
                attributes.add(f"role:{role}")
                
                # Add predicate frequency category
                freq_cat = self._categorize_frequency(predicate)
                attributes.add(f"freq:{freq_cat}")
        
        return sorted(list(attributes))
    
    def _get_semantic_role(self, predicate: str) -> str:
        """Categorize predicate into semantic role."""
        role_patterns = {
            'location': ['place', 'location', 'birthplace', 'birthPlace'],
            'time': ['date', 'born', 'died', 'founded', 'year'],
            'relationship': ['knows', 'friend', 'spouse', 'parent', 'related'],
            'attribute': ['name', 'label', 'title', 'description', 'abstract'],
            'work': ['work', 'wrote', 'created', 'author', 'composed'],
            'organization': ['member', 'organization', 'company', 'team'],
        }
        
        pred_name = predicate.split('/')[-1].split(':')[-1].lower()
        for role, patterns in role_patterns.items():
            if any(pattern in pred_name for pattern in patterns):
                return role
        return 'other'
    
    def _categorize_frequency(self, predicate: str) -> str:
        """Categorize predicate by frequency: rare, common, or frequent."""
        freq_count = sum(1 for t in self.triples 
                        if predicate in t.split(maxsplit=2)[1] if len(t.split()) >= 2)
        total = len(self.triples)
        freq_ratio = freq_count / max(1, total)
        
        if freq_ratio < 0.1:
            return "rare"
        elif freq_ratio < 0.3:
            return "common"
        else:
            return "frequent"
    
    def _build_context_matrix(self) -> np.ndarray:
        """
        Build binary context matrix (triples × attributes).
        
        Returns:
            Binary matrix where matrix[i,j] = 1 if triple i has attribute j
        """
        n_triples = len(self.triples)
        n_attrs = len(self.attributes)
        
        matrix = np.zeros((n_triples, n_attrs), dtype=int)
        
        for i, triple in enumerate(self.triples):
            parts = triple.split(maxsplit=2)
            if len(parts) >= 2:
                predicate = parts[1]
                
                # Mark predicate attribute
                if f"pred:{predicate}" in self.attributes:
                    j = self.attributes.index(f"pred:{predicate}")
                    matrix[i, j] = 1
                
                # Mark role attribute
                role = self._get_semantic_role(predicate)
                if f"role:{role}" in self.attributes:
                    j = self.attributes.index(f"role:{role}")
                    matrix[i, j] = 1
                
                # Mark frequency attribute
                freq_cat = self._categorize_frequency(predicate)
                if f"freq:{freq_cat}" in self.attributes:
                    j = self.attributes.index(f"freq:{freq_cat}")
                    matrix[i, j] = 1
        
        return matrix
    
    def closure(self, triple_indices: Set[int]) -> Set[int]:
        """
        Compute attribute closure: common attributes of given triples.
        
        Args:
            triple_indices: Set of triple indices
            
        Returns:
            Set of attribute indices common to all triples
        """
        if not triple_indices:
            return set(range(len(self.attributes)))
        
        # Find attributes common to all triples
        common_attrs = set(range(len(self.attributes)))
        for idx in triple_indices:
            triple_attrs = set(np.where(self.context_matrix[idx, :] == 1)[0])
            common_attrs = common_attrs.intersection(triple_attrs)
        
        return common_attrs


class ConceptLattice:
    """
    Formal Concept Lattice: partially ordered set of formal concepts.
    
    Each concept = (extent, intent) where:
    - extent: set of triples with the concept's attributes
    - intent: set of attributes shared by all triples in extent
    """
    
    def __init__(self, context: FCAContext):
        """
        Generate concept lattice from formal context.
        
        Args:
            context: FCAContext object
        """
        self.context = context
        self.concepts = self._generate_concepts()
        self.lattice_graph = self._build_lattice_structure()
    
    def _generate_concepts(self) -> List[Tuple[Set[int], Set[int]]]:
        """
        Generate all formal concepts using simple algorithm.
        
        Each concept is (extent, intent) pair where:
        - extent: triple indices sharing attributes
        - intent: attributes common to extent
        
        Returns:
            List of (extent, intent) tuples
        """
        concepts = []
        
        # Generate concepts by examining all attribute subsets
        # (simplified: only generate from actual attribute combinations)
        seen_extents = set()
        
        for attr_subset in self._generate_attribute_subsets():
            extent = self._find_extent(attr_subset)
            extent_key = frozenset(extent)
            
            if extent_key not in seen_extents:
                seen_extents.add(extent_key)
                intent = self.context.closure(extent)
                concepts.append((extent, intent))
        
        return concepts
    
    def _generate_attribute_subsets(self):
        """
        Generate relevant attribute subsets.
        For efficiency, generate from single attributes and their intersections.
        """
        n_attrs = len(self.context.attributes)
        
        # Single attributes
        for attr_idx in range(n_attrs):
            yield {attr_idx}
        
        # Pairs of related attributes
        for i in range(n_attrs):
            for j in range(i + 1, min(i + 5, n_attrs)):  # Limit to efficiency
                yield {i, j}
    
    def _find_extent(self, intent: Set[int]) -> Set[int]:
        """
        Find extent: all triples having ALL attributes in intent.
        
        Args:
            intent: Set of attribute indices
            
        Returns:
            Set of triple indices forming the extent
        """
        extent = set()
        
        for i in range(len(self.context.triples)):
            triple_attrs = set(np.where(self.context.context_matrix[i, :] == 1)[0])
            if intent.issubset(triple_attrs):
                extent.add(i)
        
        return extent
    
    def _build_lattice_structure(self) -> Dict:
        """
        Build partial order relations between concepts.
        Returns adjacency info for lattice.
        """
        lattice = {}
        
        for i, (extent_i, intent_i) in enumerate(self.concepts):
            lattice[i] = {
                'extent': extent_i,
                'intent': intent_i,
                'extent_size': len(extent_i),
                'intent_size': len(intent_i),
                'parents': [],
                'children': []
            }
        
        # Build subsumption relations (A ≤ B if extent_A ⊆ extent_B)
        for i, (extent_i, intent_i) in enumerate(self.concepts):
            for j, (extent_j, intent_j) in enumerate(self.concepts):
                if i != j and extent_i.issubset(extent_j) and extent_i != extent_j:
                    # i is child of j (i has more specific attributes)
                    if j not in lattice[i]['parents']:
                        lattice[i]['parents'].append(j)
                    if i not in lattice[j]['children']:
                        lattice[j]['children'].append(i)
        
        return lattice
    
    def get_concept_for_triple(self, triple_idx: int) -> Tuple[Set[int], Set[int]]:
        """
        Find the formal concept containing a given triple.
        Returns the most specific concept (maximal intent).
        """
        best_concept = None
        best_intent_size = -1
        
        for extent, intent in self.concepts:
            if triple_idx in extent and len(intent) > best_intent_size:
                best_concept = (extent, intent)
                best_intent_size = len(intent)
        
        return best_concept if best_concept else (set(), set())


class FCAScorerImpl(HeuristicScorer):
    """
    FCA-based heuristic scorer using concept lattice.
    
    Provides semantic-aware scoring based on lattice properties:
    - Relatedness: Concept extent (centrality in lattice)
    - Informativeness: Concept intent rarity
    - Coverage: Concept diversity across lattice
    """
    
    def __init__(self, all_triples: List[str]):
        """
        Initialize FCA scorer.
        
        Args:
            all_triples: List of RDF triples
        """
        super().__init__(all_triples)
        self.context = FCAContext(all_triples)
        self.lattice = ConceptLattice(self.context)
        self._triple_to_idx = {triple: idx for idx, triple in enumerate(all_triples)}
    
    def _get_triple_idx(self, triple: str) -> int:
        """Get index of triple in all_triples."""
        return self._triple_to_idx.get(triple, -1)
    
    def score_relatedness(self, triple: str) -> float:
        """
        Relatedness via concept extent.
        
        Triple in large extent (many objects share attributes)
        → more central/core to entity
        
        Args:
            triple: RDF triple string
            
        Returns:
            Score in [0, 1]
        """
        idx = self._get_triple_idx(triple)
        if idx < 0:
            return 0.5
        
        extent, intent = self.lattice.get_concept_for_triple(idx)
        
        # Normalized extent size
        extent_score = len(extent) / max(1, len(self.all_triples))
        return min(1.0, extent_score)
    
    def score_informativeness(self, triple: str) -> float:
        """
        Informativeness via attribute rarity in concept.
        
        Triple with rare attributes (specific intent)
        → more informative
        
        Args:
            triple: RDF triple string
            
        Returns:
            Score in [0, 1]
        """
        idx = self._get_triple_idx(triple)
        if idx < 0:
            return 0.5
        
        extent, intent = self.lattice.get_concept_for_triple(idx)
        
        if not intent:
            return 0.5
        
        # Rarity of intent: fewer triples share these attributes = more informative
        # Inverse of extent relative to intent
        rarity = 1.0 - (len(extent) / max(1, len(self.all_triples)))
        
        # Also consider intent size: more specific (larger intent) = more informative
        specificity = min(1.0, len(intent) / max(1, len(self.context.attributes)))
        
        return (rarity + specificity) / 2.0
    
    def score_coverage(self, triple: str, selected_triples: List[str]) -> float:
        """
        Coverage via lattice diversity.
        
        Triple in underrepresented concept region
        → adds more diversity
        
        Args:
            triple: RDF triple string
            selected_triples: Already selected triples
            
        Returns:
            Score in [0, 1]
        """
        idx = self._get_triple_idx(triple)
        if idx < 0:
            return 0.5
        
        extent_triple, intent_triple = self.lattice.get_concept_for_triple(idx)
        
        if not selected_triples:
            return 1.0  # First triple = good diversity
        
        # Find concepts of selected triples
        selected_concepts = set()
        for s_triple in selected_triples:
            s_idx = self._get_triple_idx(s_triple)
            if s_idx >= 0:
                s_extent, s_intent = self.lattice.get_concept_for_triple(s_idx)
                selected_concepts.add(frozenset(s_extent))
        
        # Diversity = how different this triple's concept is from selected ones
        triple_concept = frozenset(extent_triple)
        
        if triple_concept in selected_concepts:
            # Same concept as already selected = low diversity
            diversity = 0.2
        else:
            # Different concept = high diversity
            diversity = 1.0 - (len(selected_triples) / (len(selected_triples) + 1))
        
        return min(1.0, diversity)
