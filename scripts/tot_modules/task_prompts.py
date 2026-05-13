#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Specific Prompt Factories
Implements decomposed prompts for Relatedness, Informativeness, and Diversity
"""

from typing import Callable, List, Optional


def make_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    dataset_name: Optional[str] = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for RELATEDNESS-focused triple selection with explicit criteria.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        predicate_frequencies: Dict mapping predicates to their occurrence count
                              (used to identify core/central predicates)
        
    Returns:
        Function that generates relatedness-focused prompts
    """
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            return parts[1]
        return ""
    
    def _get_core_predicates(top_n: int = 8) -> str:
        """Get most common/core predicates from corpus statistics."""
        if not predicate_frequencies:
            return "Not available in this run."
        
        sorted_preds = sorted(
            predicate_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        core = [p for p, _ in sorted_preds[:top_n]]
        return ", ".join(core) if core else "Analysis shows no clear core predicates"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        core_preds_text = _get_core_predicates()

        return f"""
You are evaluating triples for RELATEDNESS to the entity.

Entity: {entity_label}

RELATEDNESS DEFINITION:
A triple is RELATED/CENTRAL if it:
1. CENTRALITY: Uses core/frequent predicates that define entity types (identity, classification, basic properties)
2. SPECIFICITY: Provides distinctive values NOT generic descriptions
3. ESSENTIALITY: Best answers "What fundamentally IS this entity?"

DOMAIN CONTEXT - Core/frequent predicates (entity-defining):
{core_preds_text}

SELECTION CRITERION:
Choose the candidate combining: Centrality (frequent predicate?) + Specificity (distinctive value?) + Essentiality (defines the entity?)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

For each candidate, briefly evaluate: predicate frequency + value specificity + whether it defines the entity.
Select ONE triple index that is MOST RELATED/CENTRAL to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    selected_triples: List[int] = None,
    dataset_name: Optional[str] = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for INFORMATIVENESS-focused triple selection with explicit criteria.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        predicate_frequencies: Dict mapping predicates to their occurrence count in corpus
                              (used to identify rare predicates)
        selected_triples: List of indices already selected (for topic coverage tracking)
        
    Returns:
        Function that generates informativeness-focused prompts
    """
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            return parts[1]
        return ""
    
    def _get_selected_predicates(selected_ids: List[int]) -> set:
        """Get set of predicates already selected."""
        predicates = set()
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred = _extract_predicate(all_triples[idx - 1])
                predicates.add(pred)
        return predicates
    
    def _get_rare_predicates(top_n: int = 8) -> str:
        """Get rare predicates from corpus statistics."""
        if not predicate_frequencies:
            return "Not available in this run."
        
        sorted_preds = sorted(
            predicate_frequencies.items(),
            key=lambda x: x[1]
        )
        rare = [p for p, _ in sorted_preds[:top_n]]
        return ", ".join(rare) if rare else "Analysis shows no clear rare predicates"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        
        covered_predicates = _get_selected_predicates(selected_ids)
        covered_predicates_text = ", ".join(covered_predicates) if covered_predicates else "None yet"
        rare_preds_text = _get_rare_predicates()

        return f"""
You are evaluating triples for INFORMATIVENESS.

Entity: {entity_label}

INFORMATIVENESS DEFINITION:
A triple is informative if it combines:
1. RARITY: Uses uncommon predicates (not generic like rdf:type, rdfs:label, or rdf:comment)
2. NOVELTY: Introduces predicates NOT already in your selection
3. SPECIFICITY: Provides concrete, detailed values (not generic categories/descriptions)

DOMAIN CONTEXT - Rare predicates in this dataset:
{rare_preds_text}

CURRENT STATE - Already selected predicates:
{covered_predicates_text}

SELECTION CRITERION:
Choose the candidate with highest information gain:
- PredicateRarity (is it uncommon?) + TopicNovelty (covers new predicate?) + Specificity (concrete value?)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

For each candidate, briefly evaluate: predicate rarity + topic coverage + value specificity.
Select ONE triple index that is MOST INFORMATIVE (best combines rarity + novelty + specificity).{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
    semantic_roles: dict = None,
    dataset_name: Optional[str] = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for DIVERSITY/COVERAGE-focused triple selection with explicit coverage analysis.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        semantic_roles: Dict mapping predicates to semantic role categories
                        (e.g., {"dbpedia:birthPlace": "location", "dbo:birthDate": "time"})
        
    Returns:
        Function that generates diversity-focused prompts
    """
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            return parts[1]
        return ""
    
    def _get_semantic_roles_coverage(selected_ids: List[int]) -> str:
        """Analyze coverage of semantic roles in selection."""
        if not semantic_roles:
            return "Not available in this run."
        
        covered_roles = {}
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred = _extract_predicate(all_triples[idx - 1])
                role = semantic_roles.get(pred, "other")
                covered_roles[role] = covered_roles.get(role, 0) + 1
        
        if not covered_roles:
            return "None yet (no roles covered)"
        
        return ", ".join([f"{role}({count})" for role, count in sorted(covered_roles.items())])
    
    def _get_available_roles(selected_ids: List[int]) -> str:
        """Get semantic roles NOT yet covered in candidates."""
        if not semantic_roles:
            return "Not available in this run."
        
        covered_roles = set()
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred = _extract_predicate(all_triples[idx - 1])
                role = semantic_roles.get(pred, "other")
                covered_roles.add(role)
        
        all_roles = set(semantic_roles.values())
        available_roles = all_roles - covered_roles
        
        return ", ".join(sorted(available_roles)) if available_roles else "All roles already covered"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        coverage_text = _get_semantic_roles_coverage(selected_ids)
        available_roles_text = _get_available_roles(selected_ids)

        return f"""
You are evaluating triples for DIVERSITY and coverage.

Entity: {entity_label}

DIVERSITY DEFINITION:
A triple MAXIMIZES DIVERSITY if it:
1. ROLE VARIETY: Covers semantic roles NOT yet represented (location, time, relationship, attribute, etc.)
2. PREDICATE NOVELTY: Uses predicates different from already selected (no redundant predicates)
3. PERSPECTIVE BREADTH: Views entity from different aspects (not just repeating the same type of info)

CURRENT STATE - Semantic roles covered:
{coverage_text}

AVAILABLE OPPORTUNITIES - Roles NOT yet covered:
{available_roles_text}

SELECTION CRITERION:
Choose the candidate with highest diversity impact:
- RoleNovelty (covers missing semantic role?) + PredicateNovelty (different predicate?) + PerspectiveBreadth (new aspect?)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

For each candidate, briefly evaluate: semantic role novelty + predicate distinctness + perspective breadth.
Select ONE triple index that MAXIMIZES DIVERSITY and coverage.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_combined_evaluation_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """
    Create evaluation prompt that assesses all three criteria.
    Enhanced with flexible format that works for any number of states.
    """
    
    def _inner(input_seq: str, states: List[str]) -> str:
        formatted_states = []
        n_triples = len(all_triples)

        for idx, state in enumerate(states):
            triple_ids: List[int] = []
            if state.strip():
                triple_ids = [
                    int(x) for x in state.strip().splitlines() if x.strip().isdigit()
                ]

            triple_ids = [tid for tid in triple_ids if 1 <= tid <= n_triples]

            if triple_ids:
                triples_txt = "\n".join(
                    f"- {tid}. {all_triples[tid - 1]}" for tid in triple_ids
                )
            else:
                triples_txt = "(empty summary)"

            formatted_states.append(f"SUMMARY {idx}:\n{triples_txt}")

        states_block = "\n\n".join(formatted_states)
        n_states = len(states)

        return f"""EVALUATE: {entity_label}

RATE each on 3 criteria (0.0-1.0):
1. RELATEDNESS: How central/defining to entity?
2. INFORMATIVENESS: How much unique valuable info?
3. COVERAGE: How diverse are the aspects?

SUMMARIES TO EVALUATE ({n_states} total):

{states_block}

RESPONSE FORMAT:
Output ONLY valid JSON array with {n_states} objects.
Each object: {{"idx": N, "relatedness": X, "informativeness": Y, "coverage": Z}}
WHERE: N is 0 to {n_states-1}, X/Y/Z are decimals 0.0-1.0

EXAMPLE FORMAT (for reference):
[
{{"idx": 0, "relatedness": 0.9, "informativeness": 0.8, "coverage": 0.7}},
{{"idx": 1, "relatedness": 0.7, "informativeness": 0.9, "coverage": 0.65}}
]

CRITICAL:
- ONLY JSON output. NOTHING else before/after.
- EXACTLY {n_states} objects (indices 0 to {n_states-1}).
- All values between 0.0 and 1.0 (decimals, not integers).
- NO text, NO explanations, NO markdown.
- Start with '[', end with ']'.
- Valid JSON syntax required.

BEGIN RESPONSE WITH '[':""".strip()

    return _inner
