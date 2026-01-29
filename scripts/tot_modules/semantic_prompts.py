#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantically-Enhanced Task Prompts
Uses semantic analysis to guide triple selection with DL principles
"""

from typing import Callable, List
from .semantic_analyzer import SemanticAnalyzer


def make_semantic_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create SEMANTICALLY-ENHANCED relatedness prompt.
    
    Uses DL principles to identify core/defining triples.
    """
    analyzer = SemanticAnalyzer(all_triples)
    relatedness_scores = analyzer.get_relatedness_scores()
    categories = analyzer.get_triple_categories()
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        # Annotate candidates with semantic info
        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                rel_score = relatedness_scores.get(idx, 0.5)
                cat = categories.get(idx, {})
                
                # Add semantic hints
                hints = []
                if cat.get('is_defining'):
                    hints.append("⭐DEFINING")
                if rel_score > 0.7:
                    hints.append("CENTRAL")
                
                hint_str = f" [{', '.join(hints)}]" if hints else ""
                candidate_lines.append(f"{idx}. {triple}{hint_str}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are a SEMANTIC RELATEDNESS expert for Knowledge Graph entity summarization.

Entity: {entity_label}

Your goal: Select the triple that is MOST CENTRAL and DEFINING for this entity.

Semantic Guidance:
- DEFINING triples (rdf:type, rdfs:label, name) = highest priority
- CENTRAL triples = frequently used predicates that characterize the entity type
- SPECIFIC values = concrete, identifying information

Prioritize:
1. ⭐DEFINING predicates that establish what the entity IS
2. Common predicates typical for this entity type
3. Values that uniquely identify this specific entity
4. Properties that answer "What is the essence of this entity?"

Avoid:
- Generic linking predicates (sameAs, seeAlso)
- Wikipedia metadata (wikiPageID, etc.)
- Overly rare predicates that may be accidental

Already selected:
{selected_text}

Remaining candidates (with semantic hints):
{candidates_text}

Task: Select ONE triple index that is MOST RELATED/CENTRAL to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_semantic_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create SEMANTICALLY-ENHANCED informativeness prompt.
    
    Uses predicate specificity and functional properties.
    """
    analyzer = SemanticAnalyzer(all_triples)
    informativeness_scores = analyzer.get_informativeness_scores()
    categories = analyzer.get_triple_categories()
    
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
                info_score = informativeness_scores.get(idx, 0.5)
                cat = categories.get(idx, {})
                
                hints = []
                if cat.get('is_functional'):
                    hints.append("🔑UNIQUE")
                if info_score > 0.7:
                    hints.append("HIGHLY_INFORMATIVE")
                if cat.get('is_entity_link'):
                    hints.append("ENTITY_LINK")
                
                hint_str = f" [{', '.join(hints)}]" if hints else ""
                candidate_lines.append(f"{idx}. {triple}{hint_str}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are a SEMANTIC INFORMATIVENESS expert for entity summarization.

Entity: {entity_label}

Your goal: Select the triple with MAXIMUM INFORMATIVE VALUE (specific, not general).

Semantic Guidance:
- 🔑UNIQUE values = functional properties (birthDate, birthPlace) provide distinctive info
- RARE predicates = less common properties are more informative
- ENTITY_LINK = connections to other entities add knowledge
- SPECIFIC details = concrete facts, not generic statements

Prioritize:
1. Rare predicates that are uncommon across entities
2. Functional properties with unique values (dates, places, IDs)
3. Entity links that connect to other informative resources
4. Specific factual information that distinguishes THIS entity

Avoid:
- Very common predicates everyone has
- Generic category assignments without specific details
- Redundant information already implied by other triples
- Wikipedia/DBpedia metadata

Already selected:
{selected_text}

Remaining candidates (with semantic hints):
{candidates_text}

Task: Select ONE triple index that is MOST INFORMATIVE (specific and distinctive).{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_semantic_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create SEMANTICALLY-ENHANCED diversity prompt.
    
    Uses predicate variety and type diversity.
    """
    analyzer = SemanticAnalyzer(all_triples)
    categories = analyzer.get_triple_categories()
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)
        
        # Get diversity hints
        diversity_scores = analyzer.get_diversity_hints(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                div_score = diversity_scores.get(idx, 0.5)
                cat = categories.get(idx, {})
                
                hints = []
                if div_score > 0.7:
                    hints.append("🌈DIVERSE")
                if cat.get('is_literal'):
                    hints.append("LITERAL")
                if cat.get('is_entity_link'):
                    hints.append("ENTITY")
                
                hint_str = f" [{', '.join(hints)}]" if hints else ""
                candidate_lines.append(f"{idx}. {triple}{hint_str}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are a SEMANTIC DIVERSITY expert for entity summarization.

Entity: {entity_label}

Your goal: Select a triple that MAXIMIZES DIVERSITY (covers different aspects).

Semantic Guidance:
- 🌈DIVERSE = different predicate from those already selected
- COMPLEMENTARY = adds new type of information (if selected has literals, add entity links, and vice versa)
- MULTI-ASPECT = cover different facets (biographical, professional, geographical, etc.)

Prioritize:
1. Different predicate types from those already selected
2. Balance between literals (concrete values) and entity links (relationships)
3. Cover multiple aspects: identity, location, time, relationships, attributes
4. Avoid redundancy with already selected information

Avoid:
- Same predicates as already selected
- Similar information type (if already have birthDate, avoid deathDate in same step)
- Narrow focus on one aspect only

Already selected:
{selected_text}

Remaining candidates (with semantic hints):
{candidates_text}

Task: Select ONE triple index that MAXIMIZES DIVERSITY (most different from selected).{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner
