#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Specific Prompt Factories
Implements decomposed prompts for Relatedness, Informativeness, and Diversity
"""

from typing import Callable, List


def make_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for RELATEDNESS-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates relatedness-focused prompts
    """
    
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

        return f"""
You are a RELATEDNESS expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that is MOST RELATED/CENTRAL to this entity.

Focus on:
1. Core predicates that define the entity's identity (e.g., rdf:type, rdfs:label)
2. Properties frequently used to describe this type of entity
3. Values that are highly specific and central to this entity
4. Triples that best answer "What is this entity?"

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that is MOST RELATED to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for INFORMATIVENESS-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates informativeness-focused prompts
    """
    
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

        return f"""
You are an INFORMATIVENESS expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that provides the MOST INFORMATIVE content.

Focus on:
1. Rare/uncommon predicates (not generic like rdf:type)
2. Specific, detailed values (not generic categories)
3. Facts that provide unique, non-obvious information
4. Triples with deep ontological specificity
5. Information NOT already covered by selected triples

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that is MOST INFORMATIVE.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for DIVERSITY/COVERAGE-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates diversity-focused prompts
    """
    
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

        return f"""
You are a DIVERSITY/COVERAGE expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that MAXIMIZES DIVERSITY and coverage.

Focus on:
1. Different predicate types than already selected
2. Different semantic roles (location, time, relationship, attribute, etc.)
3. Values that are dissimilar to already selected values
4. Covering different aspects of the entity (biography, work, relations, etc.)
5. Avoiding redundancy with existing selections

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that MAXIMIZES DIVERSITY.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_combined_evaluation_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """
    Create evaluation prompt that assesses all three criteria.
    Uses simpler format for better compatibility with smaller models.
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

        return f"""
You are evaluating RDF triple summaries for entity: {entity_label}

Rate each summary on 3 criteria (scale 0.0 to 1.0):
- RELATEDNESS (R): How central are triples to entity identity?
- INFORMATIVENESS (I): How unique/valuable is the information?
- COVERAGE (C): How diverse are the aspects covered?

IMPORTANT: Output EXACTLY one line per summary in this format:
SUMMARY_0: R=0.8 I=0.7 C=0.9

Below are {len(states)} summaries to rate:

{states_block}

Now output your ratings (one line per summary):
        """.strip()

    return _inner
