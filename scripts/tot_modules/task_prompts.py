#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-specific prompt factories.

Rolled back to the simpler pre-enhancement prompts that work directly with
raw RDF triples.
"""

from typing import Callable, Dict, List, Optional


def _parse_selected_ids(state: str) -> List[int]:
    """Parse newline-separated triple indices from the current state."""

    if not state.strip():
        return []
    return [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]


def _format_remaining_triples(all_triples: List[str], selected_ids: List[int]) -> str:
    """Format remaining candidate triples using the raw RDF triple strings."""

    selected_set = set(selected_ids)
    lines = []
    for idx, triple in enumerate(all_triples, start=1):
        if idx not in selected_set:
            lines.append(f"{idx}. {triple}")
    return "\n".join(lines) if lines else "<no candidates>"


def make_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """Create prompt for RELATEDNESS-focused triple selection."""

    _ = predicate_frequencies, dataset_name

    def _inner(input_seq: str, state: str) -> str:
        selected_ids = _parse_selected_ids(state)
        selected_text = (
            "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            if selected_ids
            else "None yet."
        )
        candidates_text = _format_remaining_triples(all_triples, selected_ids)
        exclusion_note = (
            f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
            if selected_ids
            else ""
        )

        return f"""
You are evaluating triples for RELATEDNESS to the entity.

Entity: {entity_label}

RELATEDNESS DEFINITION:
A triple is related if it is central to the entity, defines what it is,
or is clearly more important than the alternatives.

SELECTION CRITERION:
Choose the candidate that is most central and essential to the entity.

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Select ONE triple index that is MOST RELATED/CENTRAL to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    selected_triples: List[int] = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """Create prompt for INFORMATIVENESS-focused triple selection."""

    _ = predicate_frequencies, selected_triples, dataset_name

    def _inner(input_seq: str, state: str) -> str:
        selected_ids = _parse_selected_ids(state)
        selected_text = (
            "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            if selected_ids
            else "None yet."
        )
        candidates_text = _format_remaining_triples(all_triples, selected_ids)
        exclusion_note = (
            f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
            if selected_ids
            else ""
        )

        return f"""
You are evaluating triples for INFORMATIVENESS.

Entity: {entity_label}

INFORMATIVENESS DEFINITION:
A triple is informative if it adds concrete, non-generic, and valuable information.

SELECTION CRITERION:
Choose the candidate that gives the highest information gain and is not redundant.

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Select ONE triple index that is MOST INFORMATIVE.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
    semantic_roles: dict = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """Create prompt for DIVERSITY/COVERAGE-focused triple selection."""

    _ = dataset_name

    def _get_role_summary(selected_ids: List[int]) -> str:
        if not semantic_roles:
            return "Not available in this run."

        covered_roles = {}
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                predicate = all_triples[idx - 1].split(maxsplit=2)[1]
                role = semantic_roles.get(predicate, "other")
                covered_roles[role] = covered_roles.get(role, 0) + 1

        if not covered_roles:
            return "None yet (no roles covered)"

        return ", ".join(f"{role}({count})" for role, count in sorted(covered_roles.items()))

    def _get_available_roles(selected_ids: List[int]) -> str:
        if not semantic_roles:
            return "Not available in this run."

        covered_roles = set()
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                predicate = all_triples[idx - 1].split(maxsplit=2)[1]
                covered_roles.add(semantic_roles.get(predicate, "other"))

        available_roles = set(semantic_roles.values()) - covered_roles
        return ", ".join(sorted(available_roles)) if available_roles else "All roles already covered"

    def _inner(input_seq: str, state: str) -> str:
        selected_ids = _parse_selected_ids(state)
        selected_text = (
            "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            if selected_ids
            else "None yet."
        )
        candidates_text = _format_remaining_triples(all_triples, selected_ids)
        exclusion_note = (
            f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
            if selected_ids
            else ""
        )
        coverage_text = _get_role_summary(selected_ids)
        available_roles_text = _get_available_roles(selected_ids)

        return f"""
You are evaluating triples for DIVERSITY and coverage.

Entity: {entity_label}

DIVERSITY DEFINITION:
A triple MAXIMIZES DIVERSITY if it covers a new semantic role,
a new predicate family, or a new aspect of the entity.

CURRENT STATE - Semantic roles covered:
{coverage_text}

AVAILABLE OPPORTUNITIES - Roles NOT yet covered:
{available_roles_text}

SELECTION CRITERION:
Choose the candidate with the highest diversity and coverage impact.

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Select ONE triple index that MAXIMIZES DIVERSITY and coverage.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_combined_evaluation_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """Create evaluation prompt that assesses all three criteria."""

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
You are evaluating RDF triple summaries for: {entity_label}

Rate each summary on three criteria (0.0–1.0):

1. RELATEDNESS: How central/core are the triples to the entity?
2. INFORMATIVENESS: How much unique, valuable information is provided?
3. COVERAGE/DIVERSITY: How diverse are the aspects covered?

There are {len(states)} summaries. Return JSON array:

[
  {{
    "idx": 0,
    "relatedness": 0.0,
    "informativeness": 0.0,
    "coverage": 0.0
  }},
  ...
]

Return ONLY the JSON array, no explanations.

Summaries:

{states_block}
        """.strip()

    return _inner
